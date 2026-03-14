#!/usr/bin/env python3
"""
BF16 -> FP8 Block-wise Quantization Script for Kimi-K2.5

Strategy:
- Quantize ALL linear proj weights to FP8 E4M3FN with 128x128 block-wise scales.
- Quantized: gate_proj, up_proj, down_proj (dense layer, MoE experts, shared_experts)
- NOT quantized: attn weights, layernorm, embed_tokens, lm_head, vision_tower, mm_projector
- Scale format: weight_scale_inv[i,j] = amax_of_block(i,j) / 448.0  (float32)
- Output per shard: .safetensors with FP8 weights + float32 weight_scale_inv tensors
- config.json: add quantization_config with correct ignored_layers
"""

import os
import sys
import json
import shutil
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import safetensors.torch as st

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
BLOCK_SIZE = 128
FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
FP8_DTYPE = torch.float8_e4m3fn

# Names of weight suffixes that should be quantized.
# We match by checking if the key ends with one of these suffixes.
QUANTIZE_SUFFIXES = (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")

# Names patterns to SKIP (not quantize) regardless.
SKIP_PATTERNS = (
    "self_attn.",       # all attention weights
    "layernorm.",       # layernorms
    "input_layernorm",
    "post_attention_layernorm",
    "embed_tokens",     # token embedding
    "lm_head",          # language model head
    "vision_tower",     # vision encoder (kimi vit)
    "mm_projector",     # multimodal projector
    ".bias",            # biases
    "norm.weight",      # root norm
)


def should_quantize(name: str) -> bool:
    """Return True if this weight tensor should be FP8 quantized."""
    # Must end with a quantizable suffix
    if not any(name.endswith(s) for s in QUANTIZE_SUFFIXES):
        return False
    # Must not match any skip pattern
    if any(p in name for p in SKIP_PATTERNS):
        return False
    return True


def quantize_weight_fp8_block(weight_bf16: torch.Tensor):
    """
    Quantize a 2D weight tensor [out_dim, in_dim] to FP8 with 128x128 block-wise scales.

    Returns:
        weight_fp8:       torch.float8_e4m3fn, shape [out_dim, in_dim]
        weight_scale_inv: torch.float32,       shape [ceil(out/128), ceil(in/128)]
                          value = amax_of_block / 448.0
    """
    assert weight_bf16.ndim == 2, f"Expected 2D weight, got shape {weight_bf16.shape}"

    out_f, in_f = weight_bf16.shape

    # Pad to multiples of BLOCK_SIZE
    pad_out = (BLOCK_SIZE - out_f % BLOCK_SIZE) % BLOCK_SIZE
    pad_in  = (BLOCK_SIZE - in_f  % BLOCK_SIZE) % BLOCK_SIZE

    weight_float = weight_bf16.float()
    if pad_out > 0 or pad_in > 0:
        weight_float = F.pad(weight_float, (0, pad_in, 0, pad_out))

    new_out, new_in = weight_float.shape
    n_blocks_out = new_out // BLOCK_SIZE
    n_blocks_in  = new_in  // BLOCK_SIZE

    # [n_blocks_out, BLOCK_SIZE, n_blocks_in, BLOCK_SIZE]
    blocks = weight_float.reshape(n_blocks_out, BLOCK_SIZE, n_blocks_in, BLOCK_SIZE)

    # amax per block -> shape [n_blocks_out, n_blocks_in]
    amax = blocks.abs().amax(dim=(1, 3))

    # scale_inv = amax / 448.0  (float32)
    scale_inv = (amax / FP8_MAX).clamp(min=1e-12).to(torch.float32)

    # Quantize: divide by scale, clamp, cast to FP8
    scale_expanded = scale_inv.unsqueeze(1).unsqueeze(3)  # [nBO, 1, nBI, 1]
    quantized = (blocks / scale_expanded).clamp(-FP8_MAX, FP8_MAX)
    quantized = quantized.reshape(new_out, new_in)

    # Trim padding
    if pad_out > 0 or pad_in > 0:
        quantized = quantized[:out_f, :in_f]

    return quantized.to(FP8_DTYPE), scale_inv


def process_shard(
    src_path: str,
    dst_path: str,
    shard_idx: int,
    total_shards: int,
) -> dict:
    """
    Process one safetensors shard: quantize eligible weights, pass through the rest.

    Returns a dict: {weight_name: "fp8" | "bf16"} for logging/verification.
    """
    t0 = time.time()
    data = st.load_file(src_path)
    out_tensors = {}
    log = {}

    for name, tensor in data.items():
        if should_quantize(name):
            assert tensor.ndim == 2, f"Unexpected shape for {name}: {tensor.shape}"
            # Ensure BF16 input
            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)
            w_fp8, scale_inv = quantize_weight_fp8_block(tensor)
            # Store quantized weight
            out_tensors[name] = w_fp8
            # Store scale as <name_without_.weight>_scale_inv
            scale_name = name.replace(".weight", ".weight_scale_inv")
            out_tensors[scale_name] = scale_inv
            log[name] = "fp8"
        else:
            # Pass through as-is
            out_tensors[name] = tensor
            log[name] = str(tensor.dtype)

    st.save_file(out_tensors, dst_path)
    elapsed = time.time() - t0
    n_fp8 = sum(1 for v in log.values() if v == "fp8")
    print(
        f"  [{shard_idx+1:03d}/{total_shards:03d}] {os.path.basename(src_path)}"
        f"  fp8={n_fp8} pass_through={len(log)-n_fp8}  ({elapsed:.1f}s)"
    )
    sys.stdout.flush()
    return log


def build_model_index(
    src_dir: str,
    dst_dir: str,
    all_logs: list[dict],
    shard_files: list[str],
) -> dict:
    """
    Build model.safetensors.index.json mapping tensor_name -> shard_file.
    Handles the addition of weight_scale_inv tensors.
    """
    # Read original index if exists
    src_index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(src_index_path):
        with open(src_index_path) as f:
            orig_index = json.load(f)
        weight_map = dict(orig_index.get("weight_map", {}))
    else:
        # Build from scratch by scanning
        weight_map = {}

    # Add scale entries: for each fp8 weight, add its scale pointing to same shard
    new_weight_map = {}
    for shard_file, log in zip(shard_files, all_logs):
        shard_name = os.path.basename(shard_file)
        for name, status in log.items():
            new_weight_map[name] = shard_name
            if status == "fp8":
                scale_name = name.replace(".weight", ".weight_scale_inv")
                new_weight_map[scale_name] = shard_name

    # Total size (approximate, not critical for loading)
    total_size = sum(
        os.path.getsize(os.path.join(dst_dir, os.path.basename(sf)))
        for sf in shard_files
        if os.path.exists(os.path.join(dst_dir, os.path.basename(sf)))
    )

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }
    return index


def build_quantization_config() -> dict:
    """
    Build the quantization_config for config.json.

    ignored_layers: attention and other non-quantized linear layers.
    We skip attn entirely via SGLang's is_layer_skipped matching.
    """
    return {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "is_checkpoint_fp8_serialized": True,
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        # Patterns matched by SGLang's is_layer_skipped (substring match on layer prefix)
        # These layers remain BF16 in the checkpoint so must be skipped by the quant config.
        "ignored_layers": [
            "self_attn",       # all MLA attention projection weights
            "lm_head",         # output projection
            "embed_tokens",    # embedding
        ],
    }


def copy_non_model_files(src_dir: str, dst_dir: str):
    """Copy tokenizer, config, and other non-weight files."""
    skip_extensions = {".safetensors"}
    skip_names = {"model.safetensors.index.json"}
    for fname in os.listdir(src_dir):
        if fname in skip_names:
            continue
        if any(fname.endswith(ext) for ext in skip_extensions):
            continue
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Quantize Kimi-K2.5 BF16 -> FP8 block-wise")
    parser.add_argument("--src", default="/data2/bf16_out", help="Source BF16 model dir")
    parser.add_argument("--dst", default="/data2/Kimi-K25-FP8-new", help="Output FP8 model dir")
    parser.add_argument("--start-shard", type=int, default=0, help="Start from shard index (0-based, for resume)")
    parser.add_argument("--dry-run", action="store_true", help="Just print what would be quantized in first shard")
    args = parser.parse_args()

    src_dir = args.src
    dst_dir = args.dst

    print(f"Source: {src_dir}")
    print(f"Dest:   {dst_dir}")

    # Collect shards
    shard_files = sorted([
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if f.endswith(".safetensors")
    ])
    total_shards = len(shard_files)
    print(f"Found {total_shards} shard(s)")

    if args.dry_run:
        print("\n--- DRY RUN: scanning first shard ---")
        data = st.load_file(shard_files[0])
        for name, tensor in sorted(data.items()):
            status = "QUANTIZE" if should_quantize(name) else "pass_through"
            print(f"  {status:12s}  {name}  {tensor.dtype}  {list(tensor.shape)}")
        return

    os.makedirs(dst_dir, exist_ok=True)

    # Copy non-weight files first (config, tokenizer, etc.)
    print("\nCopying non-weight files...")
    copy_non_model_files(src_dir, dst_dir)

    # Update config.json with quantization_config
    config_path = os.path.join(dst_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["quantization_config"] = build_quantization_config()
        # Remove dtype=bfloat16 to avoid confusion (model is FP8)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  Updated config.json with quantization_config")
        print(f"  quantization_config: {json.dumps(config['quantization_config'], indent=4)}")

    # Process shards
    print(f"\nProcessing shards (starting from index {args.start_shard})...")
    all_logs = []

    for i, shard_path in enumerate(shard_files):
        dst_shard = os.path.join(dst_dir, os.path.basename(shard_path))

        if i < args.start_shard:
            # Load log from already-processed shard for index building
            if os.path.exists(dst_shard):
                # Reconstruct minimal log from existing output
                data = st.load_file(dst_shard)
                log = {}
                for name, tensor in data.items():
                    if "weight_scale_inv" in name:
                        continue
                    if tensor.dtype == FP8_DTYPE:
                        log[name] = "fp8"
                    else:
                        log[name] = str(tensor.dtype)
                all_logs.append(log)
                print(f"  [{i+1:03d}/{total_shards:03d}] Skipped (already exists): {os.path.basename(shard_path)}")
            else:
                print(f"  WARNING: --start-shard={args.start_shard} but {dst_shard} does not exist!")
                all_logs.append({})
            continue

        log = process_shard(shard_path, dst_shard, i, total_shards)
        all_logs.append(log)

    # Build and write index
    print("\nBuilding model.safetensors.index.json...")
    index = build_model_index(src_dir, dst_dir, all_logs, shard_files)
    index_path = os.path.join(dst_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    n_fp8_total = sum(
        sum(1 for v in log.values() if v == "fp8")
        for log in all_logs
    )
    print(f"\nDone! Total FP8 quantized tensors: {n_fp8_total}")
    print(f"Output: {dst_dir}")

    # Verification summary
    print("\n--- Quantization Summary ---")
    if all_logs:
        sample_log = all_logs[1] if len(all_logs) > 1 else all_logs[0]
        for name, status in sorted(sample_log.items())[:30]:
            print(f"  {status:30s}  {name}")
        print(f"  ...")


if __name__ == "__main__":
    main()
