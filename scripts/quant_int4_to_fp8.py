#!/usr/bin/env python3
"""
Kimi-K2.5: INT4 compressed-tensors -> FP8 block-wise (128x128) W8A8 dynamic
=============================================================================
对齐 Kimi-K2 官方 HuggingFace FP8 格式（moonshotai/Kimi-K2-Instruct）

Source format (/work/models, 64 shards):
  - MoE routed experts gate/up/down_proj: INT4 packed
      .weight_packed: int32 [out, in/8]  (8 INT4 per int32)
      .weight_scale:  bf16  [out, in/32] (group_size=32)
      .weight_shape:  int32 [2] = [out, in]
  - 其余所有层 (attn proj / shared_experts / dense MLP / layernorm): bfloat16
  - 所有 key 带 "language_model." 前缀

Target format (/data2/Kimi-K25-FP8-new):
  FP8 量化（对齐 K2 官方格式，量化所有 Linear projection weights）：
    - MoE routed experts: gate/up/down_proj -> FP8
    - shared_experts:     gate/up/down_proj -> FP8
    - attention:          q_a_proj / q_b_proj / kv_a_proj_with_mqa / kv_b_proj / o_proj -> FP8
    - dense MLP (layer0): gate/up/down_proj -> FP8
  保持 BF16（非线性投影权重）：
    - layernorm weights (input_layernorm, post_attention_layernorm, kv_a_layernorm, q_a_layernorm, norm)
    - embed_tokens.weight
    - lm_head.weight
    - mlp.gate.weight (MoE router gate, shape [n_experts, hidden])
    - self_attn.rotary_emb.inv_freq
  去掉 "language_model." 前缀

quantization_config (写入 config.json):
  quant_method: fp8
  is_checkpoint_fp8_serialized: true
  activation_scheme: dynamic
  weight_block_size: [128, 128]
  （不设 ignored_layers，与 K2 官方保持一致）
"""
import os
import re
import json
import shutil
import time
import gc
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/work/models"
DST = "/data2/Kimi-K25-FP8-new"
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
BLOCK = 128

os.makedirs(DST, exist_ok=True)

# BF16 保留的 key 后缀/子串（不做 FP8 量化）
BF16_PATTERNS = (
    "layernorm.weight",      # input_layernorm, post_attention_layernorm, kv_a_layernorm, q_a_layernorm
    "layer_norm.weight",
    "norm.weight",           # model.norm.weight
    "norm.bias",
    "embed_tokens.weight",
    "lm_head.weight",
    "mlp.gate.weight",       # MoE router gate, shape [n_experts, hidden_size]
    "rotary_emb.inv_freq",
)

# 非 language_model 模块全部保持 BF16（vision encoder / multimodal projector）
BF16_PREFIXES = (
    "vision_tower.",
    "mm_projector.",
    "vision_embed.",
    "image_newline",
)


def should_keep_bf16(key):
    """判断该 key 是否保持 BF16（不量化）"""
    # vision tower 全部跳过
    for pfx in BF16_PREFIXES:
        if key.startswith(pfx):
            return True
    # 非 2D 权重（bias、1D norm、4D conv 等）全部跳过
    return False  # shape 检查在调用处做


def is_quantizable_weight(key, tensor):
    """只量化 routed experts 的 gate/up/down_proj（对齐 MooreThreads 策略）"""
    if not key.endswith(".weight"):
        return False
    if tensor.dim() != 2:
        return False
    for pfx in BF16_PREFIXES:
        if key.startswith(pfx):
            return False
    # 只有 mlp.experts.N.{gate,up,down}_proj.weight 才量化
    if re.search(r'mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$', key):
        return True
    return False


def unpack_int4(packed):
    """int32 [out, in/8] -> int8 [out, in] (signed 4-bit two's complement)"""
    vals = []
    for i in range(8):
        nibble = (packed >> (i * 4)) & 0xF
        signed = nibble.to(torch.int8)
        signed = torch.where(signed >= 8, signed - 16, signed)
        vals.append(signed)
    return torch.stack(vals, dim=-1).reshape(packed.shape[0], -1)


def dequant_int4(packed, scale, shape):
    """INT4 group_size=32 dequant -> float32"""
    out_f, in_f = shape
    w = unpack_int4(packed)[:out_f, :in_f].float()
    s = scale.float().repeat_interleave(32, dim=1)[:out_f, :in_f]
    return w * s


def quant_fp8_blockwise(w_f32):
    """
    float32 [out, in] -> (float8_e4m3fn [out, in], float32 scale_inv [nr, nc])
    scale_inv[i,j] = amax(block[i,j]) / FP8_MAX
    这与 K2 官方格式一致：dequant 时 w_fp8 * scale_inv 还原为 float
    """
    rows, cols = w_f32.shape
    pad_r = (BLOCK - rows % BLOCK) % BLOCK
    pad_c = (BLOCK - cols % BLOCK) % BLOCK
    w = torch.nn.functional.pad(w_f32, (0, pad_c, 0, pad_r))
    pr, pc = w.shape
    nr, nc = pr // BLOCK, pc // BLOCK
    blocks = w.reshape(nr, BLOCK, nc, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)   # [nr, nc]
    scale_inv = (amax / FP8_MAX).to(torch.float32)          # [nr, nc]
    scale_q = FP8_MAX / amax                                 # [nr, nc]
    sq_exp = scale_q.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:pr, :pc]
    fp8 = (w * sq_exp).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return fp8[:rows, :cols].contiguous(), scale_inv.contiguous()


def main():
    with open(os.path.join(SRC, "model.safetensors.index.json")) as f:
        orig_idx = json.load(f)

    shard_keys = {}
    for key, shard in orig_idx["weight_map"].items():
        shard_keys.setdefault(shard, []).append(key)

    new_weight_map = {}
    total = len(shard_keys)
    print(f"Total shards: {total}", flush=True)

    for i, (shard_name, _) in enumerate(sorted(shard_keys.items())):
        t0 = time.time()
        shard_path = os.path.join(SRC, shard_name)

        raw = {}
        with safe_open(shard_path, framework="pt") as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k)

        # INT4 packed 的 base keys
        int4_bases = {}
        for k in raw:
            if k.endswith(".weight_packed"):
                base = k[: -len(".weight_packed")]
                int4_bases[base] = True

        # 非 INT4 的普通 keys
        int4_suffixes = (".weight_packed", ".weight_scale", ".weight_shape")
        plain_keys = [k for k in raw if not any(k.endswith(s) for s in int4_suffixes)]

        out_tensors = {}
        quant_count = 0
        bf16_count = 0

        # ── INT4 层（routed experts）: dequant -> FP8 ──────────────────────────
        for base in int4_bases:
            packed = raw.get(base + ".weight_packed")
            scale  = raw.get(base + ".weight_scale")
            shape_t = raw.get(base + ".weight_shape")
            if packed is None or scale is None or shape_t is None:
                print(f"  WARN: incomplete INT4 for {base}", flush=True)
                continue
            shape = shape_t.tolist()
            new_base = base.removeprefix("language_model.")
            # INT4 packed 层都是 2D linear proj，直接 dequant -> FP8
            w_f32 = dequant_int4(packed.cpu(), scale.cpu(), shape)
            fp8_w, scale_inv = quant_fp8_blockwise(w_f32)
            out_tensors[new_base + ".weight"] = fp8_w
            out_tensors[new_base + ".weight_scale_inv"] = scale_inv
            quant_count += 1
            del w_f32

        # ── 普通 BF16 层 ───────────────────────────────────────────────────────
        for k in plain_keys:
            new_k = k.removeprefix("language_model.")
            v = raw[k]

            if is_quantizable_weight(k, v):
                # 2D linear proj（attn / shared_experts / dense MLP）-> FP8
                fp8_w, scale_inv = quant_fp8_blockwise(v.cpu().float())
                out_tensors[new_k] = fp8_w
                out_tensors[new_k.replace(".weight", ".weight_scale_inv")] = scale_inv
                quant_count += 1
            else:
                # layernorm / embed / lm_head / moe gate / rotary / bias /
                # vision_tower / 1D / 4D -> 原样保留 BF16
                out_tensors[new_k] = v
                bf16_count += 1

        out_path = os.path.join(DST, shard_name)
        save_file(out_tensors, out_path)
        for k in out_tensors:
            new_weight_map[k] = shard_name

        elapsed = time.time() - t0
        print(
            f"[{i+1}/{total}] {shard_name}: {len(out_tensors)} tensors, "
            f"fp8={quant_count} bf16={bf16_count} ({elapsed:.1f}s)",
            flush=True,
        )

        del raw, out_tensors, int4_bases, plain_keys
        gc.collect()

    # ── 保存 index ──────────────────────────────────────────────────────────────
    new_idx = {"metadata": {"format": "pt"}, "weight_map": new_weight_map}
    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
        json.dump(new_idx, f, indent=2)
    print("Saved index", flush=True)

    # ── 复制 config & tokenizer 文件 ────────────────────────────────────────────
    for fname in [
        "config.json",
        "configuration.json",
        "generation_config.json",
        "configuration_deepseek.py",
        "configuration_kimi_k25.py",
        "tokenizer_config.json",
        "tokenization_kimi.py",
        "preprocessor_config.json",
        "tiktoken.model",
        "chat_template.jinja",
        "kimi_k25_processor.py",
        "kimi_k25_vision_processing.py",
        "media_utils.py",
        "tool_declaration_ts.py",
        "modeling_kimi_k25.py",
        "modeling_deepseek.py",
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
    ]:
        src_p = os.path.join(SRC, fname)
        if os.path.exists(src_p):
            shutil.copy2(src_p, os.path.join(DST, fname))
            print(f"Copied {fname}", flush=True)

    # ── 更新 config.json（对齐 K2 官方格式，无 ignored_layers）────────────────
    cfg_path = os.path.join(DST, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    fp8_qcfg = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        # SGLang is_layer_skipped 用子串匹配：any(ignored in prefix ...)
        # 不支持 regex，直接用子串即可覆盖所有需要跳过的层
        "ignored_layers": [
            "lm_head",
            "self_attn",
            "shared_experts",
            "layers.0.mlp",   # dense MLP layer0 (first_k_dense_replace=1)
        ],
    }
    if "text_config" in cfg:
        cfg["text_config"]["quantization_config"] = fp8_qcfg
        cfg["text_config"].pop("ignored_layers", None)
    else:
        cfg["quantization_config"] = fp8_qcfg
        cfg.pop("ignored_layers", None)

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Updated config.json", flush=True)

    print(f"\nDone! Output: {DST}", flush=True)


if __name__ == "__main__":
    main()
