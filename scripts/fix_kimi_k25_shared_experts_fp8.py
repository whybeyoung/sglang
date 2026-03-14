"""
修复脚本: 将 shared_experts 也量化为 FP8 block-wise [128,128]
并更新 index.json 和 config.json

在量化机 26.5.27.241 上执行:
  python3 /data2/fix_add_shared_fp8.py
"""
import os
import json
import glob
import torch
import safetensors.torch as st

FP8_DIR = "/data2/Kimi-K2.5-FP8"
FP8_MAX = 448.0
BLOCK = 128


def quantize_fp8_block(w_bf16):
    """BF16 -> FP8 block-wise 量化"""
    out_f, in_f = w_bf16.shape
    pad_out = (BLOCK - out_f % BLOCK) % BLOCK
    pad_in = (BLOCK - in_f % BLOCK) % BLOCK
    wp = torch.nn.functional.pad(w_bf16.float(), (0, pad_in, 0, pad_out))
    no, ni = wp.shape[0] // BLOCK, wp.shape[1] // BLOCK
    blocks = wp.reshape(no, BLOCK, ni, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale_inv = (amax / FP8_MAX).to(torch.float32)
    s_exp = scale_inv.unsqueeze(1).unsqueeze(3)
    qw = (blocks / s_exp).clamp(-FP8_MAX, FP8_MAX).reshape(wp.shape[0], wp.shape[1])
    qw = qw[:out_f, :in_f].to(torch.float8_e4m3fn)
    return qw, scale_inv


shards = sorted(glob.glob(os.path.join(FP8_DIR, "model-*.safetensors")))
print("Total shards:", len(shards))

changed = 0
for i, fp8_path in enumerate(shards):
    shard_name = os.path.basename(fp8_path)
    fp8_shard = st.load_file(fp8_path)

    shared_bf16_keys = [
        k
        for k, v in fp8_shard.items()
        if "shared_experts" in k and k.endswith(".weight") and v.dtype == torch.bfloat16
    ]
    if not shared_bf16_keys:
        continue

    new_tensors = dict(fp8_shard)
    fix_count = 0
    for k in shared_bf16_keys:
        w_bf16 = fp8_shard[k]
        qw, s = quantize_fp8_block(w_bf16)
        new_tensors[k] = qw
        scale_key = k.replace(".weight", ".weight_scale_inv")
        new_tensors[scale_key] = s
        fix_count += 1

    st.save_file(new_tensors, fp8_path)
    changed += fix_count
    print("[%d/%d] %s: quantized %d shared_expert weights" % (i + 1, len(shards), shard_name, fix_count))

# 更新 index.json
idx_path = os.path.join(FP8_DIR, "model.safetensors.index.json")
with open(idx_path) as f:
    idx = json.load(f)

new_map = dict(idx["weight_map"])
added = 0
for k, v in list(idx["weight_map"].items()):
    if "shared_experts" in k and k.endswith(".weight"):
        scale_key = k.replace(".weight", ".weight_scale_inv")
        if scale_key not in new_map:
            new_map[scale_key] = v
            added += 1

idx["weight_map"] = new_map
with open(idx_path, "w") as f:
    json.dump(idx, f, indent=2)

# 更新 config.json: 从 ignored_layers 中移除 shared_experts
cfg_path = os.path.join(FP8_DIR, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)

tc = cfg.get("text_config", {})
qc = tc.get("quantization_config", {})
if "ignored_layers" in qc:
    old_ignored = qc["ignored_layers"]
    qc["ignored_layers"] = [x for x in old_ignored if x != "shared_experts"]
    print("ignored_layers: %s -> %s" % (old_ignored, qc["ignored_layers"]))

with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

print("Done: quantized %d shared_expert weights, added %d scale keys" % (changed, added))
