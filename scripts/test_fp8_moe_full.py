#!/usr/bin/env python3
"""
Test FP8 MoE layer via SGLang's triton kernel.
Tests that the FP8 block-wise MoE produces output consistent with BF16 dequantized computation.
"""
import sys
sys.path.insert(0, "/usr/local/src/sglang/python")

import torch
import safetensors.torch as st
import json

# Load model config
config = json.load(open("/work/models/config.json"))
tc = config["text_config"]
hidden_size = tc["hidden_size"]  # 7168
moe_intermediate_size = tc["moe_intermediate_size"]  # 2048
n_experts = tc["n_routed_experts"]  # 384

print(f"Config: hidden_size={hidden_size}, moe_intermediate_size={moe_intermediate_size}, n_experts={n_experts}")

# Load expert weights for layer 1 - first few experts
print("\nLoading weights...")
shard = st.load_file("/work/models/model-00002-of-000064.safetensors", device="cuda")
expert_keys = [k for k in shard.keys() if "layers.1.mlp.experts." in k]
print(f"Expert-related keys in shard: {len(expert_keys)}")

# Count how many experts are in this shard
expert_ids = set()
for k in expert_keys:
    if ".experts." in k:
        parts = k.split(".experts.")
        if len(parts) > 1:
            eid = parts[1].split(".")[0]
            try:
                expert_ids.add(int(eid))
            except:
                pass
print(f"Expert IDs in this shard: {sorted(expert_ids)[:5]}...{sorted(expert_ids)[-3:]}")

# Load a few experts
num_test_experts = min(8, len(expert_ids))
test_experts = sorted(expert_ids)[:num_test_experts]
print(f"Testing with experts: {test_experts}")

# Build w13_weight and w13_weight_scale_inv for testing
# w13_weight: [E, 2*moe_intermediate_size, hidden_size]
# w13_weight_scale_inv: [E, 2*ceil(moe_intermediate_size/128), ceil(hidden_size/128)]
BLOCK = 128
n_out_blocks = (moe_intermediate_size + BLOCK - 1) // BLOCK  # 16
n_in_blocks = (hidden_size + BLOCK - 1) // BLOCK  # 56

w13_weight = torch.zeros(num_test_experts, 2 * moe_intermediate_size, hidden_size, dtype=torch.float8_e4m3fn, device="cuda")
w13_scale = torch.ones(num_test_experts, 2 * n_out_blocks, n_in_blocks, dtype=torch.float32, device="cuda")
w2_weight = torch.zeros(num_test_experts, hidden_size, moe_intermediate_size, dtype=torch.float8_e4m3fn, device="cuda")
w2_scale = torch.ones(num_test_experts, n_in_blocks, n_out_blocks, dtype=torch.float32, device="cuda")

for i, eid in enumerate(test_experts):
    prefix = f"language_model.model.layers.1.mlp.experts.{eid}."
    gate_w = shard[prefix + "gate_proj.weight"]  # [2048, 7168] fp8
    gate_s = shard[prefix + "gate_proj.weight_scale_inv"]  # [16, 56]
    up_w = shard[prefix + "up_proj.weight"]  # [2048, 7168] fp8
    up_s = shard[prefix + "up_proj.weight_scale_inv"]  # [16, 56]
    down_w = shard[prefix + "down_proj.weight"]  # [7168, 2048] fp8
    down_s = shard[prefix + "down_proj.weight_scale_inv"]  # [56, 16]
    
    w13_weight[i, 0:moe_intermediate_size, :] = gate_w
    w13_weight[i, moe_intermediate_size:, :] = up_w
    w13_scale[i, 0:n_out_blocks, :] = gate_s
    w13_scale[i, n_out_blocks:, :] = up_s
    w2_weight[i, :, :] = down_w
    w2_scale[i, :, :] = down_s

print(f"w13_weight: {w13_weight.shape}")
print(f"w13_scale: {w13_scale.shape}")

# Use the triton MoE kernel
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe

# Create inputs
batch_size = 4
x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="cuda")
top_k = 2
topk_weights = torch.ones(batch_size, top_k, dtype=torch.float32, device="cuda") / top_k
topk_ids = torch.zeros(batch_size, top_k, dtype=torch.int32, device="cuda")
# All tokens go to expert 0 and 1
for i in range(batch_size):
    topk_ids[i, 0] = 0
    topk_ids[i, 1] = 1 % num_test_experts

print(f"\nInput: {x.shape}, topk_ids: {topk_ids}")

try:
    result_fp8 = fused_moe(
        hidden_states=x,
        w1=w13_weight,
        w2=w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=True,
        use_fp8_w8a8=True,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[BLOCK, BLOCK],
    )
    print(f"FP8 MoE output shape: {result_fp8.shape}")
    print(f"FP8 MoE output stats: min={result_fp8.min():.4f} max={result_fp8.max():.4f} std={result_fp8.std():.4f}")
    print(f"Any NaN: {result_fp8.isnan().any()}")
    print(f"Any Inf: {result_fp8.isinf().any()}")
except Exception as e:
    print(f"ERROR in FP8 MoE: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now compute BF16 reference
def dequant_block(w_fp8, scale, out_f, in_f):
    """Dequantize FP8 weight with block-wise scale."""
    n_b_out = (out_f + BLOCK - 1) // BLOCK
    n_b_in = (in_f + BLOCK - 1) // BLOCK
    w_float = w_fp8.float()
    w_dequant = torch.zeros(out_f, in_f, dtype=torch.float32, device="cuda")
    for i in range(n_b_out):
        for j in range(n_b_in):
            r0, r1 = i * BLOCK, min((i+1)*BLOCK, out_f)
            c0, c1 = j * BLOCK, min((j+1)*BLOCK, in_f)
            w_dequant[r0:r1, c0:c1] = w_float[r0:r1, c0:c1] * scale[i, j]
    return w_dequant

print("\nComputing BF16 reference...")
result_ref = torch.zeros(batch_size, hidden_size, dtype=torch.float32, device="cuda")

for i in range(batch_size):
    token_out = torch.zeros(hidden_size, dtype=torch.float32, device="cuda")
    for k in range(top_k):
        eid = topk_ids[i, k].item()
        w = topk_weights[i, k].item()
        
        gate_dequant = dequant_block(w13_weight[eid, :moe_intermediate_size, :], w13_scale[eid, :n_out_blocks, :], moe_intermediate_size, hidden_size)
        up_dequant = dequant_block(w13_weight[eid, moe_intermediate_size:, :], w13_scale[eid, n_out_blocks:, :], moe_intermediate_size, hidden_size)
        down_dequant = dequant_block(w2_weight[eid], w2_scale[eid], hidden_size, moe_intermediate_size)
        
        x_f = x[i].float()
        gate = x_f @ gate_dequant.t()  # [2048]
        up = x_f @ up_dequant.t()  # [2048]
        hidden = torch.sigmoid(gate) * gate * up  # SwiGLU
        out = hidden @ down_dequant.t()  # [7168]
        token_out += w * out
    result_ref[i] = token_out

print(f"BF16 ref output stats: min={result_ref.min():.4f} max={result_ref.max():.4f} std={result_ref.std():.4f}")

diff = (result_fp8.float() - result_ref).abs()
print(f"\nMax diff (FP8 MoE vs BF16 ref): {diff.max():.4f}")
print(f"Mean diff: {diff.mean():.4f}")
rel_err = diff / (result_ref.abs() + 1e-10)
print(f"Mean relative error: {rel_err.mean()*100:.2f}%")
