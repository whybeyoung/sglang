#!/usr/bin/env python3
"""
Standalone test: Load one FP8 MoE layer and verify computation.
Tests FP8 block-wise quantized expert weights produce correct output
by comparing FP8 expert matmul with dequantized BF16 matmul.
"""
import sys
sys.path.insert(0, "/usr/local/src/sglang/python")

import torch
import safetensors.torch as st

# Load a shard that has expert weights
model_dir = "/work/models"
shard = st.load_file(model_dir + "/model-00002-of-000064.safetensors", device="cuda")

# Get expert 0 weights for layer 1
gate_w = shard["language_model.model.layers.1.mlp.experts.0.gate_proj.weight"]  # [2048, 7168] fp8
gate_s = shard["language_model.model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv"]  # [16, 56]
print("gate_proj weight:", gate_w.shape, gate_w.dtype)
print("gate_proj scale:", gate_s.shape, gate_s.dtype)

# Dequantize to BF16
BLOCK = 128
out_f, in_f = gate_w.shape  # 2048, 7168
n_blocks_out = (out_f + BLOCK - 1) // BLOCK
n_blocks_in = (in_f + BLOCK - 1) // BLOCK

# Dequantize: fp8_val * scale_inv
gate_w_float = gate_w.float()
gate_dequant = torch.zeros_like(gate_w_float)
for i in range(n_blocks_out):
    for j in range(n_blocks_in):
        r_start = i * BLOCK
        r_end = min((i + 1) * BLOCK, out_f)
        c_start = j * BLOCK
        c_end = min((j + 1) * BLOCK, in_f)
        gate_dequant[r_start:r_end, c_start:c_end] = (
            gate_w_float[r_start:r_end, c_start:c_end] * gate_s[i, j]
        )

print("gate_dequant stats: min=%.4f max=%.4f std=%.4f" % (
    gate_dequant.min().item(), gate_dequant.max().item(), gate_dequant.std().item()))

# Create a random input
x = torch.randn(4, 7168, dtype=torch.bfloat16, device="cuda")  # 4 tokens, hidden_size=7168

# Reference: BF16 matmul
ref = x.float() @ gate_dequant.t()  # [4, 2048]
print("ref output stats: min=%.4f max=%.4f std=%.4f" % (
    ref.min().item(), ref.max().item(), ref.std().item()))

# Now test FP8 kernel: simulate block-wise FP8 matmul
# Using sgl_kernel per_token_group_quant_fp8 for activation
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

x_fp8, x_scale = sglang_per_token_group_quant_fp8(x, BLOCK)
print("x_fp8:", x_fp8.shape, x_fp8.dtype)
print("x_scale:", x_scale.shape, x_scale.dtype)

# Compute with torch FP8 matmul (no block)
# For block-wise, we need to use the triton kernel
# Let's do a manual block-wise matmul
result = torch.zeros(4, 2048, dtype=torch.float32, device="cuda")
for k_block in range(n_blocks_in):
    k_start = k_block * BLOCK
    k_end = min((k_block + 1) * BLOCK, in_f)
    
    a_block = x_fp8[:, k_start:k_end].float()
    a_s = x_scale[:, k_block:k_block+1]  # [4, 1]
    
    for n_block in range(n_blocks_out):
        n_start = n_block * BLOCK
        n_end = min((n_block + 1) * BLOCK, out_f)
        
        b_block = gate_w_float[n_start:n_end, k_start:k_end]  # [BLOCK, BLOCK]
        b_s = gate_s[n_block, k_block]
        
        # partial = a_block @ b_block.t() * a_s * b_s
        partial = (a_block @ b_block.t()) * a_s * b_s
        result[:, n_start:n_end] += partial

print("\nManual block FP8 result stats: min=%.4f max=%.4f std=%.4f" % (
    result.min().item(), result.max().item(), result.std().item()))

# Compare
diff = (result - ref).abs()
print("Max diff (manual FP8 vs BF16 ref): %.6f" % diff.max().item())
print("Mean diff: %.6f" % diff.mean().item())

# Check relative error
rel_err = (diff / (ref.abs() + 1e-10))
print("Mean relative error: %.4f%%" % (rel_err.mean().item() * 100))

if diff.max().item() > 1.0:
    print("WARNING: Large difference detected! FP8 computation may be incorrect.")
else:
    print("OK: FP8 block-wise computation matches BF16 reference.")
