#!/usr/bin/env python3
"""
Test to see what actually happens when loading FP8 MoE block scales
with moe_dense_tp_size=1 and use_triton_kernels=True.
"""
import sys
sys.path.insert(0, "/usr/local/src/sglang/python")

import torch

# Simulate the parameters from Kimi-K2.5
hidden_size = 7168
moe_intermediate_size = 2048
moe_tp_size = 1  # moe_dense_tp_size=1
moe_tp_rank = 0
block_n = 128
block_k = 128
n_experts = 384
num_fused_shared_experts = 0  # disabled for 384 experts

intermediate_per_tp = moe_intermediate_size // moe_tp_size  # 2048

print("=" * 60)
print("Testing FP8 block scale loading")
print("=" * 60)
print(f"intermediate_per_tp: {intermediate_per_tp}")
print(f"n_blocks for w13 gate: {(intermediate_per_tp + block_n - 1) // block_n}")  # 16
print(f"n_blocks for w13 total (gate+up): {2 * (intermediate_per_tp + block_n - 1) // block_n}")  # 32
print(f"n_blocks for K: {(hidden_size + block_k - 1) // block_k}")  # 56

# What w13_weight_scale_inv shape is allocated
n_out = 2 * ((intermediate_per_tp + block_n - 1) // block_n)  # 32
n_in = (hidden_size + block_k - 1) // block_k  # 56
print(f"\nw13_weight_scale_inv shape: [{n_experts}, {n_out}, {n_in}] = [{n_experts}, 32, 56]")

expert_data_shape = (n_out, n_in)  # [32, 56]

# What the checkpoint provides
loaded_scale_shape = ((intermediate_per_tp + block_n - 1) // block_n, n_in)  # [16, 56]
print(f"Checkpoint gate_proj scale shape: {loaded_scale_shape}")

# _load_w13 logic:
SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}
shard_id = "w1"
use_triton_kernels = True
is_transposed = False  # initial

# from line 720-723 in layer.py:
if use_triton_kernels:
    is_transposed = True
if is_transposed:
    shard_dim = int(not SHARD_ID_TO_SHARDED_DIM[shard_id])  # flip 0 → 1
else:
    shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]  # 0

print(f"\nshard_dim: {shard_dim}")
print(f"expert_data_shape[shard_dim]: {expert_data_shape[shard_dim]}")

shard_size = expert_data_shape[shard_dim] // 2  # [32,56][1]//2 = 28
print(f"shard_size = {shard_size}")

# Loaded weight for scale after potential transpose:
print(f"\nLoaded scale original: {loaded_scale_shape}")
transposed_scale_shape = (loaded_scale_shape[1], loaded_scale_shape[0])  # [56, 16]
print(f"After transpose: {transposed_scale_shape}")

print(f"\nTrying narrow(dim={shard_dim}, start=0, size={shard_size}) on shape {transposed_scale_shape}")
if shard_size > transposed_scale_shape[shard_dim]:
    print(f"ERROR: shard_size ({shard_size}) > dim size ({transposed_scale_shape[shard_dim]})")
    print("This would cause: RuntimeError: start (0) + length (28) exceeds dimension size (16)")
else:
    print(f"OK: would narrow to [{transposed_scale_shape[0]}, {shard_size}]")

print("\n" + "=" * 60)
print("CONCLUSION: The FP8 block-scale loading with is_transposed=True")
print("attempts to shard on the wrong dimension for the scale tensor,")
print("causing size overflow. The model may silently fail or produce wrong scales.")
print("=" * 60)

# Now test what the CORRECT loading should do
print("\n=== CORRECT loading (without is_transposed for scales) ===")
shard_dim_correct = SHARD_ID_TO_SHARDED_DIM[shard_id]  # = 0
shard_size_correct = expert_data_shape[shard_dim_correct] // 2  # = 32 // 2 = 16
print(f"shard_dim: {shard_dim_correct}")
print(f"shard_size: {shard_size_correct}")
print(f"Narrow({shard_dim_correct}, 0, {shard_size_correct}) on shape {loaded_scale_shape}")
print(f"→ [{shard_size_correct}, {loaded_scale_shape[1]}] = [16, 56]")
print(f"Expert data target: expert_data[0:{shard_size_correct}, :] = [16, 56]")
print("This is CORRECT: gate_proj scale placed in first half of w13 scale")
