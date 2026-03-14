#!/usr/bin/env python3
"""Test scale loading path for FP8 MoE with moe_dense_tp_size=1"""
import torch

def test_scale_loading_path():
    # Simulate parameters
    hidden_size = 7168
    moe_intermediate_size = 2048
    moe_tp_size = 1   # moe_dense_tp_size=1
    moe_tp_rank = 0
    block_n = 128
    block_k = 128
    
    intermediate_per_tp = moe_intermediate_size // moe_tp_size  # = 2048
    n_blocks_out = (intermediate_per_tp + block_n - 1) // block_n  # = 16
    n_blocks_in = (hidden_size + block_k - 1) // block_k  # = 56
    
    print(f"intermediate_per_tp: {intermediate_per_tp}")
    print(f"n_blocks_out: {n_blocks_out}, n_blocks_in: {n_blocks_in}")
    
    # w13_weight_scale_inv shape: [E, 2*n_blocks_out, n_blocks_in]
    n_experts = 384
    w13_scale = torch.ones(n_experts, 2 * n_blocks_out, n_blocks_in)
    print(f"w13_weight_scale_inv shape: {w13_scale.shape}")
    
    # expert_data for expert 0
    expert_data = w13_scale[0]  # shape [32, 56]
    print(f"expert_data shape: {expert_data.shape}")
    
    # Loaded scale from checkpoint: [n_blocks_out, n_blocks_in] = [16, 56]
    loaded_scale = torch.arange(16 * 56).float().reshape(16, 56) / (16 * 56)
    print(f"loaded_scale shape: {loaded_scale.shape}")
    
    # Simulating _load_w13 with shard_id="w1", use_triton_kernels=True
    SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}
    shard_dim = SHARD_ID_TO_SHARDED_DIM["w1"]  # = 0
    
    # is_transposed = True for triton kernels
    is_transposed = True
    if is_transposed:
        shard_dim = int(not shard_dim)  # = 1
    
    print(f"shard_dim after is_transposed: {shard_dim}")
    
    # shard_size = expert_data.shape[shard_dim] // 2
    shard_size = expert_data.shape[shard_dim] // 2  # = 56 // 2 = 28
    print(f"shard_size: {shard_size}")
    
    # start = 0 for w1 (gate_proj)
    start = 0
    
    # Transpose loaded_weight (scale)
    loaded_scale_transposed = loaded_scale.transpose(-2, -1)
    print(f"loaded_scale after transpose: {loaded_scale_transposed.shape}")  # [56, 16]
    
    # Narrow on shard_dim
    print(f"Trying narrow({shard_dim}, {shard_size * moe_tp_rank}, {shard_size}) on shape {loaded_scale_transposed.shape}")
    try:
        loaded_scale_narrowed = loaded_scale_transposed.narrow(shard_dim, shard_size * moe_tp_rank, shard_size)
        print(f"narrowed loaded_scale: {loaded_scale_narrowed.shape}")
    except Exception as e:
        print(f"ERROR during narrow: {e}")
        return
    
    # Narrow expert_data
    expert_data_narrowed = expert_data.narrow(shard_dim, start, shard_size)
    print(f"narrowed expert_data: {expert_data_narrowed.shape}")
    
    # Copy
    try:
        expert_data_narrowed.copy_(loaded_scale_narrowed)
        print("copy succeeded!")
    except Exception as e:
        print(f"ERROR during copy: {e}")
        return
    
    print(f"\nFinal w13_scale[0, 0:16, 0:5]:\n{w13_scale[0, 0:16, 0:5]}")
    print(f"\nOriginal loaded_scale[0:5, 0:5]:\n{loaded_scale[0:5, 0:5]}")
    print(f"\nExpected: loaded into expert_data[0:16, 0:56]")
    print(f"Actual: expert_data[0, 0:16, :] = {w13_scale[0, 0:16, 0:5]}")


if __name__ == "__main__":
    test_scale_loading_path()
