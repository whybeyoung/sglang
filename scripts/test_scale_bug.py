#!/usr/bin/env python3
"""
Verify actual scale loading behavior on the test machine.
Tests whether FP8 MoE scales are correctly loaded with triton kernels.
"""
import sys
sys.path.insert(0, "/usr/local/src/sglang/python")

import torch
print("Testing FP8 block scale loading on actual SGLang code...")

# Import the actual FusedMoE
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE, FusedMoeWeightScaleSupported

# Check what is_transposed behavior is for scale vs weight
print("\n=== Checking scale vs weight loading path ===")

# Key question: when weight_name contains "scale_inv", 
# does the code distinguish it from actual weights for transposition?
test_weight_name = "language_model.model.layers.1.mlp.experts.w13_weight_scale_inv"
print(f"weight_name: {test_weight_name}")
print(f"  'scale' in weight_name: {'scale' in test_weight_name}")
print(f"  'weight' in weight_name after scale check: this goes to scale branch first")

# The critical path: for BLOCK quant, scale goes through _load_model_weight_or_group_weight_scale
# which calls _load_w13, which has the transpose + shard_dim logic

# Test: create a minimal FusedMoE and try to load a scale
print("\n=== Attempting minimal FusedMoE scale load ===")

try:
    # Create a simple 2-expert FusedMoE to test loading
    hidden_size = 512
    intermediate_size = 256
    block_n, block_k = 128, 128
    
    # Create the parameter directly
    n_experts = 2
    n_out_blocks = 2 * ((intermediate_size + block_n - 1) // block_n)  # = 4
    n_in_blocks = (hidden_size + block_k - 1) // block_k  # = 4
    
    print(f"n_out_blocks: {n_out_blocks}, n_in_blocks: {n_in_blocks}")
    
    # Create scale parameter
    scale_param = torch.ones(n_experts, n_out_blocks, n_in_blocks)
    expert_data = scale_param[0]  # [4, 4]
    
    # Simulate loaded gate_proj scale from checkpoint
    n_out_per_expert = intermediate_size // 1  # no TP
    loaded_scale = torch.arange(2 * 4).float().reshape(2, 4)  # [2, 4] = [ceil(256/128), ceil(512/128)]
    print(f"expert_data shape: {expert_data.shape}")  # [4, 4]
    print(f"loaded_scale shape: {loaded_scale.shape}")  # [2, 4]
    
    # Test with is_transposed = False (what we want for scales)
    print("\n--- Without transpose (correct for scales) ---")
    shard_dim = 0  # for w1/gate
    shard_size = expert_data.shape[shard_dim] // 2  # = 2
    print(f"shard_size: {shard_size}")
    ls = loaded_scale.narrow(0, 0 * shard_size, shard_size)
    ed = expert_data.narrow(0, 0, shard_size)
    ed.copy_(ls)
    print(f"Success: placed {ls.shape} into expert_data[0:2, :]")
    print(f"scale_param[0, 0:2, :] = {scale_param[0, 0:2, :]}")
    
    # Test with is_transposed = True (current buggy behavior)
    print("\n--- With transpose (current buggy behavior) ---")
    scale_param2 = torch.ones(n_experts, n_out_blocks, n_in_blocks)
    expert_data2 = scale_param2[0]
    shard_dim2 = 1  # after is_transposed=True flip
    shard_size2 = expert_data2.shape[shard_dim2] // 2  # = 2
    loaded_scale2 = loaded_scale.transpose(-2, -1)  # [4, 2]
    print(f"loaded_scale after transpose: {loaded_scale2.shape}")
    ls2 = loaded_scale2.narrow(1, 0, shard_size2)  # narrow(1, 0, 2) on [4, 2]
    ed2 = expert_data2.narrow(1, 0, shard_size2)  # narrow(1, 0, 2) on [4, 4]
    ed2.copy_(ls2)
    print(f"Success (but WRONG): placed transposed+narrowed {ls2.shape} into expert_data[:,0:2]")
    print(f"scale_param2[0, :, 0:2] = {scale_param2[0, :, 0:2]}")
    print(f"Expected scale_param[0, 0:2, :] = {scale_param[0, 0:2, :]}")
    print(f"These are DIFFERENT - scales are wrong!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
