#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Masked MHA Kernel for DeepSeek V3.2
// Supports masked attention for sequences > 2048 tokens using MHA path

// Mask format: uint64_t bitmap
// - Each bit corresponds to one token within a K-Block
// - Mask dimension: [Batch, SeqQ, Max_K_Blocks]
// - Each uint64_t represents 64 tokens (one page)

struct MaskedMHAKernelParams {
    // Input tensors
    void* q_ptr;              // Query: [total_q_tokens, num_heads, head_dim]
    void* k_ptr;              // Key: [total_kv_tokens, num_kv_heads, head_dim]
    void* v_ptr;              // Value: [total_kv_tokens, num_kv_heads, head_dim]
    
    // Output tensor
    void* out_ptr;            // Output: [total_q_tokens, num_heads, head_dim]
    
    // Mask tensor: [batch_size, max_seq_q, max_k_blocks]
    uint64_t* mask_ptr;       // Coarse-grained mask (tile-level)
    uint64_t* fine_mask_ptr;  // Fine-grained mask (per-token)
    
    // Sequence information
    int32_t* cu_seqlens_q;    // Cumulative sequence lengths for Q: [batch_size + 1]
    int32_t* cu_seqlens_k;    // Cumulative sequence lengths for K: [batch_size + 1]
    int32_t* seq_lens;        // Sequence lengths: [batch_size]
    
    // Page table for KV cache
    int32_t* page_table;      // Page table: [batch_size, max_pages]
    int32_t* page_table_lens; // Number of pages per sequence: [batch_size]
    
    // Dimensions
    int batch_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int v_head_dim;
    int max_seq_q;
    int max_seq_k;
    int page_size;            // Typically 64 tokens
    int tile_size;            // Query tile size: 128 tokens
    
    // Scaling
    float sm_scale;           // Softmax scale (1.0 / sqrt(head_dim))
    
    // Data types
    bool is_bf16;
    bool is_fp16;
    
    // Stream
    cudaStream_t stream;
    
    // TMA parameters (for SM90 implementation)
    // These will be set up in masked_mha_attn_tma_impl
    void* tma_params_ptr;  // Pointer to MainloopParams structure
};

// Prepare mask function
// Generates coarse-grained (tile-level) and fine-grained (per-token) masks
// Returns: mask tensor of shape [batch_size, max_seq_q, max_k_blocks]
//          Each uint64_t represents 64 tokens (one page)
void prepare_mask(
    uint64_t* mask_ptr,
    uint64_t* fine_mask_ptr,
    const int32_t* cu_seqlens_q,
    const int32_t* cu_seqlens_k,
    const int32_t* seq_lens,
    const int32_t* page_table,
    const int32_t* page_table_lens,
    int batch_size,
    int max_seq_q,
    int max_seq_k,
    int page_size,
    int tile_size,
    cudaStream_t stream
);

// Main masked MHA kernel
// Based on Flash Attention 3, with mask support
void masked_mha_kernel(
    const MaskedMHAKernelParams& params
);
