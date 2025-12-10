#include "masked_mha_kernel.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <mma.h>
#include <cmath>
#include <cfloat>

namespace cg = cooperative_groups;

// Constants
constexpr int TILE_SIZE_Q = 128;  // Query tile size
constexpr int PAGE_SIZE = 64;     // KV cache page size
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;      // Number of warps per block

// Helper function to check if a tile needs a K-block
__device__ __forceinline__ bool tile_needs_block(
    int q_tile_start,
    int q_tile_end,
    int k_block_start,
    int k_block_end
) {
    // Causal attention: Q tile can only attend to K blocks up to its end position
    return k_block_end <= q_tile_end;
}

// Prepare mask kernel
// Generates coarse-grained (tile-level) mask
__global__ void prepare_mask_kernel(
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
    int tile_size
) {
    int batch_idx = blockIdx.x;
    int tile_idx = blockIdx.y;
    
    if (batch_idx >= batch_size) return;
    
    int seq_start_q = cu_seqlens_q[batch_idx];
    int seq_end_q = cu_seqlens_q[batch_idx + 1];
    int seq_len_q = seq_end_q - seq_start_q;
    
    int seq_start_k = cu_seqlens_k[batch_idx];
    int seq_end_k = cu_seqlens_k[batch_idx + 1];
    int seq_len_k = seq_end_k - seq_start_k;
    
    int q_tile_start = tile_idx * tile_size;
    int q_tile_end = min(q_tile_start + tile_size, seq_len_q);
    
    if (q_tile_start >= seq_len_q) return;
    
    int num_k_blocks = (seq_len_k + page_size - 1) / page_size;
    int max_k_blocks = (max_seq_k + page_size - 1) / page_size;
    
    int thread_idx = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Process K-blocks
    for (int k_block_idx = thread_idx; k_block_idx < max_k_blocks; k_block_idx += num_threads) {
        int k_block_start = k_block_idx * page_size;
        int k_block_end = min(k_block_start + page_size, seq_len_k);
        
        if (k_block_start >= seq_len_k) {
            // Block is beyond sequence length, mask it out
            int mask_idx = batch_idx * max_seq_q * max_k_blocks + tile_idx * max_k_blocks + k_block_idx;
            mask_ptr[mask_idx] = 0;
            fine_mask_ptr[mask_idx] = 0;
            continue;
        }
        
        // Check if tile needs this block (causal attention)
        bool needs_block = tile_needs_block(q_tile_start, q_tile_end, k_block_start, k_block_end);
        
        int mask_idx = batch_idx * max_seq_q * max_k_blocks + tile_idx * max_k_blocks + k_block_idx;
        
        if (needs_block) {
            // Generate fine-grained mask (per-token)
            uint64_t fine_mask = 0;
            for (int token_idx = 0; token_idx < page_size && (k_block_start + token_idx) < seq_len_k; token_idx++) {
                int k_token_pos = k_block_start + token_idx;
                // Causal mask: Q tokens can only attend to K tokens up to their position
                if (k_token_pos < q_tile_end) {
                    fine_mask |= (1ULL << token_idx);
                }
            }
            
            mask_ptr[mask_idx] = 1;  // Coarse-grained: block is needed
            fine_mask_ptr[mask_idx] = fine_mask;  // Fine-grained: per-token mask
        } else {
            mask_ptr[mask_idx] = 0;
            fine_mask_ptr[mask_idx] = 0;
        }
    }
}

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
) {
    dim3 grid(batch_size, (max_seq_q + tile_size - 1) / tile_size);
    dim3 block(256);  // 256 threads per block
    
    prepare_mask_kernel<<<grid, block, 0, stream>>>(
        mask_ptr,
        fine_mask_ptr,
        cu_seqlens_q,
        cu_seqlens_k,
        seq_lens,
        page_table,
        page_table_lens,
        batch_size,
        max_seq_q,
        max_seq_k,
        page_size,
        tile_size
    );
}

// Wrapper function for Python binding
void prepare_mask_masked_mha(
    torch::Tensor& coarse_mask,
    torch::Tensor& fine_mask,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    const torch::Tensor& seq_lens,
    const torch::Tensor& page_table,
    const torch::Tensor& page_table_lens,
    int64_t batch_size,
    int64_t max_seq_q,
    int64_t max_seq_k,
    int64_t page_size,
    int64_t tile_size
) {
    TORCH_CHECK(coarse_mask.is_cuda(), "coarse_mask must be on CUDA");
    TORCH_CHECK(fine_mask.is_cuda(), "fine_mask must be on CUDA");
    TORCH_CHECK(cu_seqlens_q.is_cuda(), "cu_seqlens_q must be on CUDA");
    TORCH_CHECK(cu_seqlens_k.is_cuda(), "cu_seqlens_k must be on CUDA");
    
    TORCH_CHECK(coarse_mask.dtype() == torch::kInt64, "coarse_mask must be int64");
    TORCH_CHECK(fine_mask.dtype() == torch::kInt64, "fine_mask must be int64");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must be int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must be int32");
    
    c10::cuda::CUDAGuard guard(cu_seqlens_q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const int32_t* cu_seqlens_q_ptr = cu_seqlens_q.data_ptr<int32_t>();
    const int32_t* cu_seqlens_k_ptr = cu_seqlens_k.data_ptr<int32_t>();
    const int32_t* seq_lens_ptr = seq_lens.defined() ? seq_lens.data_ptr<int32_t>() : nullptr;
    const int32_t* page_table_ptr = page_table.defined() ? page_table.data_ptr<int32_t>() : nullptr;
    const int32_t* page_table_lens_ptr = page_table_lens.defined() ? page_table_lens.data_ptr<int32_t>() : nullptr;
    
    uint64_t* coarse_mask_ptr = coarse_mask.data_ptr<uint64_t>();
    uint64_t* fine_mask_ptr = fine_mask.data_ptr<uint64_t>();
    
    prepare_mask(
        coarse_mask_ptr,
        fine_mask_ptr,
        cu_seqlens_q_ptr,
        cu_seqlens_k_ptr,
        seq_lens_ptr,
        page_table_ptr,
        page_table_lens_ptr,
        batch_size,
        max_seq_q,
        max_seq_k,
        page_size,
        tile_size,
        stream
    );
}

// Forward declaration for TMA-based kernel
// The full implementation is in masked_mha_kernel_tma.cu
// For now, we keep the fallback behavior
void masked_mha_attn_tma_impl(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& out,
    const torch::Tensor& coarse_mask,
    const torch::Tensor& fine_mask,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    const torch::Tensor& page_table,
    const torch::Tensor& page_table_lens,
    int64_t batch_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t v_head_dim,
    int64_t max_seq_q,
    int64_t max_seq_k,
    double sm_scale
);

// Main masked MHA attention kernel wrapper
// Tries TMA implementation first, falls back to error (Python layer will use Flash Attention 3)
void masked_mha_attn(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& out,
    const torch::Tensor& coarse_mask,
    const torch::Tensor& fine_mask,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    const torch::Tensor& page_table,
    const torch::Tensor& page_table_lens,
    int64_t batch_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t v_head_dim,
    int64_t max_seq_q,
    int64_t max_seq_k,
    double sm_scale
) {
    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be on CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be on CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
    
    // Check device capability for TMA
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Try TMA implementation if available and device supports it
    if (prop.major >= 9) {
        try {
            masked_mha_attn_tma_impl(
                q, k, v, out,
                coarse_mask, fine_mask,
                cu_seqlens_q, cu_seqlens_k,
                page_table, page_table_lens,
                batch_size, num_heads, num_kv_heads,
                head_dim, v_head_dim,
                max_seq_q, max_seq_k,
                sm_scale
            );
            return;
        } catch (...) {
            // Fall through to error message
        }
    }
    
    // If TMA implementation is not available, raise informative error
    // The Python layer will catch this and use Flash Attention 3 as fallback
    TORCH_CHECK(false,
        "Masked MHA TMA kernel is not yet fully implemented.\n"
        "Current status:\n"
        "- prepare_mask function: ✓ Implemented\n"
        "- TMA kernel structure: ✓ Created (masked_mha_kernel_tma.cu)\n"
        "- TMA kernel implementation: ⚠️  In progress (needs completion)\n"
        "\n"
        "The Python layer will use Flash Attention 3 as fallback.\n"
        "\n"
        "To complete TMA implementation:\n"
        "1. Implement TMA-based Q tile loading (128 tokens)\n"
        "2. Implement K/V page loading with coarse mask check\n"
        "3. Implement CUTLASS GEMM operations\n"
        "4. Implement fine-grained mask application\n"
        "5. Implement online softmax\n"
        "6. Implement output writing\n"
        "\n"
        "Reference: cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp");
}
