"""Utilities for building sparse_mask_fine bitmap from NSA topk indices (PR#24, SM90).

sparse_mask_fine shape: [total_q, max_k_blocks_padded, num_int32_per_block]
  - max_k_blocks_padded: padded so that (max_k_blocks_padded * num_int32_per_block * 4) % 128 == 0
    (TMA 128-byte alignment requirement)
  - num_int32_per_block = kBlockN // 32  (kBlockN is always a multiple of 32)
  - bit kb*32+b is set iff K-token (kb*kBlockN + b) is in the attend set for this Q token.

For NSA, topk selects whole K-blocks, so setting an entire block = setting all its bits.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_indices_to_sparse_mask_kernel(
    topk_indices_ptr,   # [total_q, topk]  int32, -1 = padding
    mask_ptr,           # [total_q, max_k_blocks_padded, num_int32_per_block]  int32, zeroed
    total_q,
    topk: tl.constexpr,
    kBlockN: tl.constexpr,
    max_k_blocks_padded,
    num_int32_per_block: tl.constexpr,  # = kBlockN // 32
    mask_row_stride,    # = max_k_blocks_padded * num_int32_per_block
):
    """Each program handles one Q-token row.

    For each valid topk index, set the corresponding K-block's bits to all-1.
    """
    q_idx = tl.program_id(0)
    if q_idx >= total_q:
        return

    topk_base = topk_indices_ptr + q_idx * topk
    mask_row_base = mask_ptr + q_idx * mask_row_stride

    # Load all topk indices for this Q token
    offsets = tl.arange(0, topk)
    indices = tl.load(topk_base + offsets)  # [topk]

    valid_mask = indices >= 0
    # K-block index for each topk entry
    kb = indices // kBlockN  # [topk]

    # For each valid (q, topk) pair: set block kb's num_int32_per_block int32s to -1 (all bits 1)
    # We scatter-write to mask_row_base[kb * num_int32_per_block + word] = -1 for each word.
    # Use atomicOr to handle multiple topk indices mapping to the same block safely.
    word_offsets = tl.arange(0, num_int32_per_block)  # [num_int32_per_block]

    for i in tl.static_range(topk):
        is_valid = tl.load(topk_base + i) >= 0
        if is_valid:
            kb_i = tl.load(topk_base + i) // kBlockN
            base = mask_row_base + kb_i * num_int32_per_block
            tl.atomic_or(base + word_offsets, tl.full([num_int32_per_block], -1, dtype=tl.int32))


def topk_indices_to_sparse_mask(
    topk_indices: torch.Tensor,   # [total_q, topk], int32, -1=padding, values are ragged K positions
    kBlockN: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    """Convert NSA topk_indices to sparse_mask_fine bitmap for FA3 varlen sparse attention.

    Args:
        topk_indices: [total_q, topk] int32 tensor. Each row lists the K-token positions
            (relative to the sequence start in ragged KV layout) that Q token i attends to.
            -1 entries are padding.
        kBlockN: K-tile block size from flash_attn_get_tile_size (typically 64 or 128).
        max_seqlen_k: Maximum K sequence length across all sequences in the batch.

    Returns:
        sparse_mask_fine: [total_q, max_k_blocks_padded, num_int32_per_block] int32 tensor,
            TMA-128-byte-aligned (zeroed then filled).
    """
    assert topk_indices.dtype == torch.int32
    assert topk_indices.is_contiguous()

    total_q, topk = topk_indices.shape
    assert kBlockN % 32 == 0, f"kBlockN must be multiple of 32, got {kBlockN}"
    num_int32_per_block = kBlockN // 32
    max_k_blocks = (max_seqlen_k + kBlockN - 1) // kBlockN

    # TMA requires (max_k_blocks * num_int32_per_block * 4) % 128 == 0
    # i.e. row_int32_count % 32 == 0
    row_int32_count = max_k_blocks * num_int32_per_block
    row_int32_padded = (row_int32_count + 31) // 32 * 32
    max_k_blocks_padded = row_int32_padded // num_int32_per_block

    mask = torch.zeros(
        (total_q, max_k_blocks_padded, num_int32_per_block),
        dtype=torch.int32,
        device=topk_indices.device,
    )

    if total_q == 0:
        return mask

    # Round topk up to the next power of 2 for triton constexpr
    topk_pow2 = triton.next_power_of_2(topk)
    # Pad topk_indices to topk_pow2 if needed
    if topk_pow2 != topk:
        pad = torch.full(
            (total_q, topk_pow2 - topk),
            -1,
            dtype=torch.int32,
            device=topk_indices.device,
        )
        topk_indices_padded = torch.cat([topk_indices, pad], dim=1)
    else:
        topk_indices_padded = topk_indices

    # Also round num_int32_per_block to next power of 2 for constexpr
    nib_pow2 = triton.next_power_of_2(num_int32_per_block)

    mask_row_stride = max_k_blocks_padded * num_int32_per_block
    grid = (total_q,)

    _topk_indices_to_sparse_mask_kernel[grid](
        topk_indices_padded,
        mask,
        total_q,
        topk=topk_pow2,
        kBlockN=kBlockN,
        max_k_blocks_padded=max_k_blocks_padded,
        num_int32_per_block=nib_pow2,
        mask_row_stride=mask_row_stride,
    )

    return mask
