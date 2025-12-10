"""
Masked MHA Kernel Python Bindings

This module provides Python bindings for the masked MHA kernel,
allowing short sequences (> 2048 tokens) to use MHA path with masked attention.
"""

import torch
from typing import Optional

try:
    from sgl_kernel import flash_ops  # C++ extension via torch library
    _masked_mha_available = True
except ImportError:
    _masked_mha_available = False
    flash_ops = None


def prepare_mask(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: Optional[torch.Tensor],
    page_table_lens: Optional[torch.Tensor],
    max_seq_q: int,
    max_seq_k: int,
    page_size: int = 64,
    tile_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare coarse-grained and fine-grained masks for masked MHA.
    
    Args:
        cu_seqlens_q: Cumulative sequence lengths for Q: [batch_size + 1]
        cu_seqlens_k: Cumulative sequence lengths for K: [batch_size + 1]
        seq_lens: Sequence lengths: [batch_size]
        page_table: Page table for KV cache: [batch_size, max_pages]
        page_table_lens: Number of pages per sequence: [batch_size]
        max_seq_q: Maximum sequence length for Q
        max_seq_k: Maximum sequence length for K
        page_size: Size of each KV cache page (default: 64)
        tile_size: Size of each Q tile (default: 128)
    
    Returns:
        tuple: (coarse_mask, fine_mask)
            - coarse_mask: [batch_size, max_seq_q // tile_size, max_seq_k // page_size]
            - fine_mask: [batch_size, max_seq_q // tile_size, max_seq_k // page_size]
                Each uint64_t represents 64 tokens (one page)
    """
    if not _masked_mha_available:
        raise RuntimeError("Masked MHA kernel is not available. Please compile the C++ extension.")
    
    batch_size = len(cu_seqlens_q) - 1
    max_q_tiles = (max_seq_q + tile_size - 1) // tile_size
    max_k_blocks = (max_seq_k + page_size - 1) // page_size
    
    # Allocate mask tensors
    coarse_mask = torch.zeros(
        (batch_size, max_q_tiles, max_k_blocks),
        dtype=torch.int64,
        device=cu_seqlens_q.device
    )
    
    fine_mask = torch.zeros(
        (batch_size, max_q_tiles, max_k_blocks),
        dtype=torch.int64,
        device=cu_seqlens_q.device
    )
    
    # Ensure tensors are contiguous and on correct device
    cu_seqlens_q = cu_seqlens_q.contiguous().to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.contiguous().to(torch.int32)
    seq_lens = seq_lens.contiguous().to(torch.int32) if seq_lens is not None else None
    page_table = page_table.contiguous().to(torch.int32) if page_table is not None else None
    page_table_lens = page_table_lens.contiguous().to(torch.int32) if page_table_lens is not None else None
    
    # Call C++ function via torch library
    torch.ops.sgl_kernel.prepare_mask_masked_mha(
        coarse_mask,
        fine_mask,
        cu_seqlens_q,
        cu_seqlens_k,
        seq_lens if seq_lens is not None else torch.empty(0, dtype=torch.int32, device=cu_seqlens_q.device),
        page_table if page_table is not None else torch.empty(0, dtype=torch.int32, device=cu_seqlens_q.device),
        page_table_lens if page_table_lens is not None else torch.empty(0, dtype=torch.int32, device=cu_seqlens_q.device),
        batch_size,
        max_seq_q,
        max_seq_k,
        page_size,
        tile_size,
    )
    
    return coarse_mask, fine_mask


def masked_mha_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    coarse_mask: torch.Tensor,
    fine_mask: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_table_lens: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    max_seq_q: Optional[int] = None,
    max_seq_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Masked MHA attention kernel.
    
    Args:
        q: Query tensor: [total_q_tokens, num_heads, head_dim]
        k: Key tensor: [total_kv_tokens, num_kv_heads, head_dim]
        v: Value tensor: [total_kv_tokens, num_kv_heads, head_dim]
        cu_seqlens_q: Cumulative sequence lengths for Q: [batch_size + 1]
        cu_seqlens_k: Cumulative sequence lengths for K: [batch_size + 1]
        coarse_mask: Coarse-grained mask: [batch_size, max_q_tiles, max_k_blocks]
        fine_mask: Fine-grained mask: [batch_size, max_q_tiles, max_k_blocks]
        page_table: Page table for KV cache: [batch_size, max_pages]
        page_table_lens: Number of pages per sequence: [batch_size]
        sm_scale: Softmax scale (default: 1.0 / sqrt(head_dim))
        max_seq_q: Maximum sequence length for Q
        max_seq_k: Maximum sequence length for K
    
    Returns:
        Output tensor: [total_q_tokens, num_heads, head_dim]
    """
    if not _masked_mha_available:
        raise RuntimeError("Masked MHA kernel is not available. Please compile the C++ extension.")
    
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    if max_seq_q is None:
        max_seq_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    
    if max_seq_k is None:
        max_seq_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
    
    batch_size = len(cu_seqlens_q) - 1
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    v_head_dim = v.shape[2]
    
    # Ensure tensors are contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous().to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.contiguous().to(torch.int32)
    coarse_mask = coarse_mask.contiguous()
    fine_mask = fine_mask.contiguous()
    
    # Allocate output tensor
    out = torch.empty_like(q)
    
    # Call C++ kernel via torch library
    torch.ops.sgl_kernel.masked_mha_attn(
        q,
        k,
        v,
        out,
        coarse_mask,
        fine_mask,
        cu_seqlens_q,
        cu_seqlens_k,
        page_table if page_table is not None else torch.empty(0, dtype=torch.int32, device=q.device),
        page_table_lens if page_table_lens is not None else torch.empty(0, dtype=torch.int32, device=q.device),
        batch_size,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        max_seq_q,
        max_seq_k,
        sm_scale,
    )
    
    return out
