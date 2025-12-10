"""
Masked MHA integration for NSA Backend

This module adds masked MHA support to NativeSparseAttnBackend,
allowing sequences > 2048 tokens to use MHA path with masked attention.

Current implementation: Uses Flash Attention 3 as fallback until full TMA kernel is implemented.
This allows sequences > 2048 to work, but doesn't optimally apply fine-grained masks.
"""

from typing import Optional
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.layers.attention.nsa_backend import NSAMetadata


def _forward_masked_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    metadata: NSAMetadata,
) -> torch.Tensor:
    """
    Masked MHA using custom kernel for sequences > 2048.
    
    This allows sequences > 2048 tokens to use MHA path
    with masked attention, avoiding the 3.4x performance penalty
    of forcing them through sparse MLA.
    
    Current implementation: Uses Flash Attention 3 as fallback.
    Flash Attention 3 handles causal masking internally, which works
    for sequences > 2048, but doesn't apply the fine-grained masks
    optimally. Full TMA-based implementation will be added later.
    """
    # Try to use masked_mha_attn kernel first
    # If it's not implemented yet, fall back to Flash Attention 3
    try:
        from sgl_kernel.masked_mha import masked_mha_attn
        
        # Reshape tensors
        q_reshaped = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k_reshaped = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v_reshaped = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        
        # Prepare masks (even though they might not be used in fallback)
        from sgl_kernel.masked_mha import prepare_mask
        page_size = getattr(forward_batch, 'page_size', 64)
        tile_size = 128
        coarse_mask, fine_mask = prepare_mask(
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            seq_lens=metadata.cache_seqlens_int32,
            page_table=forward_batch.page_table if hasattr(forward_batch, 'page_table') else None,
            page_table_lens=None,
            max_seq_q=metadata.max_seq_len_q,
            max_seq_k=metadata.max_seq_len_k,
            page_size=page_size,
            tile_size=tile_size,
        )
        
        # Try to call masked_mha_attn
        # This will fail if kernel is not implemented, triggering fallback
        out = masked_mha_attn(
            q=q_reshaped,
            k=k_reshaped,
            v=v_reshaped,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            coarse_mask=coarse_mask,
            fine_mask=fine_mask,
            sm_scale=layer.scaling,
            max_seq_q=metadata.max_seq_len_q,
            max_seq_k=metadata.max_seq_len_k,
        )
        
        return out
        
    except (RuntimeError, ImportError) as e:
        # Fall back to Flash Attention 3
        # This works for sequences > 2048 but doesn't use fine-grained masks
        from sgl_kernel.flash_attn import flash_attn_varlen_func
        
        # Reshape tensors
        q_reshaped = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k_reshaped = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v_reshaped = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        
        # Get sequence information
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        max_seqlen_k = metadata.max_seq_len_k
        
        # Use Flash Attention 3 with causal masking
        # Note: Flash Attention 3 already handles causal masking internally
        # This works for sequences > 2048, but doesn't use our fine-grained masks
        # Full implementation will use masked_mha_attn with TMA and proper mask support
        out = flash_attn_varlen_func(
            q=q_reshaped,
            k=k_reshaped,
            v=v_reshaped,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=layer.scaling,
            causal=True,
            ver=3,
        )
        
        return out
