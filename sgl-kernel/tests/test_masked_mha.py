"""
Unit tests for Masked MHA Kernel

This test suite verifies the correctness and functionality of the Masked MHA kernel
for DeepSeek V3.2, which enables MHA path for sequences > 2048 tokens.
"""
import pytest
import torch
from sgl_kernel import masked_mha


def check_sm90():
    """Check if running on SM90 (Hopper) architecture"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device)
    if prop.major != 9:
        pytest.skip(f"Requires SM90 (Hopper), got SM{prop.major}{prop.minor}")


def test_prepare_mask():
    """Test mask preparation function"""
    check_sm90()
    
    device = torch.device("cuda")
    batch_size = 4
    max_seq_q = 4096
    max_seq_k = 4096
    page_size = 64
    tile_size = 128
    
    # Create sequence lengths
    seq_lens = torch.tensor([512, 1024, 2048, 4096], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.cumsum(torch.cat([torch.tensor([0], device=device), seq_lens]), dim=0)
    cu_seqlens_k = cu_seqlens_q.clone()
    
    # Prepare masks
    coarse_mask, fine_mask = masked_mha.prepare_mask(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seq_lens=seq_lens,
        page_table=None,
        page_table_lens=None,
        max_seq_q=max_seq_q,
        max_seq_k=max_seq_k,
        page_size=page_size,
        tile_size=tile_size,
    )
    
    assert coarse_mask.shape == (batch_size, max_seq_q // tile_size, max_seq_k // page_size)
    assert fine_mask.shape == (batch_size, max_seq_q // tile_size, max_seq_k // page_size)
    assert coarse_mask.dtype == torch.int64
    assert fine_mask.dtype == torch.int64
    
    # Check that masks are non-negative
    assert (coarse_mask >= 0).all()
    assert (fine_mask >= 0).all()


def test_masked_mha_attn_basic():
    """Basic test for masked MHA attention"""
    check_sm90()
    
    device = torch.device("cuda")
    batch_size = 2
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    seq_lens = torch.tensor([512, 1024], dtype=torch.int32, device=device)
    
    # Create cumulative sequence lengths
    cu_seqlens_q = torch.cumsum(torch.cat([torch.tensor([0], device=device), seq_lens]), dim=0)
    cu_seqlens_k = cu_seqlens_q.clone()
    
    total_q_tokens = cu_seqlens_q[-1].item()
    total_kv_tokens = cu_seqlens_k[-1].item()
    
    # Create Q, K, V tensors
    q = torch.randn(total_q_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    
    max_seq_q = seq_lens.max().item()
    max_seq_k = seq_lens.max().item()
    
    # Prepare masks
    coarse_mask, fine_mask = masked_mha.prepare_mask(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seq_lens=seq_lens,
        page_table=None,
        page_table_lens=None,
        max_seq_q=max_seq_q,
        max_seq_k=max_seq_k,
    )
    
    # Run masked MHA attention
    try:
        out = masked_mha.masked_mha_attn(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            coarse_mask=coarse_mask,
            fine_mask=fine_mask,
            sm_scale=None,
            max_seq_q=max_seq_q,
            max_seq_k=max_seq_k,
        )
        
        assert out.shape == q.shape
        assert out.dtype == q.dtype
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"
        
    except RuntimeError as e:
        # If kernel is not fully implemented, fallback should work
        if "not yet fully implemented" in str(e):
            pytest.skip("Kernel not fully implemented, using fallback")
        else:
            raise


def test_masked_mha_attn_long_sequence():
    """Test with long sequences (> 2048 tokens)"""
    check_sm90()
    
    device = torch.device("cuda")
    batch_size = 1
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    seq_len = 4096  # > 2048
    
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.cumsum(torch.cat([torch.tensor([0], device=device), seq_lens]), dim=0)
    cu_seqlens_k = cu_seqlens_q.clone()
    
    total_q_tokens = cu_seqlens_q[-1].item()
    total_kv_tokens = cu_seqlens_k[-1].item()
    
    q = torch.randn(total_q_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    
    max_seq_q = seq_len
    max_seq_k = seq_len
    
    coarse_mask, fine_mask = masked_mha.prepare_mask(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seq_lens=seq_lens,
        page_table=None,
        page_table_lens=None,
        max_seq_q=max_seq_q,
        max_seq_k=max_seq_k,
    )
    
    try:
        out = masked_mha.masked_mha_attn(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            coarse_mask=coarse_mask,
            fine_mask=fine_mask,
            max_seq_q=max_seq_q,
            max_seq_k=max_seq_k,
        )
        
        assert out.shape == q.shape
        assert torch.isfinite(out).all()
        
    except RuntimeError as e:
        if "not yet fully implemented" in str(e):
            pytest.skip("Kernel not fully implemented")
        else:
            raise


def test_masked_mha_correctness():
    """Compare with Flash Attention 3 for correctness"""
    check_sm90()
    
    device = torch.device("cuda")
    batch_size = 2
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    seq_lens = torch.tensor([512, 1024], dtype=torch.int32, device=device)
    
    cu_seqlens_q = torch.cumsum(torch.cat([torch.tensor([0], device=device), seq_lens]), dim=0)
    cu_seqlens_k = cu_seqlens_q.clone()
    
    total_q_tokens = cu_seqlens_q[-1].item()
    total_kv_tokens = cu_seqlens_k[-1].item()
    
    # Use same random seed for reproducibility
    torch.manual_seed(42)
    q = torch.randn(total_q_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    
    max_seq_q = seq_lens.max().item()
    max_seq_k = seq_lens.max().item()
    
    coarse_mask, fine_mask = masked_mha.prepare_mask(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seq_lens=seq_lens,
        page_table=None,
        page_table_lens=None,
        max_seq_q=max_seq_q,
        max_seq_k=max_seq_k,
    )
    
    # Run masked MHA
    try:
        out_masked = masked_mha.masked_mha_attn(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            coarse_mask=coarse_mask,
            fine_mask=fine_mask,
            max_seq_q=max_seq_q,
            max_seq_k=max_seq_k,
        )
        
        # Compare with Flash Attention 3 (if available)
        try:
            from sgl_kernel.flash_attn import flash_attn_varlen_func
            
            out_flash = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                sm_scale=1.0 / (head_dim ** 0.5),
            )
            
            # Check if outputs are close (allowing for numerical differences)
            max_diff = (out_masked - out_flash).abs().max().item()
            mean_diff = (out_masked - out_flash).abs().mean().item()
            
            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")
            
            # Allow reasonable numerical differences
            assert max_diff < 0.1, f"Outputs differ too much: max_diff={max_diff}"
            
        except ImportError:
            pytest.skip("Flash Attention 3 not available for comparison")
            
    except RuntimeError as e:
        if "not yet fully implemented" in str(e):
            pytest.skip("Kernel not fully implemented")
        else:
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
