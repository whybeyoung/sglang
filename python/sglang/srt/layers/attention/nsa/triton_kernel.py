from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# Triton implementation
@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for activation quantization.

    Each block processes BLOCK_M rows and group_size columns.
    """
    # Get block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # FP8 constants
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    # Calculate row and column offsets
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    # Create offset arrays
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Mask for valid rows and columns
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Load input data
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute absolute max along columns (group_size dimension) for each row
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)  # Shape: (BLOCK_M,)

    # Clamp amax to avoid division by zero
    amax = tl.maximum(amax, 1e-4)

    # Compute scale
    if round_scale:
        # Fast round scale using bit manipulation approximation
        # This is a simplified version - the exact bit manipulation is harder in Triton
        # Using log2 + ceil + pow2 as approximation
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    # Store quantized output
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    # Store scales
    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # Flatten all dims except last
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    # Allocate output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)

    # Launch kernel
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y, s


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 16},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 4},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 4},
            num_stages=2,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_weights_proj_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    Q_Scale_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_outm,
    stride_outn,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Pointers to X and W
    # X: [M, K], W: [N, K]
    # Out: [M, N]
    
    # Calculate ranges
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Split K dimension
    total_k = K
    k_chunk = tl.cdiv(total_k, SPLIT_K)
    k_start = pid_k * k_chunk
    k_end = tl.minimum(k_start + k_chunk, total_k)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Masks
    mask_m = rm < M
    mask_n = rn < N
    
    # Pointers
    # x_ptrs = X_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    # w_ptrs = W_ptr + (rn[:, None] * stride_wn + rk[None, :] * stride_wk)
    
    # Loop over K
    for k in range(k_start, k_end, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        mask_k = rk < k_end # Check against k_end which is capped at total_k
        
        # Load blocks
        # x shape: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # w shape: [BLOCK_N, BLOCK_K]
        w_ptrs = W_ptr + (rn[:, None] * stride_wn + rk[None, :] * stride_wk)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Dot product
        # x: [M, K], w: [N, K]. dot(x, w.T)
        acc += tl.dot(x, w.T)
        
    # Apply scales
    # q_scale: [M]
    qs_ptrs = Q_Scale_ptr + rm
    qs = tl.load(qs_ptrs, mask=mask_m, other=0.0)
    
    acc = acc * qs[:, None] * softmax_scale
    
    # Store with atomic add
    out_ptrs = Out_ptr + (rm[:, None] * stride_outm + rn[None, :] * stride_outn)
    tl.atomic_add(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def fused_weights_proj(
    x: torch.Tensor,
    weights: torch.Tensor,
    q_scale: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Fused implementation of:
    1. x.float() @ weights.T
    2. * q_scale
    3. * softmax_scale
    
    x: [M, K] (bf16/fp16/fp32)
    weights: [N, K] (bf16/fp16/fp32)
    q_scale: [M] (fp32)
    """
    M, K = x.shape
    N = weights.shape[0]
    assert weights.shape[1] == K
    
    # Output buffer (zeroed for atomic add)
    out = torch.zeros((M, N), dtype=torch.float32, device=x.device)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
        META['SPLIT_K']
    )
    
    _fused_weights_proj_kernel[grid](
        x, weights, out, q_scale,
        M, N, K,
        x.stride(0), x.stride(1),
        weights.stride(0), weights.stride(1),
        out.stride(0), out.stride(1),
        softmax_scale,
    )
    
    return out
