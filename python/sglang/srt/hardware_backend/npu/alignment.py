# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""2 MB alignment helpers for CANN HCCL IPC RMA registration.

HCCL IPC RMA needs every registered ``(ptr, len)`` to be 2 MB-aligned (base and
length). Non-NPU devices pass through to ``torch.zeros``.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, List, Sequence, Tuple, Union

import torch

from sglang.srt.utils import is_npu

ALIGNMENT_BLOCK_2M = 2 * 1024 * 1024

# One block absorbs the base-align shift, the other lets a region's length round
# up to 2 MB while staying inside the (tensor-owned) allocation.
_OVERALLOC_2M = 2 * ALIGNMENT_BLOCK_2M


def _device_type(device: Union[str, torch.device, None]) -> str:
    """Bare device type ('npu' / 'cuda' / 'cpu' / ...)."""
    if device is None:
        return ""
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def _is_npu_device(device: Any) -> bool:
    return is_npu() and _device_type(device) == "npu"


def zeros_2m_aligned(shape, dtype, device) -> torch.Tensor:
    """``torch.zeros`` with 2 MB-aligned ``data_ptr()`` on NPU; passthrough elsewhere."""
    if not _is_npu_device(device):
        return torch.zeros(shape, dtype=dtype, device=device)

    if isinstance(shape, Integral):
        view_shape = (int(shape),)
    else:
        view_shape = tuple(int(s) for s in shape)
    n_elems = 1
    for s in view_shape:
        n_elems *= s

    elem_size = torch.empty(0, dtype=dtype).element_size()
    if ALIGNMENT_BLOCK_2M % elem_size != 0:
        raise RuntimeError(f"ALIGNMENT_BLOCK_2M not divisible by elem_size={elem_size}")

    # Over-allocate two 2 MB blocks so that after the forward shift to a
    # 2 MB-aligned base there is still >= requested_bytes rounded up to the
    # next 2 MB boundary of valid memory (see _OVERALLOC_2M).
    requested_bytes = n_elems * elem_size
    mem_size = requested_bytes + _OVERALLOC_2M
    flat = torch.zeros(mem_size // elem_size, dtype=dtype, device=device)

    addr = flat.data_ptr()
    aligned_addr = (
        (addr + ALIGNMENT_BLOCK_2M - 1) // ALIGNMENT_BLOCK_2M * ALIGNMENT_BLOCK_2M
    )
    head_offset_bytes = aligned_addr - addr
    aligned_mem_size = mem_size - head_offset_bytes

    if head_offset_bytes % elem_size != 0:
        raise RuntimeError(f"NPU base 0x{addr:x} not aligned to elem_size={elem_size}")
    if aligned_mem_size < requested_bytes:
        raise RuntimeError(
            f"Aligned region too small: aligned_mem_size={aligned_mem_size}"
            f" < requested_bytes={requested_bytes}"
        )

    head_offset_elems = head_offset_bytes // elem_size
    out = flat[head_offset_elems : head_offset_elems + n_elems].view(view_shape)

    if out.data_ptr() != aligned_addr:
        raise RuntimeError(
            f"Self-align view data_ptr mismatch: aligned=0x{aligned_addr:x}"
            f" out=0x{out.data_ptr():x}"
        )
    if out.data_ptr() % ALIGNMENT_BLOCK_2M != 0:
        raise RuntimeError(f"Self-align failed: out=0x{out.data_ptr():x}")
    # `out` keeps `flat`'s storage alive via tensor refcount.
    return out


def ipc_register_regions(
    ptrs: Sequence[int], lengths: Sequence[int]
) -> Tuple[List[int], List[int]]:
    """Per-layer ``(ptr, len)`` -> 2 MB-aligned HCCL IPC registration regions.

    The KV pool is one contiguous ``[layer, ...]`` tensor, so per-layer ptrs are
    not 2 MB-aligned. Merge the adjacent per-layer runs back into their parent
    allocation (2 MB-aligned base) and round each length up to 2 MB; transfers
    still address sub-ranges inside these regions.
    """
    if not ptrs:
        return [], []

    order = sorted(range(len(ptrs)), key=lambda i: int(ptrs[i]))
    merged: List[List[int]] = []  # [start, end)
    for i in order:
        start = int(ptrs[i])
        end = start + int(lengths[i])
        if merged and start == merged[-1][1]:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    reg_ptrs: List[int] = []
    reg_lens: List[int] = []
    for start, end in merged:
        if start % ALIGNMENT_BLOCK_2M != 0:
            raise RuntimeError(
                f"NPU IPC region base 0x{start:x} is not 2 MB-aligned; "
                f"buffer must be allocated via zeros_2m_aligned."
            )
        length = end - start
        aligned_len = (
            (length + ALIGNMENT_BLOCK_2M - 1) // ALIGNMENT_BLOCK_2M
        ) * ALIGNMENT_BLOCK_2M
        reg_ptrs.append(start)
        reg_lens.append(aligned_len)
    return reg_ptrs, reg_lens
