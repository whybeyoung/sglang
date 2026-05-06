# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""NPU-specific 2 MB alignment helper for HCCL IPC RMA registration.

Why this exists
---------------
CANN HCCL's IPC RMA buffer export path (rtsIpcMemGetExportKey ->
halShmemCreateHandle) requires every registered buffer's start address
to be aligned to the HCCL page size (2 MB / page_size=2097152).
Otherwise ADXL Connect fails later with status 503900.

The PyTorch NPU caching allocator does not reliably guarantee 2 MB
alignment on Ascend 910C, even for allocations >= 2 MB
(observed data_ptr()=0x12d3a4144000 for a 2 MB alloc).

Strategy (per CANN HCCL FAQ)
----------------------------
The vendor recommends two options when registering small buffers:

1. Use ACL_MEM_MALLOC_HUGE_ONLY when allocating, so the driver returns
   a 2 MB-aligned huge page directly. This requires bypassing the
   PyTorch caching allocator and is invasive.

2. Allocate normally, then self-align using:

       aligned_addr = (addr + ALIGNMENT_BLOCK - 1)
                      // ALIGNMENT_BLOCK * ALIGNMENT_BLOCK
       aligned_mem_size = mem_size - (aligned_addr - addr)

   Register [aligned_addr, aligned_addr + aligned_mem_size) instead of
   [addr, addr + mem_size).

We use option 2 to stay inside PyTorch. To guarantee the aligned region
is large enough to hold the caller's requested ``shape``, we
over-allocate by exactly ``ALIGNMENT_BLOCK`` extra bytes; after moving
the start forward by at most ``ALIGNMENT_BLOCK - 1`` bytes there is
always >= ``shape`` bytes left.

The returned tensor's logical shape / dtype / nbytes are exactly what
the caller asked for, so Mooncake's ``get_buf_infos`` (which uses
``.data_ptr()`` and ``.nbytes`` of the view) registers the aligned
region of the correct length.

Non-NPU runtimes and non-NPU devices return ``torch.zeros(...)``
unchanged. GPU and CPU paths are bit-exact identical to before.
"""

from __future__ import annotations

from numbers import Integral
from typing import Union

import torch

from sglang.srt.utils import is_npu

# CANN HCCL IPC RMA page size, per the FAQ formula.
ALIGNMENT_BLOCK = 2 * 1024 * 1024


def _device_type(device: Union[str, torch.device, None]) -> str:
    """Return the bare device type ('npu' / 'cuda' / 'cpu' / ...)."""
    if device is None:
        return ""
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def zeros(shape, dtype, device):
    """torch.zeros(shape, dtype, device) with a 2 MB aligned data_ptr() on NPU.

    Non-NPU runtimes and non-NPU devices return
    ``torch.zeros(shape, dtype=dtype, device=device)`` unchanged.
    """
    # Only enter the alignment path when both the runtime IS NPU and the
    # caller explicitly requests an NPU device. This protects the CPU
    # fallback used by MetadataBuffers (custom_mem_pool / NVLink) on hosts
    # where is_npu() may still be true.
    if not is_npu() or _device_type(device) != "npu":
        return torch.zeros(shape, dtype=dtype, device=device)

    # Normalize shape to a tuple of plain Python ints so .view() never
    # trips on list / torch.Size / np.int64 inputs.
    if isinstance(shape, Integral):
        view_shape = (int(shape),)
    else:
        view_shape = tuple(int(s) for s in shape)

    n_elems = 1
    for s in view_shape:
        n_elems *= s

    elem_size = torch.empty(0, dtype=dtype).element_size()
    if ALIGNMENT_BLOCK % elem_size != 0:
        # All standard torch dtypes (1/2/4/8 bytes) divide 2 MB; assert
        # explicitly so a future exotic dtype fails loudly here rather
        # than producing a misaligned offset later.
        raise RuntimeError(
            f"ALIGNMENT_BLOCK={ALIGNMENT_BLOCK} not divisible by "
            f"elem_size={elem_size}"
        )

    requested_bytes = n_elems * elem_size

    # Over-allocate by exactly ALIGNMENT_BLOCK so that, after shifting
    # the start forward by up to (ALIGNMENT_BLOCK - 1) bytes to reach
    # the next 2 MB boundary, the remaining aligned region is still
    # >= requested_bytes.
    mem_size = requested_bytes + ALIGNMENT_BLOCK
    total_elems = mem_size // elem_size
    flat = torch.zeros(total_elems, dtype=dtype, device=device)

    # CANN HCCL FAQ alignment formula:
    #     aligned_addr = (addr + ALIGNMENT_BLOCK - 1)
    #                    // ALIGNMENT_BLOCK * ALIGNMENT_BLOCK
    #     aligned_mem_size = mem_size - (aligned_addr - addr)
    addr = flat.data_ptr()
    aligned_addr = (
        (addr + ALIGNMENT_BLOCK - 1) // ALIGNMENT_BLOCK * ALIGNMENT_BLOCK
    )
    head_offset_bytes = aligned_addr - addr
    aligned_mem_size = mem_size - head_offset_bytes

    if head_offset_bytes % elem_size != 0:
        # Cannot happen for standard dtypes (elem_size divides
        # ALIGNMENT_BLOCK and addr is elem_size aligned), but guard.
        raise RuntimeError(
            f"NPU base 0x{addr:x} not aligned to elem_size={elem_size}"
        )
    if aligned_mem_size < requested_bytes:
        raise RuntimeError(
            "Aligned region too small: "
            f"aligned_mem_size={aligned_mem_size} < "
            f"requested_bytes={requested_bytes}"
        )

    head_offset_elems = head_offset_bytes // elem_size
    out = flat[head_offset_elems : head_offset_elems + n_elems].view(view_shape)

    if out.data_ptr() != aligned_addr:
        raise RuntimeError(
            "Self-align view produced an unexpected data_ptr(): "
            f"addr=0x{addr:x}, aligned_addr=0x{aligned_addr:x}, "
            f"out=0x{out.data_ptr():x}"
        )
    if out.data_ptr() % ALIGNMENT_BLOCK != 0:
        raise RuntimeError(
            "Self-align failed despite over-allocation: "
            f"addr=0x{addr:x}, head_offset_bytes=0x{head_offset_bytes:x}, "
            f"out=0x{out.data_ptr():x}"
        )
    # `out` keeps `flat`'s storage alive via tensor storage refcounting,
    # so the parent allocation is not freed when this function returns.
    return out
