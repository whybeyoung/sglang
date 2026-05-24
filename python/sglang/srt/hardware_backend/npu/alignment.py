# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Registration-time 2 MB alignment safety net for CANN HCCL IPC RMA.

The 2 MiB alignment guarantee itself is now provided by torch_npu's
process-wide ``sub_block_alignment`` setting (bound in
``init_npu_backend()`` in ``utils.py``). After that, plain
``torch.zeros(..., device='npu:0')`` already returns a 2 MiB-aligned,
huge-page-backed tensor — there is no longer any allocator helper here.

This module only keeps a cheap fail-fast assertion that is invoked at
HCCL IPC RMA registration time (see
``disaggregation/mooncake/conn.py::register_buffer_to_engine``). It
catches mis-configured deployments — e.g. running on a torch_npu build
without process-wide sub_block_alignment support — and surfaces a
clear error pointing at the misaligned ptr, instead of an opaque
``ADXL Connect 503900`` from the driver.
"""

from __future__ import annotations

ALIGNMENT_BLOCK_2M = 2 * 1024 * 1024


def assert_2m_aligned_kv_args(kv_args) -> None:
    """Fail fast on misaligned kv/aux/state ptrs before HCCL IPC RMA registration."""
    groups = (
        ("kv_data_ptrs", getattr(kv_args, "kv_data_ptrs", None)),
        ("aux_data_ptrs", getattr(kv_args, "aux_data_ptrs", None)),
        ("state_data_ptrs", getattr(kv_args, "state_data_ptrs", None)),
    )
    misaligned = [
        (name, i, ptr)
        for name, ptrs in groups
        if ptrs
        for i, ptr in enumerate(ptrs)
        if ptr % ALIGNMENT_BLOCK_2M != 0
    ]
    if misaligned:
        details = ", ".join(f"{name}[{i}]=0x{ptr:x}" for name, i, ptr in misaligned[:8])
        raise RuntimeError(
            f"NPU PD: {len(misaligned)} buffer(s) not 2 MB-aligned. First: {details}. "
            "Expected torch_npu to provide process-wide sub_block_alignment via "
            "init_npu_backend(); verify the torch_npu build supports the "
            "`sub_block_alignment_kb` allocator setting."
        )
