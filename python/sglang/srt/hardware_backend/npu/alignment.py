# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fail-fast 2 MiB alignment check for CANN HCCL IPC RMA.

Alignment is provided process-wide by torch_npu's sub_block_alignment
(see init_npu_backend). This assert is the safety net invoked at
register-time (mooncake/conn.py::register_buffer_to_engine) to surface
a clear pointer error instead of an opaque ADXL 503900.
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
            f"NPU PD: {len(misaligned)} buffer(s) not 2 MiB-aligned. First: {details}. "
            "Upgrade torch_npu to a build with sub_block_alignment_kb support."
        )
