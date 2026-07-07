"""NPU HCCL IPC 2MB alignment helpers (Mooncake ascend / NPU device memory)."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

NPU_IPC_PAGE_SIZE_2MB = 2097152


def needs_npu_ipc_alignment() -> bool:
    """NPU Mooncake registers device memory via HCCL IPC (2MB page rules)."""
    try:
        from sglang.srt.utils.common import is_npu

        return is_npu()
    except (RuntimeError, ImportError):
        return False


def ipc_region_aligned(ptr: int, length: int) -> bool:
    page = NPU_IPC_PAGE_SIZE_2MB
    return ptr % page == 0 and length % page == 0 and length > 0


def align_npu_ipc_length(length: int) -> int:
    page = NPU_IPC_PAGE_SIZE_2MB
    return ((length + page - 1) // page) * page


def align_npu_ipc_region(ptr: int, length: int) -> Tuple[int, int]:
    """Expand [ptr, ptr+length) to a 2MB-aligned superset for HCCL IPC registration."""
    page = NPU_IPC_PAGE_SIZE_2MB
    aligned_ptr = ptr & ~(page - 1)
    end = ptr + length
    aligned_end = ((end + page - 1) // page) * page
    return aligned_ptr, aligned_end - aligned_ptr


def align_npu_ipc_regions(
    ptrs: Sequence[int], lengths: Sequence[int]
) -> Tuple[List[int], List[int]]:
    aligned_ptrs: List[int] = []
    aligned_lengths: List[int] = []
    for ptr, length in zip(ptrs, lengths):
        ap, al = align_npu_ipc_region(int(ptr), int(length))
        aligned_ptrs.append(ap)
        aligned_lengths.append(al)
    return aligned_ptrs, aligned_lengths


def coalesce_npu_ipc_regions(
    ptrs: Sequence[int], lengths: Sequence[int]
) -> Tuple[List[int], List[int]]:
    """Align each buffer then merge overlapping 2MB-padded intervals for HCCL IPC.

    Padding adjacent layers independently causes HCCL overlap errors like:
    new [0x12d36dc00000, +24MB) overlaps existing [0x12d36c600000, +24MB).
    """
    if not ptrs:
        return [], []

    intervals: List[Tuple[int, int]] = []
    for ptr, length in zip(ptrs, lengths):
        ap, al = align_npu_ipc_region(int(ptr), int(length))
        intervals.append((ap, ap + al))

    intervals.sort(key=lambda item: item[0])
    merged: List[Tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    merged_ptrs = [start for start, _ in merged]
    merged_lens = [end - start for start, end in merged]
    return merged_ptrs, merged_lens


def prepare_npu_ipc_register_regions(
    ptrs: Sequence[int], lengths: Sequence[int]
) -> Tuple[List[int], List[int]]:
    """Regions to pass to Mooncake/HCCL register_memory (aligned + coalesced)."""
    return coalesce_npu_ipc_regions(ptrs, lengths)


def should_log_npu_ipc_alignment() -> bool:
    if envs.SGLANG_NPU_IPC_ALIGN_DEBUG.get():
        return True
    return needs_npu_ipc_alignment()


def log_ipc_regions(
    source: str,
    ptrs: Sequence[int],
    lengths: Sequence[int],
    labels: Optional[Sequence[str]] = None,
) -> int:
    """Log 2MB alignment status. Returns count of misaligned regions."""
    if not should_log_npu_ipc_alignment():
        return 0

    page = NPU_IPC_PAGE_SIZE_2MB
    debug_all = envs.SGLANG_NPU_IPC_ALIGN_DEBUG.get()
    misaligned = 0
    n = len(ptrs)

    for i in range(n):
        ptr = int(ptrs[i])
        length = int(lengths[i])
        label = labels[i] if labels is not None and i < len(labels) else f"buffer[{i}]"
        ptr_ok = ptr % page == 0
        len_ok = length % page == 0
        if ptr_ok and len_ok:
            if debug_all:
                logger.info(
                    "[NPU IPC align] %s %s: ptr=0x%x size=%d OK",
                    source,
                    label,
                    ptr,
                    length,
                )
            continue

        misaligned += 1
        ptr_rem = ptr % page
        len_rem = length % page
        ap, al = align_npu_ipc_region(ptr, length)
        logger.warning(
            "[NPU IPC align] %s %s: ptr=0x%x (offset=%d) size=%d (remainder=%d) "
            "NOT 2MB aligned -> register ptr=0x%x size=%d",
            source,
            label,
            ptr,
            ptr_rem,
            length,
            len_rem,
            ap,
            al,
        )

    if misaligned:
        logger.warning(
            "[NPU IPC align] %s: %d/%d region(s) misaligned (will pad at register)",
            source,
            misaligned,
            n,
        )
    elif debug_all and n:
        logger.info(
            "[NPU IPC align] %s: all %d region(s) are 2MB aligned",
            source,
            n,
        )
    return misaligned
