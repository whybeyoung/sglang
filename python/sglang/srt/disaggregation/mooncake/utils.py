# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mooncake-specific utilities for custom memory pool management."""

import logging
from contextlib import nullcontext
from typing import Any, Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Global constants for custom memory pool types
SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES = [
    "NVLINK",
    "BAREX",
    "INTRA_NODE_NVLINK",
    "ASCEND",
]


def init_mooncake_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Initialize custom memory pool based on environment variable.

    Args:
        device: The device to allocate memory on

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type)
    """
    enable_custom_mem_pool, custom_mem_pool_type = (
        check_mooncake_custom_mem_pool_enabled()
    )

    custom_mem_pool = None

    if enable_custom_mem_pool:
        try:
            # TODO(shangming): abstract custom allocator class for more backends
            if custom_mem_pool_type == "NVLINK":
                from mooncake.allocator import NVLinkAllocator

                allocator = NVLinkAllocator.get_allocator(device)
                custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
            elif custom_mem_pool_type == "BAREX":
                from mooncake.allocator import BarexAllocator

                allocator = BarexAllocator.get_allocator(device)
                custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
            elif custom_mem_pool_type == "INTRA_NODE_NVLINK":
                return False, None, None
            elif custom_mem_pool_type == "ASCEND":
                # NPU path: route every tensor allocated inside the
                # use_mem_pool scope through mooncake's
                # ascend_allocator.so (aclrtMalloc HUGE_ONLY), which
                # returns 2 MB-aligned NPU memory. That satisfies CANN
                # HCCL IPC RMA's 2 MB-per-ptr registration contract
                # transparently, so callers (KV pool, MetadataBuffers)
                # can keep using plain torch.zeros and the dense layout.
                if not str(device).startswith("npu"):
                    raise ValueError(
                        f"ASCEND custom mem pool requires an NPU device, "
                        f"got device={device!r}"
                    )
                from mooncake.allocator import AscendAllocator

                allocator = AscendAllocator.get_allocator(device)
                import torch_npu  # noqa: F401

                custom_mem_pool = torch_npu.npu.MemPool(allocator.allocator())
            else:
                # This should not happen due to the enable_custom_mem_pool check above
                raise ValueError(
                    f"Unsupported custom mem pool type: {custom_mem_pool_type}"
                )

            logger.debug(
                f"Initialized custom memory pool: {custom_mem_pool_type} on device {device}"
            )
        except ImportError as e:
            logger.warning(
                f"Failed to import mooncake allocator for {custom_mem_pool_type}: {e}. "
                f"Falling back to default memory pool."
            )
            enable_custom_mem_pool = False
            custom_mem_pool = None
            custom_mem_pool_type = None
        except Exception as e:
            logger.error(
                f"Failed to initialize custom memory pool {custom_mem_pool_type}: {e}. "
                f"Falling back to default memory pool."
            )
            enable_custom_mem_pool = False
            custom_mem_pool = None
            custom_mem_pool_type = None
    else:
        return False, None, None

    return enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type


def check_mooncake_custom_mem_pool_enabled() -> Tuple[bool, Optional[str]]:
    """
    Check if custom memory pool is enabled without importing allocators.

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool_type)
    """
    custom_mem_pool_type = envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get()

    if custom_mem_pool_type is not None:
        # Handle boolean True as NVLINK
        if custom_mem_pool_type.lower() == "true":
            custom_mem_pool_type = "NVLINK"
        enable_custom_mem_pool = (
            custom_mem_pool_type in SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES
        )
    else:
        enable_custom_mem_pool = False
        custom_mem_pool_type = None

    return enable_custom_mem_pool, custom_mem_pool_type


def use_custom_mem_pool_for_device(custom_mem_pool: Any, device: Any):
    """Return a context manager that binds ``custom_mem_pool`` for tensor
    allocations on ``device``.

    Dispatches to the correct ``use_mem_pool`` based on the device type:

      * ``cuda`` / ``hip`` -> ``torch.cuda.use_mem_pool``
      * ``npu``            -> ``torch_npu.npu.use_mem_pool``
      * anything else      -> ``nullcontext()`` (the mem pool is silently
                              ignored because torch has no scoped allocator
                              hook for that device type)

    A ``None`` ``custom_mem_pool`` always degrades to ``nullcontext()``.
    """
    if custom_mem_pool is None:
        return nullcontext()
    device_type = device.type if isinstance(device, torch.device) else str(device)
    if device_type.startswith("npu"):
        import torch_npu

        return torch_npu.npu.use_mem_pool(custom_mem_pool)
    return torch.cuda.use_mem_pool(custom_mem_pool)
