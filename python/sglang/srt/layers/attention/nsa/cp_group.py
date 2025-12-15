"""
CP (Context Parallelism) communication group management.

This module provides functions to create and manage CP communication groups
for true TP+CP mode where weights are sharded (TP) and sequences are parallelized (CP).
"""

from typing import Optional

import torch

from sglang.srt.distributed import (
    GroupCoordinator,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group, get_attention_tp_size
from sglang.srt.server_args import get_global_server_args


_CP_GROUP: Optional[GroupCoordinator] = None
_CP_RANK: Optional[int] = None
_CP_SIZE: Optional[int] = None


def init_cp_group():
    """Initialize CP communication group."""
    global _CP_GROUP, _CP_RANK, _CP_SIZE
    
    from sglang.srt.layers.attention.nsa.utils import get_cp_size
    
    cp_size_config = get_cp_size()
    if cp_size_config is None:
        # Backward compatibility: use atten_tp_size (original mode)
        _CP_SIZE = get_attention_tp_size()
        _CP_GROUP = get_attention_tp_group()  # Reuse atten_tp_group
        _CP_RANK = get_attention_tp_group().rank_in_group
        return
    
    # True TP+CP mode: create independent CP groups
    _CP_SIZE = cp_size_config
    atten_tp_size = get_attention_tp_size()
    pp_size = get_global_server_args().pp_size
    tp_size = get_global_server_args().tp_size
    
    # Validate CP size divides atten_tp_size
    assert atten_tp_size % _CP_SIZE == 0, (
        f"CP size ({_CP_SIZE}) must divide atten_tp_size ({atten_tp_size})"
    )
    
    # Calculate number of CP groups per PP stage
    num_cp_groups_per_pp = atten_tp_size // _CP_SIZE
    
    # Get current ranks
    # Note: get_attention_tp_rank() returns atten_tp_rank within current PP stage
    # This is already computed as tp_rank % atten_tp_size in compute_dp_attention_world_info
    atten_tp_rank = get_attention_tp_rank()  # PP stage 内的 atten_tp_rank (0 到 atten_tp_size - 1)
    pp_rank = get_pipeline_model_parallel_rank()
    tp_rank_global = get_tensor_model_parallel_rank()  # 全局 TP rank，用于计算 base_rank
    
    # Calculate CP group ID within current PP stage
    cp_group_id = atten_tp_rank // _CP_SIZE
    
    # Calculate CP rank within CP group
    _CP_RANK = atten_tp_rank % _CP_SIZE
    
    # Calculate global ranks for this CP group
    # base_rank is the starting rank of current PP stage
    base_rank = pp_rank * tp_size
    # cp_group_ranks are the ranks within the CP group in current PP stage
    cp_group_ranks = [
        base_rank + cp_group_id * _CP_SIZE + i
        for i in range(_CP_SIZE)
    ]
    
    # Get TP group for backend configuration
    tp_group = get_attention_tp_group()
    
    # Create CP group coordinator
    _CP_GROUP = GroupCoordinator(
        [cp_group_ranks],
        tp_group.local_rank if _CP_RANK == 0 else -1,  # Only first rank in CP group has valid local_rank
        torch.distributed.get_backend(tp_group.device_group),
        use_pynccl=tp_group.use_pynccl,
        use_pymscclpp=tp_group.use_pymscclpp,
        use_custom_allreduce=tp_group.use_custom_allreduce,
        use_torch_symm_mem_all_reduce=tp_group.use_torch_symm_mem_all_reduce,
        use_hpu_communicator=tp_group.use_hpu_communicator,
        use_xpu_communicator=tp_group.use_xpu_communicator,
        use_npu_communicator=tp_group.use_npu_communicator,
        group_name="context_parallel",
    )


def get_cp_group() -> GroupCoordinator:
    """Get CP communication group.
    
    CP group must be initialized in initialize_dp_attention() if CP is enabled.
    """
    assert _CP_GROUP is not None, (
        "CP group not initialized! Make sure CP is enabled (--enable-nsa-prefill-context-parallel) "
        "and initialize_dp_attention() is called."
    )
    return _CP_GROUP


def get_cp_rank() -> int:
    """Get CP rank within CP group.
    
    CP group must be initialized in initialize_dp_attention() if CP is enabled.
    """
    assert _CP_RANK is not None, (
        "CP rank not initialized! Make sure CP is enabled (--enable-nsa-prefill-context-parallel) "
        "and initialize_dp_attention() is called."
    )
    return _CP_RANK


def get_cp_size() -> int:
    """Get CP size.
    
    CP group must be initialized in initialize_dp_attention() if CP is enabled.
    """
    assert _CP_SIZE is not None, (
        "CP size not initialized! Make sure CP is enabled (--enable-nsa-prefill-context-parallel) "
        "and initialize_dp_attention() is called."
    )
    return _CP_SIZE
