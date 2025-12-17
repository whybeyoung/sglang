# temp NSA debugging environ
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.server_args import get_global_server_args

# Import CP group functions (will be used when true TP+CP mode is enabled)
from sglang.srt.layers.attention.nsa.cp_group import (
    get_cp_group as get_cp_comm_group,
    get_cp_rank as get_cp_comm_rank,
    get_cp_size as get_cp_comm_size,
)
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


NSA_DUAL_STREAM = get_bool_env_var("SGLANG_NSA_DUAL_STREAM", "true")
NSA_FUSE_TOPK = get_bool_env_var("SGLANG_NSA_FUSE_TOPK", "true")

NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8 = get_bool_env_var(
    "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8", "true"
)
NSA_QUANT_K_CACHE_FAST = get_bool_env_var("SGLANG_NSA_QUANT_K_CACHE_FAST", "true")
NSA_DEQUANT_K_CACHE_FAST = get_bool_env_var("SGLANG_NSA_DEQUANT_K_CACHE_FAST", "true")


def print_nsa_bool_env_vars():
    msg = ""
    for k, v in globals().items():
        if k.startswith("NSA_") and isinstance(v, bool):
            msg += f"{k}={v} "
    print(msg, flush=True)


def is_nsa_enable_prefill_cp():
    return get_global_server_args().enable_nsa_prefill_context_parallel


def get_cp_size():
    """Get CP size from server args. If not set, returns None (will use atten_tp_size)."""
    server_args = get_global_server_args()
    if not server_args.enable_nsa_prefill_context_parallel:
        return None
    return server_args.nsa_prefill_context_parallel_size


def is_nsa_prefill_cp_mode0():
    return (
        is_nsa_enable_prefill_cp() and get_global_server_args().nsa_prefill_cp_mode == 0
    )


def is_nsa_prefill_cp_mode1():
    return (
        is_nsa_enable_prefill_cp() and get_global_server_args().nsa_prefill_cp_mode == 1
    )


def can_nsa_prefill_cp_mode1(forward_batch: "ForwardBatch"):
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    cp_size = get_attention_tp_size()
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return is_nsa_prefill_cp_mode1() and seq_len > 0 and cp_size > 1


def enable_prefill_cp(forward_batch: "ForwardBatch", nsa_enable_prefill_cp: Optional[bool] = None) -> bool:
    """Check if context parallelism should be enabled for a forward batch.
    
    Args:
        forward_batch: The forward batch to check
        nsa_enable_prefill_cp: Optional flag indicating if CP is enabled globally.
                              If None, uses is_nsa_enable_prefill_cp() to check.
    
    Returns:
        True if CP should be enabled for this batch, False otherwise.
    """
    if forward_batch is None:
        return False
    
    # Check if CP is enabled globally
    if nsa_enable_prefill_cp is None:
        nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
    
    if not nsa_enable_prefill_cp:
        return False
    
    # Check if forward mode supports CP
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    
    # Check if CP metadata exists (indicates CP is active for this batch)
    if forward_batch.nsa_cp_metadata is None:
        return False
    
    return True


def nsa_cp_mode1_split_data(input_: Union[torch.Tensor, List]):
    cp_size = get_attention_tp_size()
    cp_rank = get_attention_tp_rank()
    if isinstance(input_, (tuple, list)) or len(input_) % cp_size != 0:
        indices = range(cp_rank, len(input_), cp_size)
        return input_[indices]
    # for torch device tensor
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()


@dataclass
class NSAContextParallelMetadata:

    split_list: List[int] = None
    max_rank_len: List[int] = None
    zigzag_index: List[int] = None
    per_rank_actual_token: List[int] = None
    reverse_split_len: List[int] = None
    cp_reverse_index: List[int] = None
    kv_len_prev: int = -1
    kv_len_next: int = -1
    actual_seq_q_prev: int = -1
    actual_seq_q_next: int = -1
    kv_len_prev_tensor: torch.Tensor = None
    kv_len_next_tensor: torch.Tensor = None
    actual_seq_q_prev_tensor: torch.Tensor = None
    actual_seq_q_next_tensor: torch.Tensor = None
    total_seq_lens: torch.Tensor = None


def can_cp_split(seq_len: int, cp_size: int, use_nsa: bool, forward_batch):
    if is_nsa_prefill_cp_mode1():
        cur_cp_seq_len = seq_len // cp_size
        assert (
            seq_len % cp_size == 0
        ), f"seq_len {seq_len} is not divisible by cp_size {cp_size} when nsa_prefill_cp_mode is 1"
    else:
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        # Note: (self.cp_size * 2) To achieve load balancing for seq computation,
        # the seq data needs to be divided and recombined at twice the size of cp_size.
        cur_cp_seq_len = seq_len // (cp_size * 2)
    if (
        cur_cp_seq_len != 0
        and cp_size > 1
        and use_nsa
        and forward_batch.forward_mode.is_context_parallel_extend()
        and is_nsa_enable_prefill_cp()
    ):
        return True
    else:
        return False


def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):
    if is_nsa_prefill_cp_mode1():
        cp_size = get_attention_tp_size()
        assert (
            input_.shape[0] % cp_size == 0
        ), f"Expect input shape 0 can divided by cp size, but got input shape {input_.shape}, cp size {cp_size}"
        return nsa_cp_mode1_split_data(input_)

    input_list = list(
        torch.split(input_, forward_batch.nsa_cp_metadata.split_list, dim=0)
    )
    result = torch.cat(
        [input_list[i] for i in forward_batch.nsa_cp_metadata.zigzag_index], dim=0
    ).view(-1, input_.shape[-1])
    return result


def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):
    if is_nsa_prefill_cp_mode1():
        cp_size = get_attention_tp_size()
        assert positions.shape[0] % cp_size == 0, (
            f"Expect positions shape 0 can divided by cp size, but got positions shape {positions.shape}, "
            f"cp size {cp_size}"
        )
        return nsa_cp_mode1_split_data(positions)

    position_id_list = list(
        torch.split(positions, forward_batch.nsa_cp_metadata.split_list, dim=-1)
    )
    positions = torch.cat(
        [position_id_list[i] for i in forward_batch.nsa_cp_metadata.zigzag_index],
        dim=-1,
    )
    return positions


def cp_all_gather_rerange_output(
    input_: torch.Tensor,
    cp_size: int,
    forward_batch: "ForwardBatch",
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Allgather and rerange output for context parallelism.
    
    This function performs allgather across CP ranks and then reranges the output
    according to CP metadata to restore the original sequence order.
    
    Args:
        input_: Input tensor split across CP ranks (shape: [split_seq_len, ...])
        cp_size: Context parallelism size
        forward_batch: Forward batch containing CP metadata
        stream: Optional CUDA stream for async operations
    
    Returns:
        Reranged output tensor with full sequence (shape: [full_seq_len, ...])
    """
    from sglang.srt.layers.attention.nsa.cp_group import get_cp_group
    
    if forward_batch.nsa_cp_metadata is None:
        # No CP metadata, just do regular allgather
        cp_group = get_cp_group()
        output = torch.empty(
            (input_.shape[0] * cp_size, *input_.shape[1:]),
            dtype=input_.dtype,
            device=input_.device,
        )
        if stream is not None:
            cp_group.cp_all_gather_into_tensor_async(output, input_, stream=stream)
            stream.synchronize()
        else:
            cp_group.all_gather_into_tensor(output, input_)
        return output
    
    # Get CP group for allgather
    cp_group = get_cp_group()
    
    # Allgather: collect data from all CP ranks
    output = torch.empty(
        (input_.shape[0] * cp_size, *input_.shape[1:]),
        dtype=input_.dtype,
        device=input_.device,
    )
    
    if stream is not None:
        cp_group.cp_all_gather_into_tensor_async(output, input_, stream=stream)
        stream.synchronize()
    else:
        cp_group.all_gather_into_tensor(output, input_)
    
    # Rerange according to CP metadata
    if is_nsa_prefill_cp_mode1():
        # Mode 1: simple split, no rerange needed (already in correct order after allgather)
        return output
    else:
        # Mode 0: zigzag mode, need to rerange using reverse_index
        if forward_batch.nsa_cp_metadata.cp_reverse_index is not None:
            # Use reverse_index to restore original order
            reverse_index = forward_batch.nsa_cp_metadata.cp_reverse_index
            if isinstance(reverse_index, list):
                reverse_index = torch.tensor(reverse_index, device=input_.device)
            return output[reverse_index]
        else:
            # Fallback: if no reverse_index, return as-is
            return output
