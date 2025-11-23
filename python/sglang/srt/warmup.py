from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}

# Warmup request ID prefix
PP_WARMUP_RID_PREFIX = "PP_WARMUP_"


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
        return fn

    return decorator


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


@warmup("voice_chat")
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in range(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


@warmup("pp_chunk_tuning")
async def pp_chunk_tuning(
    disaggregation_mode: str, tokenizer_manager: "TokenizerManager"
):
    """
    Warmup for PP dynamic chunk size: collect data points to fit f(l) = al^2 + bl + c.
    
    Per pseudocode:
    1. Send requests with different total lengths that will be split into multiple chunks
    2. Collect (sequence_length, total_latency) pairs from all chunks
    3. Fit f(l) = al^2 + bl + c to get coefficients a, b, c
    4. Calculate target_chunk_time = f(base_chunk_size) - f(0)
    """
    server_args = get_global_server_args()
    
    # Check if PP mode and dynamic chunk size are enabled
    if server_args.pp_size <= 1:
        logger.info("PP chunk tuning warmup skipped: PP mode is not enabled.")
        return
    if server_args.chunked_prefill_size is None or server_args.chunked_prefill_size <= 0:
        logger.warning(
            "PP chunk tuning warmup skipped: chunked_prefill_size is not set or invalid."
        )
        return
    
    base_chunk_size = server_args.chunked_prefill_size
    context_len = tokenizer_manager.context_len
    
    # Generate requests with total lengths that are multiples of base_chunk_size
    # to ensure diverse total_seq_lens for fitting
    max_total_len_for_warmup = min(base_chunk_size * 5, context_len - 100)
    
    request_sizes_to_test = [
        base_chunk_size,
        base_chunk_size * 2,
        base_chunk_size * 3,
        base_chunk_size * 4,
        max_total_len_for_warmup,
    ]
    
    # Ensure all sizes are within limits and at least base_chunk_size
    request_sizes_to_test = sorted(list(set([
        min(max(size, base_chunk_size), context_len - 100) for size in request_sizes_to_test
    ])))
    
    logger.info(
        f"PP chunk tuning warmup: testing {len(request_sizes_to_test)} requests "
        f"with total lengths: {request_sizes_to_test} (context_len={context_len}, "
        f"base_chunk_size={base_chunk_size})"
    )
    
    valid_token_id = 1
    for i, total_length in enumerate(request_sizes_to_test):
        input_ids = [valid_token_id] * total_length
        warmup_rid = f"{PP_WARMUP_RID_PREFIX}{i}_{total_length}"
        generate_req_input = GenerateReqInput(
            rid=warmup_rid,
            input_ids=input_ids,
            sampling_params={
                "max_new_tokens": 0,  # Prefill only for timing measurement
                "temperature": 0.0,
                "ignore_eos": True,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST
        
        try:
            logger.info(
                f"PP chunk tuning warmup [{i+1}/{len(request_sizes_to_test)}]: "
                f"sending request with {total_length} tokens"
            )
            async for _ in tokenizer_manager.generate_request(generate_req_input, None):
                pass
        except Exception as e:
            logger.warning(
                f"PP chunk tuning warmup request {i+1} failed: {e}. Continuing..."
            )
    
    logger.info(
        f"PP chunk tuning warmup completed. "
        f"Scheduler will fit coefficients from collected timing data."
    )
