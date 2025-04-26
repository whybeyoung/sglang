import logging
from typing import List

import numpy as np
import tqdm
import os
import aiohttp

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str) -> callable:
    def decorator(fn: callable):
        _warmup_registry[name] = fn
        return fn

    return decorator


async def execute_warmups(warmup_names: List[str], tokenizer_manager: TokenizerManager):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](tokenizer_manager)


@warmup("voice_chat")
async def voice_chat(tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
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
        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


@warmup("compile-deep-gemm")
async def warm_up_compile(tokenizer_manager: TokenizerManager):
    # Reduce warning
    os.environ["SGL_IN_DEEPGEMM_PRECOMPILE_STAGE"] = "1"
    # Force enable deep gemm
    os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = "1"
    # Force enable mha chunked kv for DeepSeek V3 to avoid missing kv_b_proj DeepGEMM case
    os.environ["SGL_CHUNKED_PREFIX_CACHE_THRESHOLD"] = "0"
    
    print("\n Starting Generate warm up request for compiling DeepGEMM...\n")
    generate_req_input = GenerateReqInput(
        input_ids=[0, 1, 2, 3],
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 8,
            "ignore_eos": True,
        },
        bootstrap_host="2.2.2.2",
        bootstrap_room=-1,
    )

    # Convert to JSON for POST request
    json_data = {
        "input_ids": generate_req_input.input_ids,
        "sampling_params": generate_req_input.sampling_params,
        "bootstrap_host": generate_req_input.bootstrap_host,
        "bootstrap_room": generate_req_input.bootstrap_room
    }
    url = tokenizer_manager.server_args.url()
    # Make POST request to the generate endpoint
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{url}/generate",
            json=json_data,
            headers={"Authorization": f"Bearer {tokenizer_manager.server_args.api_key}"} if tokenizer_manager.server_args.api_key else {}
        ) as response:
            if response.status != 200:
                raise Exception(f"Warmup request failed with status {response.status}")
            await response.json()
    
    print("\n End Generate warm up request for compiling DeepGEMM...\n")