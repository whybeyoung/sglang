# PD disaggregation: Triton JIT on alloc_extend during prealloc causes ~15s TTFT delay on one EP rank

## Summary

In **prefill-decode (PD) disaggregation**, the decode-side **prealloc** path calls `alloc_extend` (paged KV allocator). The first call triggers **Triton JIT compilation** of `alloc_extend_kernel` (and related kernels). Because compilation happens on **one EP rank** at the time of its first prealloc, that rank becomes a **straggler**: other EP ranks reach the **DP all_gather** (MLP sync) and block until this rank arrives, causing **~15s TTFT delay** for the whole decode batch. After warmup, the delay disappears.

## Environment

- **Mode**: PD (prefill-decode) disaggregation, decode with **EP (e.g. EP16) + DP**.
- **Path**: Decode prealloc → `alloc_extend()` → `alloc_extend_kernel` (Triton) → first-time JIT compile (~15s).
- **Observed**: One decode/EP rank’s first prealloc triggers Triton compile; other ranks wait in `all_gather` → ~15s head latency.

## Root cause (brief)

1. Decode prealloc allocates KV for the first request via **paged allocator** `alloc_extend()`.
2. `alloc_extend` uses **Triton** kernels (`alloc_extend_kernel`, and in some paths `alloc_decode_kernel`) which are **JIT-compiled on first use**.
3. Compilation (e.g. `make_cubin` via subprocess) takes on the order of **tens of seconds**.
4. All EP ranks must participate in the **DP all_gather** (e.g. in `prepare_mlp_sync_batch`). The rank that is still in Triton compile **arrives late**; others block in the collective → **~15s delay**.
5. Once kernels are cached, subsequent requests no longer compile → delay only on first request / cold start.

## Request for help

We need to **warm up the allocator extend (and decode) Triton kernels at startup** so that the first prealloc does not trigger JIT and no EP rank is a straggler.

- **Goal**: Trigger JIT for `alloc_extend_kernel` and `alloc_decode_kernel` (and any related Triton in this path) **before** the first real prealloc, using **dummy tensors only** (no change to real allocator state, to avoid affecting memory-leak checks).
- **Scope**: Decode-side, paged allocator; parameters should be driven by launch config (e.g. `--cuda-graph-max-bs`, `--max-running-requests`, context length) so that the first real request hits cached kernels.
- **Constraint**: Warmup must not significantly increase **peak GPU memory** (e.g. avoid large allocations or release promptly) so that it can run together with other warmup (e.g. spec) without OOM.

We have a **draft implementation** (standalone `warmup_triton_alloc_kernels(device, page_size, max_bs, max_context_len)` with dummy tensors, called from `model_runner.kernel_warmup()`, and an extend cap + `empty_cache` to reduce OOM risk). We are seeking **review and suggestions** for:

1. **Correctness**: Ensuring all kernel variants that prealloc can hit are warmed (e.g. `bs_upper`, `max_num_extend_tokens`, page_size).
2. **Memory safety**: No change to scheduler/allocator state; minimal peak memory during warmup.
3. **Integration**: Best place and timing for this warmup (e.g. decode-only, after allocator init, before spec warmup).
4. **Config**: Whether to expose a switch (e.g. env or server arg) to disable this warmup when needed.

Any help or pointers to similar warmup patterns in the codebase would be appreciated.

## References

- Decode prealloc: `sglang/srt/disaggregation/decode.py` (`_alloc_kv_for_prebuilt_req` → `token_to_kv_pool_allocator.alloc_extend`).
- Allocator Triton kernels: `sglang/srt/mem_cache/allocator.py` (`alloc_extend_kernel`, `alloc_decode_kernel`).
- DP all_gather: `sglang/srt/managers/scheduler_dp_attn_mixin.py` (`prepare_mlp_sync_batch` → `mlp_sync_info.all_gather`).
