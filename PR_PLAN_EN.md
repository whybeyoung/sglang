# PP HiCache Consistency for Disaggregated Prefill — PR Plan

## Background

### Motivation

In agentic and long-context reasoning scenarios, requests frequently share long prefixes (system prompts, tool definitions, multi-turn history) that can span tens of thousands of tokens. Recomputing these shared prefixes for every request is prohibitively expensive. SGLang's **HiCache** addresses this with a 3-level KV cache hierarchy — **L1** (GPU device memory) → **L2** (CPU host memory) → **L3** (external distributed storage, e.g. Mooncake) — enabling prefix reuse across requests and even across restarts via persistent storage.

Meanwhile, large models (e.g. DeepSeek-V3 with 60 layers) require **Pipeline Parallelism (PP)** to split layers across multiple GPU groups, combined with **disaggregated prefill** where prefill and decode run on separate server pools. This PP + HiCache + disaggregated prefill stack is critical for production agentic workloads: it maximizes both throughput (via disaggregation) and cache hit rates (via L3 persistence) on large models.

However, combining PP with L3 storage introduces a class of consistency bugs that do not exist in simpler configurations. Understanding why requires examining the cache hierarchy level by level.

### Why L3 Storage Breaks PP Consistency

In PP mode, each rank runs an independent scheduler with its own radix tree. The key question is: why does adding L3 cause divergence when L1 and L2 alone do not?

**L1-only + PP (no HiCache)**: Both ranks receive the same requests in the same order via P2P. `match_prefix` operates on the device radix tree, which is fully deterministic — insert and evict are driven by the same batch picks, and all operations are synchronous within each scheduler cycle. **No divergence possible.**

**L1 + L2 + PP (HiCache without storage)**: Adding the host cache introduces `write-through` (GPU→CPU backup) and `load-back` (CPU→GPU restore). These are still **synchronous** operations within each cycle — `writing_check()` and `loading_check()` are called at deterministic points in the event loop, and the host tree mutations they produce are driven by the same batch sequence. **No divergence possible.**

**L1 + L2 + L3 + PP (HiCache with storage)**: Adding L3 storage introduces the **prefetch thread** — an asynchronous background thread that queries external storage independently on each rank. This is where consistency breaks:

1. **Async completion timing**: PP0 and PP1's prefetch threads may complete at different wall-clock times. When PP0's prefetch finishes first, it inserts nodes into its host tree immediately, while PP1's tree has not yet been updated. The next `match_prefix` on each rank sees different host tree states → different `host_hit_length` → different `prefix_indices`.

2. **Anchor divergence**: The L3 query uses a hash chain starting from an anchor node in the host tree. If PP0 has already inserted nodes from a previous prefetch (giving it `host_hit=896`), its anchor and token range differ from PP1's (which has `host_hit=0`). The two ranks compute **completely different hash chains** for the same request → fetch different (or wrong) data from storage.

3. **Eviction divergence from wall-clock LRU**: Even after fixing async finalization (making PP1 event-driven), the eviction layer itself can diverge. LRU eviction uses `last_access_time = time.monotonic()`, which differs by microseconds between ranks → different victim node selection → different GPU→CPU demotion → different host memory pressure → different `evictable_host_leaves` candidate sets. In production (PP2 + NSA + disagg prefill), this caused: PP0 finalized a prefetch → inserted host node → immediately triggered `evict_host` (memory pressure) → evicted the node. PP1 replayed the same finalize event one cycle later → inserted the node → no eviction pressure → node remained. Result: PP0 `matched_host=0`, PP1 `matched_host=17664` → shape mismatch crash.

4. **Amplification cascade**: Once the host tree diverges (even by one node), all subsequent operations compound the difference — different eviction decisions, different write-through timing, different prefetch anchors for the next request — until a shape mismatch crash occurs.

In essence: **L1 and L2 operations are synchronous and deterministic within each cycle, but L3 prefetch is asynchronous and state-dependent. The async completion timing, state-dependent query parameters, and wall-clock-dependent eviction ordering create a feedback loop where small divergences amplify into crashes.**

## Solution Architecture

We introduce **3 synchronization channels** between PP ranks to ensure L1/L2/L3 cache consistency:

```
 ┌───────────────────────────┐            ┌───────────────────────────┐
 │       PP0 Scheduler       │            │       PP1 Scheduler       │
 │                           │            │                           │
 │  Event Loop               │            │  Event Loop               │
 │  ├─ batch pick            │ ═A.event═> │  ├─ replay events (L2)    │
 │  ├─ writing_check         │ ─B.count─> │  ├─ budget-cap check (L1) │
 │  └─ prefetch issue        │            │  └─ prefetch issue        │
 │                           │            │                           │
 │  Prefetch Thread          │ ─C.hit───> │  Prefetch Thread          │
 │  └─ query L3, hash chain  │            │  └─ wait result, trim,   │
 │                           │            │     align anchor (L3)     │
 │                           │            │                           │
 │  Radix Tree (host+device) │            │  Radix Tree (host+device) │
 └──────┬──────────┬─────────┘            └──────┬──────────┬─────────┘
 ┌──────▼──────┐ ┌─▼────────┐            ┌──────▼──────┐ ┌─▼────────┐
 │ L1 GPU KV   │ │ L2 Host  │            │ L1 GPU KV   │ │ L2 Host  │
 │ (layer 0-29)│ │ KV Pool  │            │ (layer30-59)│ │ KV Pool  │
 └─────────────┘ └────┬─────┘            └─────────────┘ └────┬─────┘
                      │ backup                                │ backup
                      ▼                                       ▼
       ┌─────────────────────────────────────────────────────────┐
       │            L3 Mooncake Storage (shared)                 │
       │  KV key:      {hash}_pp_size_X_pp_rank_Y_cp_Z          │
       │  Indexer key:  {hash}_{mla_suffix}_indexer              │
       │  MLA+CP: only CP0 writes; other CP ranks read CP0 key  │
       └─────────────────────────────────────────────────────────┘
```

### Synchronization Channels

| Channel | Direction | Strategy | Guarantees |
|---|---|---|---|
| **A. Event Replay** | PP0 → PP1 | FIFO queue, per-request events (FINALIZE / REVOKE / SKIP) | L2 host tree structure consistency |
| **B. Count Sync** | PP0 → PP1 | Scalar `write_ack_count` per cycle | L1 device tree eviction parity |
| **C. Hit Delegation** | PP0 → PP1 | `(hit_count, anchor_hash, token_len)` via prefetch thread | L3 hash chain and data consistency |

### Why Two Sync Strategies?

Events A (FINALIZE/REVOKE/SKIP) are **low-frequency** (≤1 per request), **order-dependent** (affect radix tree structure), and must be replayed in sequence.

Write-back acks were originally emitted as `WRITE_BACKUP_COMMITTED` events in the same FIFO queue. However, they are **high-frequency** (10–75 per cycle) and only require **count parity** (not exact ordering). When PP1's IO was slower than PP0's, these events caused **FIFO head-of-line blocking**: unprocessed write-back events at the queue head prevented downstream FINALIZE/REVOKE events from being replayed, causing 75+ event backlog → device tree lock-up → prefix divergence → crash.

**Solution**: Replace `WRITE_BACKUP_COMMITTED` events with a per-cycle scalar count (Channel B). PP1 caps its `dec_lock_ref` consumption at `min(local_ready, upstream_count)`. This eliminates HOL blocking while preserving L1 eviction consistency.

## Task List

> Base: `97adf8a2`. PR #20977 (CP support) already merged.

- [ ] **Logical clock for PP-deterministic eviction** — Replace `time.monotonic()` with a logical batch-step counter for `last_access_time` in PP>1 mode, making both GPU (LRU) and host eviction fully deterministic across PP ranks. Addresses root cause #3 above.
- [ ] Mooncake CP write control + PP storage key isolation (`_should_skip_local_write`, `pp_suffix`)
- [ ] Hybrid cache (Mamba/DSA) enhancements (`hybrid_pool_assembler`, `batch_get_v2`/`batch_set_v2`)
- [ ] Channel A: Host tree event replay infrastructure (`PPHostTreeEvent`, emit/enqueue/replay pipeline, payload V2)
- [ ] Channel B: Write-back count sync (budget-capped `writing_check`, replaces FIFO event replay)
- [ ] Channel C: L3 hit delegation + anchor alignment (PP0 publishes hit result, PP1 trims + walks anchor)
- [ ] Zero-hit deferred revoke (1-cycle defer to prevent PP deadlock on L3 zero-hit)
- [ ] PP prefill diagnostics (ENV-gated, optional)
- [ ] Independent fixes (FlashInfer PP warmup, HTTP 400 logging, router guard, HiCache monitoring, tool content normalization)

## Acknowledgments

Special thanks to @ShangmingCai @hzh0425 for their guidance on the HiCache architecture, PP synchronization design, and code review throughout this effort.
CC @merrymercy @hnyls2002 @xiezhq-hermann @Fridge003
