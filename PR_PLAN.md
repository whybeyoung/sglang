# PR 拆分计划 (v2)

**Base**: `97adf8a2` (main/hicache-refactor 基线)
**HEAD**: `5201c44da` (debug)
**总量**: ~144 commits, 20 files, +4497/-367

---

## PP HiCache 一致性保障架构

PP 模式下两个 rank 独立运行 scheduler event loop，radix tree 状态会因为 wall-clock 时间差异而分歧。我们通过 **5 层同步机制** 保证 PP0 和 PP1 的 batch pick、host tree 结构、device tree 行为完全一致：

```
┌─────────────────────────────────────────────────────────────────┐
│                    PP 一致性保障 — 5 层同步                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ① Host Tree Event Sync (PP0 → PP1)                            │
│     PP0 发送 PREFETCH_FINALIZE / REVOKE / WRITE_BACKUP 事件     │
│     PP1 按序 replay → host tree 结构一致                        │
│                                                                 │
│  ② L3 Storage Hit Delegation (PP0 → PP1, prefetch thread)      │
│     PP1 不独立查 L3 storage，等 PP0 发布                        │
│     payload: (hit_count, anchor_hash, token_ids_len)            │
│     PP1 对齐 token range + 采纳 PP0 的 anchor hash              │
│                                                                 │
│  ③ Prefetch Finalize Anchor Alignment (PP1 本地)                │
│     PP1 finalize 时检测 operation 的 token trim offset          │
│     walk host tree 到正确 anchor 节点，用裁后数据 insert         │
│     保证 insert 后的 host tree 路径与 PP0 一致                   │
│                                                                 │
│  ④ Write-Back Ack Count Sync (PP0 → PP1)                       │
│     PP0 发送 write_ack_count (每 cycle 消费的 write-through 数)  │
│     PP1 用作 cumulative budget 上限                              │
│     防止 dec_lock_ref 消费速度分歧 → device tree 一致            │
│                                                                 │
│  ⑤ Launch Frontier Ack (PP1 → PP0, 反向同步)                   │
│     PP1 batch pick 后回传选中的 request ID 列表                  │
│     PP0 下一 cycle 按 ack_rids 顺序约束 pick                    │
│     rid 不匹配 → break; ack 用完 → break                        │
│     保证两侧 batch 完全一致                                      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  时序:                                                          │
│                                                                 │
│  Cycle N:                                                       │
│    PP0 ──send(reqs + events + hits + ack_count)──→ PP1          │
│    PP1: replay events → prefetch → batch pick B                 │
│    PP1 ──send(reqs + launch_ack{B})──→ PP0 (piggyback)         │
│                                                                 │
│  Cycle N+1:                                                     │
│    PP0: activate launch_ack → pick batch constrained by B       │
│    结果: PP0.batch == PP1.batch (前一 cycle 的)                  │
│                                                                 │
│  注: PP0 的 batch 落后 PP1 一个 cycle，但两侧                   │
│      最终 pick 的请求集合和顺序完全相同                          │
└─────────────────────────────────────────────────────────────────┘
```

### 各层缓存的一致性保障

```
┌─────────────────────────────────────────────────────────────────┐
│  L3 Storage (Mooncake) 一致性 — 层②                             │
│                                                                 │
│  问题: PP0/PP1 各自查 L3，因 host tree 分歧导致 anchor_hash     │
│        和 token_ids 范围不同 → hash chain 分歧 → 取回错误数据    │
│                                                                 │
│  方案: PP1 不独立查 L3，等 PP0 发布结果                         │
│                                                                 │
│  PP0 prefetch_thread:                                           │
│    query L3 → hit_count=24256                                   │
│    publish (24256, anchor_hash="64147d...", token_len=24256)     │
│                                                                 │
│  PP1 prefetch_thread:                                           │
│    wait pp0_storage_hit_results[rid]                             │
│    local_tokens=25152, pp0_tokens=24256                         │
│    token_offset = 25152 - 24256 = 896                           │
│    trim operation.token_ids[:896]  (释放前 896 个 host slots)   │
│    operation.last_hash = pp0_anchor_hash                        │
│    → 两侧 hash chain 完全一致                                   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  L2 Host Tree (CPU Radix Tree) 一致性 — 层①③                   │
│                                                                 │
│  问题: prefetch finalize 在 scheduler event loop 中执行，       │
│        PP0 先 finalize 并 insert 节点，PP1 落后 → 结构不同      │
│                                                                 │
│  方案 A (层①): PP0 发送 host tree 事件，PP1 按序 replay         │
│    PP0: insert node → emit PREFETCH_FINALIZE event              │
│    PP0: revoke prefetch → emit REVOKE event                     │
│    PP1: _pp_apply_hicache_sync_before_batch() 中 replay        │
│                                                                 │
│  方案 B (层③): PP1 finalize 时 anchor 对齐                     │
│    ongoing_prefetch 中的 last_host_node 是 PP1 自己的 (node344) │
│    但 operation.token_ids 已被 prefetch thread 裁掉 896 tokens  │
│    → 检测 token_offset = orig_len - op_len                      │
│    → walk host tree forward 896 tokens 找到正确 anchor          │
│    → 用裁后的 token_ids + 正确 anchor 做 insert                 │
│    → insert 结果与 PP0 一致                                     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  L1 Device Tree (GPU KV Cache) 一致性 — 层④⑤                   │
│                                                                 │
│  问题 A: write-through ack 消费速度不同                         │
│    PP1 IO 慢 → ack queue 堆积 → PP0 已 dec_lock_ref 释放节点   │
│    PP1 还锁着 → eviction 不一致 → match_prefix 不同 prefix     │
│                                                                 │
│  方案 (层④): PP0 发送 write_ack_count = N                      │
│    PP1: budget = min(local_ready, upstream_budget)               │
│    PP1 最多消费 N 个 ack → dec_lock_ref 与 PP0 同步             │
│                                                                 │
│  问题 B: batch pick 不一致                                      │
│    waiting_queue 顺序因 prefetch/revoke 时序不同 → pick 不同请求│
│    → 不同 prefix_indices → shape mismatch crash                 │
│                                                                 │
│  方案 (层⑤): PP1 batch pick 后回传 rid 列表                    │
│    PP0: for req in waiting_queue:                                │
│           if req.rid != ack_rids[idx]: break  # 不匹配就停      │
│           if idx >= len(ack_rids): break      # 用完就停        │
│    → PP0 pick 结果 == PP1 pick 结果                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 已合入 main 的 PR（不再单独提交）

| PR | 状态 | 说明 |
|---|---|---|
| #20977 [HiCache] Add CP support for HiCache | ✅ 已合入 | `hicache_storage.py` 的 `attn_cp_rank/size`、`cache_init_params.py` 的 CP 字段已在 main |

> **注意**：`hicache_storage.py` main 侧还做了 hash 工具函数迁移（#22214），我们的分支没有这个变化。提 PR 时需要 rebase 到最新 main。

---

## PR 1: Mooncake CP 写入控制 + PP Storage Key 语义

**主题**: MLA+CP 模式下只有 CP0 写入共享 storage key；PP 模式下 storage key 加入 `pp_size_X_pp_rank_Y` 前缀以避免跨 PP rank key 冲突。

**涉及文件** (~190 行新增):
- `mooncake_store.py` — `mla_cp_writer_rank`、`is_mla_cp_mode`、`_should_skip_local_write()`、PP suffix 重构（`pp_suffix`）、`_tag_keys()`
- `hicache_storage.py` — `PoolName.INDEXER` 枚举值（为 hybrid cache 准备）

**注意**: CP 基础字段（`attn_cp_rank/size`、`enable_cp`）已随 #20977 合入 main，本 PR 只含增量逻辑。

**独立性**: ✅ 完全独立

---

## PR 2: Hybrid Cache (Mamba/DSA) 增强

**主题**: `memory_pool_host` 支持 Mamba 状态的 `page_first_direct` layout、`hybrid_pool_assembler` 组装 NSA hybrid stack、`hybrid_cache_controller` 多 pool transfer。

**涉及文件** (~340 行新增):
- `memory_pool_host.py` — `MambaPoolHost` page_first layout、`get_hybrid_pool_buffer()`、element size 计算
- `hybrid_cache/hybrid_pool_assembler.py` — **新文件**，`build_nsa_hybrid_stack()`
- `hybrid_cache/hybrid_cache_controller.py` — 多 pool transfer、`enable_storage_metrics`
- `hi_mamba_radix_cache.py` — `_get_extra_pools()` 接口
- `mooncake_store.py` — `register_mem_host_pool_v2()`、`batch_get_v2`/`batch_set_v2`、`_batch_io_v2`、`_batch_postprocess`、`_resolve_shared_pool_transfers()`

**独立性**: ✅ 与 PP 修复独立。依赖 PR 1（`PoolName` 枚举）。

---

## PR 3: PP HiCache 事件同步框架 + Write-Back Barrier

**主题**: PP 模式下 host tree 事件同步基础设施 + write-back barrier + count-based ack sync。这是核心 PR。

**涉及文件** (~2200 行新增，建议进一步拆分，见下):
- `hiradix_cache.py` — `PPHostTreeEvent`、事件队列、`replay_pp_host_tree_events()`、`writing_check()` PP barrier、count-based write ack sync、`_finalize_prefetch_progress` anchor 对齐
- `scheduler_pp_mixin.py` — `_pp_build_req_payload()` V2 格式、`_pp_apply_hicache_sync_before_batch()`、`event_loop_pp_disagg_prefill()` 完整重写
- `scheduler.py` — `recv_requests()` PP payload 解包、`get_new_batch_prefill()` PP barrier、`_prefetch_kvcache()` PP follow rank
- `cache_controller.py` — PP0→PP1 storage hit delegation（三元组 payload）、anchor sync、token_offset trim
- `base_prefix_cache.py` — PP 接口声明
- `schedule_batch.py` — `Req` 扩展字段
- `disaggregation/prefill.py` — PP disagg prefill 适配

**对应根因**: 1 (write-back barrier), 2 (ack defer→count sync), 3 (prefetch finalize barrier), 4 (L3 completion timing), 5 (empty batch phase alignment), 8 (anchor divergence), 11 (finalize anchor 对齐)

### 建议子拆分

| 子 PR | 内容 | 行数(估) |
|---|---|---|
| 3a: 事件基础设施 | `PPHostTreeEvent` 定义、发送/接收/序列化、`_pp_build_req_payload` V2、`recv_requests` 解包、`_pp_apply_hicache_sync_before_batch` | ~400 |
| 3b: Write-back barrier + count-based ack sync（层④） | `writing_check()` PP 分支、`set_pp_upstream_write_ack_count`、`get_pp_last_write_ack_consumed`、`broadcast_pyobj` wrapping | ~200 |
| 3c: Prefetch finalize + L3 delegation（层②③） | `check_prefetch_progress()` PP 等待、PP0 storage hit publish、PP1 `pp_follow_hit` anchor sync、`_finalize_prefetch_progress` anchor walk 对齐 | ~300 |
| 3d: Launch Frontier Ack + event loop 适配（层⑤） | `launch_frontier_ack` 机制（PP1→PP0 batch pick 回传）、`_pp_record/activate/consume_launch_frontier_ack`、`get_new_batch_prefill` 中按 ack_rids 约束 pick、event loop 中 `has_batch` 传递 | ~140 |
| 3e: PP prefill 诊断工具 | 20 个 `_pp_prefill_*` 诊断 helper + event loop 中 ~110 行诊断调用插桩 | ~500 |

> ⚠️ 3a-3d 逻辑紧耦合，如 reviewer 接受可合为一个 PR。3e 纯诊断代码可独立。
> 
> 注：PR 3d 原估 ~800 行，其中 ~110 行为 event loop 内的诊断日志调用（已移入 3e），~140 行为 launch_frontier_ack 核心逻辑 + event loop 适配。event loop 本身结构未重写，只是在原有流程中插入了同步点。

**独立性**: ⚠️ 依赖 PR 1（CP 字段）

---

## PR 4: PP Zero-Hit Deferred Revoke 机制

**主题**: PP 模式下 L3 zero-hit 请求的 1-cycle deferred revoke + promote 机制。

**涉及文件** (~300 行新增):
- `hiradix_cache.py` — `pp_zero_hit_deferred_req_ids`、`pp_zero_hit_pending_promote_req_ids`、`_drain_single_revoke_req()` deferred 分支、`_drain_revoke()` promote/clear、`_try_replay_revoke_event()` 上游 REVOKE 清除、`_clear_zero_hit_deferred_state()`
- `scheduler.py` — `process_bootstrapped_queue()` deferred barrier 清理 (`cc9ffa6ec`)

**对应根因**: 6 (zero-hit 死锁), 7 (zero-hit prefix 不一致), 9 (deferred promote timing)

**独立性**: ⚠️ 依赖 PR 3

---

## PR 5: 独立小改

### 5a: PP FlashInfer Sampling Warmup (~40 行)
- `model_runner.py` — PP 模式 flashinfer sampling 预热
- 对应 `e340ea395`
- **独立性**: ✅

### 5b: HTTP 400 日志增强 (~30 行)
- `http_server.py` — `_log_http_exception()`、`_stringify_http_error_detail()`、`_format_validation_error_message()`
- 对应 `bdd8fcb96` (merge branch 21959)
- **独立性**: ✅

### 5c: Router Rust Extension 兼容 (~6 行)
- `sgl-model-gateway/.../router_args.py` — try/except guard
- 对应 `72dce8852`
- **独立性**: ✅

### 5d: HiCache 状态监控日志 (~70 行)
- `scheduler_metrics_mixin.py` — `_maybe_log_hicache_state_growth()`
- `scheduler_output_processor_mixin.py` — `_release_request_ephemeral_state()`
- **独立性**: ✅ (但引用 hiradix_cache 接口，建议在 PR 3 之后)

### 5e: Tool Message Content 归一化 (~20 行)
- `openai/serving_chat.py` — tool role content list→string 归一化
- 对应 `de34b97f7`
- **独立性**: ✅ 与 HiCache/PP 完全无关

---

## 死代码 / ENV-Gated 代码备注

| 代码 | 状态 | 说明 |
|---|---|---|
| `_try_replay_write_backup_event()` + `_write_backup_event_affects_req()` + `_write_commit_event_matches()` | 🔴 默认死代码 | `SGLANG_ENABLE_PP_WRITE_BACKUP_REPLAY` 默认 `"0"`，已被 count-based sync 替代。保留作为 fallback，建议 PR 中标注为 deprecated |
| `WRITE_BACKUP_COMMITTED` 事件发射 | 🔴 默认不发射 | `emit_event = not _pp_write_backup_count_sync_enabled()` 条件下不发射 |
| 20 个 `_pp_prefill_*` 诊断 helper | 🟡 ENV 门控 | `SGLANG_DEBUG_PP_PREFILL_DIAG` 默认 `"0"`，仅调试时启用。~400 行，建议作为独立 PR 3e |
| `SGLANG_DEBUG_HICACHE_HOST_DRIFT` 相关日志 | 🟡 ENV 门控 | 默认关闭，~50 行调试日志 |
| `SGLANG_DEBUG_HICACHE_MATCH_CHAIN` 相关日志 | 🟡 ENV 门控 | 默认关闭，~30 行调试日志 |
| `SGLANG_DEBUG_PP_PREFETCH_TRACE` 相关日志 | 🟡 ENV 门控 | 默认关闭，~40 行调试日志 |
| `_HiCacheDebugFilter` | 🟢 活代码 | 默认过滤 `[HiCache` 和 `[PPReqPhase]` 日志，`SGLANG_DEBUG_HICACHE_VERBOSE=1` 时全部输出 |

## 调试日志分级策略

所有 PP/HiCache 调试日志通过两层门控机制控制，默认**全部静默**：

### 第一层：logging.Filter（默认静默，`SGLANG_DEBUG_HICACHE_VERBOSE=1` 全部可见）

| 过滤器 | 所在文件 | 过滤的前缀 |
|---|---|---|
| `_HiCacheDebugFilter` | `hiradix_cache.py` | `[HiCache*]`、`[PPReqPhase]`、`[PPHiCacheSync]` |
| `_PPPrefillDebugFilter` | `scheduler_pp_mixin.py` | `[PPReqPhase]`、`[PPPrefillDiag]`、`[PPPrefillProblem]`、`[PP Dynamic Chunk]` |

### 第二层：ENV 变量独立开关（更细粒度）

| ENV | 控制的日志 | 用途 |
|---|---|---|
| `SGLANG_DEBUG_PP_PREFETCH_TRACE` | `[PPPrefetchTrace]` phase=prepare/issue | 追踪 prefetch anchor/host_hit/token 分歧 |
| `SGLANG_DEBUG_PP_PREFILL_DIAG` | `[PPPrefillDiag]` recv/batch_pick/ack | PP prefill 事件循环诊断 |
| `SGLANG_DEBUG_HICACHE_HOST_DRIFT` | `[HiCacheFinalizeInsert]`、`[HiCacheHostInsert]`、`[HiCacheNodeSplit]` | Host tree 漂移定位 |
| `SGLANG_DEBUG_HICACHE_MATCH_CHAIN` | `[HiCacheMatchChain]` init/finalize | Match 链路完整跟踪 |
| `SGLANG_DEBUG_HICACHE_MATCH` | `[HiCacheMatch]`、`[HiCacheMatchPath]` | 单次 match 细节 |

### 定位问题时的最小开启组合

| 问题类型 | 建议 ENV | 说明 |
|---|---|---|
| **Shape mismatch crash** | `SGLANG_DEBUG_HICACHE_VERBOSE=1` | 一次性开启所有日志，直接看 `[HiCachePrefetchThread][pp_follow_hit]` 确认 token_offset 和 anchor 分歧 |
| **Prefix 长度分歧** | `SGLANG_DEBUG_PP_PREFETCH_TRACE=1` | 看 PP0 vs PP1 的 prefix/host_hit/last_hash |
| **Prefetch 死锁** | `SGLANG_DEBUG_HICACHE_VERBOSE=1` | 看 `[HiCachePrefetchWait]`、`[HiCachePrefetchDecision]`、`[HiCachePPZeroHitDeferred]` |
| **Write-through ack 堆积** | `SGLANG_DEBUG_HICACHE_VERBOSE=1` | 看 `[HiCacheWriteBackup]`、`[HiCachePPEvent]` |
| **PP event loop 卡住** | `SGLANG_DEBUG_PP_PREFILL_DIAG=1` | 看 `[PPPrefillDiag]` recv/batch_pick 时序 |

---

## 不提交的 commit（调试/被覆盖）

| Commit | 原因 |
|---|---|
| `3b4c2e041` log event | 纯调试日志 |
| `20a459e20` fix | 无描述的中间修复 |
| `a2eddba82` log revoke | 纯调试日志 |
| `13ecfd989` add revoke log | 纯调试日志 |
| `f9c25f72d` ad log | 纯调试日志 |
| `6f6b7f55e` log | 纯调试日志 |
| `f61b030e6` log | 纯调试日志 |
| `da4ee7dbb` update | 无描述 |
| `419ebd357` add log for oiik | 纯调试日志 |
| `8dd2287e5` / `75b15b16e` / `98048c530` fix | 被后续覆盖 |
| `875bb2462` Remove PP zero-hit local revoke barrier | 被 `125cec4b1` deferred 机制替代 |
| `de1debb61` Apply local PP zero-hit revokes immediately | 被 deferred 机制替代 |
| `da89eab04` Block PP1 batch pick while local revokes settle | 被后续重构替代 |
| 约 30+ 个中间探索 commit | 最终被 deferred/delegation 机制替代 |

---

## 提交顺序建议

```
PR 5a/5b/5c/5e (独立小改，最先合，无依赖)
    ↓
PR 1 (Mooncake CP+PP key，rebase 到最新 main)
    ↓
PR 2 (Hybrid Cache，依赖 PR 1 PoolName)
    ↓
PR 3a (PP 事件基础设施)
    ↓
PR 3b (Write-back barrier + count-based ack)
    ↓
PR 3c (Prefetch L3 delegation + anchor sync)
    ↓
PR 3d (PP disagg prefill event loop 重写)
    ↓
PR 3e (PP prefill 诊断工具，可选)
    ↓
PR 4 (Zero-Hit Deferred Revoke)
    ↓
PR 5d (HiCache 监控日志)
```

## 行数总结

| PR | 新增行数(估) | 文件数 |
|---|---|---|
| PR 1 | ~190 | 2 |
| PR 2 | ~340 | 5 |
| PR 3a | ~400 | 5 |
| PR 3b | ~200 | 3 |
| PR 3c | ~300 | 3 |
| PR 3d | ~140 | 3 |
| PR 3e | ~500 | 2 |
| PR 4 | ~300 | 2 |
| PR 5a-5e | ~170 | 5 |
| **合计** | **~3100** | 20 |

> 相比 144 commit 的 4497 行原始 diff，整理后去除调试日志和被覆盖代码约 ~1400 行。
