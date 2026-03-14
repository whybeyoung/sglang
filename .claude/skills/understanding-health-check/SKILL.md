---
name: understanding-health-check
description: SGLang 健康检查机制详解。涵盖 health_generate 流程、multi-tokenizer-worker 路由问题、PD 分离下的健康检查行为、相关 PR（#20191/#20227/#20256/#20296）分析。Use when asked about health check, liveness probe, last_receive_tstamp, HealthCheckOutput, or K8s pod killed under high prefill load.
---

# SGLang Health Check 机制详解

## 核心组件

### `/health_generate` 端点（http_server.py）

```python
tic = time.time()
while time.time() < tic + HEALTH_CHECK_TIMEOUT:  # 默认 20s
    await asyncio.sleep(1)
    if tokenizer_manager.last_receive_tstamp > tic:
        return Response(status_code=200)
return Response(status_code=503)
```

**判定标准**：在 20 秒内，`last_receive_tstamp` 是否被更新过。

### `last_receive_tstamp` 更新路径（tokenizer_manager.py）

```python
async def handle_loop(self):
    while True:
        recv_obj = await self.recv_from_detokenizer.recv_pyobj()
        self._result_dispatcher(recv_obj)
        self.last_receive_tstamp = real_time()  # 每次收到任何消息都更新
```

任何从 detokenizer 返回的消息（`BatchStrOutput`、`HealthCheckOutput` 等）都会更新时间戳。

---

## Scheduler 侧的健康检查逻辑

### `process_input_requests`（scheduler.py）

```python
if is_health_check_generate_req(recv_req):
    has_running_requests = (
        self.chunked_req is not None
        or self.dllm_manager.any_staging_reqs()
        or not self.running_batch.is_empty()
        or len(self.offload_tags) > 0
    )
    will_block_in_pd_queue = False
    if self.disaggregation_mode == DisaggregationMode.PREFILL:
        will_block_in_pd_queue = (
            len(self.disagg_prefill_bootstrap_queue.queue) > 0
            or len(self.disagg_prefill_inflight_queue) > 0
        )
    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        will_block_in_pd_queue = (
            len(self.disagg_decode_prealloc_queue.queue) > 0
            or len(self.disagg_decode_transfer_queue.queue) > 0
        )

    if has_running_requests or will_block_in_pd_queue:
        self.return_health_check_ipcs.append(
            getattr(recv_req, "http_worker_ipc", None)
        )
        continue  # 跳过，推迟回复
```

**当 scheduler 忙时，health check 请求被推迟**，保存其 `http_worker_ipc` 后等待下一次 batch 完成再回复。

### `maybe_send_health_check_signal`（scheduler.py）

```python
def maybe_send_health_check_signal(self):
    if self.return_health_check_ipcs:
        self.send_to_tokenizer.send_output(
            HealthCheckOutput(http_worker_ipc=self.return_health_check_ipcs.popleft())
        )
```

在 `process_batch_result` 末尾调用，每完成一个 batch 发一个 `HealthCheckOutput` 回 tokenizer。

**两条路径**：
- **Scheduler 空闲**：health check 作为正常请求处理，走完整 scheduler→detokenizer→tokenizer 流程
- **Scheduler 忙**：推迟，等 batch 完成后发 `HealthCheckOutput` 捷径回复

---

## Multi-Tokenizer-Worker 模式下的路由问题（PR #20256）

### 架构

```
K8s Health Probe → Worker 0 (pid=1001, ipc=ipc://xxx-1001)
                   Worker 1 (pid=1002, ipc=ipc://xxx-1002)
                        ↓ 汇聚
                   MultiTokenizerRouter
                        ↓
                   Scheduler
                        ↓ 响应
                   MultiTokenizerRouter._distribute_result_to_workers()
                        ↓ 根据 http_worker_ipc 路由
                   回到对应 Worker
```

### 修复前的问题

**修复前**：`HealthCheckOutput()` 不带 `http_worker_ipc`（值为 `None`）。

Router 的 `send_output`：
```python
def send_output(self, ipc_name: str, output: Any):
    if ipc_name is None:
        logger.warning(f"IPC name is None, output type={type(output)}, skipping...")
        return  # 直接丢弃！
```

**结果**：`HealthCheckOutput` 被 Router 丢弃，发起 health check 的 Worker 0 永远收不到响应，`last_receive_tstamp` 不更新。

### 修复后（PR #20256）

将 `return_health_check_ct`（整数计数器）改为 `return_health_check_ipcs`（Deque），保存每个被推迟请求的 `http_worker_ipc`：

```python
# Before
self.return_health_check_ct = 0
self.return_health_check_ct += 1
self.send_to_tokenizer.send_output(HealthCheckOutput())

# After
self.return_health_check_ipcs: Deque[Optional[str]] = deque()
self.return_health_check_ipcs.append(getattr(recv_req, "http_worker_ipc", None))
self.send_to_tokenizer.send_output(
    HealthCheckOutput(http_worker_ipc=self.return_health_check_ipcs.popleft())
)
```

### 触发条件

- **低流量必现**：Worker 长时间无业务请求，health check 是唯一触发更新的机会，被 skip 后 20s 超时 → 503 → K8s 连续 4 次失败后杀 Pod
- **高流量低概率**：业务响应持续更新 `last_receive_tstamp`，掩盖了 health check skip 的问题

---

## PD 分离场景分析

### `will_block_in_pd_queue` 的问题（PR #20191）

**问题**：当 PD 队列有积压时，health check 被推迟后通过 `maybe_send_health_check_signal` 立即回复 healthy。这意味着即使系统真正死锁（如 Decode 端挂掉导致 KV transfer 永远不完成），health check 依然返回 200，**掩盖死锁**。

**Lianmin 的观点**：`HealthCheckOutput` 语义是"我还活着（liveness）"，应该基于系统真正有进展（`process_batch_result` 被调用），而不是仅仅 event loop 在空转。

**正确的 liveness 判断**：Prefill 端最近一段时间内是否有 batch 完成或 KV transfer 完成，而不是队列是否有积压。

### Prefill 端 DS32EncodingError 400 的影响

- DS32EncodingError 在 HTTP 层（`serving_base.py`）抛出，请求从未进入 scheduler
- **Decode 端有独立的超时机制**（`SGLANG_DISAGGREGATION_WAITING_TIMEOUT`，默认 300s），receiver poll 超时后返回 `KVPoll.Failed`，自动 abort 清理

---

## Idle 状态检查统一（PR #20296）

**修复前**：3 套不一致的 idle 检查（`_is_no_request`、`_is_idle_for_hicache_storage_op`、health check 内联逻辑），都有不同的漏检项。

**修复后**：统一为 `is_fully_idle(for_health_check=False)`，覆盖：
- `running_batch`、`last_batch`、`cur_batch`、`result_queue`（overlap）
- `chunked_req`、`dllm_manager`
- `waiting_queue`、`grammar_queue`
- PP 的 `running_mbs`
- PD disagg 的 `bootstrap_queue`、`inflight_queue`、`prealloc_queue`、`transfer_queue`

`for_health_check=True` 时跳过 `grammar_queue` 和 `prefill_inflight_queue`（这些队列中的请求本身就携带进度信息）。

**不修复的影响**：
- `flush_cache` 在 waiting_queue 有 retracted 请求时清空 KV cache → 请求失败
- `update_weights` 在 chunked_req 处理中时更新权重 → 输出损坏
- HiCache storage attach/detach 时机错误

---

## 相关环境变量

| 变量 | 默认值 | 含义 |
|------|--------|------|
| `SGLANG_HEALTH_CHECK_TIMEOUT` | 20s | health_generate 等待超时 |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | 300s | Prefill bootstrap 超时 |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | 300s | Decode KV transfer 等待超时 |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | ≥2s | 心跳间隔 |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` | ≥1 | 最大心跳失败次数 |

---

## 关键文件

- `python/sglang/srt/entrypoints/http_server.py` - `health_generate` 端点
- `python/sglang/srt/managers/tokenizer_manager.py` - `handle_loop`、`last_receive_tstamp`
- `python/sglang/srt/managers/scheduler.py` - `process_input_requests`、`maybe_send_health_check_signal`、`return_health_check_ipcs`
- `python/sglang/srt/managers/multi_tokenizer_mixin.py` - `MultiTokenizerRouter`、`SocketMapping.send_output`
- `python/sglang/srt/disaggregation/prefill.py` - `process_disagg_prefill_inflight_queue`
- `python/sglang/srt/disaggregation/decode.py` - `pop_transferred`、超时机制
- `python/sglang/srt/disaggregation/common/conn.py` - `bootstrap_timeout`、`waiting_timeout`
