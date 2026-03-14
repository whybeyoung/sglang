---
name: understanding-dp-load-balance
description: SGLang DP（数据并行）负载均衡策略详解。涵盖 PD 分离下 decode 侧调度策略、total_tokens/round_robin/follow_bootstrap_room 原理、负载上报机制、启动参数。Use when asked about DP load balancing, decode routing, or PD disaggregation scheduling strategies in SGLang.
---

# SGLang DP 负载均衡策略详解

## 策略一览

| 策略 | 枚举值 | 适用场景 |
|------|--------|----------|
| `round_robin` | `ROUND_ROBIN` | 默认（non-PD / PD decode） |
| `follow_bootstrap_room` | `FOLLOW_BOOTSTRAP_ROOM` | PD prefill 默认，按 bootstrap_room 哈希路由 |
| `total_requests` | `TOTAL_REQUESTS` | 最少请求数优先 |
| `total_tokens` | `TOTAL_TOKENS` | 最少 token 数优先（推荐用于 decode 侧负载均衡） |

## 默认行为（auto）

代码位置：`python/sglang/srt/server_args.py:807-822`

```python
if self.load_balance_method == "auto":
    # - non-PD:      round_robin
    # - PD prefill:  follow_bootstrap_room
    # - PD decode:   round_robin
    self.load_balance_method = (
        "follow_bootstrap_room"
        if self.disaggregation_mode == "prefill"
        else "round_robin"
    )
```

## total_tokens 策略详解

### 调度逻辑

代码位置：`python/sglang/srt/managers/data_parallel_controller.py:93-107`

- 选 `num_tokens` 最小的 worker 派发
- `num_tokens` 相同时用 `num_reqs` 做 tie-break
- 选中后 `total_requests[target_rank] += 1`（启发式预更新）

### 负载数据来源（piggyback 机制）

PR #11469 实现，链路如下：

1. **scheduler** 每次 `stream_output_generation` 调用 `get_load()`
   - 代码：`scheduler_output_processor_mixin.py:914`
2. **`get_load()`** 收集（代码：`observability/scheduler_metrics_mixin.py:591-618`）：
   - running batch 的 tokens
   - waiting_queue tokens
   - PD 分离下：bootstrap_queue / prealloc_queue / transfer_queue 的 tokens
3. **piggyback** 到 batch output，随响应发给 tokenizer_manager
4. **tokenizer_manager** 收到后构造 `WatchLoadUpdateReq` 发给 dp_controller（`tokenizer_manager.py:1613-1618`）
5. **dp_controller** 更新 `DPBudget`（`data_parallel_controller.py:87-91`）

### `num_tokens` 统计范围

```python
# scheduler_metrics_mixin.py:609
num_tokens += sum(req.seqlen for queue in waiting_queues for req in queue)
```

包含：running batch tokens + 所有等待队列（含 PD 分离专属队列）的 tokens，统计非常全面。

## 外部显式指定 dp rank

请求可携带 `routed_dp_rank` 字段直接路由到指定 rank，优先级高于任何调度策略：

```python
# data_parallel_controller.py:496-501
def maybe_external_dp_rank_routing(self, req):
    if req.routed_dp_rank is not None:
        self.workers[req.routed_dp_rank].send_pyobj(req)
        return True
    return False
```

不传 `routed_dp_rank`（默认 None）则由 dp_controller 按策略自动分配，**PD 分离下 decode 侧无需显式指定**。

## 启动参数

```bash
--load-balance-method total_tokens
```

可选值：`auto`（默认）、`round_robin`、`follow_bootstrap_room`、`total_requests`、`total_tokens`

### PD 分离 decode 侧推荐配置

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --dp-size <N> \
  --disaggregation-mode decode \
  --load-balance-method total_tokens
```

## 策略演进历史

| PR | 内容 |
|----|------|
| #7379 | 最早引入 minimum token load balance（`total_tokens`） |
| #11170 | 删除 minimum token balance（废弃旧实现） |
| #10201 | 重新实现 `total_tokens` / `total_requests`，引入 watching 上报机制 |
| #11469 | piggyback 负载上报，idle batch 也触发上报，负载数据更实时 |
| #16110 | 新增 `follow_bootstrap_room`，引入 `auto` 默认策略逻辑 |

## 关键文件

- `python/sglang/srt/managers/data_parallel_controller.py` — 策略实现、dispatch 逻辑
- `python/sglang/srt/observability/scheduler_metrics_mixin.py:591` — `get_load()` 实现
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py:914` — piggyback 触发点
- `python/sglang/srt/managers/tokenizer_manager.py:1613` — 负载上报转发
- `python/sglang/srt/server_args.py:807` — auto 默认策略逻辑
