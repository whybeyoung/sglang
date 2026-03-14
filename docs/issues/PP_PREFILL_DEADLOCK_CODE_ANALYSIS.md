# PP2 Prefill Disaggregation 死锁：基于当前代码的收发顺序分析

## 启动参数摘要

与本次分析相关的参数：

- `--pp-size 2`：Pipeline 2 阶段（PP0, PP1）
- `--tp 8`
- `--disaggregation-mode prefill`：使用 `event_loop_pp_disagg_prefill`
- `--nnodes 2`，`--node-rank 0`（当前为 node 0）
- 未传 `--pp-async-batch-depth`，默认 **0**

因此：

- `pp_loop_size = pp_size + pp_async_batch_depth = 2`
- 每轮 `while True` 内 `for mb_id in range(2)`，顺序为 **mb_id=0 → mb_id=1**
- PP0 = first rank，PP1 = last rank

---

## 单轮 for 循环内各步骤（与通信相关）

下面只列会 **send/recv** 或 **阻塞** 的步骤，便于看出谁在等谁。

### PP0（first rank，非 last rank）每轮迭代

| 步骤 | mb_id | 操作 | 说明 |
|------|-------|------|------|
| 1 | 0,1 | `recv_requests()` | 从 tokenizer（NOBLOCK），不跨 PP |
| 2 | 0,1 | `_pp_commit_comm_work(send_req_work)` | 等待上一轮对 PP1 的 req 发送完成 |
| 3 | 0,1 | `_pp_pd_get_bootstrapped_ids()` | 只从本地 queue 取，**不 recv** |
| 4 | 0,1 | `_pp_pd_get_prefill_transferred_ids()` | 只本地 get_rids，**不 recv** |
| 5 | 0,1 | `get_new_batch_prefill()`，设 `cur_batch` | - |
| 6 | 0,1 | `_pp_recv_proxy_tensors()` | **first rank 不调用 recv**，直接跳过 |
| 7 | 0,1 | `_pp_launch_batch()` → `run_batch()` | `forward_ct += 1` |
| 8 | 0,1 | `send_consensus_bootstrapped_work` / `send_release_work` | 仅入队 async send，未 wait |
| 9 | 0,1 | **`if bmbs[next_mb_id]: _pp_recv_pyobj_from_prev_stage()`** | **阻塞：从 PP1 收 consensus/release** |
| 10 | 0,1 | `_pp_commit_comm_work(send_consensus_bootstrapped_work)` 等 | 把本轮 consensus/release 发往 PP1 并 wait |
| 11 | 0,1 | `if tmbs[next_mb_id]: _pp_recv_pyobj_from_prev_stage()` | 再次从 PP1 收 release |
| 12 | 0,1 | 末尾 `send_req_work = _pp_send_pyobj_to_next_stage(recv_reqs)` 等 | **向 PP1 发本轮的 recv_reqs** |

对 PP0，`next_mb_id = (mb_id + 1) % 2`，所以：

- **mb_id=0**：`next_mb_id=1`，步骤 9 为 **`if bmbs[1]: recv from PP1`**
- **mb_id=1**：`next_mb_id=0`，步骤 9 为 **`if bmbs[0]: recv from PP1`**

`bmbs[1]` 在本轮 for 里是在 **mb_id=1** 时才被赋值；当 PP0 正在执行 **mb_id=0** 时，`bmbs[1]` 仍是 **上一轮** 里 mb_id=1 赋的值。因此：

- PP0 在 **当前轮、mb_id=0** 的步骤 9 中，实际是在收 **上一轮 PP1 在 mb_id=1 时发来的** consensus/release。
- 若两 rank 严格同轮、同 mb_id 步进，则：当 PP0 处于「当前轮、mb_id=0、步骤 9」时，PP1 在同一轮里只可能处于 mb_id=0（或更早），**还没执行到 mb_id=1 的发送** → PP0 的这次 recv 等的是「上一轮」PP1 的发送；若上一轮 PP1 尚未发（例如上一轮 PP1 卡在 mb_id=0），就会死锁。

### PP1（非 first rank，last rank）每轮迭代

| 步骤 | mb_id | 操作 | 说明 |
|------|-------|------|------|
| 1 | 0,1 | **`recv_requests()`** | **阻塞：从 PP0 收 requests（point_to_point_pyobj）** |
| 2 | 0,1 | `_pp_pd_get_bootstrapped_ids()` | **`_pp_recv_pyobj_from_prev_stage()`：从 PP0 收 bootstrap** |
| 3 | 0,1 | `_pp_pd_get_prefill_transferred_ids()` | **`_pp_recv_pyobj_from_prev_stage()`：从 PP0 收 transferred** |
| 4 | 0,1 | `get_new_batch_prefill()`，设 `cur_batch` | - |
| 5 | 0,1 | **`_pp_recv_proxy_tensors()`** | **阻塞：从 PP0 收 proxy tensors（recv_tensor_dict）** |
| 6 | 0,1 | `_pp_launch_batch()` → `run_batch()` | - |
| 7 | 0,1 | `send_consensus_bootstrapped_work` / `send_release_work` | last rank：向 PP0 发 consensus/release（async） |
| 8 | 0,1 | **`if bmbs[next_mb_id]: _pp_recv_pyobj_from_prev_stage()`** | **从 PP0 收下一微批的 consensus/release** |
| 9 | 0,1 | 末尾 `send_req_work` 等 | 向下一 stage 发 recv_reqs（last rank 时下一 stage 为 PP0，ring） |

PP1 在 **mb_id=0** 时：先 `recv_requests()` 等 PP0 发 **本轮的 requests**；PP0 只有在**完成当前轮 mb_id=0 的步骤 12** 才会发这批 requests。而 PP0 要完成步骤 12，必须先完成步骤 9（recv from PP1）。因此：

- PP0 完成步骤 9 需要：PP1 在 **上一轮 mb_id=1** 已发出 consensus/release。
- PP1 完成上一轮 mb_id=1 需要：PP1 先完成上一轮 mb_id=0。
- PP1 完成上一轮 mb_id=0 需要：在步骤 1 收到 PP0 的 requests，即 PP0 在 **更上一轮 mb_id=0 末尾** 已发送。

若两 rank 严格同轮推进，则会出现：

- **当前轮、mb_id=0**：PP0 在步骤 9 等 PP1 的「上一轮 mb_id=1」的发送；PP1 在步骤 1 等 PP0 的「当前轮 mb_id=0 末尾」的发送。
- PP0 当前轮 mb_id=0 未完成 → 不会发当前轮 requests → PP1 的步骤 1 一直拿不到 → PP1 无法走到上一轮 mb_id=1 的发送（若上一轮 PP1 也卡在 mb_id=0，则根本没有「上一轮 mb_id=1」的发送）→ **PP0 步骤 9 永远等不到** → 死锁。

---

## 死锁条件归纳

1. **PP0** 在 **当前轮、mb_id=0** 的步骤 9 阻塞：`_pp_recv_pyobj_from_prev_stage()` 等 PP1 发 consensus/release（对应的是 **上一轮** PP1 在 mb_id=1 的发送）。
2. **PP1** 在 **当前轮、mb_id=0** 的步骤 1 阻塞：`recv_requests()` 等 PP0 发本轮的 requests（PP0 只在 **当前轮、mb_id=0 步骤 12** 才发）。
3. PP0 必须**先完成步骤 9** 才能执行到步骤 12 并发 requests；步骤 9 又依赖 PP1 在上一轮 mb_id=1 的发送。
4. 若因负载、调度或首轮/冷启动导致 **PP1 比 PP0 慢一拍**（PP1 上一轮未完成 mb_id=1，或尚未发出），则 PP0 的步骤 9 会一直等；PP0 不发 requests → PP1 步骤 1 一直等 → **双向等待，死锁**。

你提供的堆栈与上述一致：

- **PP1** 卡在 `_pp_recv_proxy_tensors()`（等 PP0 的 proxy）。
- **PP0** 卡在 `_pp_recv_pyobj_from_prev_stage()`（等 PP1 的 consensus/release）。

---

## 与 `pp_loop_size=2`、同序 for 的关系

- `pp_loop_size = 2`，for 顺序固定为 **mb_id=0 → mb_id=1**。
- 同一轮内，**PP1 的 mb_id=1 发送** 一定晚于 **PP1 的 mb_id=0**；而 **PP0 在 mb_id=0 的步骤 9** 需要的是「上一轮 PP1 mb_id=1」的发送。
- 因此协议依赖「**PP1 上一轮已完整跑完**」；若两 rank 不同步（PP1 落后或首轮未发），就会触发上述死锁。

---

## 建议改动方向（不改协议语义的前提下）

1. **放宽 PP0 步骤 9 的阻塞条件**  
   - 仅在「确定 PP1 上一轮已发送」时才做 blocking recv；否则先跳过或做一次非阻塞/带超时的 recv，未就绪则本轮不卡住（例如本微批 consensus 延后处理），避免 PP0 永远卡在步骤 9 导致无法发 requests。

2. **或：保证「PP1 先于 PP0 发出上一轮 mb_id=1」**  
   - 例如在 PP0 步骤 9 前加一次轻量同步/barrier，或由 PP1 在完成上一轮 mb_id=1 的 send 后再通知 PP0（若现有协议允许），确保 PP0 的 recv 不会等一个尚未发生的发送。

3. **或：调整 for 顺序/分工**  
   - 让「需要对方上一轮 mb_id=1 发送」的 recv 发生在对方已经发完的时机（例如不同 rank 使用不同 mb_id 顺序，或拆成两阶段 loop），需要结合现有 consensus/release 语义做设计。

4. **观测与定位**  
   - 已在 `event_loop_pp_disagg_prefill` 中加上诊断日志（见下），死锁时看**最后一条** `[PP prefill deadlock trace]` 即可判断卡点。

---

## 诊断日志用法（已加在代码里）

在 `scheduler_pp_mixin.py` 的 `event_loop_pp_disagg_prefill` 里已加 **debug** 级日志，统一前缀：`[PP prefill deadlock trace]`。

**启用方式**：启动参数里已有 `--log-level debug` 即可；若没有，加上 `--log-level debug`。

**日志含义**（每条都带 `pp_rank`、`mb_id`）：

| 日志 step | 含义 |
|-----------|------|
| `before recv_requests` | 即将从上一 stage 收 requests（PP1 会阻塞在这里等 PP0） |
| `after recv_requests` | 已收完 requests |
| `before _pp_recv_proxy_tensors` | 即将收 proxy（PP1 会阻塞在这里等 PP0） |
| `after _pp_recv_proxy_tensors` | 已收完 proxy |
| `before recv_pyobj consensus (from prev_stage)` | 即将从 prev_stage 收 consensus（PP0 会阻塞在这里等 PP1） |
| `after recv_pyobj consensus` | 已收完 consensus |
| `before recv_pyobj release (from prev_stage)` | 即将从 prev_stage 收 release |
| `after recv_pyobj release` | 已收完 release |
| `after send_req/send_proxy (posted async)` | 本 mb_id 已把 req/proxy 异步发往下一 stage |

**排查死锁时**：对 PP0 和 PP1 的进程/ Pod 分别看**最后一条** `[PP prefill deadlock trace]`：

- PP1 最后是 `before recv_requests` → PP1 在等 PP0 发 requests。
- PP1 最后是 `before _pp_recv_proxy_tensors` → PP1 在等 PP0 发 proxy。
- PP0 最后是 `before recv_pyobj consensus` 或 `before recv_pyobj release` → PP0 在等 PP1 发 consensus/release。

结合两边的 `mb_id` 可判断是否处于「PP0 在 mb_id=0 等 PP1，PP1 在 mb_id=0 等 PP0」的死锁。

---

## 小结

在你当前的启动方式下（PP2、disaggregation prefill、`pp_async_batch_depth=0`），`event_loop_pp_disagg_prefill` 的 **同一轮 for 内**：

- **PP0** 在 mb_id=0 的 **步骤 9** 会阻塞 recv PP1 的 consensus/release（依赖上一轮 PP1 mb_id=1 的发送）。
- **PP1** 在 mb_id=0 的 **步骤 1** 会阻塞 recv PP0 的 requests（依赖 PP0 当前轮 mb_id=0 末尾的发送）。

一旦 PP1 未及时完成上一轮 mb_id=1 的发送（或首轮/冷启动未发），就会形成 **PP0 等 PP1、PP1 等 PP0** 的死锁，与现有堆栈现象一致。建议按上面方向在协议或实现上打破这一双向等待（放宽 PP0 的 recv 条件、或保证 PP1 先发、或调整 loop 顺序），并加日志确认卡点与轮次。
