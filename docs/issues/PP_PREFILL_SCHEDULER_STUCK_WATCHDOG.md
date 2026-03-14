# PP2+TP8+CP8 Prefill 调度卡死与 Watchdog 超时

## 现象

- **拓扑**：PP2 + TP8 + CP8（Pipeline Parallel 2，Tensor Parallel 8，Context Parallel 8）。
- **现象**：Prefill 调度有一定概率卡住，随后 **watchdog 约 5 分钟超时**，**PP rank1** 表现尤为明显。
- **日志特征**：
  - 多次出现 `scheduler.cur_batch.batch_size()=1`，且该 request 的 `input_ids` 极长（长序列 prefill）。
  - 时间上存在明显空档：例如 scheduler debug 在 `12:06:11.440`，下一批 prefill batch 日志在 `12:08:48.xxx`，中间约 2.5 分钟无进展。
  - Watchdog 触发时 dump 出 `cur_batch.reqs` 为单个超长请求。

## 堆栈信息（两轮截图综合）

### 堆栈一：PP1 卡在等 PP0

1. **`point_to_point_pyobj (utils/common.py:1364)`**  
   - 对应 `work.wait()`：在 **torch.distributed** 的 `irecv` 后 **阻塞等待** 对端数据。
2. **`recv_multipart (zmq/sugar/socket.py:799)`**  
   - 某线程在 **ZMQ 的阻塞 recv** 上等待消息。
3. **`_watchdog_thread (utils/watchdog.py:125)`**  
   - Watchdog 线程在同一快照中出现（多线程 dump）。

→ **PP rank1** 主线程卡在「等上一 stage（PP0）发数据」的 recv 上（如 `recv_requests()`、`_pp_recv_proxy_tensors()` 或 `_pp_recv_pyobj_from_prev_stage()` 里的 `point_to_point_pyobj`）。

### 堆栈二（PP1 明确卡点）：`_pp_recv_proxy_tensors` → `recv_tensor_dict`

第三张堆栈（Pyspy dump for PID 142, **PP1 ATTN_CP0 TP0**）精确显示：

- **`_pp_recv_proxy_tensors (scheduler_pp_mixin.py:941)`**  
  - 对应 `self.pp_group.recv_tensor_dict(...)`，即 PP1 在等 **上一 stage（PP0）发来的 proxy tensors**（前一阶段的 hidden states，供当前 stage 做 forward 输入）。
- **`recv_tensor_dict (distributed/parallel_state.py:1299)`**  
  - 底层在 `recv_object(src=src)` 等 PP0（`src = (rank_in_group - 1) % world_size`），阻塞在分布式通信。
- **`_watchdog_once (utils/watchdog.py:150)`**  
  - 同进程内 watchdog 线程在监控，主线程长时间不返回 → 约 5min 后触发超时。

结论：**PP1 的卡死位置可以确定为「已组好 `cur_batch` 后，在 `_pp_recv_proxy_tensors()` 内等 PP0 发送 proxy」**。PP0 若因长 prefill、调度瓶颈或自身卡在等 PP1 的 consensus/release 而迟迟不发送 proxy，PP1 就会一直停在这里，`forward_ct` 不涨，最终被 watchdog 判定为卡死。

### 堆栈三：PP0 卡在等 PP1（重新分析）

第二张堆栈明确显示：

1. **Process 211: `scheduler_PP0_ATTN_CP0_TP0`**  
   - 卡住的是 **PP0（Pipeline rank 0）**，不是 PP1。
2. **`_pp_recv_pyobj_from_prev_stage (scheduler_pp_mixin.py:877)`**  
   - PP0 在 **从「上一 stage」收 Python 对象** 时阻塞。  
   - 代码里 `prev = (pp_rank - 1) % pp_size`，对 PP0 即 **prev = PP1**，因此 **PP0 在等 PP1 发送**（consensus bootstrapped 或 release 等）。
3. **`PyThread_acquire_lock_timed` / `lock_PyThread_acquire_lock`**  
   - 与 **`zmq/backend/cython/_zmq.abi3.so`** 相邻，说明阻塞/等待发生在 ZMQ 或 dist 底层，并伴随 GIL/锁竞争。
4. **`_watchdog_once (utils/watchdog.py:150)`**  
   - 同进程内 watchdog 线程在 `_watchdog_once`（约 150 行为 sleep 或超时分支），与主线程「长时间无进度」一致，最终触发 5min 超时。

结论：**PP0 主线程卡在 `_pp_recv_pyobj_from_prev_stage()`，等待 PP1 发来的 consensus bootstrapped / release 等 pyobj**。与 PP1 堆栈结合可知：**PP0 等 PP1 发 → PP1 等 PP0 发（PP1 明确卡在 `_pp_recv_proxy_tensors` 等 PP0 的 proxy）**，形成 **PP0↔PP1 双向等待，即死锁**。

## 根因分析

### 1. Watchdog 的判定方式

- 逻辑在 `scheduler_runtime_checker_mixin.py` 的 `create_scheduler_watchdog`：
  - **进度**：`get_counter() => scheduler.forward_ct`
  - **活跃**：`is_active() => cur_batch is not None` 或初始化中
- `forward_ct` 只在 **`run_batch()` 入口** 自增（`scheduler.py` 约 2298 行）。

因此：只要在进入 `run_batch()` 之前发生**任意长时间阻塞**，进度就不会更新，但 `cur_batch` 可能已设置，watchdog 会认为「活跃但无进展」→ 超时。

### 2. PP Prefill 事件循环中 rank1 的阻塞点

`event_loop_pp_disagg_prefill`（`scheduler_pp_mixin.py`）中，**非首 PP rank** 的每轮迭代会依次：

| 步骤 | 可能阻塞点 |
|------|------------|
| `recv_requests()` | **blocking**：`point_to_point_pyobj` 从 rank0 收请求（`scheduler.py` 1261–1267） |
| `_pp_pd_get_bootstrapped_ids()` / `_pp_pd_get_prefill_transferred_ids()` | 内部可能 recv |
| `process_prefill_chunk()` / `get_new_batch_prefill()` | 一般较短，长序列时可能变慢 |
| `cur_batch = mbs[mb_id]` | 已设置 `cur_batch` |
| **`_pp_recv_proxy_tensors()`** | **blocking**：`pp_group.recv_tensor_dict()` 等 rank0 的 proxy |
| `_pp_recv_pyobj_from_prev_stage()`（consensus bootstrapped / release） | **blocking**：`point_to_point_pyobj` 再收上一 stage 的 pyobj |
| `_pp_launch_batch()` → `run_batch()` | 这里才 `forward_ct += 1` |

因此：

- **Rank1 已构造出 `cur_batch`（例如 batch_size=1 的长序列）后**，若在：
  - `recv_requests()`（等 rank0 发下一批 req），或
  - `_pp_recv_proxy_tensors()`（等 rank0 发当前 batch 的 proxy），或
  - `_pp_recv_pyobj_from_prev_stage()`（等 rank0 发 bootstrap/release 等）
  中任意一处阻塞，则本 rank 不会进入 `run_batch()`，**`forward_ct` 一直不变**。
- Rank0 若因**长 prefill、调度/显存问题或自身卡住**而迟迟不发送，rank1 就会长时间停在上述 recv，最终被 watchdog 判为卡死（约 5min）。

堆栈里出现 **`point_to_point_pyobj` 的 `work.wait()`** 与（若存在）**ZMQ `recv_multipart`** 与这一分析一致：**PP rank1 在「等上一 stage」的 recv 上卡住**。

### 3. 为何既有 PP1 卡住也有 PP0 卡住：死锁

- **PP0**：在 `event_loop_pp_disagg_prefill` 里，每轮会执行 `if bmbs[next_mb_id] is not None: _pp_recv_pyobj_from_prev_stage()` 和 `if tmbs[next_mb_id] is not None: _pp_recv_pyobj_from_prev_stage()`。对 PP0 而言 `prev = (0-1)%2 = 1`，即 **PP0 在等 PP1 发** consensus bootstrapped / release。
- **PP1**：在 `recv_requests()`（等 PP0 发请求）或 `_pp_recv_proxy_tensors()`（等 PP0 发 proxy）或 `_pp_recv_pyobj_from_prev_stage()`（等 PP0 发 consensus/release）里阻塞。
- 若 **PP0 先进入「等 PP1 的 recv」**，而 **PP1 尚未发**（例如 PP1 还在本轮更早的 `recv_requests()` / `_pp_recv_proxy_tensors()` 里等 PP0），则两边都在等对方 → **PP0↔PP1 死锁**。谁先触发 5min 取决于谁先被 watchdog 判定无进度（PP0/PP1 都可能被 dump）。
- 锁与 ZMQ：堆栈中的 `PyThread_acquire_lock_timed` 与 ZMQ 库相邻，说明阻塞发生在通信/序列化层，可能放大这种双向等待（例如一端持锁导致对端 send 无法完成）。

### 4. 为何容易表现为「PP rank1 卡 5min」或「PP0 卡 5min」

- **单边慢**：Rank0 长 prefill 或调度/显存瓶颈时，PP1 在 `_pp_recv_proxy_tensors()` / `recv_requests()` 里长时间等 PP0 → 表现为 PP1 卡、watchdog 在 PP1 上触发。
- **死锁**：PP0 在 `_pp_recv_pyobj_from_prev_stage()` 等 PP1，PP1 在 `recv_requests()` / `_pp_recv_proxy_tensors()` / `_pp_recv_pyobj_from_prev_stage()` 等 PP0 → 两边都无 `forward_ct` 更新，任一 rank 或先到 5min 的 rank 触发 watchdog（你提供的两张堆栈分别对应 PP1 与 PP0 的卡住）。

## 建议方向

### 1. 短期：缓解误杀（PP 下放宽 watchdog 进度）【已实现】

- **思路**：在 PP 且非首 rank 时，让「进度」不仅看 `forward_ct`，也看「事件循环步进」`scheduler_loop_step_ct`，避免单纯因「等上一 stage」导致误杀。
- **实现**（已合入）：
  - `Scheduler` 增加 `scheduler_loop_step_ct`；在 `event_loop_pp` / `event_loop_pp_disagg_prefill` / `event_loop_pp_disagg_decode` 的**每轮 `mb_id` 迭代开始**处自增。
  - `create_scheduler_watchdog` 中，当 `pp_size > 1 and pp_rank > 0` 时，`get_counter()` 使用 `max(forward_ct, scheduler_loop_step_ct)`。这样只要本 rank 能**完成迭代并进入下一轮**（步进会自增），就不会仅因 `forward_ct` 不涨而 5min 超时。
- **局限**：若某次迭代内**一直阻塞在单个 recv**（如 `_pp_recv_proxy_tensors()`）超过 5 分钟，本轮不会走到下一轮，步进不会再次自增，watchdog 仍会触发。要彻底避免此类误杀，需在 recv 路径上加**超时 + 重试**，并在重试中自增步进或使用更长 PP 专用超时。

### 2. 中期：打破 PP0↔PP1 死锁与 rank0 瓶颈

- **死锁**：两段堆栈表明存在 **PP0 等 PP1 发 consensus/release、PP1 等 PP0 发 req/proxy** 的循环等待。需要核对 `event_loop_pp_disagg_prefill` 的**收发顺序**：是否能在不依赖「对方先发」的前提下，保证至少一方先完成 send（例如 last rank 先发 consensus/release，或 first rank 先发 req），避免双方同时进入 recv。必要时调整 **consensus/release 与 req/proxy 的先后或异步提交顺序**。
- **Rank0 瓶颈**：在 rank0 上对以下做 profiling / 打点：
  - `get_new_batch_prefill()` / `process_prefill_chunk()` 在**长序列**下的耗时；
  - `_pp_launch_batch()` / `run_batch()` 的耗时；
  - 是否有显存/调度导致 rank0 长时间拿不到 batch 或无法发 send。
- 确认 PP 两端的 **mb_id / 迭代顺序** 一致，避免错拍导致一方永远等不到预期消息。

### 3. 长期：通信可观测性与超时

- 为 `point_to_point_pyobj` 或 `recv_tensor_dict` 增加**可选超时**（若底层 dist 支持），超时后打日志并重试或 abort，避免无限阻塞。
- 对「等上一 stage」的 recv 打**耗时日志**（例如超过 30s 打 WARNING），便于区分「正常长 prefill」与「异常卡死」。

---

**总结**：结合两段堆栈可以确认：

1. **PP1 卡住**：主线程在 `point_to_point_pyobj`（或 `_pp_recv_proxy_tensors`）里等 PP0 发请求/proxy，`forward_ct` 不涨 → 5min 后 watchdog 在 PP1 上触发。
2. **PP0 卡住**：主线程在 **`_pp_recv_pyobj_from_prev_stage`（scheduler_pp_mixin.py:877）** 里等 **PP1** 发 consensus bootstrapped / release（对 PP0 而言 prev stage = PP1），同样无 `forward_ct` 更新 → 5min 后 watchdog 可在 PP0 上触发。
3. **死锁**：PP0 等 PP1 发、PP1 等 PP0 发，形成 **PP0↔PP1 双向等待**，与「Prefill 调度概率卡死」一致。锁与 ZMQ 堆栈说明阻塞发生在通信/锁层面。

建议：在 PP 非首 rank 用 `scheduler_loop_step_ct` 缓解单边「等上一 stage」的误杀；**重点排查并调整 consensus/release 与 req/proxy 的收发顺序以打破死锁**；并为跨 stage 通信加超时与可观测性。
