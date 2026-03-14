# PP Disagg Prefill KeyError: 'hidden_states' 深度分析

## 1. 问题描述

在 prefill-decode 分离部署 (PP2+TP8+CP8) 模式下，长短请求混合压测时出现以下错误：

```
File "scheduler_pp_mixin.py", line 245, in event_loop_pp_disagg_prefill
    result, self.launch_event = self._pp_launch_batch(...)
  -> scheduler_pp_mixin.py:1069  run_batch(self.cur_batch, pp_proxy_tensors)
  -> model_runner.py:2399        forward_extend -> model.forward()
  -> deepseek_v2.py:1891         hidden_states = pp_proxy_tensors["hidden_states"]
KeyError: 'hidden_states'
```

PP Stage 1 (pp_rank=1) 在执行模型前向时，从 `pp_proxy_tensors` 字典中查找 `hidden_states` 键失败。

---

## 2. 正常工作流

### 2.1 PP 通信链路

PP2 模式下，模型被切分为两半：

- **Stage 0 (pp_rank=0)**：执行 embedding + 前半部分 layers，产出 `{"hidden_states": tensor, "residual": tensor}`
- **Stage 1 (pp_rank=1)**：接收 hidden_states，执行后半部分 layers + lm_head，产出 logits

```
Stage 0                              Stage 1
────────────────────                 ────────────────────
embedding(input_ids)
    │
前半层 forward
    │
PPProxyTensors {
  "hidden_states": ...,              _pp_recv_proxy_tensors()
  "residual": ...                         │
}                                    hidden_states = pp_proxy_tensors["hidden_states"]  <-- 此处出错
    │ _pp_send_dict_to_next_stage        │
    └──────────────────────────────→ 后半层 forward
                                         │
                                     logits output
```

### 2.2 关键代码路径 (deepseek_v2.py:1874-1891)

```python
def forward(self, input_ids, positions, forward_batch, ..., pp_proxy_tensors=None):
    if self.pp_group.is_first_rank:
        # Stage 0: 从 embedding 产生 hidden_states
        hidden_states = self.embed_tokens(input_ids)
        residual = None
    else:
        # Stage 1: 从 pp_proxy_tensors 获取 hidden_states (出错位置)
        hidden_states = pp_proxy_tensors["hidden_states"]   # <-- KeyError
        residual = pp_proxy_tensors["residual"]
```

---

## 3. 根因分析：PP 两阶段的 batch 调度不一致

### 3.1 核心问题

**根本原因：在 `event_loop_pp_disagg_prefill` 中，Stage 0 和 Stage 1 对于同一个 microbatch slot 可能产生不同的调度决策——Stage 0 认为没有 batch 需要运行（不发送 proxy tensors），而 Stage 1 认为有 batch 需要运行（尝试接收 proxy tensors）。**

这导致 Stage 1 的 `_pp_recv_proxy_tensors()` 接收到了错误迭代的数据，或者接收到了非 proxy tensor 类型的通信数据（如 output tensors），该字典中不包含 `hidden_states` 键。

### 3.2 对比：普通 event_loop_pp vs event_loop_pp_disagg_prefill

#### 普通 event_loop_pp（正确的同步方式）

```python
# scheduler_pp_mixin.py:80-95
recv_reqs = self.recv_requests()
self.process_input_requests(recv_reqs)
if not self.pp_group.is_last_rank:
    self._pp_commit_comm_work(self.send_req_work)
    # 注意：send_req_work 在此处立即发送，commit 在下一次迭代开始
    self.send_req_work = self._pp_send_pyobj_to_next_stage(recv_reqs, async_send=True)
# 关键：使用 get_next_batch_to_run()，该方法同时处理 prefill 和 decode
self.mbs[mb_id] = self.get_next_batch_to_run()
```

在普通 PP 模式中：
1. **请求转发是即时的**：Stage 0 在 `recv_requests` 之后立即异步发送给 Stage 1
2. **请求先到达再调度**：Stage 1 先接收 Stage 0 转发的请求，然后调度 batch
3. **相同的调度函数**：所有阶段使用相同的 `get_next_batch_to_run()`，在相同的 waiting_queue 状态下产生确定性的调度结果

#### Disagg Prefill 事件循环（问题代码）

```python
# scheduler_pp_mixin.py:211-249
recv_reqs = self.recv_requests()
self.process_input_requests(recv_reqs)
if not self.pp_group.is_last_rank:
    self._pp_commit_comm_work(self.send_req_work)       # [A] 等待上一轮的 req 发送完成

bootstrapped_rids = self._pp_pd_get_bootstrapped_ids()   # [B] bootstrap 共识
self._pp_commit_comm_work(send_bootstrapped_work)

transferred_rids = self._pp_pd_get_prefill_transferred_ids()  # [C] transfer 共识
self._pp_commit_comm_work(send_transfer_work)

self.process_prefill_chunk()                              # [D] 处理 chunked prefill
batch = self.get_new_batch_prefill()                      # [E] 调度新 batch
self.mbs[mb_id] = batch

self.cur_batch = self.mbs[mb_id]
if self.cur_batch:
    pp_proxy_tensors = self._pp_recv_proxy_tensors()      # [F] Stage 1 阻塞等待

self._pp_commit_comm_work(self.send_proxy_work)           # [G] 等待上一轮 proxy 发送完成
if self.cur_batch:
    result, self.launch_event = self._pp_launch_batch(    # [H] 执行 batch
        mb_id, pp_proxy_tensors, ...)
```

以及循环末尾的发送逻辑：
```python
# scheduler_pp_mixin.py:294-309
if not self.pp_group.is_last_rank:
    self.send_req_work = self._pp_send_pyobj_to_next_stage(recv_reqs, async_send=True)
    send_bootstrapped_work = self._pp_send_pyobj_to_next_stage(bootstrapped_rids, ...)
    send_transfer_work = self._pp_send_pyobj_to_next_stage(transferred_rids, ...)
    if self.cur_batch:                                    # [I] 只有有 batch 时才发送 proxy
        torch.cuda.current_stream().wait_event(self.launch_event)
        self.send_proxy_work = self._pp_send_dict_to_next_stage(
            result.pp_hidden_states_proxy_tensors.tensors, async_send=True)
```

### 3.3 问题触发机制详解

问题出在 **请求转发的延迟** 和 **bootstrap 共识带来的 waiting_queue 不一致**。

#### 时序分析

```
迭代 N:
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 0                            │ Stage 1                       │
├────────────────────────────────────┼───────────────────────────────┤
│                                    │                               │
│ 1. recv_requests (新请求 R1,R2)     │ 1. recv_requests              │
│ 2. process_input_requests          │ 2. process_input_requests     │
│ 3. commit(prev send_req_work)      │ 3. commit(prev send_req_work) │
│                                    │                               │
│ 4. bootstrap 共识 (本地决定)         │ 4. bootstrap 共识 (接收 Stage 0)│
│ 5. transfer 共识 (本地决定)          │ 5. transfer 共识 (接收 Stage 0) │
│                                    │                               │
│ ★ waiting_queue 可能有 R1,R2       │ ★ waiting_queue 状态可能不同    │
│   因为 bootstrap 共识刚完成将       │   因为请求转发要到循环末尾才发   │
│   某些 req 从 bootstrap_queue      │   送(步骤12)，但 bootstrap     │
│   移到 waiting_queue               │   共识中的 req 是上一轮的       │
│                                    │                               │
│ 6. process_prefill_chunk()         │ 6. process_prefill_chunk()    │
│ 7. get_new_batch_prefill()         │ 7. get_new_batch_prefill()    │
│    → batch = [R1, R2] (有batch)    │    → batch = None (无batch!)  │
│                                    │    或反之                      │
│ 8. cur_batch = batch (非空)        │ 8. cur_batch = batch          │
│ 9. [跳过 recv，是 first rank]       │ 9. cur_batch 非空时:           │
│                                    │    _pp_recv_proxy_tensors()   │
│                                    │    (阻塞等待 Stage 0 发送)     │
│ 10. _pp_launch_batch()             │ 10. 如果 cur_batch 非空:       │
│     → 产生 hidden_states           │     _pp_launch_batch() 出错!   │
│                                    │                               │
│ -- 循环末尾 --                      │                               │
│ 11. send_req_work(R1,R2→Stage1)   │                               │
│ 12. send bootstrap/transfer        │                               │
│ 13. if cur_batch:                  │                               │
│     send_proxy_work(hidden_states) │                               │
└─────────────────────────────────────┴───────────────────────────────┘
```

#### 关键差异：请求转发时序

在普通 `event_loop_pp` 中：
```python
# 步骤 2 就立即发送请求给下一阶段
self.send_req_work = self._pp_send_pyobj_to_next_stage(recv_reqs, async_send=True)
# 步骤 3 调度 batch (此时 Stage 1 已经有了相同的请求)
self.mbs[mb_id] = self.get_next_batch_to_run()
```

在 `event_loop_pp_disagg_prefill` 中：
```python
# 步骤 2: commit 上一轮的 send_req_work
self._pp_commit_comm_work(self.send_req_work)
# 步骤 3-5: bootstrap 和 transfer 共识 (期间 waiting_queue 在变化！)
# 步骤 6-7: 调度 batch
batch = self.get_new_batch_prefill()
# ...
# 步骤 12 (循环末尾才发送请求！):
self.send_req_work = self._pp_send_pyobj_to_next_stage(recv_reqs, async_send=True)
```

**在 disagg prefill 中，recv_reqs 被延迟到循环末尾才发送给 Stage 1，但 Stage 0 已经在步骤 7 根据新请求做了调度决策。**

### 3.4 batch 不一致的具体场景

场景：**长短请求混合导致 bootstrap 时序差异**

1. 在 **迭代 N-1** 末尾，Stage 0 完成了对一批新请求的 bootstrap 共识，将这些请求加入 `waiting_queue`
2. 在 **迭代 N** 的 bootstrap 共识步骤中（步骤4），Stage 0 又有新的 bootstrap 请求就绪
3. Stage 0 执行 `process_bootstrapped_queue`，将新就绪的请求加入 `waiting_queue`
4. Stage 0 的 `get_new_batch_prefill()` 看到 `waiting_queue` 非空，构建了一个 batch
5. **但 Stage 1 在同一迭代的 bootstrap 共识中**，由于：
   - 上一轮 Stage 0 转发的请求还没到（send_req_work 延迟发送）
   - 或 bootstrap queue 中的请求因为本地 KV sender 的状态差异，导致共识结果不同
   - 或 `process_prefill_chunk()` 的行为不同（因为 chunked_req 状态可能不一致）
6. Stage 1 的 `get_new_batch_prefill()` 返回 None（或返回了 batch，但 Stage 0 返回了 None）

### 3.5 不一致产生后的错误传播

当 Stage 0 和 Stage 1 的 `cur_batch` 不一致时，会出现以下两种情况之一：

#### 情况 A：Stage 0 有 batch, Stage 1 无 batch

```
Stage 0: cur_batch ≠ None → launch_batch → send_proxy_work (发送 hidden_states)
Stage 1: cur_batch = None → 不执行 recv_proxy_tensors，不执行 launch_batch
```

这种情况下，Stage 0 发送了 proxy tensors，但 Stage 1 没有接收。这些数据会滞留在通信通道中。

在**下一个迭代**，如果 Stage 1 有了 batch：
```
Stage 1: cur_batch ≠ None → _pp_recv_proxy_tensors() → 接收到上一轮 Stage 0 的遗留数据
```

如果上一轮的数据和当前 batch 的形状匹配，可能不会立即报错，但计算结果是错误的。如果形状不匹配，则可能在后续计算中报错。

#### 情况 B：Stage 0 无 batch, Stage 1 有 batch（更致命）

```
Stage 0: cur_batch = None → 不执行 send_proxy_work
Stage 1: cur_batch ≠ None → _pp_recv_proxy_tensors() → 阻塞等待...
```

此时 Stage 1 会接收到**下一个通信操作**的数据，而不是 proxy tensors。由于 `send_tensor_dict` / `recv_tensor_dict` 基于 NCCL 点对点通信，通道中的下一个消息可能是：
- 下一轮迭代 Stage 0 发送的 output tensors（包含 `next_token_ids` 等键）
- 或完全不相关的 tensor dict

**这正是 `KeyError: 'hidden_states'` 的直接原因：Stage 1 接收到的 dict 中没有 `hidden_states` 键，因为它接收到了错误的数据。**

---

## 4. 为什么触发概率苛刻？

### 4.1 需要同时满足多个条件

1. **bootstrap 时序窗口**：需要恰好在某个迭代中，Stage 0 和 Stage 1 的 bootstrap 共识结果导致 `waiting_queue` 内容不同。这要求 bootstrap 事件（KV sender 状态变化）恰好发生在两个 stage 之间的处理间隙中。

2. **长短请求混合触发**：短请求完成 bootstrap 更快（KV 传输数据量小），增加了两个 stage 之间 bootstrap 状态不一致的窗口。长请求的 chunked prefill 增加了 `chunked_req` 状态不一致的概率。混合负载使得 `get_new_batch_prefill()` 的决策更加敏感。

3. **chunked_req 状态差异**：在 PP 模式下，chunked request 可以在一个 microbatch 中开始，在另一个中结束。`process_prefill_chunk()` 的行为取决于 `self.chunked_req` 和 `self.last_batch` 的状态，这些状态在不同 stage 之间可能有微妙差异。

4. **内存压力和 batch_is_full**：`get_new_batch_prefill()` 中有多个提前返回 None 的条件：
   ```python
   if self.running_batch.batch_is_full or len(self.waiting_queue) == 0:
       return None
   if self.get_num_allocatable_reqs(running_bs) <= 0 and ...:
       return None
   if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
       self.running_batch.batch_is_full = True
   ```
   这些条件中的 `available_size()`、`running_batch.batch_is_full` 等状态在不同 stage 之间可能不同步。

5. **PP async batch depth = 0**：当 `pp_async_batch_depth > 0` 时，microbatch 交错执行，增加了对通信顺序的容错性。当 `pp_async_batch_depth = 0` 时，两个 stage 严格串行，更容易暴露时序问题。

### 4.2 竞态窗口分析

```
              时间线
              ──────→

Stage 0:  [recv_req] [bootstrap共识] [transfer共识] [schedule] ──── [send_req]
                                                        ↑
                                              此处 Stage 0 的
                                              waiting_queue 已更新

Stage 1:  [recv_req] [bootstrap共识] [transfer共识] [schedule]
                          ↑                              ↑
                  接收 Stage 0 的                 如果 waiting_queue
                  上一轮 bootstrap               与 Stage 0 不一致
                  结果                           → batch 决策不一致
```

竞态窗口 = bootstrap 共识到 schedule 之间的时间。在此窗口内，任何导致 `waiting_queue` 变化的事件（新请求 bootstrap 完成、KV transfer 完成等）都可能导致不一致。

### 4.3 长短请求混合为何增加触发概率

1. **短请求的 bootstrap 更快完成**：短请求的 KV sender 初始化更快，更容易在一个 stage 上完成而另一个 stage 上还未完成
2. **长请求的 chunked prefill**：长请求被分成多个 chunk，每个 chunk 作为一个 extend batch 执行。`chunked_req` 的存在改变了 `get_new_batch_prefill()` 的行为路径，增加了两个 stage 之间调度不一致的可能性
3. **内存竞争**：长请求占用大量 KV cache，导致 `available_size()` 接近阈值，使得小的差异就能决定 `batch_is_full` 的值
4. **请求到达不均匀**：长短请求混合时，请求完成和到达的模式更加不规则，增加了时序窗口中事件的随机性

---

## 5. 已实施的最小化修复

### 5.1 修复策略：保持 tensor-dict 通道始终同步

核心思想：**无论 Stage 0 是否有 batch，都在 proxy tensor 通道上发送数据**。Stage 0 无 batch 时发送空字典 `{}` 作为哨兵。Stage 1 **无条件接收**，检测到哨兵则强制跳过 batch。

这样 proxy tensor 通道的 send/recv 在每个 microbatch 迭代中永远成对出现，彻底消除通道错位。

### 5.2 具体改动 (`scheduler_pp_mixin.py`)

**改动一：接收侧（原 line 231-234）— 无条件 recv + 哨兵检测**

```python
self.cur_batch = self.mbs[mb_id]
if self.cur_batch:
    server_is_idle = False

# Always recv proxy tensors on non-first rank to keep the
# PP tensor-dict channel in sync.  When Stage 0 had no batch
# it sends an empty sentinel dict; we detect that here and
# force-skip the batch on this stage as well.
pp_proxy_tensors = self._pp_recv_proxy_tensors()
if pp_proxy_tensors is not None and not pp_proxy_tensors.tensors:
    # Received empty sentinel – previous stage had no batch
    self.cur_batch = None
    self.mbs[mb_id] = None
```

- `_pp_recv_proxy_tensors()` 对 first rank 返回 `None`（不做通信），对 non-first rank 始终执行 `recv_tensor_dict`
- 收到空字典时 `not pp_proxy_tensors.tensors` 为 `True`，强制当前 stage 也跳过 batch

**改动二：发送侧（原 line 304-309）— 无条件 send**

```python
if not self.pp_group.is_last_rank:
    ...
    if self.cur_batch:
        torch.cuda.current_stream().wait_event(self.launch_event)
        self.send_proxy_work = self._pp_send_dict_to_next_stage(
            result.pp_hidden_states_proxy_tensors.tensors, async_send=True)
    else:
        # Send empty sentinel so the next stage's recv stays
        # in sync with our send on the tensor-dict channel.
        self.send_proxy_work = self._pp_send_dict_to_next_stage(
            {}, async_send=True)
```

- 有 batch：正常发送 `{"hidden_states": ..., "residual": ...}`
- 无 batch：发送空字典 `{}`，`send_tensor_dict({})` 仅传递空 metadata，无实际 tensor 数据

### 5.3 覆盖的场景

| 场景 | 修复前行为 | 修复后行为 |
|------|-----------|-----------|
| Stage 0 有 batch, Stage 1 有 batch | 正常 | 正常（无变化） |
| Stage 0 无 batch, Stage 1 无 batch | 正常（均不通信） | Stage 0 发送空哨兵, Stage 1 收到哨兵确认跳过 |
| Stage 0 有 batch, Stage 1 无 batch | **通道错位**：Stage 0 发送了数据但 Stage 1 未接收 | Stage 1 无条件接收并消费数据，但 cur_batch=None 不执行 launch_batch |
| Stage 0 无 batch, Stage 1 有 batch | **KeyError/Hang**：Stage 1 阻塞等待或收到错误数据 | Stage 1 收到空哨兵，强制 cur_batch=None，跳过 batch |

### 5.4 开销分析

- **有 batch 时**：零额外开销，行为与修复前完全一致
- **无 batch 时**：额外一次空 dict 的 `send_tensor_dict` / `recv_tensor_dict`（仅传递空 metadata list，无 GPU tensor 通信），开销为微秒级 CPU P2P，可忽略不计

---

## 6. 修复评估

### 6.1 修复正确性

**结论：最新修复（commit `03f163baa`）方向正确，思路合理，能有效解决 proxy tensor 通道错位问题。**

**修复的核心不变式（invariant）：**
> 对于每一个 microbatch slot，非 first rank 恰好执行一次 `recv_tensor_dict`，non-last rank 恰好执行一次 `send_tensor_dict`（有 batch 时传真实数据，无 batch 时传空哨兵 `{}`）。

这保证了 proxy tensor 通道的 send/recv 在任何情况下都是 **1:1 对应** 的。

### 6.2 潜在边界情况分析

#### ❶ `server_is_idle` 标记的问题
在修复代码中，`server_is_idle = False` 的设置发生在 recv proxy tensors **之前**：
```python
self.cur_batch = self.mbs[mb_id]
if self.cur_batch:
    server_is_idle = False          # 此时还未检查哨兵

pp_proxy_tensors = self._pp_recv_proxy_tensors()
if pp_proxy_tensors is not None and not pp_proxy_tensors.tensors:
    self.cur_batch = None           # 强制跳过，但 server_is_idle 已被设为 False
    self.mbs[mb_id] = None
```
当 Stage 1 收到哨兵时，`self.cur_batch` 被置 None，但 `server_is_idle` 已经是 False。在普通 PP 中，这会导致 idle 路径不被触发，但在 disagg prefill 中 `event_loop_pp_disagg_prefill` 的 idle 判断是 `server_is_idle and len(self.disagg_prefill_inflight_queue) == 0`，影响有限，因为此时 Stage 0 确实有 batch 在跑（否则哨兵不会出现这种情况；若 Stage 0 也无 batch 则直接发送哨兵，Stage 1 收到后 cur_batch = None，server_is_idle 保持 True，符合预期）。**这是一个轻微的状态不一致，不影响正确性，只会让 idle 检查稍微保守一些。**

#### ❷ 普通 `event_loop_pp` 未添加哨兵逻辑
`event_loop_pp`（第 65-144 行）保留了原有的 "只有 cur_batch 非空才 recv/send" 的逻辑，没有添加哨兵机制。这是 **正确的**——因为普通 PP 在 `process_input_requests` 后立即发送请求给下一 stage，两个 stage 的调度决策是同步的，不存在不一致的问题。

#### ❸ output tensor 通道是否有类似问题？
`_pp_send_output_to_next_stage`（第 1009-1036 行）也有类似的条件发送：
```python
# last rank: only send if mbs[next_first_rank_mb_id] is not None
if mbs[next_first_rank_mb_id] is not None:
    send_output_work = self._pp_send_dict_to_next_stage(...)
# non-last rank: only send if pp_outputs is not None
if pp_outputs:
    send_output_work = self._pp_send_dict_to_next_stage(...)
```
对应的接收在 `_pp_commit_send_output_work_and_preprocess_output_tensors`（第 1038-1073 行）：
```python
if mbs[next_mb_id] is not None:
    if not mbs[next_mb_id].forward_mode.is_prebuilt():
        next_pp_outputs = PPProxyTensors(self._pp_recv_dict_from_prev_stage())
```
**output tensor 通道通过 `mbs[next_mb_id]` 来同步守卫**，发送和接收都用同一个 `mbs` 数组，不存在两侧独立判断的问题。**output 通道不受此 bug 影响。**

#### ❹ PP > 2 的情况（中间 stage）
当 PP > 2 时，中间 stage 既是 non-first rank（需要 recv proxy）也是 non-last rank（需要 send proxy）。修复代码：
- recv 侧：无条件 recv，收到哨兵则 cur_batch = None
- send 侧：cur_batch = None 时发送哨兵 `{}`

这保证了哨兵信号能沿 PP 链路正确传播。✓

### 6.3 修复完整性

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| Stage 0 有 batch, Stage 1 有 batch | ✓ 正常 | ✓ 正常（无变化）|
| Stage 0 无 batch, Stage 1 无 batch | ✓ 正常（均不通信）| ✓ 正常（发哨兵/收哨兵）|
| **Stage 0 有 batch, Stage 1 无 batch** | ⚠️ 通道积压 → 下一迭代数据错乱 | ✓ Stage 1 无条件 recv 消费数据，cur_batch=None 不执行 launch |
| **Stage 0 无 batch, Stage 1 有 batch** | ❌ KeyError/Hang | ✓ Stage 1 收到哨兵，强制 cur_batch=None |

**结论：修复逻辑完整，覆盖了所有 4 种 batch 不一致场景。**

---

## 7. 总结

| 项目 | 说明 |
|------|------|
| **根因** | `event_loop_pp_disagg_prefill` 中 Stage 0 和 Stage 1 独立调度 batch，bootstrap/transfer 共识和请求转发延迟导致 waiting_queue 不一致，进而导致 batch 调度决策不同 |
| **直接表现** | Stage 1 接收到了非 proxy tensor 的通信数据（或上一轮残留数据），缺少 `hidden_states` 键 |
| **触发条件** | 长短请求混合 + bootstrap 时序窗口内的状态不一致 + 内存压力接近阈值 |
| **触发概率** | 低（需要多个条件同时满足），但在高并发混合负载下可复现 |
| **影响范围** | 仅影响 PP + disagg prefill 模式，普通 PP 和 output tensor 通道不受影响 |
| **修复方案（03f163baa）** | 保持 proxy tensor 通道始终同步（空哨兵 + 无条件 recv），2 处改动共 ~13 行 |
| **修复评估** | ✅ 正确且完整，覆盖所有不一致场景；有一处轻微的 server_is_idle 保守化，不影响正确性 |
