---
name: understanding-pp-flow
description: "SGLang Pipeline Parallelism (PP) 完整流程详解。涵盖 PP 调度循环、PP+PD 分离部署、PP+CP 协同、proxy tensor 通信、microbatch 调度、共识机制等。Use when asked about PP flow, PP architecture, PP scheduling, PP disaggregation, or PP+CP+TP parallelism in SGLang."
---

# SGLang Pipeline Parallelism (PP) 完整流程详解

## 1. PP 核心概念

### 1.1 什么是 Pipeline Parallelism

PP 将模型按**层**切分到多个 stage（通常每个 stage 一个节点），每个 stage 处理模型的一部分层，通过 P2P 通信传递中间结果（hidden_states）。

```
PP2 模式示例 (DeepSeek-V3, 64 layers):

Stage 0 (Node 0): embedding + Layers 0-31
    ↓ hidden_states (P2P 通信)
Stage 1 (Node 1): Layers 32-63 + lm_head → logits
```

### 1.2 PP + TP + CP 并行层次

```
PP (Pipeline Parallelism) - 按模型层分割
  └── TP (Tensor Parallelism) - 按权重矩阵分割
       └── CP (Context Parallelism) - 按序列长度分割
```

- **PP**: 跨节点，按 layer 分割
- **TP**: 节点内，按权重矩阵列/行分割
- **CP**: 节点内，按序列长度分割（用于长序列 prefill）

### 1.3 关键数据结构

```python
# PPProxyTensors: PP 阶段间传递的中间张量
class PPProxyTensors:
    tensors: Dict[str, torch.Tensor]
    # Stage 0 → Stage 1: {"hidden_states": tensor, "residual": tensor}
    # Stage N (last) → Stage 0: {"next_token_ids": tensor, ...logprob tensors}

# PPBatchMetadata: 每个 microbatch 的元信息
@dataclass
class PPBatchMetadata:
    can_run_cuda_graph: bool
```

**关键文件**:
- `python/sglang/srt/managers/scheduler_pp_mixin.py` — PP 调度主逻辑
- `python/sglang/srt/managers/scheduler.py` — Scheduler 基类
- `python/sglang/srt/model_executor/forward_batch_info.py` — ForwardBatch 和 PPProxyTensors
- `python/sglang/srt/models/deepseek_v2.py` — 模型 forward 中的 PP 处理

---

## 2. PP 调度循环 (`event_loop_pp`)

### 2.1 核心循环结构

```python
# scheduler_pp_mixin.py::event_loop_pp (Line 46-145)
def event_loop_pp(self: Scheduler):
    self.init_pp_loop_state()
    while True:
        for mb_id in range(self.pp_loop_size):
            # pp_loop_size = pp_size + pp_async_batch_depth
            # 每个 microbatch slot 按顺序执行
            ...
```

### 2.2 单次 microbatch 迭代流程

```
┌─ 1. recv_requests() ─────────────── 从 tokenizer_manager 接收新请求
│
├─ 2. process_input_requests() ────── 预处理请求并加入 waiting_queue
│
├─ 3. send reqs to next stage ─────── (非 last rank) 将请求转发给下一 stage
│       ↑ 关键：普通 PP 在调度前就发送请求，保证各 stage 调度一致
│
├─ 4. get_next_batch_to_run() ─────── 调度下一个 batch (prefill 或 decode)
│
├─ 5. _pp_recv_proxy_tensors() ────── (非 first rank) 接收上一 stage 的 hidden_states
│
├─ 6. _pp_launch_batch() ─────────── 执行模型 forward
│       ├─ run_batch(cur_batch, pp_proxy_tensors)
│       ├─ Stage 0: embedding → 前半层 → 输出 {hidden_states, residual}
│       └─ Stage 1: 接收 hidden_states → 后半层 → lm_head → logits
│
├─ 7. _pp_send_recv_and_preprocess_output_tensors()
│       ├─ last rank: 发送 {next_token_ids, ...} 给 rank 0
│       └─ rank 0: 接收并预处理 output
│
├─ 8. _pp_process_batch_result() ──── 处理上一轮 batch 的结果
│       └─ 更新 req 状态、检查完成条件、发送 output
│
└─ 9. send proxy tensors ─────────── (非 last rank) 发送 hidden_states 给下一 stage
```

### 2.3 Microbatch 交错与 Async Batch Depth

```
pp_loop_size = pp_size + pp_async_batch_depth

示例 PP2, async_batch_depth=1:
  pp_loop_size = 3, microbatch slots: [mb0, mb1, mb2]

  循环中:
  mb0 → mb1 → mb2 → mb0 → mb1 → ...

  Stage 0 计算 mb0 时，Stage 1 可以处理 mb2 的结果
  → 实现计算与通信 overlap，减少 PP bubble
```

### 2.4 通信原语

| 函数 | 用途 | 通信方式 | async? |
|------|------|---------|--------|
| `_pp_send_pyobj_to_next_stage` | 发送 Python 对象（请求、rid 列表等） | CPU P2P (pickle) | 可选 |
| `_pp_recv_pyobj_from_prev_stage` | 接收 Python 对象 | CPU P2P (pickle) | 同步 |
| `_pp_send_dict_to_next_stage` | 发送 tensor dict (hidden_states / output) | NCCL P2P | 可选 |
| `_pp_recv_proxy_tensors` | 接收 proxy tensors | NCCL P2P | 同步阻塞 |
| `_pp_recv_dict_from_prev_stage` | 接收 tensor dict | NCCL P2P | 同步阻塞 |

**关键约束**：
- pyobj 通信仅在 `attn_tp_rank == 0` 的进程上执行，然后 broadcast 给其他 TP ranks
- 所有走同一 `(src, dst, group)` 的 pyobj 消息共享一条 **FIFO 有序** 管道
- **发送顺序必须与接收顺序严格对齐**，否则消息错位导致 mbs 槽位混乱

### 2.5 async send 的 commit 模式

PP 中所有 async pyobj 消息遵循统一的 **"底部发送 / 顶部 commit"** 模式：

```
mb_id=k 底部（循环末尾）:
    send_req_work      = send(recv_reqs,     async=True)   ← 消息入管道 #1
    send_bootstrap     = send(bootstrapped,  async=True)   ← 消息入管道 #2
    send_transfer_work = send(transferred,   async=True)   ← 消息入管道 #3
    send_has_batch_work= send([has_batch],   async=True)   ← 消息入管道 #4
    send_proxy_work    = send_dict(proxy,    async=True)   ← NCCL tensor dict

mb_id=k+1 顶部（循环开头）:
    commit send_req_work       (line 216: _pp_commit_comm_work)
    commit send_bootstrapped   (line 220)
    commit send_transfer_work  (line 223)
    ...接收对应消息...
    commit send_has_batch_work (line 239)
    commit send_proxy_work     (line 274)
```

`_pp_commit_comm_work` 调用 `work.wait()` 确保发送完成后再 recv 下一条，防止缓冲区重用冲突。

下游在 **mb_id=k+1** 顶部按同样顺序 recv：
```
recv recv_reqs       (line 212: recv_requests())
recv bootstrapped    (line 218: _pp_pd_get_bootstrapped_ids())
recv transferred     (line 222: _pp_pd_get_prefill_transferred_ids())
recv has_batch       (line 236: _pp_recv_has_batch_from_prev_stage())
```

---

## 3. PP + PD 分离部署 (Disaggregation)

### 3.1 Disagg Prefill 事件循环

```python
# scheduler_pp_mixin.py::event_loop_pp_disagg_prefill (Line 147-336)
```

**与普通 PP 的关键区别**:

1. **额外的队列管理**:
   - `disagg_prefill_bootstrap_queue`: 等待 KV sender 初始化
   - `disagg_prefill_inflight_queue`: KV 正在传输中
   - `waiting_queue`: 已就绪可执行 prefill

2. **共识机制 (Consensus)**:
   - Bootstrap 共识：各 stage 独立检查 KV sender 状态，取交集确定就绪请求
   - Transfer 共识：各 stage 独立检查 KV 传输完成状态，取交集确定完成请求
   - Release 共识：last rank → first rank 的环形共识

3. **请求转发延迟**:
   - **普通 PP**: 在调度前就发送请求给下一 stage
   - **Disagg Prefill**: 请求在循环末尾才发送（因为中间插入了 bootstrap/transfer 共识步骤）

### 3.2 Disagg Prefill 单次迭代流程

```
┌─ 1. recv_requests + process_input_requests
│
├─ 2. _pp_pd_get_bootstrapped_ids() ── Bootstrap 共识
│       ├─ First rank: 本地 poll KV sender 状态
│       └─ Other ranks: recv 上一 stage 结果 → 取交集
│
├─ 3. _pp_pd_get_prefill_transferred_ids() ── Transfer 共识
│       ├─ First rank: 本地检查 KV 传输完成
│       └─ Other ranks: recv 上一 stage 结果 → 取交集
│
├─ 4. process_prefill_chunk()
│
├─ 5. recv has_batch signal ── 调度决策同步（非 first_rank）
│       └─ prev_stage_has_batch=False → 跳过调度，batch=None
│
├─ 6. get_new_batch_prefill() ── 仅当 has_batch=True（或 first_rank）
│
├─ 7. recv proxy tensors ── 仅当 prev_stage_has_batch=True（非 first_rank）
│       └─ 注意：即使本 stage 的 cur_batch=None 也要 recv（保通道同步）
│
├─ 8. _pp_launch_batch() ── 执行 prefill forward（仅当 cur_batch）
│
├─ 9. 发送/接收 consensus bootstrapped/release rids（环形）
│
├─ 10. _pp_process_batch_result()
│
└─ 11. 循环末尾：发送 reqs, bootstrap rids, transfer rids, has_batch, proxy tensors
```

### 3.3 调度决策同步机制 (has_batch signal)

**问题**: Disagg Prefill 中各 stage 独立调度 batch（不同于普通 PP 在调度前先转发请求），
bootstrap/transfer 共识导致各 stage 的 waiting_queue 状态不一致。Stage 0 可能无 batch
但 Stage 1 有 batch（或反之），导致两个通道错位：
- **pyobj 通道**: 消息数量不对（如 consensus rids 取决于 bmbs/tmbs 是否非空）
- **tensor-dict 通道**: Stage 0 不 send proxy 但 Stage 1 在 recv → 死锁

**根因修复**: Stage 0 通过 pyobj 通道广播其 `has_batch` 决策，下游 stage 在调度**之前**接收该信号。

```python
# 循环顶部：recv has_batch（非 first_rank）
if not self.pp_group.is_first_rank:
    prev_stage_has_batch = self._pp_recv_has_batch_from_prev_stage()
else:
    prev_stage_has_batch = True  # Stage0 自己决定

# 调度逻辑
if self.pp_group.is_first_rank:
    batch = self.get_new_batch_prefill()  # Stage0 正常调度
    prev_stage_has_batch = batch is not None
elif prev_stage_has_batch:
    batch = self.get_new_batch_prefill()  # Stage1+ 仅在上游有 batch 时调度
else:
    batch = None  # 上游无 batch → 完全跳过，不分配 KV

# proxy tensor recv 条件：上游是否 send 了（而非本 stage 是否有 batch）
if not self.pp_group.is_first_rank and prev_stage_has_batch:
    pp_proxy_tensors = self._pp_recv_proxy_tensors()

# 循环底部：async send has_batch 给下一 stage
if not self.pp_group.is_last_rank:
    send_has_batch_work = self._pp_send_pyobj_to_next_stage(
        [prev_stage_has_batch], async_send=True)
```

**关键设计约束**：

1. **has_batch 必须用 async send + top recv 模式**（和 recv_reqs 等一样），不能用 sync send。因为所有 pyobj 消息走同一条 FIFO 管道，sync send 如果插在循环中部会打乱消息顺序。

2. **proxy tensor recv 条件是 `prev_stage_has_batch`，不是 `cur_batch`**。当上游有 batch（send proxy）但本 stage 自己调度返回 None 时，仍需 recv 以保持 tensor-dict 通道同步。

3. **PP > 2 时逐级传播**：中间 stage 根据自己的调度结果（而非上游信号）向下游发送 has_batch，确保如果中间 stage 自己无法调度，下游也会跳过。

| 场景 | S0 send proxy | S1 recv proxy | S1 调度 | 平衡？ |
|------|-------------|-------------|---------|--------|
| S0 有 batch, S1 有 batch | ✓ | ✓ (prev=True) | ✓ | ✓ |
| S0 有 batch, S1 无 batch | ✓ | ✓ (prev=True, 丢弃) | ✓ 但返回 None | ✓ |
| S0 无 batch, S1 有请求 | ✗ | ✗ (prev=False) | 完全跳过 | ✓ |
| S0 无 batch, S1 无请求 | ✗ | ✗ (prev=False) | 完全跳过 | ✓ |

### 3.4 Disagg Decode 事件循环

```python
# scheduler_pp_mixin.py::event_loop_pp_disagg_decode (Line 338-524)
```

**额外队列**:
- `disagg_decode_prealloc_queue`: 预分配 KV cache 等待接收
- `disagg_decode_transfer_queue`: KV 正在接收中
- Retract 机制：内存不足时回退请求

**共识类型**:
- Retract 共识：确定需要回退的请求
- Prealloc 共识：确定预分配成功的请求
- Transfer/Release 共识：确定 KV 接收完成的请求

---

## 4. PP 中的模型 Forward 流程

### 4.1 DeepSeek-V2 模型的 PP 处理

```python
# deepseek_v2.py::DeepseekV2ForCausalLM.forward
def forward(self, input_ids, positions, forward_batch, pp_proxy_tensors=None):
    if self.pp_group.is_first_rank:
        hidden_states = self.embed_tokens(input_ids)  # Stage 0: embedding
        residual = None
    else:
        hidden_states = pp_proxy_tensors["hidden_states"]  # Stage 1+: 从上一 stage 接收
        residual = pp_proxy_tensors["residual"]

    # 各 stage 执行自己负责的 layers
    for layer in self.layers[start_layer:end_layer]:
        hidden_states, residual = layer(hidden_states, residual, ...)

    if not self.pp_group.is_last_rank:
        # 非最后一个 stage: 返回 proxy tensors 给下一 stage
        return GenerationBatchResult(
            pp_hidden_states_proxy_tensors=PPProxyTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })
        )
    else:
        # 最后一个 stage: norm → lm_head → logits
        hidden_states = self.norm(hidden_states + residual)
        logits = self.lm_head(hidden_states)
        return logits
```

### 4.2 Layer 分配

```python
# 每个 PP stage 处理的 layer 范围
start_layer = pp_rank * (num_layers // pp_size)
end_layer = (pp_rank + 1) * (num_layers // pp_size)

# PP2, 64 layers:
# Stage 0: layers[0:32]
# Stage 1: layers[32:64]
```

---

## 5. PP + CP (Context Parallelism) 协同

### 5.1 CP 在 PP 内的位置

```
PP Stage 0 (Node 0, 8 GPUs):
  ├─ Embedding: 在 CP rank 0 执行，然后 CP split
  ├─ Layers 0-31: 每个 GPU 处理 seq_len/cp_size 的 tokens
  ├─ CP allgather: 收集所有 rank 的输出
  └─ 发送 hidden_states 给 PP Stage 1

PP Stage 1 (Node 1, 8 GPUs):
  ├─ 接收 hidden_states
  ├─ CP split: 再次按序列长度分割
  ├─ Layers 32-63: 每个 GPU 处理部分 tokens
  ├─ CP allgather: 收集输出
  └─ lm_head → logits
```

### 5.2 CP 两种模式

**Mode 0 (Zigzag)**: 用于 batch_size=1
```
原始: [0,1,2,3,4,5,6,7] → 8 blocks
Zigzag 重排: [block0, block7, block1, block6, block2, block5, block3, block4]
Rank 0: block0 + block7 (早期+晚期, 负载均衡)
```

**Mode 1 (Interleave)**: 用于 multi-batch
```
Rank 0: tokens[0, 4, 8, 12]  (idx % cp_size == 0)
Rank 1: tokens[1, 5, 9, 13]  (idx % cp_size == 1)
```

### 5.3 关键代码路径

| 步骤 | 位置 | 功能 |
|------|------|------|
| CP 元数据准备 | `deepseek_v2.py` forward 入口 | 计算 zigzag index, split_list |
| Input Split | `nsa/utils.py::cp_split_and_rebuild_data` | 按 CP 分割输入 |
| Forward | 各 layer 独立计算 | 每个 rank 处理部分 tokens |
| Output Gather | `nsa/utils.py::cp_all_gather_rerange_output` | allgather + 恢复顺序 |

---

## 6. PP 初始化与状态管理

### 6.1 init_pp_loop_state

```python
def init_pp_loop_state(self: Scheduler):
    self.pp_loop_size = self.pp_size + self.server_args.pp_async_batch_depth
    self.mbs = [None] * self.pp_loop_size          # 当前 microbatch
    self.last_mbs = [None] * self.pp_loop_size     # 上一轮 microbatch
    self.running_mbs = [ScheduleBatch(reqs=[], batch_is_full=False)
                        for _ in range(self.pp_loop_size)]  # 运行中 batch
    self.mb_metadata = [None] * self.pp_loop_size   # batch 元信息
    self.pp_outputs = None                          # 当前 PP 输出
    self.last_rank_comm_queue = deque()             # last rank 缓冲队列
    self.send_req_work = []                         # 异步发送 work
    self.send_proxy_work = []
    self.send_output_work = []
```

### 6.2 PP 通信组

```python
# scheduler.py 中初始化
self.pp_group       # PP 通信组 (NCCL)
self.pp_rank        # 当前 PP rank
self.pp_size        # PP stages 数量
self.tp_size        # TP 大小
self.attn_tp_group  # Attention TP 通信组
self.attn_tp_rank   # Attention TP rank
self.attn_cp_group  # CP 通信组
self.attn_cp_rank   # CP rank
```

---

## 7. 动态 Chunk Size 预测

PP 模式下使用二次模型预测最优 chunk size，使各 stage 的 prefill 延迟一致。

```python
# ChunkSizePredictor: f(l) = a*l^2 + b*l + c
# 目标：找到 x 使得 f(L+x) - f(L) = target_latency

def predict_next_chunk_size(self, history_len, base_chunk_size, ...):
    # 解方程 ax^2 + (2aL+b)x - T = 0
    A = a
    B = 2*a*L + b
    C = -T
    x = (-B + sqrt(B^2 - 4AC)) / (2A)
```

**Profile 阶段** (`profile_and_init_predictor`):
1. 仅在 PP0 执行 128 个不同长度的 prefill
2. 记录 (seq_len, latency) 数据点
3. 拟合二次曲线
4. 广播系数到所有 PP ranks

---

## 8. 已知问题与修复

### 8.1 Disagg Prefill KeyError: 'hidden_states' + KV Cache 泄漏

**根因**: Stage 0 和 Stage 1 独立调度 batch，bootstrap/transfer 共识和请求转发延迟导致各 stage 的 waiting_queue 不一致。Stage 0 无 batch 时不 send proxy tensors，Stage 1 有 batch 时阻塞 recv → 死锁或 KeyError。

**修复演进**:

1. **初始方案（空哨兵）**: Stage 0 无 batch 时发送空 dict `{}` 作为哨兵，Stage 1 无条件 recv 后检测空哨兵并丢弃 batch。
   - **问题**: 哨兵检测在 `get_new_batch_prefill()` 之后，KV cache 已分配，需要复杂的 rollback（释放 KV、回退 chunked_req.is_chunked、重新入 waiting_queue）。rollback 不完整导致 KV 泄漏（检测到 memory leak, 差 256 tokens）。

2. **错误修复（移动 recv 位置）**: 将 `_pp_recv_proxy_tensors()` 移到 `get_new_batch_prefill()` 之前，收到空哨兵后跳过调度。
   - **问题**: 破坏了 `pp_async_batch_depth > 0` 时的 mbs 槽位对应关系。proxy tensor send 在 mb_id=k 底部、recv 在 mb_id=k+1 顶部，移动 recv 位置导致收到错误 mb_id 的 proxy。

3. **正确修复（has_batch signal）**: Stage 0 通过 async pyobj 在循环底部发送 `[has_batch]` 信号，下游在循环顶部 recv 信号后决定是否调度。见 3.3 节详述。
   - 完全避免 KV 分配 → 无需 rollback
   - 遵循 send-at-bottom / recv-at-top 的统一 pyobj 模式 → 无 FIFO 消息错位
   - proxy tensor recv 以 `prev_stage_has_batch` 为条件 → tensor-dict 通道始终平衡

### 8.2 PP + Disagg 通信约束备忘

修改 PP disagg 循环时必须注意的约束：

- **pyobj FIFO 约束**: 同一 `(src, dst, cpu_group)` 管道上所有消息严格 FIFO，不能在循环中部插入 sync send（会排在其后的 async send 之前到达接收方）
- **tensor-dict 通道平衡**: Stage N send proxy tensors 的条件与 Stage N+1 recv 的条件必须一致（都用 `prev_stage_has_batch` 或都用 `cur_batch`），否则 mbs 槽位错乱
- **process_prefill_chunk() 不可跳过**: 它管理 chunked_req 和 last_batch 状态，即使无 batch 也必须执行
- **`_pp_commit_comm_work` 调用 `work.clear()`**: commit 后列表清空，下一轮 commit 同一变量不会重复 wait
- Output tensor 通道由 `mbs[]` 数组双边同步守卫，不受 batch 不一致影响
- 普通 `event_loop_pp` 不需要 has_batch 机制（请求在调度前已转发，各 stage 调度结果一致）

---

## 9. 典型部署配置

### PP2 + TP8 + CP8 (双机 16 GPU)

```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --tp-size 8 --pp-size 2 \
    --dist-init-addr <node0_ip>:5000 \
    --nnodes 2 --node-rank 0 \
    --enable-dp-attention \
    --enable-nsa-prefill-context-parallel \
    --nsa-prefill-context-parallel-size 8 \
    --chunked-prefill-size 16384
```

### PP2 + TP8 + Disagg (PD 分离)

Prefill server 和 Decode server 各自独立启动，都支持 PP：
- Prefill: `event_loop_pp_disagg_prefill`
- Decode: `event_loop_pp_disagg_decode`

---

## 10. 快速代码导航

| 概念 | 文件 | 函数/行 |
|------|------|---------|
| PP 主循环 | `scheduler_pp_mixin.py` | `event_loop_pp` (L46) |
| PP Disagg Prefill 循环 | `scheduler_pp_mixin.py` | `event_loop_pp_disagg_prefill` (L147) |
| PP Disagg Decode 循环 | `scheduler_pp_mixin.py` | `event_loop_pp_disagg_decode` (L338) |
| PP 初始化 | `scheduler_pp_mixin.py` | `init_pp_loop_state` (L526) |
| PP 通信：发送 pyobj | `scheduler_pp_mixin.py` | `_pp_send_pyobj_to_next_stage` (L876) |
| PP 通信：接收 pyobj | `scheduler_pp_mixin.py` | `_pp_recv_pyobj_from_prev_stage` (L906) |
| PP 通信：接收 has_batch | `scheduler_pp_mixin.py` | `_pp_recv_has_batch_from_prev_stage` (L904) |
| PP 通信：发送 tensor dict | `scheduler_pp_mixin.py` | `_pp_send_dict_to_next_stage` (L936) |
| PP 通信：接收 proxy tensors | `scheduler_pp_mixin.py` | `_pp_recv_proxy_tensors` (L953) |
| PP launch batch | `scheduler_pp_mixin.py` | `_pp_launch_batch` (L1075) |
| PP output 处理 | `scheduler_pp_mixin.py` | `_pp_send_recv_and_preprocess_output_tensors` (L1038) |
| PP 共识：bootstrap | `scheduler_pp_mixin.py` | `_pp_pd_get_bootstrapped_ids` (L750) |
| PP 共识：transfer | `scheduler_pp_mixin.py` | `_pp_pd_get_prefill_transferred_ids` (L780) |
| 动态 chunk 预测 | `scheduler_pp_mixin.py` | `ChunkSizePredictor` (L1240) |
| 模型 PP forward | `models/deepseek_v2.py` | `DeepseekV2ForCausalLM.forward` |
| CP split/gather | `layers/attention/nsa/utils.py` | `cp_split_and_rebuild_data`, `cp_all_gather_rerange_output` |
| ForwardBatch PP | `model_executor/forward_batch_info.py` | `PPProxyTensors` |
