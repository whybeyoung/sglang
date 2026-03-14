# MTP (Multi-Token Prediction) 流程详解

## 核心问题

用户疑问：假设 `steps = 5`，`topk = 1`：
- Draft 生成 5 个 draft token
- Target 验证这 5 个位置
- 如果全部接受：在第 6 个位置（`spec_steps + 1`）生成一个新 token

**为什么全部接受后还要在第 6 个位置单独生成 1 个 token？为什么不直接进入下一次 step5？**

## 答案：Verify 是一个 Prefill-like 操作

**关键理解：Verify 的流程可以理解成一个 prefill。**

### 详细流程分析

#### 1. Draft 阶段（生成 5 个 draft tokens）

```
当前序列: [t0, t1, t2, ..., t_n]
Draft 模型生成: [d0, d1, d2, d3, d4]  (5 个 draft tokens)
```

**代码位置：**
- `python/sglang/srt/speculative/multi_layer_eagle_worker.py::draft()`
- `python/sglang/srt/speculative/eagle_worker.py::draft()`

#### 2. Verify 阶段（Target 模型验证 + 生成新 token）

**这是关键！** Verify 不是简单的"验证 5 个位置"，而是一个 **prefill-like 的 forward pass**：

```
输入到 Target 模型: [d0, d1, d2, d3, d4]  (5 个 draft tokens)
Target 模型 Forward: 
  - Position 0: 验证 d0，输出 logits_0
  - Position 1: 验证 d1，输出 logits_1
  - Position 2: 验证 d2，输出 logits_2
  - Position 3: 验证 d3，输出 logits_3
  - Position 4: 验证 d4，输出 logits_4
  - Position 5: 生成新 token，输出 logits_5  ← 这就是第 6 个位置！
```

**代码证据：**

1. **`num_tokens_per_batch = spec_steps + 1`**
   ```python
   # python/sglang/srt/speculative/eagle_worker.py:681
   spec_info.num_tokens_per_batch = self.speculative_num_steps + 1
   
   # python/sglang/srt/speculative/eagle_worker_v2.py:658
   verify_input.num_tokens_per_batch = self.speculative_num_steps + 1
   ```

2. **`accept_index` 的形状是 `(bs, spec_steps + 1)`**
   ```python
   # python/sglang/srt/speculative/eagle_info.py:262
   accept_index = torch.full(
       (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=batch.device
   )
   ```

3. **`predict` 的形状比 `logits` 多 1**
   ```python
   # python/sglang/srt/speculative/eagle_info.py:258-259
   predict_shape = list(logits_output.next_token_logits.shape)[:-1]
   predict_shape[-1] += 1  # 多 1 个位置！
   predict = torch.empty(predict_shape, dtype=torch.int32, device=batch.device)
   ```

#### 3. 为什么需要第 6 个位置？

**原因：Target 模型需要基于所有已接受的 tokens 生成下一个 token。**

假设全部接受 5 个 draft tokens：
- 序列变为：`[t0, t1, ..., t_n, d0, d1, d2, d3, d4]`
- 下一个 token 应该基于这个完整序列生成
- 第 6 个位置的 logits 就是基于 `[t0, ..., t_n, d0, d1, d2, d3, d4]` 计算的

**如果不生成第 6 个位置：**
- 下次 draft 时，需要先 forward 一次 target 模型来生成新 token
- 这会增加一次 forward pass，降低效率

**生成第 6 个位置的优势：**
- 在 verify 的 prefill-like forward 中，**一次性**完成验证 + 生成
- 避免额外的 forward pass
- 提高吞吐量

### 完整流程示例（steps=5, topk=1）

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Draft 生成                                           │
├─────────────────────────────────────────────────────────────┤
│ 当前序列: [t0, t1, ..., t_n]                                │
│ Draft 模型生成: [d0, d1, d2, d3, d4]                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Target Verify (Prefill-like)                        │
├─────────────────────────────────────────────────────────────┤
│ 输入: [d0, d1, d2, d3, d4]                                  │
│                                                             │
│ Target Forward (一次性处理 6 个位置):                        │
│   Pos 0: 验证 d0 → logits_0 → accept/reject d0             │
│   Pos 1: 验证 d1 → logits_1 → accept/reject d1             │
│   Pos 2: 验证 d2 → logits_2 → accept/reject d2             │
│   Pos 3: 验证 d3 → logits_3 → accept/reject d3             │
│   Pos 4: 验证 d4 → logits_4 → accept/reject d4             │
│   Pos 5: 生成新 token → logits_5 → sample new_token        │
│                                                             │
│ 假设全部接受:                                                │
│   接受的 tokens: [d0, d1, d2, d3, d4]                       │
│   新生成的 token: new_token (从 logits_5 采样)              │
│                                                             │
│ 最终输出: [d0, d1, d2, d3, d4, new_token]                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 更新序列，准备下一轮                                │
├─────────────────────────────────────────────────────────────┤
│ 新序列: [t0, ..., t_n, d0, d1, d2, d3, d4, new_token]      │
│ 下一轮 Draft 基于这个新序列生成新的 draft tokens             │
└─────────────────────────────────────────────────────────────┘
```

### 代码流程追踪

#### 1. Draft 生成
```python
# python/sglang/srt/speculative/multi_layer_eagle_worker.py:504
def draft(self, batch: ScheduleBatch):
    # ... 生成 draft tokens
    # 返回 EagleVerifyInput，包含 draft_token (5 个 tokens)
    return EagleVerifyInput(
        draft_token=draft_tokens,  # shape: (bs * topk * spec_steps,)
        spec_steps=self.speculative_num_steps,  # 5
        draft_token_num=self.server_args.speculative_num_draft_tokens,  # 5
        ...
    )
```

#### 2. Verify Forward
```python
# python/sglang/srt/speculative/eagle_worker.py:680-705
def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
    spec_info.prepare_for_verify(batch, self.page_size)
    spec_info.num_tokens_per_batch = self.speculative_num_steps + 1  # 6!
    
    # Target 模型 forward，处理 6 个位置
    batch_result = self.target_worker.forward_batch_generation(
        model_worker_batch, is_verify=True
    )
    logits_output = batch_result.logits_output
    # logits_output.next_token_logits.shape = (bs * 6, vocab_size)
```

#### 3. Verify 采样和接受
```python
# python/sglang/srt/speculative/eagle_info.py:216-377
def verify(self, batch, logits_output, ...):
    # accept_index 形状: (bs, spec_steps + 1) = (bs, 6)
    accept_index = torch.full(
        (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=batch.device
    )
    
    # predict 形状: (bs, spec_steps + 1 + 1) = (bs, 7)
    # 前 6 个位置存储验证结果，第 7 个位置存储新生成的 token
    predict_shape[-1] += 1
    predict = torch.empty(predict_shape, dtype=torch.int32, device=batch.device)
    
    # 验证前 5 个 draft tokens
    # 从 logits_0 到 logits_4 验证 d0 到 d4
    
    # 从 logits_5 生成新 token
    # 存储到 predict 的第 6 个位置
```

### 关键代码位置总结

1. **设置 `num_tokens_per_batch = spec_steps + 1`**:
   - `python/sglang/srt/speculative/eagle_worker.py:681`
   - `python/sglang/srt/speculative/eagle_worker_v2.py:658`
   - `python/sglang/srt/speculative/multi_layer_eagle_worker.py:916`

2. **`accept_index` 形状 `(bs, spec_steps + 1)`**:
   - `python/sglang/srt/speculative/eagle_info.py:262`
   - `python/sglang/srt/speculative/eagle_info_v2.py:297`

3. **Verify Forward (Prefill-like)**:
   - `python/sglang/srt/speculative/eagle_worker.py:703`
   - `python/sglang/srt/speculative/multi_layer_eagle_worker.py:641`

4. **Post-forward 处理**:
   - `python/sglang/srt/model_executor/forward_batch_info.py:967-972`
   ```python
   elif self.forward_mode.is_target_verify():  # verify
       num_tokens = bs * self.spec_info.draft_token_num  # bs * 5
       # 但实际 forward 了 bs * 6 个 tokens
   ```

### 总结

**为什么需要第 6 个位置？**

1. **Verify 是 Prefill-like 操作**：一次性 forward 所有 draft tokens + 生成新 token
2. **效率优化**：避免额外的 forward pass 来生成新 token
3. **数学正确性**：新 token 必须基于完整序列（包括所有接受的 draft tokens）生成

**如果不生成第 6 个位置会怎样？**
- 全部接受 5 个 draft tokens 后，序列变为 `[..., d0, d1, d2, d3, d4]`
- 下次 draft 前，需要先 forward target 模型生成新 token
- 这会增加一次 forward pass，降低吞吐量

**生成第 6 个位置的优势：**
- 在 verify 的 prefill-like forward 中，**一次性**完成验证 + 生成
- 提高吞吐量，减少延迟

---

## MTP 一句话推理循环流程（steps=5, topk=1）

### 完整循环图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MTP 循环开始                                      │
│                     当前序列: [t0, t1, ..., t_n]                          │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1: DRAFT (Decode Mode)                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Draft 模型基于当前序列生成 5 个 draft tokens                            │
│                                                                         │
│  输入序列: [t0, t1, ..., t_n]                                            │
│  Draft Forward (Decode, 5 steps):                                       │
│    Step 0: [t_n] → logits → sample d0                                  │
│    Step 1: [t_n, d0] → logits → sample d1                               │
│    Step 2: [t_n, d0, d1] → logits → sample d2                           │
│    Step 3: [t_n, d0, d1, d2] → logits → sample d3                       │
│    Step 4: [t_n, d0, d1, d2, d3] → logits → sample d4                  │
│                                                                         │
│  输出: draft_tokens = [d0, d1, d2, d3, d4]                              │
│  代码: multi_layer_eagle_worker.py::draft()                              │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 2: VERIFY (Target Verify Mode - Prefill-like)                     │
├─────────────────────────────────────────────────────────────────────────┤
│  Target 模型验证 draft tokens 并生成新 token                             │
│                                                                         │
│  输入: draft_tokens = [d0, d1, d2, d3, d4]                              │
│  Target Forward (Prefill-like, 一次性处理 6 个位置):                    │
│    Position 0: [t0, ..., t_n, d0] → logits_0 → verify d0               │
│    Position 1: [t0, ..., t_n, d0, d1] → logits_1 → verify d1            │
│    Position 2: [t0, ..., t_n, d0, d1, d2] → logits_2 → verify d2        │
│    Position 3: [t0, ..., t_n, d0, d1, d2, d3] → logits_3 → verify d3    │
│    Position 4: [t0, ..., t_n, d0, d1, d2, d3, d4] → logits_4 → verify d4 │
│    Position 5: [t0, ..., t_n, d0, d1, d2, d3, d4] → logits_5 → new_token│
│                                                                         │
│  验证结果: accept_length = 5 (假设全部接受)                               │
│  新 token: new_token (从 logits_5 采样)                                 │
│  代码: eagle_worker.py::verify()                                        │
│        eagle_info.py::verify()                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 3: 更新序列和输出                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  接受的 tokens: [d0, d1, d2, d3, d4]                                     │
│  新生成的 token: new_token                                               │
│                                                                         │
│  输出给用户: [d0, d1, d2, d3, d4, new_token]                            │
│  更新序列: [t0, t1, ..., t_n, d0, d1, d2, d3, d4, new_token]            │
│                                                                         │
│  代码: GenerationBatchResult                                            │
│        verified_id = [d0, d1, d2, d3, d4, new_token]                   │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 4: DRAFT EXTEND (如果有接受的 tokens)                             │
├─────────────────────────────────────────────────────────────────────────┤
│  更新 Draft 模型的 KV Cache，使其包含接受的 tokens                      │
│                                                                         │
│  输入: verified_id = [d0, d1, d2, d3, d4, new_token]                     │
│  Draft Extend Forward:                                                   │
│    基于新序列 [..., d0, d1, d2, d3, d4, new_token]                       │
│    更新 draft model 的 hidden states                                     │
│                                                                         │
│  目的: 为下一轮 draft 准备正确的状态                                    │
│  代码: multi_layer_eagle_worker.py::forward_draft_extend_after_decode() │
│        eagle_worker.py::forward_draft_extend_after_decode()            │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │   是否结束？     │
                    │  (EOS token?)   │
                    └─────────────────┘
                         ↙        ↘
                    Yes /          \ No
                     ↙              ↘
            ┌─────────────┐    ┌──────────────────────────────┐
            │   结束      │    │  回到 Step 1 (DRAFT)          │
            │  返回结果   │    │  新序列作为输入                │
            └─────────────┘    └──────────────────────────────┘
```

### 循环关键点

1. **Draft 阶段（Decode Mode）**
   - Draft 模型基于当前序列，**逐步**生成 5 个 tokens
   - 每个 step 都是 decode（单 token forward）
   - 输出：`[d0, d1, d2, d3, d4]`

2. **Verify 阶段（Prefill-like Mode）**
   - Target 模型**一次性** forward 6 个位置
   - 前 5 个位置验证 draft tokens
   - 第 6 个位置生成新 token
   - 这是 prefill-like 操作，不是 decode

3. **Draft Extend 阶段**
   - 如果有接受的 tokens，需要更新 draft model 的状态
   - 确保下一轮 draft 基于正确的序列

4. **循环**
   - 更新后的序列作为下一轮的输入
   - 重复 Draft → Verify → Extend 流程

### 代码调用链

```python
# 主循环入口
forward_batch_generation(batch)
    ↓
# Step 1: Draft
draft(batch)  # 生成 draft tokens
    ↓
# Step 2: Verify  
verify(batch, spec_info)  # 验证 + 生成新 token
    ↓
# Step 3: Draft Extend (如果有接受的 tokens)
forward_draft_extend_after_decode(batch)  # 更新 draft model 状态
    ↓
# 返回结果，准备下一轮
return GenerationBatchResult(...)
```

### 序列增长示例

假设初始序列长度为 `n`，steps=5：

```
Round 1:
  输入: [t0, ..., t_n]                    (长度: n)
  Draft: [d0, d1, d2, d3, d4]            (生成 5 个)
  Verify: 全部接受 + new_token_1          (接受 5 + 1 = 6 个)
  输出: [d0, d1, d2, d3, d4, new_token_1] (长度: n + 6)
  
Round 2:
  输入: [t0, ..., t_n, d0, d1, d2, d3, d4, new_token_1]  (长度: n + 6)
  Draft: [d5, d6, d7, d8, d9]                            (生成 5 个)
  Verify: 全部接受 + new_token_2                          (接受 5 + 1 = 6 个)
  输出: [d5, d6, d7, d8, d9, new_token_2]                (长度: n + 12)
  
Round 3:
  输入: [t0, ..., t_n, ..., new_token_1, d5, d6, d7, d8, d9, new_token_2]
  ...
```

### 关键理解

1. **Draft 是 Decode**：逐步生成，每个 step 是单 token forward
2. **Verify 是 Prefill-like**：一次性处理多个位置，类似 prefill
3. **第 6 个位置的必要性**：在 verify 的 prefill-like forward 中生成新 token，避免额外的 forward pass
4. **Draft Extend**：确保 draft model 的状态与 target model 同步，为下一轮准备

这就是为什么全部接受后还要生成第 6 个 token 的原因：**它在 verify 的 prefill-like forward 中一次性完成，而不是等到下一轮**。

