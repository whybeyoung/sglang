# PP2TP8CP8 架构原理分析 - Context Parallel (CP) 详解

## 1. 架构概览

### 1.1 配置说明
- **PP (Pipeline Parallelism)**: 2 stages
- **TP (Tensor Parallelism)**: 8 ranks per PP stage  
- **CP (Context Parallelism)**: 8 ranks (等于 atten_tp_size = tp_size / dp_size)
- **总 GPU 数**: 16 GPUs (每机 8 GPU)

### 1.2 进程分布

```
Node 0 (PP Stage 0):
├── PP0 TP0-7: GPU 0-7
└── 处理 Layers 0-31（模型层的前半部分）
    └── 每个 rank 通过 CP 处理序列的一部分（~16k tokens per rank）

Node 1 (PP Stage 1):
├── PP1 TP0-7: GPU 0-7
└── 处理 Layers 32-63（模型层的后半部分）
    └── 每个 rank 通过 CP 处理序列的一部分（~16k tokens per rank）
```

**关键理解**：
- **PP**: 按模型层分割（Layer 0-31 vs 32-63），每个 PP stage 处理**完整的序列**
- **CP**: 在每个 PP stage 内部，按序列长度分割（131k tokens → 8 ranks，每个 rank ~16k tokens）
- **TP**: 在每个 rank 内部，按权重矩阵分割

### 1.3 并行策略层次

```
PP (Pipeline Parallelism) - 按模型层分割
  └── TP (Tensor Parallelism) - 按权重矩阵分割
       └── CP (Context Parallelism) - 按序列长度分割
```

## 2. Context Parallelism (CP) 核心原理

### 2.1 CP 的核心思想

CP 将**长序列**在**序列维度**上分割到多个 ranks，每个 rank 处理序列的一部分，然后通过 allgather 收集结果。

**关键优势**：
1. **降低单 GPU 内存压力**：每个 rank 只需要处理 `seq_len / cp_size` 的 tokens
2. **并行计算**：多个 ranks 同时处理不同部分的序列
3. **适用于长序列**：特别适合 128k+ 的长上下文场景
4. **负载均衡**：通过 zigzag 重排实现计算量均衡

### 2.2 CP 的两种模式

#### Mode 0: Zigzag 重排模式（默认，用于 batch_size=1）

**代码路径**: `python/sglang/srt/layers/attention/nsa/utils.py::prepare_input_dp_with_cp_dsa`

**工作原理**：
1. 将序列分割成 `cp_size * 2` 个 blocks
2. 使用 zigzag 方式重排 blocks，实现负载均衡
3. 每个 rank 处理两个 blocks（一个来自前半段，一个来自后半段）

**示例**（CP_SIZE=4，序列长度=16）：

```
原始序列: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

Step 1: 分割成 8 个 blocks (cp_size * 2 = 8)
block0: [0, 1]
block1: [2, 3]
block2: [4, 5]
block3: [6, 7]
block4: [8, 9]
block5: [10, 11]
block6: [12, 13]
block7: [14, 15]

Step 2: Zigzag 重排
重排后: [block0, block7, block1, block6, block2, block5, block3, block4]

Step 3: 分配给各个 ranks
Rank 0: block0 + block7 = [0,1, 14,15]  (4 tokens)
Rank 1: block1 + block6 = [2,3, 12,13]  (4 tokens)
Rank 2: block2 + block5 = [4,5, 10,11]  (4 tokens)
Rank 3: block3 + block4 = [6,7, 8,9]    (4 tokens)
```

**为什么需要 Zigzag？**

在 Causal Attention 中：
- Rank 0 处理 tokens [0,1, 14,15]，需要 attend 到所有前面的 tokens
- Rank 3 处理 tokens [6,7, 8,9]，需要 attend 到更多历史 tokens

如果简单按顺序分割：
- Rank 0: [0,1,2,3] - 只有少量历史 tokens（计算量小）
- Rank 3: [12,13,14,15] - 需要 attend 到所有前面的 tokens（计算量大）

**负载不均衡！**

Zigzag 重排后：
- 每个 rank 都处理一个"早期 block"和一个"晚期 block"
- 计算量更均衡

#### Mode 1: 简单分割模式（用于 multi-batch）

**代码路径**: `python/sglang/srt/layers/attention/nsa/utils.py::nsa_cp_mode1_split_data`

**工作原理**：
- 使用 `token_idx % cp_size` 来分配 tokens
- 更简单，但需要序列长度能被 `cp_size` 整除

**示例**（CP_SIZE=4，序列长度=16）：

```
Rank 0: tokens [0, 4, 8, 12]   (indices % 4 == 0)
Rank 1: tokens [1, 5, 9, 13]   (indices % 4 == 1)
Rank 2: tokens [2, 6, 10, 14]  (indices % 4 == 2)
Rank 3: tokens [3, 7, 11, 15]  (indices % 4 == 3)
```

**优势**：
- 支持 multi-batch prefill
- 更好的负载均衡
- 兼容 fused MoE

## 3. CP 代码路径分析

### 3.1 CP 元数据准备

**入口**: `python/sglang/srt/models/deepseek_v2.py::DeepseekV2ForCausalLM.forward`

```python
# Line 3369-3376
if self.nsa_enable_prefill_cp:
    if can_cp_split(len(input_ids), self.cp_size, self.use_nsa, forward_batch):
        forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(
            len(input_ids),
            self.cp_rank,
            self.cp_size,
            forward_batch.seq_lens_cpu.tolist(),
        )
```

**关键函数**: `prepare_input_dp_with_cp_dsa`
- **位置**: `python/sglang/srt/layers/attention/nsa/utils.py::315`
- **功能**: 计算 zigzag index、split_list、reverse_index 等元数据
- **关键计算**:
  ```python
  cp_segment_num = cp_size * 2  # 8 * 2 = 16 blocks
  seq_per_batch = kv_len // cp_segment_num  # 每个 block 的 token 数
  split_list = seq_per_batch.repeat_interleave(cp_segment_num).int().tolist()
  
  # Zigzag index: [0, 7, 1, 6, 2, 5, 3, 4] for cp_size=4
  zigzag_index = list(range(cp_rank, ..., cp_segment_num)) + 
                 list(range(cp_segment_num - cp_rank - 1, ..., -cp_segment_num))
  
  # Reverse index: [0, 1, 2, 3, 4, 5, 6, 7] 用于恢复原始顺序
  cp_reverse_index = ...
  ```

### 3.2 Input Split（输入分割）

**位置**: `python/sglang/srt/models/deepseek_v2.py::DeepseekV2Model.forward`

```python
# Line 3159-3162
if enable_prefill_cp(forward_batch, self.nsa_enable_prefill_cp):
    if self.pp_group.is_first_rank:
        hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
    positions = cp_split_and_rebuild_position(forward_batch, positions)
```

**关键函数**: `cp_split_and_rebuild_data`
- **位置**: `python/sglang/srt/layers/attention/nsa/utils.py::126`
- **功能**: 
  - Mode 0: 使用 zigzag_index 重排数据
  - Mode 1: 使用 `token_idx % cp_size` 分割

**实现细节**:
```python
def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):
    if is_nsa_prefill_cp_mode1():
        return nsa_cp_mode1_split_data(input_)  # Mode 1: 简单分割
    
    # Mode 0: Zigzag 重排
    input_list = list(torch.split(input_, forward_batch.nsa_cp_metadata.split_list, dim=0))
    result = torch.cat([input_list[i] for i in forward_batch.nsa_cp_metadata.zigzag_index], dim=0)
    return result.view(-1, input_.shape[-1])
```

### 3.3 Forward Pass（前向传播）

每个 rank 独立处理自己分配到的 tokens：
- 计算 attention（只 attend 到自己的 KV cache）
- 计算 MLP
- 所有计算都是并行的

**关键点**：
- 每个 rank 的 `hidden_states` 形状是 `(seq_len / cp_size, hidden_size)`
- Attention 计算时，每个 rank 只处理自己的 tokens
- KV cache 也按 CP 分割存储

### 3.4 Output Gather（输出收集）

**位置**: `python/sglang/srt/models/deepseek_v2.py::DeepseekV2Model.forward`

```python
# Line 3230-3239 (在 forward 的最后)
if self.pp_group.is_last_rank and enable_prefill_cp(forward_batch, self.nsa_enable_prefill_cp):
    # allgather + rerrange
    hidden_states = cp_all_gather_rerange_output(
        hidden_states,
        self.cp_size,
        forward_batch,
        torch.cuda.current_stream(),
    )
```

**关键函数**: `cp_all_gather_rerange_output`
- **位置**: `python/sglang/srt/layers/attention/nsa/utils.py::216`
- **功能**:
  1. **Allgather**: 收集所有 ranks 的输出
  2. **Rerange**: 使用 `cp_reverse_index` 重新排列，恢复原始顺序

**Allgather 过程**（Mode 0）:

```
Before Allgather:
Rank 0: [block0, block7]
Rank 1: [block1, block6]
Rank 2: [block2, block5]
Rank 3: [block3, block4]

After Allgather (每个 rank 都有完整数据):
[block0, block7, block1, block6, block2, block5, block3, block4]

After Rerange (使用 cp_reverse_index):
[block0, block1, block2, block3, block4, block5, block6, block7]
```

## 4. CP 与 NSA Indexer 的配合

### 4.1 Key Cache Gather

**位置**: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py::_get_q_k_bf16`

```python
# Line 259-265
if forward_batch.nsa_cp_metadata is not None and self.nsa_enable_prefill_cp:
    key = cp_all_gather_rerange_output(
        key.contiguous(),
        self.cp_size,
        forward_batch,
        torch.cuda.current_stream(),
    )
```

**原因**：
- 在 CP 模式下，每个 rank 只处理部分 tokens
- 但 NSA indexer 需要完整的 key cache 来计算 attention scores
- 因此需要 allgather key cache

### 4.2 KV Length 计算

**位置**: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py::forward_indexer`

```python
# Line 887-893
if forward_batch.nsa_cp_metadata is not None and is_nsa_prefill_cp_mode0():
    kv_len_prev = forward_batch.nsa_cp_metadata.kv_len_prev
    kv_len_next = forward_batch.nsa_cp_metadata.kv_len_next
    actual_seq_q_prev = forward_batch.nsa_cp_metadata.actual_seq_q_prev
    actual_seq_q_next = forward_batch.nsa_cp_metadata.actual_seq_q_next
```

**用途**：
- `kv_len_prev`: 当前 rank 处理的前一个 block 的累计长度
- `kv_len_next`: 当前 rank 处理的后一个 block 的累计长度
- 用于正确计算 attention 的 KV cache 索引

## 5. CP 的优势分析

### 5.1 内存优势

**单 GPU 内存需求对比**：

```
无 CP (CP_SIZE=1):
- 序列长度: 131k tokens
- Hidden size: 4096
- KV cache: 131k * 4096 * 2 (K+V) * 2 (bf16) = ~2.1 GB per layer
- 64 layers: ~134 GB (仅 KV cache)

有 CP (CP_SIZE=8):
- 每个 rank 序列长度: 131k / 8 ≈ 16k tokens
- KV cache: 16k * 4096 * 2 * 2 = ~262 MB per layer
- 64 layers: ~16.8 GB (仅 KV cache)
- **内存减少 8 倍！**
```

### 5.2 计算优势

**并行度提升**：
- 8 个 ranks 同时处理不同部分的序列
- 理论上可以提升 8 倍吞吐量（受通信开销限制）

**负载均衡**：
- Zigzag 重排确保每个 rank 的计算量相近
- 避免某些 rank 成为瓶颈

### 5.3 通信开销

**Allgather 通信量**：
- 每个 layer 的输出: `(seq_len, hidden_size)` = `(131k, 4096)` = ~2.1 GB
- Allgather 通信: 8 ranks × 2.1 GB = 16.8 GB
- 但这是**并行通信**，实际时间取决于最慢的 rank

**优化**：
- 使用异步 allgather (`cp_all_gather_into_tensor_async`)
- 与计算 overlap

## 6. PP + TP + CP 协同工作

### 6.1 数据流（131k tokens 输入）

```
Input (131k tokens)
  │
  ├─ PP Stage 0 (Node 0)
  │   ├─ Split by CP (8 ranks)
  │   │   ├─ Rank 0: ~16k tokens (zigzag: block0 + block15)
  │   │   ├─ Rank 1: ~16k tokens (zigzag: block1 + block14)
  │   │   ├─ ...
  │   │   └─ Rank 7: ~16k tokens (zigzag: block7 + block8)
  │   │
  │   ├─ Forward (Layers 0-31)
  │   │   ├─ Embedding (split by CP)
  │   │   ├─ Layer 0-31 (split by TP + CP)
  │   │   │   ├─ Attention (TP: 8 ranks, CP: 8 ranks)
  │   │   │   ├─ MLP (TP: 8 ranks, CP: 8 ranks)
  │   │   │   └─ Allgather outputs (CP)
  │   │   └─ Allgather final outputs (CP)
  │   │
  │   └─ Send to PP Stage 1 (PP communication)
  │
  └─ PP Stage 1 (Node 1)
      ├─ Receive from PP Stage 0
      ├─ Forward (Layers 32-63)
      │   ├─ Layer 32-63 (split by TP + CP)
      │   └─ Allgather final outputs (CP)
      └─ Output (131k tokens)
```

### 6.2 关键代码路径

1. **CP 元数据准备**: `deepseek_v2.py::3369-3376`
2. **Input Split**: `deepseek_v2.py::3159-3162`
3. **Forward with CP**: `deepseek_v2.py::3117-3243`
4. **Output Gather**: `deepseek_v2.py::3230-3239`
5. **NSA Indexer CP**: `nsa_indexer.py::259-265` (key gather)
6. **KV Length**: `nsa_indexer.py::887-893`

## 7. CP 的限制和注意事项

### 7.1 限制

1. **只支持 Prefill 阶段**：
   - Decode 阶段不需要 CP（每次只生成 1 个 token）
   - CP 只在 `is_context_parallel_extend()` 模式下启用

2. **Mode 0 限制**：
   - 只支持 `batch_size=1`
   - 需要 zigzag 重排，实现复杂

3. **Mode 1 限制**：
   - 需要序列长度能被 `cp_size` 整除
   - 否则需要 padding

4. **通信开销**：
   - 每个 layer 都需要 allgather
   - 64 layers × 2.1 GB = 134 GB 通信量
   - 但这是并行通信，实际时间取决于网络带宽

### 7.2 性能优化建议

1. **使用 Mode 1**（如果支持）：
   - 更好的负载均衡
   - 支持 multi-batch
   - 实现更简单

2. **异步通信**：
   - 使用 `cp_all_gather_into_tensor_async`
   - 与计算 overlap

3. **合理设置 cp_size**：
   - 通常等于 `atten_tp_size = tp_size / dp_size`
   - 在 PP2TP8CP8 中，`cp_size = 8 / 1 = 8`

4. **结合 Chunked Prefill**：
   - 进一步降低内存压力
   - 例如：`--chunked-prefill-size 16384`

## 8. 总结

### 8.1 CP 的核心价值

1. **突破单 GPU 内存限制**：支持超长序列（128k+）
2. **提升并行度**：多个 ranks 同时处理不同部分
3. **负载均衡**：Zigzag 重排确保计算量均衡
4. **与 TP/PP 协同**：三种并行策略互补，最大化 GPU 利用率

### 8.2 适用场景

- ✅ 长序列 prefill（>32k tokens）
- ✅ 内存受限的场景
- ✅ NSA (Native Sparse Attention) 模型
- ✅ 需要支持 128k+ 上下文的场景

### 8.3 性能权衡

**优势**：
- 内存减少 8 倍（CP_SIZE=8）
- 并行度提升 8 倍
- 支持超长序列

**代价**：
- Allgather 通信开销
- 实现复杂度增加
- 只支持 prefill 阶段

**总体**：对于长序列场景，CP 的优势远大于代价，是必要的优化。

