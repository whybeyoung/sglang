# CP Continuous-Split KV Cache 优化方案

## 问题描述

当前在 `continuous-split` 模式下，每个 CP rank 虽然只处理自己的 tokens（通过 `token_idx % cp_size` 分割），但在 attention 计算时，key 被 allgather 了，导致每个 rank 都能看到**所有 KV cache**，而不是只看到自己的部分。

**问题**：是否可以让每个 rank 只看到自己的 KV cache？

**分析**：
- ✅ **计算并行性**：每个 rank 可以并行处理自己的 tokens
- ❌ **Attention 正确性**：在 causal attention 中，每个 token 需要 attend 到所有之前的 tokens，因此需要 allgather
- ❌ **内存优化限制**：由于需要 allgather，每个 rank 仍然需要存储完整的 KV cache（在 allgather 后）

**结论**：**当前的 allgather 实现是必要的**，因为：
1. Causal attention 要求每个 token 能看到所有之前的 tokens
2. Continuous-split 只是将计算分配到不同 ranks，但 attention 计算仍需要完整的上下文
3. 无法避免 allgather 的通信开销

**可能的优化方向**：
1. 优化 allgather 的实现（异步、overlap 等）
2. 如果模型支持特殊的 attention 模式，可以考虑修改 attention 计算逻辑
3. 探索部分 allgather 的可能性（只 allgather 需要的 tokens）

## 当前实现分析

### 1. Key Allgather 的位置

**文件**: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`

```python
# Line 258-266
if forward_batch.nsa_cp_metadata is not None and self.nsa_enable_prefill_cp:
    key = cp_all_gather_rerange_output(
        key.contiguous(),
        self.cp_size,
        forward_batch,
        torch.cuda.current_stream(),
    )
```

**问题**：这个 allgather 在 `continuous-split` 模式下是不必要的，因为：
- 每个 rank 的 query 只对应自己的 tokens
- Attention 计算时，每个 rank 只需要 attend 到自己的 KV cache
- Allgather 会增加通信开销和内存占用

### 2. Continuous-Split 的数据分割

**文件**: `python/sglang/srt/layers/attention/nsa/utils.py`

```python
# Line 77-105
def nsa_cp_continuous_split_data(input_: Union[torch.Tensor, List]):
    """
    # for continuous-split, split the tokens evenly according to the rule of token_idx % cp_size.
    """
    cp_size = get_attention_tp_size()
    cp_rank = get_attention_tp_rank()
    # ... 按照 token_idx % cp_size 分割
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()
```

**特点**：
- 每个 rank 只处理 `token_idx % cp_size == cp_rank` 的 tokens
- 例如：CP_SIZE=4，Rank 0 处理 tokens [0, 4, 8, 12, ...]，Rank 1 处理 [1, 5, 9, 13, ...]

### 3. KV Cache 写入

**文件**: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`

```python
# Line 857-864 (在 forward_indexer 中)
forward_batch.token_to_kv_pool.set_index_k_scale_buffer(
    layer_id=layer_id,
    loc=forward_batch.out_cache_loc,
    index_k=k_fp8,
    index_k_scale=k_scale,
)
```

**当前行为**：
- 每个 rank 写入自己的 KV cache（`out_cache_loc` 对应自己的 tokens）
- 但读取时，通过 allgather 可以看到所有 ranks 的 KV cache

## 优化方案

### 方案 1：条件性跳过 Key Allgather（推荐）

在 `continuous-split` 模式下，跳过 key 的 allgather，让每个 rank 只使用自己的 KV cache。

#### 修改点 1: `nsa_indexer.py::_get_q_k_bf16`

```python
# 文件: python/sglang/srt/layers/attention/nsa/nsa_indexer.py
# Line 258-266

# 修改前：
if forward_batch.nsa_cp_metadata is not None and self.nsa_enable_prefill_cp:
    key = cp_all_gather_rerange_output(
        key.contiguous(),
        self.cp_size,
        forward_batch,
        torch.cuda.current_stream(),
    )

# 修改后：
if forward_batch.nsa_cp_metadata is not None and self.nsa_enable_prefill_cp:
    # 在 continuous-split 模式下，不 allgather key，每个 rank 只使用自己的 KV cache
    if not is_nsa_prefill_cp_continuous_split():
        key = cp_all_gather_rerange_output(
            key.contiguous(),
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )
    # continuous-split 模式下，key 已经是每个 rank 自己的部分，不需要 allgather
```

#### 修改点 2: `deepseek_v2.py::rebuild_cp_kv_cache`

**文件**: `python/sglang/srt/models/deepseek_v2.py`

```python
# Line 1838-1850
def rebuild_cp_kv_cache(self, latent_cache, forward_batch, k_nope, k_pe):
    # support allgather+rerrange
    latent_cache[..., : self.kv_lora_rank] = k_nope.squeeze(1)
    latent_cache[..., self.kv_lora_rank :] = k_pe.squeeze(1)
    
    # 修改：在 continuous-split 模式下跳过 allgather
    if is_nsa_prefill_cp_continuous_split():
        # continuous-split 模式：不 allgather，直接返回 local KV cache
        return k_nope, k_pe
    else:
        # in-seq-split 模式：需要 allgather
        latent_cache_output = cp_all_gather_rerange_output(
            latent_cache.contiguous(),
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )
        k_nope = latent_cache_output[..., : self.kv_lora_rank].unsqueeze(1)
        k_pe = latent_cache_output[..., self.kv_lora_rank :].unsqueeze(1)
        return k_nope, k_pe
```

#### 修改点 3: `nsa_indexer.py::forward_npu` (NPU backend)

**文件**: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`

```python
# Line 1025-1036
if (
    is_prefill
    and self.nsa_enable_prefill_cp
    and forward_batch.nsa_cp_metadata is not None
):
    # 修改：在 continuous-split 模式下跳过 allgather
    if not is_nsa_prefill_cp_continuous_split():
        k = cp_all_gather_rerange_output(
            k.contiguous().view(-1, self.head_dim),
            self.cp_size,
            forward_batch,
            torch.npu.current_stream(),
        )
    # continuous-split 模式下，k 已经是每个 rank 自己的部分
```

#### 修改点 4: `nsa_backend.py` 中的类似逻辑

检查是否有其他地方也需要类似的修改。

**文件**: `python/sglang/srt/layers/attention/nsa_backend.py`

需要检查 `_get_topk_ragged` 或其他 attention 计算函数，确保它们能正确处理 local KV cache。

### 方案 2：修改 Attention 计算逻辑

确保 attention 计算时，每个 rank 只使用自己的 KV cache。

#### 关键点：KV Cache 读取范围

在 `_get_topk_ragged` 中，确保只读取当前 rank 对应的 KV cache：

```python
# 文件: python/sglang/srt/layers/attention/nsa/nsa_indexer.py
# Line 372-426 (_get_topk_ragged)

# 需要确保：
# 1. k_fp8 只包含当前 rank 的 KV cache
# 2. attention 计算时，query 只 attend 到自己的 KV cache
# 3. 不需要 allgather
```

### 方案 3：修改 KV Cache 存储结构

如果需要在 decode 阶段也能使用 local KV cache，可能需要：
1. 每个 rank 独立存储自己的 KV cache
2. 在需要时（如 decode 阶段）再进行 allgather

但 decode 阶段通常不需要 CP，所以这个方案可能不必要。

## 实现细节

### 1. 检查 Continuous-Split 模式

使用现有的函数：

```python
from sglang.srt.layers.attention.nsa.utils import is_nsa_prefill_cp_continuous_split

if is_nsa_prefill_cp_continuous_split():
    # continuous-split 模式：不 allgather key
    pass
else:
    # in-seq-split 模式：需要 allgather key
    key = cp_all_gather_rerange_output(...)
```

### 2. Attention 计算的影响 ⚠️ **关键问题**

在 `continuous-split` 模式下：
- **Query**: 每个 rank 的 query 只对应自己的 tokens（已实现）
- **Key**: 每个 rank 的 key 只对应自己的 tokens（需要修改，当前被 allgather）
- **Attention**: 每个 rank 的 query 需要 attend 到哪些 key？

#### ⚠️ **问题：Causal Attention 的正确性**

在 **causal attention**（自回归模型）中，token i 需要 attend 到所有之前的 tokens [0, 1, ..., i]。

**示例**（CP_SIZE=4）：
- Rank 0: tokens [0, 4, 8, 12, ...]
- Rank 1: tokens [1, 5, 9, 13, ...]
- Rank 2: tokens [2, 6, 10, 14, ...]
- Rank 3: tokens [3, 7, 11, 15, ...]

**对于 token 4 (Rank 0)**：
- 需要 attend 到 [0, 1, 2, 3, 4]
- Token 0 在 Rank 0 ✓
- Token 1 在 Rank 1 ✗ **需要 allgather**
- Token 2 在 Rank 2 ✗ **需要 allgather**
- Token 3 在 Rank 3 ✗ **需要 allgather**
- Token 4 在 Rank 0 ✓

**结论**：每个 rank 需要看到**完整的 KV cache** 才能正确计算 causal attention！

#### 🤔 **可能的解决方案**

**方案 A：保持 Allgather（当前实现）**
- ✅ 正确性保证：每个 rank 都能看到完整的 KV cache
- ❌ 内存占用：每个 rank 存储完整的 KV cache
- ❌ 通信开销：需要 allgather key

**方案 B：修改 Attention 模式（如果可行）**
- 如果模型支持某种特殊的 attention 模式，允许每个 rank 只 attend 到自己的 tokens
- 需要确认 continuous-split 是否设计为支持这种模式
- 可能需要修改 attention mask 或计算逻辑

**方案 C：部分 Allgather（优化方案）**
- 每个 rank 只 allgather 需要的 KV cache（例如，只 allgather 之前的 tokens）
- 但这可能比全量 allgather 更复杂

#### 📝 **当前实现的合理性**

**当前实现（allgather key）是合理的**，因为：
1. Causal attention 要求每个 token 能看到所有之前的 tokens
2. Continuous-split 只是将 tokens 分配到不同 ranks 处理，但 attention 计算仍需要完整的上下文
3. Allgather 是必要的通信开销，无法避免

**Attention 矩阵**（实际）：
```
Rank 0: Q[0,4,8,...] × K[0,1,2,3,4,5,6,7,8,...]^T  (allgather 后)
Rank 1: Q[1,5,9,...] × K[0,1,2,3,4,5,6,7,8,...]^T  (allgather 后)
Rank 2: Q[2,6,10,...] × K[0,1,2,3,4,5,6,7,8,...]^T  (allgather 后)
Rank 3: Q[3,7,11,...] × K[0,1,2,3,4,5,6,7,8,...]^T  (allgather 后)
```

**注意**：虽然每个 rank 的 query 只对应自己的 tokens，但 key 需要包含所有 tokens（通过 allgather）。

### 3. KV Cache 内存优化

**优化前**（每个 rank）：
- KV cache 大小：`seq_len * hidden_size * 2`（包含所有 tokens）
- 内存占用：`seq_len * hidden_size * 2 * dtype_size`

**优化后**（每个 rank）：
- KV cache 大小：`(seq_len / cp_size) * hidden_size * 2`（只包含自己的 tokens）
- 内存占用：`(seq_len / cp_size) * hidden_size * 2 * dtype_size`
- **内存减少 `cp_size` 倍**

### 4. 通信开销优化

**优化前**：
- 每个 layer 需要 allgather key：`seq_len * hidden_size` 的数据
- 通信量：`seq_len * hidden_size * cp_size`（allgather 的总通信量）

**优化后**：
- 不需要 allgather key
- 通信量：0（每个 rank 独立计算）

## 验证方法

### 1. 功能验证

1. **正确性检查**：
   - 运行相同的输入，比较优化前后的输出是否一致
   - 确保 attention 计算正确

2. **内存检查**：
   - 监控每个 rank 的 KV cache 内存占用
   - 确认内存减少 `cp_size` 倍

3. **性能检查**：
   - 测量通信时间（应该减少）
   - 测量整体 prefill 时间（应该减少或不变）

### 2. 边界情况

1. **序列长度不能被 cp_size 整除**：
   - 确保每个 rank 的 token 数量正确
   - 确保 KV cache 索引正确

2. **Multi-batch**：
   - 确保每个 batch 的 tokens 正确分割
   - 确保 KV cache 写入位置正确

## 潜在问题

### 1. Decode 阶段

Decode 阶段通常不需要 CP，但如果需要：
- 可能需要 allgather KV cache（因为 decode 需要看到完整的上下文）
- 或者使用其他策略

### 2. 与其他功能的兼容性

需要确保与以下功能的兼容性：
- FP8 KV cache
- HiCache
- Prefix sharing
- Speculative decoding

### 3. Attention 计算

确保 attention 计算时：
- Query 和 Key 的维度匹配
- Attention mask 正确
- 输出形状正确

## 实施步骤

1. **第一步**：修改所有 key allgather 的位置：
   - `nsa_indexer.py::_get_q_k_bf16` (Line 258-266)
   - `nsa_indexer.py::forward_npu` (Line 1025-1036)
   - `deepseek_v2.py::rebuild_cp_kv_cache` (Line 1838-1850)

2. **第二步**：添加必要的 import：
   ```python
   from sglang.srt.layers.attention.nsa.utils import is_nsa_prefill_cp_continuous_split
   ```

3. **第三步**：验证功能正确性（输出一致）
   - 运行相同的输入，比较优化前后的输出
   - 确保 attention 计算正确

4. **第四步**：验证内存优化（内存减少）
   - 监控每个 rank 的 KV cache 内存占用
   - 确认内存减少 `cp_size` 倍

5. **第五步**：验证性能优化（通信时间减少）
   - 测量通信时间（应该减少）
   - 测量整体 prefill 时间（应该减少或不变）

6. **第六步**：测试边界情况和兼容性
   - 序列长度不能被 cp_size 整除的情况
   - Multi-batch 情况
   - 与其他功能的兼容性（FP8 KV cache, HiCache, etc.）

## 相关代码位置

### 需要修改的位置：

1. **`python/sglang/srt/layers/attention/nsa/nsa_indexer.py`**:
   - Line 258-266 (`_get_q_k_bf16`): Key allgather
   - Line 1025-1036 (`forward_npu`): NPU backend key allgather

2. **`python/sglang/srt/models/deepseek_v2.py`**:
   - Line 1838-1850 (`rebuild_cp_kv_cache`): MLA attention KV cache rebuild
   - Line 2010-2014: 调用 `rebuild_cp_kv_cache` 的地方

3. **`python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`**:
   - Line 202, 348: 调用 `rebuild_cp_kv_cache` 的地方（可能需要检查）

### 参考代码位置：

- `python/sglang/srt/layers/attention/nsa/utils.py`:77-105 (`nsa_cp_continuous_split_data`)
- `python/sglang/srt/layers/attention/nsa/utils.py`:62-65 (`is_nsa_prefill_cp_continuous_split`)
- `python/sglang/srt/layers/attention/nsa_backend.py`:492-506 (continuous-split 处理)

## 总结

### ⚠️ **重要发现**

经过分析，**跳过 key 的 allgather 会导致 attention 计算错误**，因为：

1. **Causal Attention 的要求**：
   - 每个 token i 需要 attend 到所有之前的 tokens [0, 1, ..., i]
   - 在 continuous-split 模式下，这些 tokens 分布在不同的 ranks
   - 因此每个 rank 需要看到完整的 KV cache

2. **当前实现的合理性**：
   - Allgather key 是**必要的**，不是可以优化的部分
   - Continuous-split 的优势在于**计算并行性**，而不是内存优化
   - 每个 rank 可以并行处理自己的 tokens，但 attention 计算时需要完整的上下文

### 📊 **Continuous-Split 的实际优势**

1. **计算并行性**：
   - 每个 rank 并行处理自己的 tokens
   - 减少单 rank 的计算量

2. **内存分布**：
   - KV cache 写入时，每个 rank 只写入自己的部分
   - 虽然 allgather 后需要完整 cache，但写入时是分布的

3. **通信开销**：
   - Allgather 是必要的通信开销
   - 可以通过异步、overlap 等方式优化

### 🔍 **可能的优化方向**

1. **优化 Allgather 实现**：
   - 使用异步 allgather
   - 与计算 overlap
   - 使用更高效的通信库

2. **探索部分 Allgather**：
   - 只 allgather 需要的 tokens（例如，只 allgather 之前的 tokens）
   - 但这可能比全量 allgather 更复杂

3. **修改 Attention 模式**（如果模型支持）：
   - 如果模型支持特殊的 attention 模式，允许每个 rank 只 attend 到自己的 tokens
   - 需要确认 continuous-split 是否设计为支持这种模式

### ✅ **结论**

**当前的 allgather 实现是正确的和必要的**。Continuous-split 的主要优势在于计算并行性，而不是内存优化。如果要优化，应该关注：
1. 优化 allgather 的性能（异步、overlap）
2. 优化计算和通信的 overlap
3. 探索部分 allgather 的可能性（如果可行）

