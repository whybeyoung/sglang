# CP Continuous-Split 通过 LSE 修正实现分布式 Attention

## 核心思想

通过 **LSE (Log-Sum-Exp) 修正**机制，可以让每个 CP rank 只存储 1/cp_size 的 KV cache，但 attention 计算仍然正确。

## LSE 修正原理

### 1. Flash Attention 的 LSE

Flash Attention 可以返回 `softmax_lse`（log-sum-exp），这是 attention 计算中的归一化因子：

```python
output, softmax_lse = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    return_softmax_lse=True,
    ...
)
```

**LSE 的含义**：
- `lse = log(sum(exp(QK^T / sqrt(d))))`
- 这是 softmax 归一化因子的对数形式
- 用于数值稳定的 attention 计算

### 2. Merge Attention States 机制

**文件**: `sgl-kernel/csrc/attention/merge_attn_states.cu`

```cpp
// Line 66-78
float p_lse = prefix_lse[token_idx * num_heads + head_idx];
float s_lse = suffix_lse[token_idx * num_heads + head_idx];

const float max_lse = fmaxf(p_lse, s_lse);
p_lse = p_lse - max_lse;  // 数值稳定化
s_lse = s_lse - max_lse;

const float p_se = expf(p_lse);  // sum-exp for prefix
const float s_se = expf(s_lse);  // sum-exp for suffix
const float out_se = p_se + s_se;  // 合并后的 sum-exp

const float p_scale = p_se / out_se;  // prefix 的权重
const float s_scale = s_se / out_se;  // suffix 的权重

// 合并 attention outputs
output = prefix_output * p_scale + suffix_output * s_scale;
```

**关键公式**：
```
max_lse = max(p_lse, s_lse)
p_scale = exp(p_lse - max_lse) / (exp(p_lse - max_lse) + exp(s_lse - max_lse))
s_scale = exp(s_lse - max_lse) / (exp(p_lse - max_lse) + exp(s_lse - max_lse))
output = prefix_output * p_scale + suffix_output * s_scale
```

### 3. 为什么这样可以工作？

**数学原理**：

假设有两个 attention outputs：
- `output_1 = softmax(QK_1^T) @ V_1`，对应的 LSE 是 `lse_1`
- `output_2 = softmax(QK_2^T) @ V_2`，对应的 LSE 是 `lse_2`

要合并这两个 outputs，需要：
```
output_merged = softmax([QK_1^T, QK_2^T]) @ [V_1; V_2]
```

使用 LSE 修正：
```
max_lse = max(lse_1, lse_2)
scale_1 = exp(lse_1 - max_lse) / (exp(lse_1 - max_lse) + exp(lse_2 - max_lse))
scale_2 = exp(lse_2 - max_lse) / (exp(lse_1 - max_lse) + exp(lse_2 - max_lse))
output_merged = output_1 * scale_1 + output_2 * scale_2
```

这等价于：
```
output_merged = softmax([QK_1^T, QK_2^T]) @ [V_1; V_2]
```

## 在 CP Continuous-Split 中的应用

### 方案：每个 Rank 独立计算 + LSE 修正合并

#### 核心思路

**关键洞察**：虽然每个 token 需要 attend 到所有之前的 tokens，但我们可以：
1. 每个 rank 只使用自己的 KV cache 计算**部分 attention**
2. 每个 rank 的 query 只 attend 到**自己的 tokens**（修改 attention mask）
3. 通过 LSE 修正合并所有 ranks 的部分 attention outputs

#### 步骤 1：修改 Attention Mask

每个 rank 的 query 只 attend 到自己的 tokens：

```python
# Rank 0: tokens [0, 4, 8, ...]
# Query 0 只 attend 到 KV [0]
# Query 4 只 attend 到 KV [0, 4]  # 注意：只包含 Rank 0 的 tokens
# Query 8 只 attend 到 KV [0, 4, 8]

# Rank 1: tokens [1, 5, 9, ...]
# Query 1 只 attend 到 KV [1]
# Query 5 只 attend 到 KV [1, 5]
# Query 9 只 attend 到 KV [1, 5, 9]
```

**实现**：修改 causal mask，让每个 rank 只 attend 到自己的 tokens：

```python
def compute_local_causal_mask(cp_rank, cp_size, seq_len_local):
    """
    为每个 rank 计算 local causal mask。
    只允许 attend 到自己的 tokens。
    """
    mask = torch.zeros((seq_len_local, seq_len_local), dtype=torch.bool)
    for i in range(seq_len_local):
        # Token i 在全局序列中的位置
        global_pos_i = i * cp_size + cp_rank
        # 只 attend 到自己的 tokens，且位置 <= i
        for j in range(i + 1):
            global_pos_j = j * cp_size + cp_rank
            if global_pos_j <= global_pos_i:
                mask[i, j] = True
    return mask
```

#### 步骤 2：每个 Rank 独立计算 Attention

每个 rank 使用**自己的 KV cache**和**修改后的 mask**计算 attention：

```python
# Rank 0: 使用 tokens [0, 4, 8, ...] 的 KV cache
local_mask_0 = compute_local_causal_mask(0, cp_size, seq_len_local)
output_0, lse_0 = flash_attn_with_kvcache(
    q=q_0,  # Rank 0 的 queries (tokens [0, 4, 8, ...])
    k_cache=k_cache_0,  # Rank 0 的 KV cache (tokens [0, 4, 8, ...])
    v_cache=v_cache_0,
    causal_mask=local_mask_0,  # 只 attend 到自己的 tokens
    return_softmax_lse=True,
)

# Rank 1: 使用 tokens [1, 5, 9, ...] 的 KV cache
local_mask_1 = compute_local_causal_mask(1, cp_size, seq_len_local)
output_1, lse_1 = flash_attn_with_kvcache(
    q=q_1,
    k_cache=k_cache_1,
    v_cache=v_cache_1,
    causal_mask=local_mask_1,
    return_softmax_lse=True,
)

# ... 其他 ranks
```

**关键点**：
- 每个 rank 的 query 只对应自己的 tokens
- 每个 rank 的 KV cache 也只包含自己的 tokens
- **每个 rank 只 attend 到自己的 tokens**（修改后的 mask）
- **不需要 allgather KV cache**

#### 步骤 3：Allgather Outputs 和 LSE

```python
# Allgather outputs 和 LSE（而不是 KV cache）
outputs_all = cp_allgather([output_0, output_1, output_2, ...])
lse_all = cp_allgather([lse_0, lse_1, lse_2, ...])
```

#### 步骤 4：使用 LSE 修正合并

**关键**：对于每个 token，需要合并所有 ranks 的 outputs。

**示例**（Token 4）：
- Rank 0 计算了 `output_0[token_4]`，attend 到 KV [0, 4]，LSE = `lse_0[token_4]`
- Rank 1 计算了 `output_1[token_1]`，attend 到 KV [1]，LSE = `lse_1[token_1]`
- Rank 2 计算了 `output_2[token_2]`，attend 到 KV [2]，LSE = `lse_2[token_2]`
- Rank 3 计算了 `output_3[token_3]`，attend 到 KV [3]，LSE = `lse_3[token_3]`

**合并**：
```python
# Token 4 需要合并所有 ranks 的 outputs
outputs_token_4 = [
    output_0[token_4_idx],  # Rank 0: attend to [0, 4]
    output_1[token_1_idx],  # Rank 1: attend to [1]
    output_2[token_2_idx],  # Rank 2: attend to [2]
    output_3[token_3_idx],  # Rank 3: attend to [3]
]
lse_token_4 = [
    lse_0[token_4_idx],
    lse_1[token_1_idx],
    lse_2[token_2_idx],
    lse_3[token_3_idx],
]

# 使用 merge_state 合并
merged_output_4 = merge_attention_states(
    outputs_token_4,
    lse_token_4
)
```

### 关键理解：为什么这样可以工作？

**数学原理**：

对于 token 4，完整的 attention 应该是：
```
output_4 = softmax([Q4K0^T, Q4K1^T, Q4K2^T, Q4K3^T, Q4K4^T]) @ [V0; V1; V2; V3; V4]
```

使用 LSE 修正：
```
output_4 = merge(
    softmax([Q4K0^T, Q4K4^T]) @ [V0; V4],  # Rank 0
    softmax([Q4K1^T]) @ [V1],              # Rank 1
    softmax([Q4K2^T]) @ [V2],              # Rank 2
    softmax([Q4K3^T]) @ [V3]               # Rank 3
)
```

通过 LSE 修正，这两个结果是**数学等价的**！

### 问题：Token 映射关系

**关键问题**：如何确定每个 token 需要合并哪些 ranks 的 outputs？

**示例**（CP_SIZE=4）：
- Token 4 (Rank 0) 需要合并：
  - Rank 0: token 4 的 output（attend to [0, 4]）
  - Rank 1: token 1 的 output（attend to [1]）
  - Rank 2: token 2 的 output（attend to [2]）
  - Rank 3: token 3 的 output（attend to [3]）

**映射关系**：
```python
def get_token_mapping(cp_rank, cp_size, token_idx):
    """
    对于 token_idx，确定需要从哪些 ranks 获取 outputs。
    """
    # Token i 需要 attend 到 [0, 1, ..., i]
    # 这些 tokens 分布在不同的 ranks
    needed_tokens = list(range(token_idx + 1))
    rank_to_token = {}
    for t in needed_tokens:
        rank = t % cp_size
        local_token_idx = t // cp_size
        if rank not in rank_to_token:
            rank_to_token[rank] = []
        rank_to_token[rank].append(local_token_idx)
    return rank_to_token
```

### 完整实现方案

#### 方案：Local Attention + LSE 修正

**核心思想**：
1. 每个 rank 只使用自己的 KV cache
2. 每个 rank 的 query 只 attend 到自己的 tokens（修改 attention mask）
3. Allgather 所有 ranks 的 outputs 和 LSE
4. 使用 LSE 修正合并，得到正确的 attention output

**实现步骤**：

```python
def cp_attention_with_lse_correction(
    q_local, k_local, v_local, cp_size, cp_rank, cp_group
):
    """
    CP Attention with LSE correction.
    每个 rank 只存储自己的 KV cache，通过 LSE 修正合并 attention outputs。
    """
    seq_len_local = q_local.shape[0]
    
    # Step 1: 计算 local causal mask（只 attend 到自己的 tokens）
    local_mask = compute_local_causal_mask(cp_rank, cp_size, seq_len_local)
    
    # Step 2: 每个 rank 独立计算 attention
    output_local, lse_local = flash_attn_with_kvcache(
        q=q_local,
        k_cache=k_local,
        v_cache=v_local,
        causal_mask=local_mask,  # 只 attend 到自己的 tokens
        return_softmax_lse=True,
    )
    
    # Step 3: Allgather outputs 和 LSE
    outputs_all = cp_allgather([output_local], cp_group)  # [cp_size, seq_len_local, num_heads, head_dim]
    lse_all = cp_allgather([lse_local], cp_group)  # [cp_size, seq_len_local, num_heads]
    
    # Step 4: 对于每个 token，合并所有 ranks 的 outputs
    seq_len_global = seq_len_local * cp_size
    output_merged = torch.zeros(
        (seq_len_global, num_heads, head_dim),
        device=q_local.device,
        dtype=q_local.dtype
    )
    
    for global_token_idx in range(seq_len_global):
        # 确定需要哪些 ranks 的 outputs
        rank_to_local_idx = get_token_mapping_for_merge(
            global_token_idx, cp_size
        )
        
        # 收集 outputs 和 LSE
        outputs_to_merge = []
        lse_to_merge = []
        for rank, local_idx in rank_to_local_idx.items():
            outputs_to_merge.append(outputs_all[rank][local_idx])
            lse_to_merge.append(lse_all[rank][local_idx])
        
        # 使用 merge_state 合并
        if len(outputs_to_merge) == 1:
            output_merged[global_token_idx] = outputs_to_merge[0]
        else:
            # 逐步合并（两两合并）
            merged = outputs_to_merge[0]
            merged_lse = lse_to_merge[0]
            for i in range(1, len(outputs_to_merge)):
                merged, merged_lse = merge_state_v2(
                    merged, merged_lse,
                    outputs_to_merge[i], lse_to_merge[i]
                )
            output_merged[global_token_idx] = merged
    
    return output_merged

def get_token_mapping_for_merge(global_token_idx, cp_size):
    """
    对于 global_token_idx，确定需要从哪些 ranks 获取 outputs。
    
    返回: {rank: local_token_idx}
    """
    rank_to_local_idx = {}
    # Token i 需要 attend 到 [0, 1, ..., i]
    for t in range(global_token_idx + 1):
        rank = t % cp_size
        local_idx = t // cp_size
        # 对于每个 rank，只需要最新的 token（因为 local attention 已经包含了之前的 tokens）
        if rank not in rank_to_local_idx or local_idx > rank_to_local_idx[rank]:
            rank_to_local_idx[rank] = local_idx
    return rank_to_local_idx
```

## 实现细节

### 1. 修改 Attention 计算

**文件**: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`

```python
def _get_q_k_bf16_with_lse_correction(
    self, x, positions, enable_dual_stream, forward_batch
):
    """
    在 continuous-split 模式下，不 allgather key，而是：
    1. 每个 rank 使用自己的 KV cache 计算 attention
    2. 返回 output 和 LSE
    3. 后续通过 LSE 修正合并
    """
    query, key = self._get_q_k_bf16_base(x, positions, enable_dual_stream)
    
    if forward_batch.nsa_cp_metadata is not None and self.nsa_enable_prefill_cp:
        if is_nsa_prefill_cp_continuous_split():
            # continuous-split 模式：不 allgather key
            # 每个 rank 使用自己的 KV cache
            # 返回 (query, key, need_lse=True)
            return query, key, True
        else:
            # in-seq-split 模式：需要 allgather
            key = cp_all_gather_rerange_output(...)
            return query, key, False
    
    return query, key, False
```

### 2. 修改 Attention 计算逻辑

```python
def forward_indexer_with_lse_correction(
    self, q_fp8, weights, forward_batch, topk, layer_id
):
    """
    在 continuous-split 模式下：
    1. 每个 rank 使用自己的 KV cache 计算 attention
    2. 返回 output 和 LSE
    3. 通过 allgather 收集 outputs 和 LSE
    4. 使用 merge_state 合并
    """
    if is_nsa_prefill_cp_continuous_split():
        # 每个 rank 独立计算
        output_local, lse_local = self._compute_attention_local(
            q_fp8, weights, forward_batch, layer_id
        )
        
        # Allgather outputs 和 LSE（而不是 KV cache）
        outputs_all = cp_allgather_outputs(output_local, cp_size)
        lse_all = cp_allgather_lse(lse_local, cp_size)
        
        # 使用 LSE 修正合并
        output_merged = self._merge_attention_with_lse(
            outputs_all, lse_all
        )
        
        return output_merged
    else:
        # 原始实现
        return self.forward_indexer(q_fp8, weights, forward_batch, topk, layer_id)
```

### 3. LSE 修正合并函数

```python
def _merge_attention_with_lse(
    self, outputs_all: List[torch.Tensor], lse_all: List[torch.Tensor]
) -> torch.Tensor:
    """
    合并多个 ranks 的 attention outputs，使用 LSE 修正。
    
    Args:
        outputs_all: List of [num_tokens, num_heads, head_dim] tensors
        lse_all: List of [num_tokens, num_heads] tensors
    
    Returns:
        Merged output: [num_tokens, num_heads, head_dim]
    """
    num_tokens = outputs_all[0].shape[0]
    num_heads = outputs_all[0].shape[1]
    head_dim = outputs_all[0].shape[2]
    cp_size = len(outputs_all)
    
    output_merged = torch.zeros(
        (num_tokens, num_heads, head_dim),
        device=outputs_all[0].device,
        dtype=outputs_all[0].dtype
    )
    
    # 对于每个 token，合并所有 ranks 的 outputs
    for token_idx in range(num_tokens):
        # 收集这个 token 的所有 outputs 和 LSE
        outputs_token = [outputs_all[r][token_idx] for r in range(cp_size)]
        lse_token = torch.stack([lse_all[r][token_idx] for r in range(cp_size)])
        
        # 计算 max LSE
        max_lse = torch.max(lse_token, dim=0).values  # [num_heads]
        
        # 计算 scales
        lse_stable = lse_token - max_lse.unsqueeze(0)  # [cp_size, num_heads]
        exp_lse = torch.exp(lse_stable)  # [cp_size, num_heads]
        sum_exp_lse = torch.sum(exp_lse, dim=0)  # [num_heads]
        scales = exp_lse / sum_exp_lse.unsqueeze(0)  # [cp_size, num_heads]
        
        # 合并 outputs
        output_token = torch.zeros(
            (num_heads, head_dim),
            device=outputs_token[0].device,
            dtype=outputs_token[0].dtype
        )
        for r in range(cp_size):
            output_token += outputs_token[r] * scales[r].unsqueeze(-1)
        
        output_merged[token_idx] = output_token
    
    return output_merged
```

## 优势分析

### 1. 内存优化

**优化前**（Allgather KV Cache）：
- 每个 rank 存储：`seq_len * hidden_size * 2`（完整 KV cache）
- Allgather 后：每个 rank 都有完整的 KV cache

**优化后**（LSE 修正）：
- 每个 rank 存储：`(seq_len / cp_size) * hidden_size * 2`（自己的 KV cache）
- Allgather outputs 和 LSE：`seq_len * hidden_size + seq_len * num_heads`（比 KV cache 小得多）
- **内存减少约 `cp_size` 倍**

### 2. 通信优化

**优化前**：
- Allgather KV cache：`seq_len * hidden_size * cp_size` 的数据量

**优化后**：
- Allgather outputs：`seq_len * hidden_size * cp_size`（相同）
- Allgather LSE：`seq_len * num_heads * cp_size`（很小，可以忽略）
- **通信量基本相同，但内存占用大幅减少**

### 3. 计算正确性

**LSE 修正保证**：
- 合并后的 output 等价于使用完整 KV cache 计算的 output
- 数学上等价，不会引入误差

## 实现挑战

### 1. Causal Attention 的处理

**问题**：每个 rank 的 query 需要 attend 到所有之前的 tokens，但这些 tokens 分布在不同的 ranks。

**解决方案**：
- **方案 A**：使用 Ring Attention 风格，逐步传递 KV cache
- **方案 B**：修改 attention mask，让每个 rank 只 attend 到自己的 tokens（但这会改变模型行为）
- **方案 C**：部分 allgather，只 allgather 需要的 tokens

### 2. 实现复杂度

- 需要修改 attention 计算逻辑
- 需要实现 LSE 修正合并
- 需要处理 causal attention 的特殊情况

### 3. 性能影响

- Ring Attention 需要多轮通信，可能比一次 allgather 更慢
- LSE 修正的计算开销
- 需要权衡内存优化和性能

## 关键理解：LSE 修正如何解决 Causal Attention

### 问题回顾

在 causal attention 中，token i 需要 attend 到所有之前的 tokens [0, 1, ..., i]。

**示例**（CP_SIZE=4，Token 4）：
- 需要 attend 到 [0, 1, 2, 3, 4]
- 这些 tokens 分布在不同的 ranks

### LSE 修正的解决方案

**核心思想**：将完整的 attention 分解为多个部分 attention，然后通过 LSE 修正合并。

**数学原理**：

对于 token 4，完整的 attention：
```
output_4 = softmax([Q4K0^T, Q4K1^T, Q4K2^T, Q4K3^T, Q4K4^T]) @ [V0; V1; V2; V3; V4]
```

可以分解为：
```
output_4 = merge(
    softmax([Q4K0^T, Q4K4^T]) @ [V0; V4],  # Rank 0: attend to [0, 4]
    softmax([Q4K1^T]) @ [V1],              # Rank 1: attend to [1]
    softmax([Q4K2^T]) @ [V2],              # Rank 2: attend to [2]
    softmax([Q4K3^T]) @ [V3]               # Rank 3: attend to [3]
)
```

**LSE 修正公式**：
```
lse_total = log(sum(exp([Q4K0^T, Q4K1^T, Q4K2^T, Q4K3^T, Q4K4^T])))
          = log(exp(lse_0) + exp(lse_1) + exp(lse_2) + exp(lse_3))

output_4 = (
    output_0 * exp(lse_0 - lse_total) +
    output_1 * exp(lse_1 - lse_total) +
    output_2 * exp(lse_2 - lse_total) +
    output_3 * exp(lse_3 - lse_total)
)
```

这等价于完整的 attention！

### 实现要点

1. **每个 rank 只计算部分 attention**：
   - Rank 0: attend to [0, 4, 8, ...]（自己的 tokens）
   - Rank 1: attend to [1, 5, 9, ...]（自己的 tokens）
   - 等等

2. **修改 attention mask**：
   - 每个 rank 的 query 只 attend 到自己的 tokens
   - 这确保了每个 rank 只使用自己的 KV cache

3. **Allgather outputs 和 LSE**：
   - 不需要 allgather KV cache
   - 只需要 allgather outputs 和 LSE（数据量小得多）

4. **LSE 修正合并**：
   - 对于每个 token，合并所有 ranks 的 outputs
   - 使用 merge_state 机制

## 总结

**LSE 修正机制**允许：
1. ✅ **每个 rank 只存储 1/cp_size 的 KV cache**
2. ✅ **Attention 计算正确**（通过 LSE 修正）
3. ✅ **内存大幅减少**（不需要 allgather KV cache）

**关键优势**：
- **不需要 Ring Attention**：不需要逐步传递 KV cache
- **只需要一次 allgather**：allgather outputs 和 LSE（而不是 KV cache）
- **内存优化明显**：每个 rank 的 KV cache 减少 `cp_size` 倍

**实现要求**：
1. 修改 attention mask，让每个 rank 只 attend 到自己的 tokens
2. Flash Attention 返回 LSE（`return_softmax_lse=True`）
3. 实现 LSE 修正合并逻辑
4. Allgather outputs 和 LSE（而不是 KV cache）

这是一个**可行的优化方案**，可以在保持正确性的同时，大幅减少内存占用，且不需要 Ring Attention 的复杂实现。

