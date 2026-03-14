# Decode Retract 触发条件分析

## 一、Retract 触发位置

### 1.1 主要触发点

**文件**: `python/sglang/srt/managers/scheduler.py`

**函数**: `update_running_batch()`

**代码位置**: 第 2048-2073 行

```python
def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
    """Update the current running decoding batch."""
    initial_bs = batch.batch_size()
    
    batch.filter_batch(v1_spec_info_filtered=True)
    if batch.is_empty():
        batch.batch_is_full = False
        return batch
    
    # Check if decode out of memory
    if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (
        TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
    ):
        # 触发 retract
        old_ratio = self.new_token_ratio
        retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
            self.server_args, self.decode_mem_cache_buf_multiplier
        )
        # ... 处理 retracted requests
```

---

## 二、触发条件详解

### 2.1 主要条件：内存不足

**条件**: `not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier)`

**含义**: 当 decode batch 的内存检查失败时触发 retract

**检查逻辑** (`schedule_batch.py:1699-1709`):

```python
def check_decode_mem(
    self, 
    buf_multiplier=1, 
    selected_indices: Optional[List[int]] = None
):
    """检查是否有足够的内存进行 decode"""
    
    # 1. 计算下一次 decode 需要的新 pages 数量
    new_page_count = self.new_page_count_next_decode(selected_indices)
    
    # 2. 计算需要的 tokens
    num_tokens = (
        new_page_count
        * buf_multiplier
        * self.token_to_kv_pool_allocator.page_size
    )
    
    # 3. 从 tree cache 中 evict tokens（释放空间）
    evict_from_tree_cache(self.tree_cache, num_tokens)
    
    # 4. 检查可用内存是否足够
    return self._is_available_size_sufficient(num_tokens)
```

**`new_page_count_next_decode()` 逻辑** (`schedule_batch.py:1670-1697`):

```python
def new_page_count_next_decode(self, selected_indices: Optional[List[int]] = None):
    """计算下一次 decode 需要分配的新 pages 数量"""
    page_size = self.token_to_kv_pool_allocator.page_size
    requests = (
        self.reqs
        if selected_indices is None
        else [self.reqs[i] for i in selected_indices]
    )
    
    if page_size == 1:
        # 非 paged 模式：每个 request 需要 1 个 slot
        return len(requests)
    
    if not self.spec_algorithm.is_none():
        # Spec 模式：考虑 draft tokens
        server_args = get_global_server_args()
        thresh = server_args.speculative_num_draft_tokens + (
            (server_args.speculative_eagle_topk or 1)
            * (server_args.speculative_num_steps or 1)
        )
        # 如果 (seqlen + thresh) % page_size <= thresh，需要新 page
        return sum(
            1 for req in requests 
            if ((req.seqlen + thresh) % page_size) <= thresh
        )
    
    # 普通 decode 模式
    if self.enable_overlap:
        # Overlap 模式：seqlen % page_size == 0 时需要新 page
        return sum(1 for req in requests if req.seqlen % page_size == 0)
    else:
        # 非 overlap 模式：(seqlen - 1) % page_size == 0 时需要新 page
        return sum(1 for req in requests if (req.seqlen - 1) % page_size == 0)
```

**关键点**:
- 基于 **page 粒度** 计算内存需求（不是简单的 token 相加）
- 计算下一次 decode 需要分配的新 pages 数量
- 考虑 page size、spec 模式、overlap 模式等因素
- 如果 `需要的 tokens > 可用内存`，返回 `False`，触发 retract

### 2.2 测试条件（可选）

**条件**: `TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0`

**含义**: 用于测试目的，定期强制触发 retract

**环境变量**:
- `SGLANG_TEST_RETRACT`: 是否启用测试 retract
- `SGLANG_TEST_RETRACT_INTERVAL`: 触发间隔（每 N 次 forward）

---

## 三、Retract 执行流程

### 3.1 `retract_decode()` 函数

**文件**: `python/sglang/srt/managers/schedule_batch.py`

**函数**: `retract_decode()` (第 1719-1793 行)

**流程**:

```python
def retract_decode(
    self,
    server_args: ServerArgs,
    buf_multiplier: int = 1,
) -> Tuple[List[Req], float, List[Req]]:
    """Retract the decoding requests when there is not enough memory."""
    
    # 1. 准备排序的索引
    sorted_indices = list(range(len(self.reqs)))
    
    # 2. 排序策略（非 spec 模式）
    if not server_args.speculative_algorithm:
        sorted_indices.sort(
            key=lambda i: (
                len(self.reqs[i].output_ids),      # 已生成的 tokens 数量（降序）
                -len(self.reqs[i].origin_input_ids),  # 输入长度（升序）
            ),
            reverse=True,
        )
    
    # 3. 循环 retract，直到内存足够
    retracted_reqs = []
    first_iter = True
    while first_iter or (
        not self.check_decode_mem(
            selected_indices=sorted_indices, 
            buf_multiplier=buf_multiplier
        )
    ):
        # 如果只剩一个 request，停止 retract
        if len(sorted_indices) == 1:
            # 确保至少有一个 request 的内存
            assert self.token_to_kv_pool_allocator.available_size() > 0
            break
        
        first_iter = False
        
        # 4. 从后往前移除 request（优先级低的先移除）
        idx = sorted_indices.pop()
        req = self.reqs[idx]
        retracted_reqs.append(req)
        
        # 5. 释放内存
        self.release_req(idx, len(sorted_indices), server_args)
    
    # 6. 过滤 batch，只保留未 retract 的 requests
    self.filter_batch(keep_indices=sorted_indices)
    
    # 7. 计算新的 token ratio
    total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
    total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)
    
    new_estimate_ratio = (
        total_decoded_tokens
        + envs.SGLANG_RETRACT_DECODE_STEPS.get() * len(self.reqs)
    ) / (total_max_new_tokens + 1)
    new_estimate_ratio = min(1.0, new_estimate_ratio)
    
    return retracted_reqs, new_estimate_ratio, []
```

### 3.2 Retract 策略

#### 3.2.1 排序策略（非 Spec 模式）

**优先级**（从低到高，先 retract）:
1. **已生成 tokens 多的** (`output_ids` 长度大)
2. **输入长度短的** (`origin_input_ids` 长度小)

**原因**:
- 已生成 tokens 多的 request 占用更多内存
- 输入短的 request 更容易重新加载

#### 3.2.2 Spec 模式

**策略**: 从后往前 retract（`sorted_indices.pop()`）

**原因**: Spec decoding 的 `filter_batch` API 只能从后往前过滤

### 3.3 内存释放 (`release_req()`)

**文件**: `python/sglang/srt/managers/schedule_batch.py`

**函数**: `release_req()` (第 1795-1808 行)

```python
def release_req(self, idx: int, remaing_req_count: int, server_args: ServerArgs):
    req = self.reqs[idx]
    
    # 1. 如果是 decode 模式，offload KV cache 到 CPU
    if server_args.disaggregation_mode == "decode":
        req.offload_kv_cache(
            self.req_to_token_pool, 
            self.token_to_kv_pool_allocator
        )
    
    # 2. 释放 KV cache（不插入 tree cache）
    release_kv_cache(req, self.tree_cache, is_insert=False)
    
    # 3. 从 tree cache 中 evict tokens
    num_tokens = remaing_req_count * envs.SGLANG_RETRACT_DECODE_STEPS.get()
    evict_from_tree_cache(self.tree_cache, num_tokens)
    
    # 4. 重置 request 状态
    req.reset_for_retract()
```

**关键操作**:
1. **Offload KV cache**: 如果是 decode 模式，将 KV cache 转移到 CPU
2. **释放内存**: 释放 GPU 上的 KV cache
3. **Evict tree cache**: 从 tree cache 中移除 tokens
4. **重置状态**: 标记 request 为 retracted

---

## 四、Retract 后的处理

### 4.1 添加到队列

**代码**: `scheduler.py:2072-2073`

```python
for req in retracted_reqs:
    self._add_request_to_queue(req, is_retracted=True)
```

**处理**: 将 retracted requests 添加到 `disagg_decode_prealloc_queue.retracted_queue`

### 4.2 Resume Retracted Requests

**文件**: `python/sglang/srt/disaggregation/decode.py`

**函数**: `resume_retracted_reqs()` (第 362-402 行)

**触发时机**: 在 `process_decode_queue()` 中定期检查

**代码**: `decode.py:999-1001`

```python
def process_decode_queue(self: Scheduler):
    # 尝试恢复 retracted requests（如果有足够空间）
    resumed_reqs = self.disagg_decode_prealloc_queue.resume_retracted_reqs()
    self.waiting_queue.extend(resumed_reqs)
    
    # 如果还有 retracted requests，不分配新的 requests
    if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:
        return
```

**恢复条件**:
1. 有足够的 `req_to_token_pool` 空间
2. 有足够的 `allocatable_tokens`（考虑预留的 decode tokens）
3. 满足单个 request 的内存需求

---

## 五、触发时机总结

### 5.1 调用链

```
update_running_batch()  # scheduler.py:2039
    ↓
检查内存: check_decode_mem()  # schedule_batch.py:1699
    ↓
如果内存不足: retract_decode()  # schedule_batch.py:1719
    ↓
释放内存: release_req()  # schedule_batch.py:1795
    ↓
添加到队列: _add_request_to_queue(is_retracted=True)  # scheduler.py:2073
    ↓
等待恢复: resume_retracted_reqs()  # decode.py:362
```

### 5.2 触发条件汇总

| 条件 | 说明 | 代码位置 |
|------|------|----------|
| **内存不足** | `check_decode_mem()` 返回 `False` | `scheduler.py:2049` |
| **测试模式** | `TEST_RETRACT` 且达到间隔 | `scheduler.py:2050` |

### 5.3 内存检查公式

```
1. 计算需要的新 pages 数量:
   new_page_count = new_page_count_next_decode(selected_indices)
   
   对于每个 request:
   - 非 paged (page_size=1): 每个 request 需要 1 个 slot
   - Spec 模式: 如果 (seqlen + thresh) % page_size <= thresh，需要新 page
   - Overlap 模式: 如果 seqlen % page_size == 0，需要新 page
   - 非 Overlap 模式: 如果 (seqlen - 1) % page_size == 0，需要新 page

2. 计算需要的 tokens:
   num_tokens = new_page_count * buf_multiplier * page_size

3. 从 tree cache evict tokens:
   evict_from_tree_cache(tree_cache, num_tokens)

4. 检查可用内存:
   可用内存 = token_to_kv_pool_allocator.available_size()
   
   触发条件: num_tokens > 可用内存
```

---

## 六、相关配置参数

### 6.1 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SGLANG_RETRACT_DECODE_STEPS` | 预留的 decode tokens 数量 | - |
| `SGLANG_TEST_RETRACT` | 是否启用测试 retract | `False` |
| `SGLANG_TEST_RETRACT_INTERVAL` | 测试 retract 间隔 | - |

### 6.2 Server Args

| 参数 | 说明 |
|------|------|
| `decode_mem_cache_buf_multiplier` | 内存缓存 buffer 倍数 |
| `disaggregation_mode` | Disaggregation 模式（"decode" 时 offload KV cache） |

---

## 七、示例场景

### 7.1 场景 1: 正常内存不足

```
1. update_running_batch() 被调用
2. check_decode_mem() 检查内存
   - 当前 batch 有 10 个 requests
   - 每个 request 需要 1000 tokens
   - 总共需要 10000 tokens
   - 可用内存只有 8000 tokens
3. check_decode_mem() 返回 False
4. 触发 retract_decode()
5. 从后往前移除 requests，直到内存足够
6. 假设移除了 3 个 requests
7. 剩余的 7 个 requests 继续 decode
```

### 7.2 场景 2: 测试模式

```
1. TEST_RETRACT = True
2. TEST_RETRACT_INTERVAL = 10
3. forward_ct = 20 (20 % 10 == 0)
4. 强制触发 retract（即使内存足够）
```

### 7.3 场景 3: Resume Retracted Requests

```
1. 之前有 3 个 requests 被 retract
2. 它们被添加到 retracted_queue
3. process_decode_queue() 被调用
4. resume_retracted_reqs() 检查是否有足够空间
5. 如果有空间，恢复这些 requests
6. 将它们添加到 waiting_queue
7. 等待重新进入 running batch
```

---

## 八、关键代码位置

### 8.1 触发检查

- **文件**: `python/sglang/srt/managers/scheduler.py`
- **函数**: `update_running_batch()` (第 2039-2085 行)
- **检查**: 第 2049-2051 行

### 8.2 内存检查

- **文件**: `python/sglang/srt/managers/schedule_batch.py`
- **函数**: `check_decode_mem()` (第 1699-1717 行)

### 8.3 Retract 执行

- **文件**: `python/sglang/srt/managers/schedule_batch.py`
- **函数**: `retract_decode()` (第 1719-1793 行)
- **函数**: `release_req()` (第 1795-1808 行)

### 8.4 Resume 处理

- **文件**: `python/sglang/srt/disaggregation/decode.py`
- **函数**: `resume_retracted_reqs()` (第 362-402 行)
- **调用**: `process_decode_queue()` (第 995-1021 行)

---

## 九、总结

### 9.1 触发条件

1. **主要条件**: `check_decode_mem()` 返回 `False`
   - 需要的 tokens > 可用内存
   - 包括：输入 tokens + 输出 tokens + 预留 decode tokens

2. **测试条件**: `TEST_RETRACT` 且达到间隔

### 9.2 Retract 策略

- **非 Spec 模式**: 优先 retract 已生成 tokens 多、输入短的 requests
- **Spec 模式**: 从后往前 retract

### 9.3 恢复机制

- 定期检查 `retracted_queue`
- 如果有足够空间，恢复 retracted requests
- 恢复的 requests 重新进入 `waiting_queue`

---

**文档创建时间**: 2025-01-XX  
**基于**: SGLang 代码库分析

