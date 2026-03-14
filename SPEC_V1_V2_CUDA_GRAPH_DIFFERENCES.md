# Spec V1 vs V2 CUDA Graph 模式区别分析

## 概述

CUDA Graph 在 Spec V1 和 V2 中的实现方式有显著差异，主要体现在 **overlap 支持**、**执行时机**和**stream 管理**上。

## 核心区别

### 1. Overlap 支持

#### Spec V1
- **Overlap**: ❌ 不支持
- **执行方式**: 同步执行，阻塞等待
- **代码位置**: `eagle_worker.py:536-554`
  ```python
  can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch)
  if can_cuda_graph:
      parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(forward_batch)
  else:
      # 同步执行，阻塞等待
      parent_list, top_scores_index, draft_tokens = self.draft_forward(forward_batch)
  ```

#### Spec V2
- **Overlap**: ✅ 支持
- **执行方式**: 异步准备 + 同步执行
- **代码位置**: `eagle_worker_v2.py:271-295`
  ```python
  forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(...)
  if can_cuda_graph:
      parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(forward_batch)
  else:
      # 支持 overlap，可以在 plan stream 中准备
      parent_list, top_scores_index, draft_tokens = self.draft_forward(forward_batch)
  ```

### 2. Graph Capture 时机

#### Spec V1
- **Capture 时机**: Worker 初始化时一次性捕获
- **代码位置**: `eagle_worker.py:220-260`
  ```python
  def init_cuda_graphs(self):
      # Capture draft graph
      if self.speculative_num_steps > 1:
          self.cuda_graph_runner = EAGLEDraftCudaGraphRunner(self)
      
      # Capture extend graph (仅 CUDA，NPU 不支持)
      if self.draft_extend_attn_backend and not _is_npu:
          self.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(self)
  ```

#### Spec V2
- **Capture 时机**: 同样在初始化时捕获，但支持多 stream 准备
- **代码位置**: `eagle_worker_v2.py:222-267`
  ```python
  def init_cuda_graphs(self):
      # 与 V1 类似的 capture 逻辑
      # 但支持在 plan stream 中准备 replay
  ```

### 3. Verify 阶段的 Graph 使用

#### Spec V1
- **执行方式**: 同步执行，直接调用 target worker 的 graph
- **代码位置**: `eagle_worker.py:699-705`
  ```python
  batch_result = self.target_worker.forward_batch_generation(
      model_worker_batch, is_verify=True
  )
  logits_output, can_run_cuda_graph = (
      batch_result.logits_output,
      batch_result.can_run_cuda_graph,  # 从 target worker 获取
  )
  ```
- **特点**: 
  - 阻塞等待 verify 完成
  - `can_run_cuda_graph` 标志由 target worker 决定

#### Spec V2
- **执行方式**: 在 plan stream 中准备，主 stream 中执行
- **代码位置**: `eagle_worker_v2.py:663-704`
  ```python
  # 在 plan stream 中准备 verify forward batch
  with self.plan_stream_ctx:
      verify_forward_batch, can_run_cuda_graph = (
          verify_input.prepare_for_v2_verify(
              self.req_to_token_pool,
              batch,
              self.target_worker,
          )
      )
  
  # 等待 plan stream 完成
  if self.plan_stream:
      torch.get_device_module(self.device).current_stream().wait_stream(
          self.plan_stream
      )
      # 更新 verify buffers（因为 draft 的输出可能依赖）
      self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(...)
  
  # 在主 compute stream 中执行
  forward_batch_output = self.target_worker.forward_batch_generation(
      model_worker_batch=None,
      forward_batch=verify_forward_batch,
      is_verify=True,
      skip_attn_backend_init=True,
  )
  ```
- **特点**:
  - ✅ 支持 overlap：plan stream 准备，compute stream 执行
  - ✅ 可以并行准备下一个 batch 的 graph
  - ✅ 更好的 GPU 利用率

### 4. Draft Extend 阶段的 Graph

#### Spec V1
- **执行方式**: 同步执行
- **代码位置**: `eagle_worker.py:932-954`
  ```python
  can_cuda_graph = (
      self.cuda_graph_runner_for_draft_extend
      and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
  )
  if can_cuda_graph:
      logits_output = self.cuda_graph_runner_for_draft_extend.replay(forward_batch)
  else:
      # 同步执行
      logits_output = self.draft_model_runner.forward(forward_batch, ...)
  ```
- **限制**: 
  - CUDA 设备支持 extend graph capture
  - NPU 设备不支持（代码: `eagle_worker.py:248`）

#### Spec V2
- **执行方式**: 在 plan stream 中准备，主 stream 中执行
- **代码位置**: `eagle_worker_v2.py:494-521`
  ```python
  # 在 plan stream 中准备
  with self.plan_stream_ctx:
      forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
          batch,
          batch_result.next_token_ids,
          self.speculative_num_draft_tokens,
          self.draft_runner,
          self.cuda_graph_runner_for_draft_extend,
      )
  
  # 等待 plan stream
  if self.plan_stream:
      torch.get_device_module(self.device).current_stream().wait_stream(
          self.plan_stream
      )
  
  # 在主 compute stream 中执行
  can_cuda_graph = (
      self.cuda_graph_runner_for_draft_extend
      and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
  )
  if can_cuda_graph:
      draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(forward_batch)
  ```
- **特点**:
  - ✅ 支持 overlap
  - ✅ 可以在 verify 执行时并行准备 draft extend

### 5. Plan Stream 机制（V2 独有）

#### Spec V2 引入 Plan Stream
- **目的**: 在单独的 stream 中准备 graph replay，与主计算 stream 并行
- **代码位置**: `eagle_worker_v2.py:63-72`
  ```python
  def _get_plan_stream(device: str):
      if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
          plan_stream = torch.get_device_module(device).Stream()
          plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
          return plan_stream, plan_stream_ctx
      else:
          return None, contextlib.nullcontext()
  ```

#### 工作流程
```
Plan Stream (准备阶段):
  ├── prepare_for_v2_draft()      # 准备 draft graph replay
  ├── prepare_for_v2_verify()    # 准备 verify graph replay
  └── prepare_for_extend_to_fill_draft_kvcache()  # 准备 extend graph replay

Compute Stream (执行阶段):
  ├── graph.replay()              # 执行 graph
  └── wait_stream(plan_stream)    # 等待 plan stream 完成
```

### 6. Graph Replay 准备差异

#### Spec V1
- **准备时机**: 执行时同步准备
- **代码位置**: `eagle_worker.py:530-538`
  ```python
  forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
  can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch)
  if can_cuda_graph:
      # 直接 replay，没有提前准备
      parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(forward_batch)
  ```

#### Spec V2
- **准备时机**: 在 plan stream 中提前准备
- **代码位置**: `eagle_info_v2.py:171-178`
  ```python
  def prepare_for_v2_draft(...):
      forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
      can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
      return forward_batch, can_cuda_graph  # 提前准备并返回
  ```
- **优势**: 
  - 可以在 compute stream 执行当前 batch 时，并行准备下一个 batch 的 graph
  - 减少 GPU 空闲时间

### 7. Verify Buffers 更新机制（V2 独有）

#### Spec V2 的特殊处理
- **问题**: Draft 的输出（tree_mask, position）依赖 draft 的执行结果
- **解决方案**: 在 plan stream 和 compute stream 同步后，更新 verify buffers
- **代码位置**: `eagle_worker_v2.py:673-688`
  ```python
  if self.plan_stream:
      torch.get_device_module(self.device).current_stream().wait_stream(
          self.plan_stream
      )
      # 因为 custom_mask 和 position 依赖 draft 的输出，
      # 所以需要在 plan stream 完成后重新计算
      self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
          verify_input,
          self.target_worker.model_runner.graph_runner.bs if can_run_cuda_graph else None,
      )
  ```

### 8. Graph 使用条件检查

#### Spec V1
- **检查方式**: 简单检查 `can_run()`
- **代码位置**: `eagle_worker.py:536`
  ```python
  can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch)
  ```

#### Spec V2
- **检查方式**: 在准备阶段检查，提前知道是否可以使用 graph
- **代码位置**: `eagle_info_v2.py:177`
  ```python
  can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
  return forward_batch, can_cuda_graph  # 返回准备结果和可用性
  ```

## 性能影响

### Spec V1
- ⚠️ **GPU 利用率**: 较低，因为同步执行
- ⚠️ **延迟**: 较高，无法 overlap
- ⚠️ **吞吐量**: 受限

### Spec V2
- ✅ **GPU 利用率**: 更高，支持 overlap
- ✅ **延迟**: 更低，可以并行准备
- ✅ **吞吐量**: 更好，充分利用 GPU

## 关键代码对比

### Draft 阶段

**V1** (`eagle_worker.py:530-554`):
```python
forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch)
if can_cuda_graph:
    # 同步 replay
    parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(forward_batch)
```

**V2** (`eagle_worker_v2.py:271-295`):
```python
# 在 prepare_for_v2_draft 中提前准备
forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
    self.req_to_token_pool,
    model_worker_batch,
    self.cuda_graph_runner,
    self.draft_runner,
    self.topk,
    self.speculative_num_steps,
)
if can_cuda_graph:
    # replay（可能已经在 plan stream 中准备）
    parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(forward_batch)
```

### Verify 阶段

**V1** (`eagle_worker.py:699-705`):
```python
# 同步执行，阻塞等待
batch_result = self.target_worker.forward_batch_generation(
    model_worker_batch, is_verify=True
)
logits_output, can_run_cuda_graph = (
    batch_result.logits_output,
    batch_result.can_run_cuda_graph,
)
```

**V2** (`eagle_worker_v2.py:663-704`):
```python
# 在 plan stream 中准备
with self.plan_stream_ctx:
    verify_forward_batch, can_run_cuda_graph = (
        verify_input.prepare_for_v2_verify(...)
    )

# 等待 plan stream，更新 buffers
if self.plan_stream:
    torch.get_device_module(self.device).current_stream().wait_stream(
        self.plan_stream
    )
    self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(...)

# 在主 stream 中执行
forward_batch_output = self.target_worker.forward_batch_generation(
    model_worker_batch=None,
    forward_batch=verify_forward_batch,
    is_verify=True,
    skip_attn_backend_init=True,
)
```

## 总结对比表

| 特性 | Spec V1 | Spec V2 |
|------|---------|---------|
| **Overlap 支持** | ❌ | ✅ |
| **Plan Stream** | ❌ | ✅ |
| **Graph 准备时机** | 执行时同步 | Plan stream 中提前准备 |
| **Verify Graph** | 同步执行 | Plan stream 准备 + Compute stream 执行 |
| **Draft Extend Graph** | 同步执行 | Plan stream 准备 + Compute stream 执行 |
| **GPU 利用率** | 较低 | 较高 |
| **吞吐量** | 受限 | 更好 |
| **延迟** | 较高 | 较低 |
| **代码复杂度** | 较低 | 较高（但更灵活） |

## 关键文件

- **V1 Graph Runner**: `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`
- **V2 Graph 准备**: `python/sglang/srt/speculative/eagle_info_v2.py`
- **Plan Stream**: `python/sglang/srt/speculative/eagle_worker_v2.py:63-72`
- **环境变量**: `python/sglang/srt/environ.py:362` (`SGLANG_ENABLE_OVERLAP_PLAN_STREAM`)

## 启用 Plan Stream

```bash
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=True
export SGLANG_ENABLE_SPEC_V2=True
```

## 注意事项

1. **V2 的限制**:
   - Plan stream 需要额外的 GPU 内存
   - 代码复杂度更高
   - 需要正确同步 plan stream 和 compute stream

2. **V1 的限制**:
   - 无法利用 GPU 并行性
   - 吞吐量受限
   - 逐步淘汰中

3. **最佳实践**:
   - 推荐使用 Spec V2 + Plan Stream
   - 确保正确设置环境变量
   - 监控 GPU 利用率



