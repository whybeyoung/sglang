# Spec V1 vs V2 CUDA Graph 数量对比

## 总结

**答案：不完全一样**

- ✅ **Draft Graph 数量：一样**
- ❌ **Draft Extend Graph 数量：不一样**（V2 在 CUDA 设备上不捕获）

## 详细对比

### 1. Draft Graph (`cuda_graph_runner`)

#### Spec V1
- **捕获条件**: `speculative_num_steps > 1`
- **代码位置**: `eagle_worker.py:233-245`
- **Graph Runner**: `EAGLEDraftCudaGraphRunner`
- **Graph 数量**: 根据 `capture_bs` 决定，每个 batch size 一个 graph

#### Spec V2
- **捕获条件**: `speculative_num_steps > 1`
- **代码位置**: `eagle_worker_v2.py:235-247`
- **Graph Runner**: `EAGLEDraftCudaGraphRunner`（相同）
- **Graph 数量**: 根据 `capture_bs` 决定，每个 batch size 一个 graph

**结论**: ✅ **数量相同**

### 2. Draft Extend Graph (`cuda_graph_runner_for_draft_extend`)

#### Spec V1
- **捕获条件**: `draft_extend_attn_backend` 存在 **且** 不是 NPU (`not _is_npu`)
- **代码位置**: `eagle_worker.py:248-260`
- **Graph Runner**: `EAGLEDraftExtendCudaGraphRunner`
- **Graph 数量**: 根据 `capture_bs` 决定，每个 batch size 一个 graph
- **支持设备**: ✅ CUDA ✅ NPU

```python
# V1: 支持 CUDA 和 NPU
if self.draft_extend_attn_backend and not _is_npu:
    self.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(self)
```

#### Spec V2
- **捕获条件**: `draft_extend_attn_backend` 存在 **且** 是 NPU (`_is_npu`)
- **代码位置**: `eagle_worker_v2.py:253-267`
- **Graph Runner**: `EAGLEDraftExtendCudaGraphRunner`
- **Graph 数量**: 根据 `capture_bs` 决定，每个 batch size 一个 graph（仅 NPU）
- **支持设备**: ❌ CUDA ✅ NPU
- **代码注释**: `# FIXME cuda not support draft_extend capture`

```python
# V2: 仅支持 NPU，CUDA 不支持
# FIXME cuda not support draft_extend capture
if self.draft_extend_attn_backend and _is_npu:
    self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
        self.target_worker.device
    ](self)
```

**结论**: ❌ **数量不同**
- **CUDA 设备**: V1 有 extend graph，V2 没有
- **NPU 设备**: V1 和 V2 都有 extend graph

## Graph 数量计算

### Graph 数量公式

每个 Graph Runner 捕获的 graph 数量 = `len(capture_bs)`

其中 `capture_bs` 由 `get_batch_sizes_to_capture()` 函数决定：
- 默认值：`server_args.cuda_graph_bs`
- 通常包含多个 batch sizes，例如：`[1, 2, 4, 8, 16, 32, 64, 128]`

### 总 Graph 数量对比

#### CUDA 设备

**Spec V1**:
```
总 Graph 数量 = Draft Graph 数量 + Draft Extend Graph 数量
             = len(capture_bs) + len(capture_bs)
             = 2 * len(capture_bs)
```

**Spec V2**:
```
总 Graph 数量 = Draft Graph 数量 + Draft Extend Graph 数量
             = len(capture_bs) + 0  (CUDA 不支持 extend)
             = len(capture_bs)
```

**差异**: V1 比 V2 多 `len(capture_bs)` 个 graph（extend graph）

#### NPU 设备

**Spec V1**:
```
总 Graph 数量 = Draft Graph 数量 + Draft Extend Graph 数量
             = len(capture_bs) + len(capture_bs)
             = 2 * len(capture_bs)
```

**Spec V2**:
```
总 Graph 数量 = Draft Graph 数量 + Draft Extend Graph 数量
             = len(capture_bs) + len(capture_bs)
             = 2 * len(capture_bs)
```

**差异**: ✅ 相同

## 代码位置

### Spec V1
- **文件**: `python/sglang/srt/speculative/eagle_worker.py`
- **Draft Graph**: 第 233-245 行
- **Extend Graph**: 第 248-260 行

### Spec V2
- **文件**: `python/sglang/srt/speculative/eagle_worker_v2.py`
- **Draft Graph**: 第 235-247 行
- **Extend Graph**: 第 253-267 行（仅 NPU）

## 实际示例

假设 `capture_bs = [1, 2, 4, 8, 16, 32, 64, 128]`（8 个 batch sizes）

### CUDA 设备

**Spec V1**:
- Draft Graph: 8 个
- Extend Graph: 8 个
- **总计**: 16 个 graph

**Spec V2**:
- Draft Graph: 8 个
- Extend Graph: 0 个（不支持）
- **总计**: 8 个 graph

**差异**: V1 比 V2 多 8 个 graph

### NPU 设备

**Spec V1**:
- Draft Graph: 8 个
- Extend Graph: 8 个
- **总计**: 16 个 graph

**Spec V2**:
- Draft Graph: 8 个
- Extend Graph: 8 个
- **总计**: 16 个 graph

**差异**: ✅ 相同

## 为什么 V2 不支持 CUDA 的 Draft Extend Graph？

根据代码注释 `# FIXME cuda not support draft_extend capture`，这是一个已知的限制：

1. **技术原因**: CUDA 设备上 draft extend 的 graph capture 可能存在技术问题
2. **优先级**: V2 架构优先支持 overlap scheduler，extend graph 的优化优先级较低
3. **性能影响**: 虽然不能使用 graph，但 V2 的 overlap 机制可以部分补偿性能损失

## 性能影响

### CUDA 设备

**Spec V1**:
- ✅ Draft 阶段可以使用 graph（更快）
- ✅ Draft Extend 阶段可以使用 graph（更快）
- ❌ 不支持 overlap scheduler

**Spec V2**:
- ✅ Draft 阶段可以使用 graph（更快）
- ❌ Draft Extend 阶段不能使用 graph（较慢）
- ✅ 支持 overlap scheduler（可以补偿性能损失）

**总体**: V2 的 overlap scheduler 可以部分补偿 extend graph 缺失的性能损失

### NPU 设备

**Spec V1** 和 **Spec V2** 都支持完整的 graph，性能差异主要来自架构优化（overlap scheduler）。

## 总结表

| 设备类型 | Graph 类型 | Spec V1 | Spec V2 | 差异 |
|---------|-----------|---------|---------|------|
| **CUDA** | Draft Graph | ✅ | ✅ | 相同 |
| **CUDA** | Extend Graph | ✅ | ❌ | V1 有，V2 无 |
| **CUDA** | **总数量** | **2×len(bs)** | **1×len(bs)** | **V1 多 1 倍** |
| **NPU** | Draft Graph | ✅ | ✅ | 相同 |
| **NPU** | Extend Graph | ✅ | ✅ | 相同 |
| **NPU** | **总数量** | **2×len(bs)** | **2×len(bs)** | **相同** |

## 相关代码

- **Graph 数量决定**: `python/sglang/srt/model_executor/cuda_graph_runner.py:189-210`
- **Draft Graph Runner**: `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`
- **Extend Graph Runner**: `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`



