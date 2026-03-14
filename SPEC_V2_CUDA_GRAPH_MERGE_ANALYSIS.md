# Spec V2 CUDA Graph 合并分析

## 答案：**没有合并 graph**

V2 CUDA **并没有合并 graph**，而是**根本不捕获 extend graph**。

## 详细分析

### 1. 标准 EAGLE V2 (eagle_worker_v2.py)

#### CUDA 设备
- ✅ **Draft Graph**: 有（`EAGLEDraftCudaGraphRunner`）
- ❌ **Extend Graph**: **没有**（不捕获）

**代码证据** (`eagle_worker_v2.py:253-255`):
```python
# Capture extend
# FIXME cuda not support draft_extend capture
if self.draft_extend_attn_backend and _is_npu:  # 注意：只有 NPU 才捕获
    self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[...](self)
```

**结论**: 
- V2 CUDA **没有 extend graph**，所以不存在"合并"的概念
- V1 CUDA 有独立的 draft graph 和 extend graph（两个独立的 graph runner）
- V2 CUDA 只有 draft graph，extend 阶段不使用 graph

### 2. Multi-Layer EAGLE V2 (multi_layer_eagle_worker_v2.py)

#### CUDA 设备
- ✅ **Draft Extend Graph**: 有（`MultiLayerEagleMultiStepDraftExtendCudaGraphRunner`）
- ⚠️ **但这不是"合并"**，而是**管理多个 step 的 graph**

**代码证据** (`multi_layer_eagle_draft_extend_cuda_graph_runner.py:545-661`):
```python
class MultiLayerEagleMultiStepDraftExtendCudaGraphRunner:
    def __init__(self, eagle_worker: MultiLayerEagleDraftWorker):
        self.runners = []  # 多个独立的 runner
        
    def _init_and_capture(self):
        # 1. Capture loop - 为每个 step 创建独立的 runner
        for step in range(self.speculative_num_steps):
            runner = MultiLayerEagleDraftExtendCudaGraphRunner(
                self.eagle_worker, step
            )
            self.runners.append(runner)  # 每个 step 一个独立的 graph
        
        # 2. Allocate buffers - 共享 buffer，但 graph 是独立的
        self.cuda_graph_buffers = {...}  # 共享的 buffer
        
        # 3. Capture each step's graph separately
        for step in range(self.speculative_num_steps - 1, -1, -1):
            self.runners[step].init_buffers_and_capture(...)
```

**特点**:
- ✅ 每个 step 有**独立的 graph**（`self.runners[step]`）
- ✅ 多个 step 的 graph **共享 buffer**（`self.cuda_graph_buffers`）
- ❌ **不是合并成一个 graph**，而是**管理多个独立的 graph**

### 3. V1 vs V2 对比

#### V1 CUDA
```
Draft Graph Runner (独立)
  └── 多个 batch size 的 graph

Extend Graph Runner (独立)
  └── 多个 batch size 的 graph

总计: 2 个 Graph Runner，每个有多个 batch size 的 graph
```

#### V2 CUDA (标准 EAGLE)
```
Draft Graph Runner (独立)
  └── 多个 batch size 的 graph

Extend Graph Runner: ❌ 不存在

总计: 1 个 Graph Runner，有多个 batch size 的 graph
```

#### V2 CUDA (Multi-Layer EAGLE)
```
Draft Graph Runner: ❌ 不存在（Multi-Layer 不使用 draft graph）

Multi-Step Extend Graph Runner (管理多个 step)
  ├── Runner[0] (step 0 的 graph)
  ├── Runner[1] (step 1 的 graph)
  ├── ...
  └── Runner[N-1] (step N-1 的 graph)
  
总计: 1 个管理器，管理多个 step 的独立 graph
```

## 为什么 V2 CUDA 不捕获 Extend Graph？

### 技术原因
1. **代码注释**: `# FIXME cuda not support draft_extend capture`
2. **可能的技术限制**: CUDA 设备上 draft extend 的 graph capture 可能存在技术问题
3. **优先级**: V2 架构优先支持 overlap scheduler，extend graph 的优化优先级较低

### 性能补偿
虽然 V2 CUDA 没有 extend graph，但通过以下方式补偿：
- ✅ **Overlap Scheduler**: 可以并行执行，提高 GPU 利用率
- ✅ **Plan Stream**: 提前准备，减少延迟
- ✅ **架构优化**: V2 的整体架构更高效

## 总结

| 版本 | Draft Graph | Extend Graph | Graph 合并 |
|------|------------|--------------|------------|
| **V1 CUDA** | ✅ 独立 | ✅ 独立 | ❌ 不合并 |
| **V2 CUDA (标准)** | ✅ 独立 | ❌ **不存在** | ❌ **不适用** |
| **V2 CUDA (Multi-Layer)** | ❌ 不使用 | ✅ **多个 step 独立** | ❌ **不合并** |

**关键点**:
1. V2 CUDA **没有合并 graph**
2. V2 CUDA **根本不捕获 extend graph**（标准 EAGLE）
3. Multi-Layer EAGLE V2 有多个 step 的 graph，但它们是**独立的**，只是**共享 buffer**

## 相关代码位置

- **V2 标准 EAGLE**: `python/sglang/srt/speculative/eagle_worker_v2.py:222-267`
- **V2 Multi-Layer EAGLE**: `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py:189-199`
- **Multi-Step Graph Runner**: `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py:545-661`



