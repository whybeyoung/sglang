# Spec vs SpecV2 代码分析

## 概述

SGLang 支持两种 Speculative Decoding 架构实现：

1. **Spec V1 (旧版)**: 继承 `TpModelWorker`，逐步淘汰
2. **Spec V2 (新版)**: 使用 `BaseSpecWorker` + `BaseDraftWorker`，推荐使用

## 核心区别

### 1. 架构设计

#### Spec V1
- **实现方式**: 直接继承 `TpModelWorker`
- **文件**: `python/sglang/srt/speculative/eagle_worker.py`
- **类**: `EAGLEWorker(TpModelWorker)`
- **特点**: 
  - 将 draft 和 target 逻辑混合在一个 worker 中
  - 代码耦合度高，难以维护

#### Spec V2
- **实现方式**: 使用组合模式，分离关注点
- **文件**: `python/sglang/srt/speculative/eagle_worker_v2.py`
- **类**: 
  - `EAGLEWorkerV2(BaseSpecWorker)` - 主协调器
  - `EagleDraftWorker(BaseDraftWorker)` - Draft 模型 worker
- **特点**:
  - 清晰的职责分离：`BaseSpecWorker` 协调，`BaseDraftWorker` 处理 draft
  - 更好的可扩展性和可维护性

### 2. Overlap Scheduler 支持

#### Spec V1
- **Overlap**: ❌ 不支持
- **原因**: 架构设计限制，无法与 overlap scheduler 良好集成
- **代码位置**: `python/sglang/srt/managers/schedule_batch.py:1855`
  ```python
  @property
  def is_spec_v2(self):
      return self.enable_overlap and self.spec_algorithm.is_eagle()
  ```
  - V1 时 `is_spec_v2 = False`，因此不走 overlap 路径

#### Spec V2
- **Overlap**: ✅ 支持
- **启用条件**: 
  - `SGLANG_ENABLE_SPEC_V2=True` (环境变量)
  - `enable_overlap=True`
  - `spec_algorithm.is_eagle()`
- **代码位置**: `python/sglang/srt/server_args.py:2005`
  ```python
  if (
      self.speculative_algorithm in ["EAGLE", "EAGLE3"]
      and envs.SGLANG_ENABLE_SPEC_V2.get()
  ):
      self.disable_overlap_schedule = False
  ```

### 3. 执行流程差异

#### Spec V1 流程
```
1. forward_batch_speculative_generation(batch)
   ├── Draft 生成 (在 worker 内部)
   ├── Verify (在 worker 内部)
   └── 同步执行，阻塞等待
```

#### Spec V2 流程
```
1. forward_batch_generation(model_worker_batch)
   ├── Prefill 阶段:
   │   ├── Target prefill (捕获 hidden states)
   │   └── Draft prefill (基于 target hidden states)
   └── Decode 阶段:
       ├── Draft 生成 (draft_worker.draft())
       ├── Verify (spec_worker.verify())
       └── Draft extend (draft_worker.draft_extend())
```

### 4. 关键代码位置

#### 判断是否为 Spec V2
**文件**: `python/sglang/srt/managers/schedule_batch.py:1855`
```python
@property
def is_spec_v2(self):
    # FIXME: finally deprecate is_spec_v2
    return self.enable_overlap and self.spec_algorithm.is_eagle()
```

#### Spec V2 的特殊处理
**文件**: `python/sglang/srt/managers/scheduler.py:2229`
```python
if batch.is_spec_v2:
    # FIXME(lsyin): tmp code for spec v2
    # We only keep future indices for next draft input
    batch.spec_info = batch_result.next_draft_input
    batch.spec_info.future_indices = future_indices
    batch.seq_lens = batch_result.next_draft_input.new_seq_lens
```

#### Spec V2 的 Decode 准备
**文件**: `python/sglang/srt/managers/schedule_batch.py:1863`
```python
if self.is_spec_v2:
    # TODO(spec-v2): all spec v2 should go through this path
    draft_input: EagleDraftInput = self.spec_info
    draft_input.prepare_for_decode(self)
```

### 5. 数据结构差异

#### Spec V1
- 使用 `EagleInfo` (旧的数据结构)
- 文件: `python/sglang/srt/speculative/eagle_info.py`

#### Spec V2
- 使用 `EagleDraftInput` (新的数据结构)
- 文件: `python/sglang/srt/speculative/eagle_info_v2.py`
- 支持 future indices，用于异步执行

### 6. 性能特性

#### Spec V1
- ❌ 不支持 overlap scheduler
- ❌ 无法与 prefill/decode overlap 并行
- ⚠️ 吞吐量受限

#### Spec V2
- ✅ 支持 overlap scheduler
- ✅ 可以与 prefill/decode overlap 并行
- ✅ 更好的吞吐量
- ⚠️ 当前限制：只支持 `topk=1` (见 `server_args.py:2015`)

### 7. 启用方式

#### 启用 Spec V2
```bash
# 设置环境变量
export SGLANG_ENABLE_SPEC_V2=True

# 启动服务器（会自动启用 overlap）
python3 -m sglang.launch_server \
    --speculative-algorithm EAGLE \
    --enable-overlap \
    ...
```

#### 代码中的自动启用
**文件**: `python/sglang/srt/server_args.py:1225`
```python
if not envs.SGLANG_ENABLE_SPEC_V2.get():
    envs.SGLANG_ENABLE_SPEC_V2.set(True)
```

### 8. 限制和注意事项

#### Spec V2 当前限制
1. **TopK 限制**: 只支持 `topk=1` (代码: `server_args.py:2015`)
2. **Grammar 支持**: 与 grammar 一起使用时需要关闭 overlap (代码: `scheduler.py:1168`)
   ```python
   need_grammar_sync = (
       batch
       and batch.is_spec_v2
       and batch.has_grammar
       and batch.forward_mode.is_decode()
       and len(self.result_queue) > 0
   )
   ```

#### Spec V1 状态
- ⚠️ 标记为逐步淘汰 (deprecated)
- 代码中有多处 `FIXME` 和 `TODO` 提到要移除 V1

### 9. 代码迁移建议

如果要从 Spec V1 迁移到 Spec V2：

1. **模型实现**: 
   - V1: 继承 `TpModelWorker`
   - V2: 实现 `BaseSpecWorker` + `BaseDraftWorker`

2. **数据结构**:
   - V1: 使用 `EagleInfo`
   - V2: 使用 `EagleDraftInput`

3. **执行流程**:
   - V1: 在 `forward_batch_speculative_generation` 中处理所有逻辑
   - V2: 分离为 `forward_batch_generation`, `verify`, `draft` 等方法

## 总结对比表

| 特性 | Spec V1 | Spec V2 |
|------|---------|---------|
| **架构** | 继承 `TpModelWorker` | `BaseSpecWorker` + `BaseDraftWorker` |
| **Overlap 支持** | ❌ | ✅ |
| **代码组织** | 耦合 | 解耦，职责分离 |
| **可维护性** | 低 | 高 |
| **吞吐量** | 受限 | 更好 |
| **状态** | 逐步淘汰 | 推荐使用 |
| **TopK 支持** | 支持多 topk | 仅 topk=1 |
| **Grammar 支持** | 支持 | 需关闭 overlap |

## 参考文件

- **V1 实现**: `python/sglang/srt/speculative/eagle_worker.py`
- **V2 实现**: `python/sglang/srt/speculative/eagle_worker_v2.py`
- **V1 数据结构**: `python/sglang/srt/speculative/eagle_info.py`
- **V2 数据结构**: `python/sglang/srt/speculative/eagle_info_v2.py`
- **基类定义**: `python/sglang/srt/speculative/base_spec_worker.py`
- **文档**: `SGLANG_SPEC_MODEL_STRUCTURE.md`



