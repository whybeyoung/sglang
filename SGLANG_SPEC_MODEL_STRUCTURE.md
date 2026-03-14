# SGLang Spec 模型需要实现的结构

## 概述

SGLang 支持两种架构来实现 Speculative Decoding：

1. **V1 架构**: 继承 `TpModelWorker`（旧版，逐步淘汰）
2. **V2 架构**: 使用 `BaseSpecWorker` + `BaseDraftWorker`（推荐）

---

## V2 架构（推荐）

### 架构设计

```
BaseSpecWorker (抽象基类)
    ├── target_worker: TpModelWorker  # Target 模型 worker
    ├── draft_worker: BaseDraftWorker # Draft 模型 worker
    ├── forward_batch_generation()    # 主入口
    ├── verify()                      # Verify 阶段
    └── clear_cache_pool()            # 清理缓存

BaseDraftWorker (抽象基类)
    ├── draft()                       # Draft 生成
    └── draft_extend()                # Draft extend (可选)
```

### 1. BaseSpecWorker 接口

**文件**: `python/sglang/srt/speculative/base_spec_worker.py`

```python
class BaseSpecWorker(ABC):
    @property
    @abstractmethod
    def target_worker(self) -> TpModelWorker:
        """Target 模型 worker"""
        pass

    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        """Draft 模型 worker"""
        pass

    @abstractmethod
    def clear_cache_pool(self):
        """清理缓存池"""
        pass
```

**必须实现的方法**:

#### 1.1 `forward_batch_generation(model_worker_batch: ModelWorkerBatch)`

**功能**: 主入口，处理整个 speculative decoding 流程

**实现模式** (参考 `EAGLEWorkerV2`):

```python
def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
    if (
        model_worker_batch.forward_mode.is_extend()
        or model_worker_batch.is_extend_in_batch
    ):
        # === Prefill 阶段 ===
        # 1. Target prefill
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch
        )

        # 2. Draft prefill (基于 target 的 hidden states)
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        batch_output.next_draft_input = (
            self.draft_worker._draft_extend_for_prefill(
                model_worker_batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
            )
        )
        return batch_output
    else:
        # === Decode 阶段 ===
        # 1. 初始化 spec_info (如果是 idle)
        if model_worker_batch.spec_info is None:
            model_worker_batch.spec_info = EagleDraftInput.create_idle_input(...)

        # 2. Draft 生成
        verify_input: EagleVerifyInput = self.draft_worker.draft(
            model_worker_batch
        )

        # 3. Verify
        model_worker_batch.spec_info = verify_input
        batch_output = self.verify(model_worker_batch)

        # 4. Draft extend (更新 draft model 状态)
        self.draft_worker._draft_extend_for_decode(
            model_worker_batch, batch_output
        )
        return batch_output
```

#### 1.2 `verify(batch: ModelWorkerBatch)`

**功能**: Verify 阶段，使用 target 模型验证 draft tokens

**实现模式**:

```python
def verify(self, batch: ModelWorkerBatch):
    # 1. 准备 verify batch
    verify_input: EagleVerifyInput = batch.spec_info
    verify_input.num_tokens_per_batch = self.speculative_num_steps + 1
    
    # 2. 准备 forward batch
    verify_forward_batch, can_run_cuda_graph = (
        verify_input.prepare_for_v2_verify(
            self.req_to_token_pool,
            batch,
            self.target_worker,
        )
    )

    # 3. Target 模型 forward (Prefill-like)
    forward_batch_output = self.target_worker.model_runner.forward(
        verify_forward_batch,
        is_verify=True,
        skip_attn_backend_init=True,
    )
    logits_output = forward_batch_output.logits_output

    # 4. 采样和接受
    predict, accept_length, accept_index = verify_input.sample(
        batch, logits_output
    )

    # 5. 构造结果
    new_seq_lens = batch.seq_lens + accept_length
    verified_id = ...  # 从 predict 和 accept_index 提取

    # 6. 构造 next_draft_input
    next_draft_input = EagleDraftInput(...)

    return GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=predict,
        can_run_cuda_graph=can_run_cuda_graph,
        next_draft_input=next_draft_input,
        accept_lens=accept_length,
    )
```

---

### 2. BaseDraftWorker 接口

**文件**: `python/sglang/srt/speculative/base_spec_worker.py`

```python
class BaseDraftWorker(ABC):
    @abstractmethod
    def draft(model_worker_batch: ModelWorkerBatch):
        """生成 draft tokens
        
        Returns:
            verify_input: EagleVerifyInput 或类似的 verify input
        """
        pass

    @abstractmethod
    def draft_extend():
        """Draft extend (可选)"""
        pass
```

**必须实现的方法**:

#### 2.1 `draft(model_worker_batch: ModelWorkerBatch)`

**功能**: 生成 draft tokens

**实现模式** (参考 `EagleDraftWorker`):

```python
def draft(self, model_worker_batch: ModelWorkerBatch):
    # 1. 准备 draft input
    draft_input: EagleDraftInput = model_worker_batch.spec_info
    
    # 2. 准备 forward batch
    forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
        self.req_to_token_pool,
        model_worker_batch,
        self.cuda_graph_runner,
        self.draft_runner_list[0],  # 或 self.draft_runner
    )

    # 3. Draft 模型 forward (多步)
    # 对于 MTP: 多步 forward
    # 对于 EAGLE: 单步 forward + tree expansion
    
    # 4. 组织 draft tokens (tree structure)
    draft_tokens = ...  # 组织成 tree
    
    # 5. 构造 verify input
    return EagleVerifyInput(
        draft_token=draft_tokens,
        custom_mask=tree_mask,
        positions=positions,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        spec_steps=self.speculative_num_steps,
        topk=self.topk,
        draft_token_num=self.speculative_num_draft_tokens,
        ...
    )
```

#### 2.2 `_draft_extend_for_prefill()` (可选)

**功能**: Prefill 阶段的 draft extend

**实现模式**:

```python
def _draft_extend_for_prefill(
    self,
    model_worker_batch: ModelWorkerBatch,
    hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
) -> EagleDraftInput:
    """基于 target prefill 的 hidden states 初始化 draft"""
    # 1. 使用 target 的 hidden states
    # 2. Draft 模型 forward (prefill-like)
    # 3. 返回 next_draft_input
    ...
```

#### 2.3 `_draft_extend_for_decode()` (可选)

**功能**: Decode 阶段的 draft extend，更新 draft model 状态

**实现模式**:

```python
def _draft_extend_for_decode(
    self,
    model_worker_batch: ModelWorkerBatch,
    batch_output: GenerationBatchResult,
):
    """基于接受的 tokens 更新 draft model 状态"""
    # 1. 获取接受的 tokens
    # 2. Draft 模型 forward (extend)
    # 3. 更新 draft model 的 KV cache
    ...
```

---

## V1 架构（旧版，逐步淘汰）

### 架构设计

```
TpModelWorker (继承)
    ├── target_worker: TpModelWorker  # Target 模型 worker
    ├── forward_batch_generation()    # 主入口
    ├── draft()                       # Draft 生成
    ├── verify()                      # Verify 阶段
    └── forward_draft_extend_after_decode()  # Draft extend
```

**示例**: `EAGLEWorker(TpModelWorker)`, `MTPWorker(TpModelWorker)`

**不推荐使用**，因为：
- 代码耦合度高
- 难以维护
- V2 架构更清晰

---

## 关键数据结构

### 1. EagleDraftInput

**功能**: Draft 阶段的输入/输出

**关键字段**:
```python
@dataclass
class EagleDraftInput(SpecInput):
    topk_p: torch.Tensor              # Top-k probabilities
    topk_index: torch.Tensor          # Top-k token indices
    hidden_states: torch.Tensor       # Hidden states
    positions: torch.Tensor           # Token positions
    capture_hidden_mode: CaptureHiddenMode  # FULL or LAST
    ...
```

### 2. EagleVerifyInput

**功能**: Verify 阶段的输入

**关键字段**:
```python
@dataclass
class EagleVerifyInput(SpecInput):
    draft_token: torch.Tensor         # Draft tokens (tree structure)
    custom_mask: torch.Tensor         # Attention mask
    positions: torch.Tensor           # Positions
    retrive_index: torch.Tensor        # Tree retrieval index
    retrive_next_token: torch.Tensor   # Next token in tree
    retrive_next_sibling: torch.Tensor # Next sibling in tree
    spec_steps: int                    # Number of speculative steps
    topk: int                          # Top-k value
    draft_token_num: int               # Number of draft tokens
    ...
```

### 3. GenerationBatchResult

**功能**: 生成结果

**关键字段**:
```python
@dataclass
class GenerationBatchResult:
    logits_output: LogitsProcessorOutput
    next_token_ids: torch.Tensor      # Generated tokens
    num_accepted_tokens: int          # Number of accepted tokens
    accept_length_per_req_cpu: List[int]  # Accept length per request
    can_run_cuda_graph: bool
    next_draft_input: Optional[EagleDraftInput]  # For next iteration
    accept_lens: Optional[torch.Tensor]  # Accept lengths
    ...
```

---

## 实现示例

### 完整实现示例 (EAGLEWorkerV2)

```python
class MySpecWorker(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.target_worker = target_worker
        self.draft_worker = MyDraftWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        self.device = server_args.device
        self.speculative_num_steps = server_args.speculative_num_steps
        self.topk = server_args.speculative_eagle_topk

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> BaseDraftWorker:
        return self._draft_worker

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        # 实现主流程
        ...

    def verify(self, batch: ModelWorkerBatch):
        # 实现 verify
        ...

    def clear_cache_pool(self):
        # 清理缓存
        pass


class MyDraftWorker(BaseDraftWorker):
    def __init__(self, ...):
        # 初始化 draft model worker
        self.draft_worker = TpModelWorker(
            ...,
            is_draft_worker=True,
        )

    def draft(self, model_worker_batch: ModelWorkerBatch):
        # 实现 draft 生成
        ...

    def draft_extend(self):
        # 实现 draft extend (可选)
        ...
```

---

## 关键要点总结

### 必须实现的接口

1. **BaseSpecWorker**:
   - ✅ `target_worker` (property)
   - ✅ `draft_worker` (property)
   - ✅ `forward_batch_generation()`
   - ✅ `verify()`
   - ✅ `clear_cache_pool()`

2. **BaseDraftWorker**:
   - ✅ `draft()`
   - ✅ `draft_extend()` (可选)

### 关键流程

1. **Prefill 阶段**:
   - Target prefill → Draft prefill (基于 target hidden states)

2. **Decode 阶段**:
   - Draft 生成 → Verify → Draft extend

### 关键数据结构

- `EagleDraftInput`: Draft 输入/输出
- `EagleVerifyInput`: Verify 输入
- `GenerationBatchResult`: 生成结果

### 注意事项

1. **Hidden States 捕获**:
   - Prefill: `CaptureHiddenMode.FULL`
   - Decode: `CaptureHiddenMode.LAST`

2. **Tree Structure**:
   - Draft tokens 需要组织成 tree structure
   - 需要 `retrive_index`, `retrive_next_token`, `retrive_next_sibling`

3. **Context Managers**:
   - 使用 `speculative_moe_backend_context()` 和 `speculative_moe_a2a_backend_context()`

4. **Forward Mode**:
   - Prefill: `ForwardMode.EXTEND`
   - Decode: `ForwardMode.DECODE`
   - Verify: `ForwardMode.TARGET_VERIFY`

---

## 参考实现

- **EAGLE V2**: `python/sglang/srt/speculative/eagle_worker_v2.py`
- **MTP V2**: `python/sglang/srt/speculative/mtp_worker_v2.py`
- **EAGLE V1**: `python/sglang/srt/speculative/eagle_worker.py` (不推荐)



