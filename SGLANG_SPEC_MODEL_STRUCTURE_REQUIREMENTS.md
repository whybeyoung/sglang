# SGLang Spec 模型（Draft Model）结构要求

## 概述

Spec 模型（Draft Model）是用于 Speculative Decoding 的小型模型，需要特定的结构来与 Target 模型协同工作。

---

## 一、模型架构类型

SGLang 支持两种主要的 Spec 模型架构：

### 1. EAGLE/EAGLE2 架构

**特点**:
- 可以有多个 Transformer 层（`num_hidden_layers`）
- 第一层跳过 `input_layernorm`
- 使用 `fc` 层融合 target 的 hidden states

**示例**: `LlamaForCausalLMEagle`

### 2. EAGLE3 架构

**特点**:
- **必须只有 1 层** (`num_hidden_layers == 1`)
- 特殊的 decoder layer 结构
- 可能有独立的 `lm_head`（不共享 target）

**示例**: `LlamaForCausalLMEagle3`

---

## 二、必须实现的接口方法

### 2.1 `get_embed_and_head()`

**功能**: 获取 embedding 和 lm_head 权重

**签名**:
```python
def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (embedding_weight, lm_head_weight)"""
    return self.model.embed_tokens.weight, self.lm_head.weight
```

**用途**: Worker 初始化时获取 target 的 embedding 和 lm_head，用于共享

### 2.2 `set_embed_and_head(embed, head)`

**功能**: 设置 embedding 和 lm_head（共享 target 的）

**签名**:
```python
def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
    """设置 embedding 和 lm_head 为 target 的权重"""
    del self.model.embed_tokens.weight
    del self.lm_head.weight
    self.model.embed_tokens.weight = embed
    self.lm_head.weight = head
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

**用途**: 
- EAGLE/EAGLE2: 共享 target 的 embedding 和 lm_head
- EAGLE3: 如果 `load_lm_head_from_target=True`，也共享

### 2.3 `set_embed(embed)` (EAGLE3 专用)

**功能**: 只设置 embedding（不共享 lm_head）

**签名**:
```python
def set_embed(self, embed: torch.Tensor) -> None:
    """只设置 embedding（用于 EAGLE3）"""
    # 注意：如果 draft hidden_size != target hidden_size，不能共享
    if (
        hasattr(self.config, "target_hidden_size")
        and self.config.target_hidden_size != self.config.hidden_size
    ):
        return
    del self.model.embed_tokens.weight
    self.model.embed_tokens.weight = embed
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

**用途**: EAGLE3 通常只共享 embedding，不共享 lm_head

### 2.4 `get_embed()` (可选)

**功能**: 获取 embedding 权重

**签名**:
```python
def get_embed(self) -> torch.Tensor:
    """返回 embedding weight"""
    return self.model.embed_tokens.weight
```

---

## 三、模型结构要求

### 3.1 EAGLE/EAGLE2 模型结构

#### 3.1.1 Model 结构

```python
class LlamaModel(nn.Module):
    def __init__(self, config, ...):
        # 1. Embedding
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        
        # 2. Transformer Layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i, ...)
            for i in range(config.num_hidden_layers)
        ])
        
        # 3. FC 层：融合 target hidden states
        # 输入: (draft_hidden, target_hidden) -> 输出: draft_hidden
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
```

#### 3.1.2 Forward 方法

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> torch.Tensor:
    # 1. 获取 embedding
    if input_embeds is None:
        hidden_states = self.embed_tokens(input_ids)
    else:
        hidden_states = input_embeds
    
    # 2. 融合 target 的 hidden states
    # forward_batch.spec_info.hidden_states 是 target 模型的 hidden states
    hidden_states = self.fc(
        torch.cat((hidden_states, forward_batch.spec_info.hidden_states), dim=-1)
    )
    
    # 3. 通过 Transformer layers
    residual = None
    for layer in self.layers:
        hidden_states, residual = layer(
            positions,
            hidden_states,
            forward_batch,
            residual,
        )
    
    return hidden_states + residual
```

#### 3.1.3 Decoder Layer 特殊处理

```python
class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_id, ...):
        super().__init__(config, layer_id, ...)
        
        # 第一层跳过 input_layernorm
        if layer_id == 0:
            del self.input_layernorm
            setattr(self, "input_layernorm", lambda x: x)
```

#### 3.1.4 LM Head

```python
class LlamaForCausalLMEagle(LlamaForCausalLM):
    def __init__(self, config, ...):
        self.model = LlamaModel(config, ...)
        
        # LM Head（可能使用 hot_vocab_size）
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                getattr(config, "hot_vocab_size", config.vocab_size),
                config.hidden_size,
                ...
            )
```

---

### 3.2 EAGLE3 模型结构

#### 3.2.1 Model 结构

```python
class LlamaModel(nn.Module):
    def __init__(self, config, ...):
        # 1. Embedding
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        
        # 2. Target hidden size（可能不同）
        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size
        
        # 3. FC 层：融合 target hidden states
        # 输入: (target_hidden * 3) -> 输出: draft_hidden
        # 3 = embeds + hidden_states + residual
        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )
        
        # 4. 只有一个 Decoder Layer
        self.midlayer = LlamaDecoderLayer(config, 0, ...)
        
        # 5. Norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

#### 3.2.2 Forward 方法

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    # 1. 获取 embedding
    if input_embeds is None:
        embeds = self.embed_tokens(input_ids)
    else:
        embeds = input_embeds
    
    # 2. 获取 target 的 hidden states
    hidden_states = forward_batch.spec_info.hidden_states
    
    # 3. 如果 hidden_size 不同，需要投影
    if hidden_states.shape[-1] != embeds.shape[-1]:
        hidden_states = self.fc(hidden_states)
    
    # 4. 通过 Decoder Layer
    # 注意：输入是 (embeds, hidden_states)，不是单一的 hidden_states
    residual = None
    hidden_states, residual = self.midlayer(
        positions,
        embeds,          # 输入 1: draft embedding
        hidden_states,   # 输入 2: target hidden states
        forward_batch,
        residual,
    )
    
    # 5. Norm 并返回
    hidden_states_to_logits, hidden_states_to_aux = self.norm(
        hidden_states, residual
    )
    
    # 返回: (用于 logits 的 hidden states, [用于 aux 的 hidden states])
    return hidden_states_to_logits, [hidden_states_to_aux]
```

#### 3.2.3 Decoder Layer 特殊结构

```python
class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_id, ...):
        super().__init__(config, layer_id, ...)
        
        # 使用 QKVParallelLinear（合并 qkv）
        self.self_attn.qkv_proj = QKVParallelLinear(...)
        
        # 添加 hidden_norm
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,        # Draft embedding
        hidden_states: torch.Tensor, # Target hidden states
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        
        # 1. Norm
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        
        # 2. 拼接 embeds 和 hidden_states
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        
        # 3. Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        
        # 4. Post-attention norm
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        
        # 5. MLP
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual
```

#### 3.2.4 LM Head 和配置

```python
class LlamaForCausalLMEagle3(LlamaForCausalLM):
    def __init__(self, config, ...):
        # 检查：必须只有 1 层
        if config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")
        
        self.model = LlamaModel(config, ...)
        
        # LM Head 配置
        self.load_lm_head_from_target = False
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.draft_vocab_size,  # 可能有独立的 vocab size
                config.hidden_size,
                ...
            )
        
        # Logits processor 使用 draft_vocab_size
        config_ = copy.deepcopy(config)
        config_.vocab_size = config_.draft_vocab_size
        self.logits_processor = LogitsProcessor(config_)
        
        # EAGLE3 特殊属性
        self.capture_aux_hidden_states = True
        self.hot_token_id = None  # 从权重加载时设置
```

#### 3.2.5 权重加载

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
    """加载权重，处理特殊的参数映射"""
    params_dict = dict(self.named_parameters())
    
    # 参数映射：处理 stacked parameters
    stacked_params_mapping = [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    
    for name, loaded_weight in weights:
        # 处理 d2t (draft to target token mapping)
        if "d2t" in name:
            self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
            continue
        
        # 跳过 t2d (target to draft)
        if "t2d" in name:
            continue
        
        # 处理 stacked parameters
        # ...
```

---

## 四、关键配置参数

### 4.1 通用配置

| 参数 | 说明 | EAGLE/EAGLE2 | EAGLE3 |
|------|------|--------------|--------|
| `num_hidden_layers` | Transformer 层数 | 任意 | **必须为 1** |
| `hidden_size` | Hidden size | 通常与 target 相同 | 可以不同 |
| `vocab_size` | Vocab size | 通常与 target 相同 | 可能有 `draft_vocab_size` |
| `hot_vocab_size` | Hot token vocab size | 可选 | 可选 |

### 4.2 EAGLE3 特殊配置

| 参数 | 说明 |
|------|------|
| `target_hidden_size` | Target 模型的 hidden size（如果不同） |
| `draft_vocab_size` | Draft 模型的 vocab size（如果不同） |
| `tie_word_embeddings` | 是否 tie embedding 和 lm_head |
| `load_lm_head_from_target` | 是否从 target 加载 lm_head |

### 4.3 Hot Token 配置

**用途**: 只预测高频 token，减少计算量

**配置方式**:
1. **通过配置文件**: `--speculative-token-map <path>`
2. **通过模型权重**: EAGLE3 模型可能包含 `d2t` 权重

**效果**:
- `hot_vocab_size` < `vocab_size`
- LM Head 只输出 `hot_vocab_size` 个 logits
- 需要 token mapping (`hot_token_id`) 映射回完整 vocab

---

## 五、权重共享机制

### 5.1 Embedding 共享

**所有架构**: 通常共享 target 的 embedding

**实现**:
```python
embed, head = target_model.get_embed_and_head()
draft_model.set_embed_and_head(embed, head)  # EAGLE/EAGLE2
# 或
draft_model.set_embed(embed)  # EAGLE3（如果 load_lm_head_from_target=False）
```

### 5.2 LM Head 共享

**EAGLE/EAGLE2**: 
- 通常共享 target 的 lm_head
- 如果使用 hot token，只共享 hot tokens 对应的权重

**EAGLE3**:
- 通常**不共享**，有自己的 lm_head
- 如果 `load_lm_head_from_target=True`，则共享

**实现**:
```python
if self.speculative_algorithm.is_eagle3():
    if hasattr(draft_model, "load_lm_head_from_target") and draft_model.load_lm_head_from_target:
        draft_model.set_embed_and_head(embed, head)
    else:
        draft_model.set_embed(embed)  # 只共享 embedding
else:
    # EAGLE/EAGLE2: 共享 embedding 和 lm_head
    if hot_token_id is not None:
        head = head[hot_token_id]  # 只取 hot tokens
    draft_model.set_embed_and_head(embed, head)
```

---

## 六、Forward 流程的特殊性

### 6.1 输入要求

**关键**: Draft model 的 forward 需要接收 **target 的 hidden states**

```python
# forward_batch.spec_info.hidden_states 是 target 模型的 hidden states
hidden_states = forward_batch.spec_info.hidden_states
```

**来源**:
- Prefill: Target prefill 时捕获的 hidden states
- Decode: Target decode 时捕获的 hidden states（通常是最后一层的）

### 6.2 EAGLE/EAGLE2 Forward

```python
# 1. Draft embedding
draft_hidden = embed_tokens(input_ids)

# 2. 融合 target hidden states
hidden_states = fc(torch.cat([draft_hidden, target_hidden], dim=-1))

# 3. 通过 Transformer layers
for layer in layers:
    hidden_states = layer(hidden_states, ...)

# 4. 输出用于 logits
return hidden_states
```

### 6.3 EAGLE3 Forward

```python
# 1. Draft embedding
embeds = embed_tokens(input_ids)

# 2. 获取 target hidden states（可能来自多个层）
target_hidden = forward_batch.spec_info.hidden_states

# 3. 如果 hidden_size 不同，投影
if target_hidden.shape[-1] != embeds.shape[-1]:
    target_hidden = fc(target_hidden)

# 4. 通过 Decoder Layer（输入是 embeds + target_hidden）
hidden_states, residual = midlayer(embeds, target_hidden, ...)

# 5. Norm
hidden_states_to_logits, hidden_states_to_aux = norm(hidden_states, residual)

# 6. 返回：用于 logits 和 aux
return hidden_states_to_logits, [hidden_states_to_aux]
```

---

## 七、模型注册

### 7.1 模型类命名

**EAGLE/EAGLE2**: `{BaseModel}ForCausalLMEagle`
- 例如: `LlamaForCausalLMEagle`

**EAGLE3**: `{BaseModel}ForCausalLMEagle3`
- 例如: `LlamaForCausalLMEagle3`

### 7.2 EntryClass

```python
EntryClass = [LlamaForCausalLMEagle]  # EAGLE/EAGLE2
# 或
EntryClass = [LlamaForCausalLMEagle3]  # EAGLE3
```

### 7.3 配置文件

**config.json**:
```json
{
    "architectures": ["LlamaForCausalLMEagle"],  // 或 LlamaForCausalLMEagle3
    "num_hidden_layers": 1,  // EAGLE3 必须为 1
    "hidden_size": 4096,
    "vocab_size": 32000,
    "draft_vocab_size": 32000,  // EAGLE3 可选
    "hot_vocab_size": 2048,  // 可选
    "target_hidden_size": 4096,  // EAGLE3 可选
    "tie_word_embeddings": false,
    ...
}
```

---

## 八、实现检查清单

### 8.1 必须实现的方法

- [ ] `get_embed_and_head()` - 获取 embedding 和 lm_head
- [ ] `set_embed_and_head(embed, head)` - 设置 embedding 和 lm_head
- [ ] `set_embed(embed)` - 设置 embedding（EAGLE3）
- [ ] `get_embed()` - 获取 embedding（可选）

### 8.2 模型结构要求

**EAGLE/EAGLE2**:
- [ ] 有 `fc` 层：`Linear(hidden_size * 2, hidden_size)`
- [ ] 第一层跳过 `input_layernorm`
- [ ] Forward 接收 `forward_batch.spec_info.hidden_states`
- [ ] Forward 融合 draft 和 target hidden states

**EAGLE3**:
- [ ] `num_hidden_layers == 1`（必须）
- [ ] 有 `fc` 层：`Linear(target_hidden_size * 3, hidden_size)`
- [ ] Decoder Layer 接收 `(embeds, hidden_states)` 两个输入
- [ ] Forward 返回 `(hidden_states_to_logits, [hidden_states_to_aux])`
- [ ] 有 `capture_aux_hidden_states = True` 属性
- [ ] 可能有 `hot_token_id` 属性

### 8.3 权重共享

- [ ] 实现权重共享机制
- [ ] 处理 hot token（如果使用）
- [ ] 处理 hidden_size 不匹配的情况（EAGLE3）

### 8.4 配置和注册

- [ ] 模型类正确命名
- [ ] 设置 `EntryClass`
- [ ] 配置文件正确

---

## 九、参考实现

### 9.1 EAGLE/EAGLE2

**文件**: `python/sglang/srt/models/llama_eagle.py`

**关键点**:
- 多层的 Transformer
- `fc` 层融合 hidden states
- 第一层特殊处理

### 9.2 EAGLE3

**文件**: `python/sglang/srt/models/llama_eagle3.py`

**关键点**:
- 只有 1 层
- 特殊的 decoder layer 结构
- 独立的 vocab size 支持

---

## 十、总结

### 核心要求

1. **必须实现的方法**:
   - `get_embed_and_head()`
   - `set_embed_and_head()` / `set_embed()`

2. **Forward 特殊性**:
   - 必须接收 `forward_batch.spec_info.hidden_states`（target 的 hidden states）
   - 需要融合 draft 和 target 的 hidden states

3. **架构差异**:
   - **EAGLE/EAGLE2**: 多层，`fc` 融合，共享 lm_head
   - **EAGLE3**: 单层，特殊结构，可能有独立 lm_head

4. **权重共享**:
   - Embedding: 通常共享
   - LM Head: EAGLE/EAGLE2 共享，EAGLE3 可能不共享

### 关键设计原则

- **轻量级**: Draft model 应该比 target model 小得多
- **快速**: 需要快速生成 draft tokens
- **协同**: 需要与 target model 协同工作（共享权重、接收 hidden states）



