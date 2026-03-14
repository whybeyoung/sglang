# Spec 模型训练指南：模型结构与权重保存格式

## 目录

1. [模型架构概览](#一模型架构概览)
2. [EAGLE/EAGLE2 模型结构](#二eagleeagle2-模型结构)
3. [EAGLE3 模型结构](#三eagle3-模型结构)
4. [权重保存格式](#四权重保存格式)
5. [参考实现代码](#五参考实现代码)
6. [训练注意事项](#六训练注意事项)

---

## 一、模型架构概览

### 1.1 两种架构对比

| 特性 | EAGLE/EAGLE2 | EAGLE3 |
|------|-------------|--------|
| **层数** | 任意（通常 1-4 层） | **必须 1 层** |
| **FC 层输入** | `hidden_size * 2` | `target_hidden_size * 3` |
| **Decoder Layer 输入** | 单一 `hidden_states` | `(embeds, hidden_states)` 两个输入 |
| **LM Head** | 通常共享 target | 通常独立 |
| **Forward 返回值** | `hidden_states` | `(hidden_states_to_logits, [hidden_states_to_aux])` |
| **特殊权重** | 无 | `d2t` (draft-to-target token mapping) |

### 1.2 核心设计思想

**Draft Model 的作用**:
- 基于 target 模型的 hidden states 快速生成 draft tokens
- 比 target model 小得多（1 层 vs 数十层）
- 与 target model 协同工作（共享 embedding，接收 hidden states）

---

## 二、EAGLE/EAGLE2 模型结构

### 2.1 完整模型结构

```
LlamaForCausalLMEagle
├── model (LlamaModel)
│   ├── embed_tokens: VocabParallelEmbedding(vocab_size, hidden_size)
│   ├── fc: Linear(hidden_size * 2, hidden_size)  # 融合层
│   ├── layers: ModuleList[LlamaDecoderLayer]
│   │   └── layers.0, layers.1, ... (num_hidden_layers 个)
│   │       ├── self_attn
│   │       │   ├── q_proj: Linear
│   │       │   ├── k_proj: Linear
│   │       │   ├── v_proj: Linear
│   │       │   └── o_proj: Linear
│   │       ├── mlp
│   │       │   ├── gate_proj: Linear
│   │       │   ├── up_proj: Linear
│   │       │   └── down_proj: Linear
│   │       ├── input_layernorm: RMSNorm  # 第 0 层跳过
│   │       └── post_attention_layernorm: RMSNorm
│   └── norm: RMSNorm (可选，通常不使用)
└── lm_head: ParallelLMHead(vocab_size 或 hot_vocab_size, hidden_size)
```

### 2.2 关键组件详解

#### 2.2.1 Embedding Layer

```python
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,      # 通常与 target 相同
    config.hidden_size,     # 通常与 target 相同
)
```

**权重名称**: `model.embed_tokens.weight`  
**形状**: `(vocab_size, hidden_size)`

#### 2.2.2 FC Fusion Layer

**作用**: 融合 draft embedding 和 target hidden states

```python
self.fc = torch.nn.Linear(
    config.hidden_size * 2,  # 输入: [draft_hidden, target_hidden]
    config.hidden_size,      # 输出: fused_hidden
    bias=False,              # 通常无 bias
)
```

**权重名称**: `model.fc.weight`, `model.fc.bias` (如果有)  
**形状**: `weight: (hidden_size, hidden_size * 2)`, `bias: (hidden_size,)`

**Forward 使用**:
```python
# draft_hidden: (batch, seq_len, hidden_size)
# target_hidden: (batch, seq_len, hidden_size)  # 来自 forward_batch.spec_info.hidden_states
fused_hidden = self.fc(torch.cat([draft_hidden, target_hidden], dim=-1))
```

#### 2.2.3 Decoder Layer

**第 0 层特殊处理**:
```python
if layer_id == 0:
    # 跳过 input_layernorm
    del self.input_layernorm
    setattr(self, "input_layernorm", lambda x: x)
```

**标准结构**:
```python
class LlamaDecoderLayer:
    def __init__(self, config, layer_id, ...):
        # Attention
        self.self_attn.q_proj: Linear(hidden_size, num_heads * head_dim)
        self.self_attn.k_proj: Linear(hidden_size, num_kv_heads * head_dim)
        self.self_attn.v_proj: Linear(hidden_size, num_kv_heads * head_dim)
        self.self_attn.o_proj: Linear(num_heads * head_dim, hidden_size)
        
        # MLP
        self.mlp.gate_proj: Linear(hidden_size, intermediate_size)
        self.mlp.up_proj: Linear(hidden_size, intermediate_size)
        self.mlp.down_proj: Linear(intermediate_size, hidden_size)
        
        # Norms
        self.input_layernorm: RMSNorm(hidden_size)  # 第 0 层跳过
        self.post_attention_layernorm: RMSNorm(hidden_size)
```

**权重命名**:
- `model.layers.{i}.self_attn.q_proj.weight`
- `model.layers.{i}.self_attn.k_proj.weight`
- `model.layers.{i}.self_attn.v_proj.weight`
- `model.layers.{i}.self_attn.o_proj.weight`
- `model.layers.{i}.mlp.gate_proj.weight`
- `model.layers.{i}.mlp.up_proj.weight`
- `model.layers.{i}.mlp.down_proj.weight`
- `model.layers.{i}.input_layernorm.weight` (第 0 层不存在)
- `model.layers.{i}.post_attention_layernorm.weight`

#### 2.2.4 LM Head

```python
if config.tie_word_embeddings:
    self.lm_head = self.model.embed_tokens  # 共享 embedding
else:
    self.lm_head = ParallelLMHead(
        getattr(config, "hot_vocab_size", config.vocab_size),
        config.hidden_size,
    )
```

**权重名称**: `lm_head.weight` (如果不 tie)  
**形状**: `(vocab_size 或 hot_vocab_size, hidden_size)`

### 2.3 Forward 流程

```python
def forward(
    self,
    input_ids: torch.Tensor,           # (batch, seq_len)
    positions: torch.Tensor,            # (batch, seq_len)
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
) -> torch.Tensor:
    # 1. Embedding
    if input_embeds is None:
        hidden_states = self.embed_tokens(input_ids)  # (batch, seq_len, hidden_size)
    else:
        hidden_states = input_embeds
    
    # 2. 获取 target hidden states
    target_hidden = forward_batch.spec_info.hidden_states  # (batch, seq_len, hidden_size)
    
    # 3. 融合
    hidden_states = self.fc(
        torch.cat([hidden_states, target_hidden], dim=-1)  # (batch, seq_len, hidden_size * 2)
    )  # -> (batch, seq_len, hidden_size)
    
    # 4. 通过 Transformer layers
    residual = None
    for layer in self.layers:
        hidden_states, residual = layer(
            positions,
            hidden_states,  # 单一输入
            forward_batch,
            residual,
        )
    
    # 5. 返回（用于 logits）
    return hidden_states + residual  # (batch, seq_len, hidden_size)
```

---

## 三、EAGLE3 模型结构

### 3.1 完整模型结构

```
LlamaForCausalLMEagle3
├── model (LlamaModel)
│   ├── embed_tokens: VocabParallelEmbedding(vocab_size, hidden_size)
│   ├── fc: Linear(target_hidden_size * 3, hidden_size)  # 投影层（如果 hidden_size 不同）
│   ├── midlayer: LlamaDecoderLayer (只有 1 个)
│   │   ├── self_attn
│   │   │   └── qkv_proj: QKVParallelLinear  # 合并的 QKV
│   │   ├── mlp
│   │   │   └── gate_up_proj: Linear  # 合并的 Gate+Up
│   │   ├── input_layernorm: RMSNorm
│   │   ├── hidden_norm: RMSNorm  # 新增
│   │   └── post_attention_layernorm: RMSNorm
│   └── norm: RMSNorm(hidden_size)
└── lm_head: ParallelLMHead(draft_vocab_size, hidden_size)
```

### 3.2 关键组件详解

#### 3.2.1 Embedding Layer

```python
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,  # 可能与 target 不同
)
```

**权重名称**: `model.embed_tokens.weight`  
**形状**: `(vocab_size, hidden_size)`

#### 3.2.2 FC Projection Layer

**作用**: 如果 `target_hidden_size != hidden_size`，投影 target hidden states

```python
if hasattr(config, "target_hidden_size"):
    self.hidden_size_in = config.target_hidden_size
else:
    self.hidden_size_in = config.hidden_size

self.fc = torch.nn.Linear(
    self.hidden_size_in * 3,  # 输入: target_hidden * 3 (embeds + hidden + residual)
    config.hidden_size,        # 输出: draft_hidden
    bias=getattr(config, "bias", False),
)
```

**权重名称**: `model.fc.weight`, `model.fc.bias` (如果有)  
**形状**: `weight: (hidden_size, target_hidden_size * 3)`

**注意**: 如果 `target_hidden_size == hidden_size`，这个层可能不使用（在 forward 中检查）

#### 3.2.3 Decoder Layer (EAGLE3 特殊结构)

**关键差异**:
1. **合并的 QKV**: 使用 `QKVParallelLinear` 而不是分开的 q/k/v
2. **合并的 Gate+Up**: MLP 使用 `gate_up_proj` 而不是分开的 `gate_proj` 和 `up_proj`
3. **新增 hidden_norm**: 对 target hidden states 进行 norm
4. **双输入**: Forward 接收 `(embeds, hidden_states)` 两个输入

```python
class LlamaDecoderLayer:
    def __init__(self, config, layer_id=0, ...):
        # 合并的 QKV
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * hidden_size,  # 输入: concat([embeds, hidden_states])
            head_dim,
            total_num_heads,
            total_num_kv_heads,
        )
        
        # 合并的 Gate+Up
        self.mlp.gate_up_proj = Linear(
            hidden_size,
            intermediate_size * 2,  # gate + up
        )
        
        # Norms
        self.input_layernorm: RMSNorm(hidden_size)      # 对 embeds
        self.hidden_norm: RMSNorm(hidden_size)           # 对 target hidden_states (新增)
        self.post_attention_layernorm: RMSNorm(hidden_size)
```

**权重命名**:
- `model.midlayer.self_attn.qkv_proj.weight` (合并的 QKV)
- `model.midlayer.self_attn.o_proj.weight`
- `model.midlayer.mlp.gate_up_proj.weight` (合并的 Gate+Up)
- `model.midlayer.mlp.down_proj.weight`
- `model.midlayer.input_layernorm.weight`
- `model.midlayer.hidden_norm.weight` (新增)
- `model.midlayer.post_attention_layernorm.weight`

**Forward 流程**:
```python
def forward(
    self,
    positions: torch.Tensor,
    embeds: torch.Tensor,           # Draft embedding
    hidden_states: torch.Tensor,    # Target hidden states
    forward_batch: ForwardBatch,
    residual: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual = hidden_states
    
    # 1. Norm
    embeds = self.input_layernorm(embeds)
    hidden_states = self.hidden_norm(hidden_states)  # 新增
    
    # 2. 拼接
    hidden_states = torch.cat([embeds, hidden_states], dim=-1)  # (batch, seq_len, hidden_size * 2)
    
    # 3. Self Attention (使用合并的 QKV)
    hidden_states = self.self_attn(hidden_states, ...)
    
    # 4. Post-attention norm
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    
    # 5. MLP (使用合并的 Gate+Up)
    hidden_states = self.mlp(hidden_states)
    
    return hidden_states, residual
```

#### 3.2.4 Final Norm

```python
self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

**权重名称**: `model.norm.weight`  
**形状**: `(hidden_size,)`

#### 3.2.5 LM Head

```python
if config.tie_word_embeddings:
    self.lm_head = self.model.embed_tokens
else:
    if config.draft_vocab_size is None:
        self.load_lm_head_from_target = True
        config.draft_vocab_size = config.vocab_size
    self.lm_head = ParallelLMHead(
        config.draft_vocab_size,  # 可能有独立的 vocab size
        config.hidden_size,
    )
```

**权重名称**: `lm_head.weight` (如果不 tie)  
**形状**: `(draft_vocab_size, hidden_size)`

### 3.3 Forward 流程

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    # 1. Embedding
    if input_embeds is None:
        embeds = self.embed_tokens(input_ids)  # (batch, seq_len, hidden_size)
    else:
        embeds = input_embeds
    
    # 2. 获取 target hidden states
    hidden_states = forward_batch.spec_info.hidden_states  # (batch, seq_len, target_hidden_size)
    
    # 3. 如果 hidden_size 不同，投影
    if hidden_states.shape[-1] != embeds.shape[-1]:
        hidden_states = self.fc(hidden_states)  # -> (batch, seq_len, hidden_size)
    
    # 4. 通过 Decoder Layer（双输入）
    residual = None
    hidden_states, residual = self.midlayer(
        positions,
        embeds,          # 输入 1: draft embedding
        hidden_states,   # 输入 2: target hidden states
        forward_batch,
        residual,
    )
    
    # 5. Final Norm
    hidden_states_to_logits, hidden_states_to_aux = self.norm(
        hidden_states, residual
    )
    
    # 6. 返回：用于 logits 和 aux
    return hidden_states_to_logits, [hidden_states_to_aux]
```

---

## 四、权重保存格式

### 4.1 标准 HuggingFace 格式

**文件结构**:
```
model_dir/
├── config.json                    # 模型配置
├── model.safetensors              # 权重文件（推荐）
├── model-00001-of-00001.safetensors  # 或分片格式
└── tokenizer.json                 # Tokenizer（可选）
```

### 4.2 权重命名规范

#### 4.2.1 EAGLE/EAGLE2 权重命名

```
# Embedding
model.embed_tokens.weight                    # (vocab_size, hidden_size)

# FC Fusion Layer
model.fc.weight                              # (hidden_size, hidden_size * 2)
model.fc.bias                                # (hidden_size,) [可选]

# Decoder Layers
model.layers.{i}.self_attn.q_proj.weight     # (num_heads * head_dim, hidden_size)
model.layers.{i}.self_attn.k_proj.weight     # (num_kv_heads * head_dim, hidden_size)
model.layers.{i}.self_attn.v_proj.weight     # (num_kv_heads * head_dim, hidden_size)
model.layers.{i}.self_attn.o_proj.weight     # (hidden_size, num_heads * head_dim)
model.layers.{i}.mlp.gate_proj.weight         # (intermediate_size, hidden_size)
model.layers.{i}.mlp.up_proj.weight           # (intermediate_size, hidden_size)
model.layers.{i}.mlp.down_proj.weight         # (hidden_size, intermediate_size)
model.layers.{i}.input_layernorm.weight       # (hidden_size,) [第 0 层不存在]
model.layers.{i}.post_attention_layernorm.weight  # (hidden_size,)

# LM Head
lm_head.weight                               # (vocab_size 或 hot_vocab_size, hidden_size)
```

#### 4.2.2 EAGLE3 权重命名

```
# Embedding
model.embed_tokens.weight                    # (vocab_size, hidden_size)

# FC Projection Layer (如果 target_hidden_size != hidden_size)
model.fc.weight                              # (hidden_size, target_hidden_size * 3)
model.fc.bias                                # (hidden_size,) [可选]

# Decoder Layer (只有 1 层，命名为 midlayer)
model.midlayer.self_attn.qkv_proj.weight     # 合并的 QKV: (num_heads * head_dim + num_kv_heads * head_dim * 2, hidden_size * 2)
model.midlayer.self_attn.o_proj.weight       # (hidden_size, num_heads * head_dim)
model.midlayer.mlp.gate_up_proj.weight       # 合并的 Gate+Up: (intermediate_size * 2, hidden_size)
model.midlayer.mlp.down_proj.weight          # (hidden_size, intermediate_size)
model.midlayer.input_layernorm.weight        # (hidden_size,)
model.midlayer.hidden_norm.weight            # (hidden_size,) [新增]
model.midlayer.post_attention_layernorm.weight  # (hidden_size,)

# Final Norm
model.norm.weight                            # (hidden_size,)

# LM Head
lm_head.weight                               # (draft_vocab_size, hidden_size)

# 特殊权重（可选）
d2t                                          # Draft-to-Target token mapping: (hot_vocab_size,)
```

### 4.3 特殊权重说明

#### 4.3.1 `d2t` (Draft-to-Target Token Mapping)

**用途**: EAGLE3 中，如果使用 hot token，需要映射 draft vocab 到 target vocab

**格式**:
```python
# d2t 存储的是 diff，实际 hot_token_id = d2t + arange(len(d2t))
d2t = torch.tensor([...])  # shape: (hot_vocab_size,)
hot_token_id = d2t + torch.arange(len(d2t))
```

**保存**:
```python
# 在权重文件中保存为
state_dict["d2t"] = d2t  # 或 "model.d2t"
```

**加载**:
```python
if "d2t" in name:
    self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
    continue
```

#### 4.3.2 `t2d` (Target-to-Draft Token Mapping)

**用途**: 反向映射（通常不需要，加载时跳过）

**处理**:
```python
if "t2d" in name:
    continue  # 跳过
```

### 4.4 权重格式示例

#### 4.4.1 EAGLE/EAGLE2 完整权重列表

```python
weights = {
    # Embedding
    "model.embed_tokens.weight": torch.Tensor,  # (vocab_size, hidden_size)
    
    # FC Layer
    "model.fc.weight": torch.Tensor,  # (hidden_size, hidden_size * 2)
    
    # Layer 0
    "model.layers.0.self_attn.q_proj.weight": torch.Tensor,
    "model.layers.0.self_attn.k_proj.weight": torch.Tensor,
    "model.layers.0.self_attn.v_proj.weight": torch.Tensor,
    "model.layers.0.self_attn.o_proj.weight": torch.Tensor,
    "model.layers.0.mlp.gate_proj.weight": torch.Tensor,
    "model.layers.0.mlp.up_proj.weight": torch.Tensor,
    "model.layers.0.mlp.down_proj.weight": torch.Tensor,
    # 注意：layers.0.input_layernorm.weight 不存在
    "model.layers.0.post_attention_layernorm.weight": torch.Tensor,
    
    # Layer 1 (如果有多层)
    "model.layers.1.self_attn.q_proj.weight": torch.Tensor,
    # ... 类似
    
    # LM Head
    "lm_head.weight": torch.Tensor,  # (vocab_size 或 hot_vocab_size, hidden_size)
}
```

#### 4.4.2 EAGLE3 完整权重列表

```python
weights = {
    # Embedding
    "model.embed_tokens.weight": torch.Tensor,  # (vocab_size, hidden_size)
    
    # FC Layer (如果 target_hidden_size != hidden_size)
    "model.fc.weight": torch.Tensor,  # (hidden_size, target_hidden_size * 3)
    
    # Decoder Layer (只有 1 层)
    "model.midlayer.self_attn.qkv_proj.weight": torch.Tensor,  # 合并的 QKV
    "model.midlayer.self_attn.o_proj.weight": torch.Tensor,
    "model.midlayer.mlp.gate_up_proj.weight": torch.Tensor,  # 合并的 Gate+Up
    "model.midlayer.mlp.down_proj.weight": torch.Tensor,
    "model.midlayer.input_layernorm.weight": torch.Tensor,
    "model.midlayer.hidden_norm.weight": torch.Tensor,  # 新增
    "model.midlayer.post_attention_layernorm.weight": torch.Tensor,
    
    # Final Norm
    "model.norm.weight": torch.Tensor,  # (hidden_size,)
    
    # LM Head
    "lm_head.weight": torch.Tensor,  # (draft_vocab_size, hidden_size)
    
    # 特殊权重（可选）
    "d2t": torch.Tensor,  # (hot_vocab_size,) - Draft-to-Target token mapping
}
```

### 4.5 权重保存代码示例

#### 4.5.1 使用 HuggingFace 格式保存

```python
from safetensors.torch import save_file
import json

# 1. 收集所有权重
state_dict = {}
for name, param in model.named_parameters():
    state_dict[name] = param.detach().cpu()

# 2. 添加特殊权重（EAGLE3）
if hasattr(model, "hot_token_id") and model.hot_token_id is not None:
    # 计算 d2t diff
    d2t = model.hot_token_id - torch.arange(len(model.hot_token_id))
    state_dict["d2t"] = d2t.cpu()

# 3. 保存权重
save_file(state_dict, "model.safetensors")

# 4. 保存配置
config_dict = {
    "architectures": ["LlamaForCausalLMEagle3"],  # 或 LlamaForCausalLMEagle
    "num_hidden_layers": 1,  # EAGLE3 必须为 1
    "hidden_size": 4096,
    "vocab_size": 32000,
    "draft_vocab_size": 32000,  # EAGLE3 可选
    "target_hidden_size": 4096,  # EAGLE3 可选
    "tie_word_embeddings": False,
    # ... 其他配置
}
with open("config.json", "w") as f:
    json.dump(config_dict, f, indent=2)
```

#### 4.5.2 处理合并的参数

**EAGLE3 的特殊情况**: 训练时可能使用分开的参数，保存时需要合并

```python
# 训练时：分开的参数
q_proj_weight = model.midlayer.self_attn.q_proj.weight
k_proj_weight = model.midlayer.self_attn.k_proj.weight
v_proj_weight = model.midlayer.self_attn.v_proj.weight

# 保存时：合并为 qkv_proj
qkv_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
state_dict["model.midlayer.self_attn.qkv_proj.weight"] = qkv_proj_weight

# 类似地处理 gate_up_proj
gate_proj_weight = model.midlayer.mlp.gate_proj.weight
up_proj_weight = model.midlayer.mlp.up_proj.weight
gate_up_proj_weight = torch.cat([gate_proj_weight, up_proj_weight], dim=0)
state_dict["model.midlayer.mlp.gate_up_proj.weight"] = gate_up_proj_weight
```

---

## 五、参考实现代码

### 5.1 EAGLE/EAGLE2 完整实现

**文件**: `python/sglang/srt/models/llama_eagle.py`

```python
from typing import Iterable, Optional, Tuple
import torch
from torch import nn
from transformers import LlamaConfig
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM

class LlamaDecoderLayer(LlamaDecoderLayer):
    """EAGLE Decoder Layer - 第 0 层跳过 input_layernorm"""
    def __init__(self, config, layer_id, ...):
        super().__init__(config, layer_id, ...)
        if layer_id == 0:
            del self.input_layernorm
            setattr(self, "input_layernorm", lambda x: x)

class LlamaModel(nn.Module):
    """EAGLE Model - 融合 target hidden states"""
    def __init__(self, config, ...):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i, ...)
            for i in range(config.num_hidden_layers)
        ])
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
    
    def forward(self, input_ids, positions, forward_batch, ...):
        hidden_states = self.embed_tokens(input_ids)
        target_hidden = forward_batch.spec_info.hidden_states
        hidden_states = self.fc(torch.cat([hidden_states, target_hidden], dim=-1))
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
        return hidden_states + residual

class LlamaForCausalLMEagle(LlamaForCausalLM):
    """EAGLE Model Wrapper"""
    def __init__(self, config, ...):
        self.model = LlamaModel(config, ...)
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                getattr(config, "hot_vocab_size", config.vocab_size),
                config.hidden_size,
            )
        self.logits_processor = LogitsProcessor(config)
    
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight
    
    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
```

### 5.2 EAGLE3 完整实现

**文件**: `python/sglang/srt/models/llama_eagle3.py`

```python
from typing import Iterable, Optional, Tuple
import torch
from torch import nn
from transformers import LlamaConfig
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.models.llama import LlamaForCausalLM, LlamaMLP

class LlamaDecoderLayer(LlamaDecoderLayer):
    """EAGLE3 Decoder Layer - 合并的 QKV 和 Gate+Up"""
    def __init__(self, config, layer_id=0, ...):
        super().__init__(config, layer_id, ...)
        # 合并的 QKV
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * config.hidden_size,  # 输入是 concat([embeds, hidden_states])
            head_dim,
            total_num_heads,
            total_num_kv_heads,
        )
        # 合并的 Gate+Up
        self.mlp.gate_up_proj = Linear(config.hidden_size, intermediate_size * 2)
        # 新增 hidden_norm
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, positions, embeds, hidden_states, forward_batch, residual):
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)  # 新增
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        hidden_states = self.self_attn(hidden_states, ...)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class LlamaModel(nn.Module):
    """EAGLE3 Model - 只有 1 层"""
    def __init__(self, config, ...):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size
        self.fc = torch.nn.Linear(self.hidden_size_in * 3, config.hidden_size)
        self.midlayer = LlamaDecoderLayer(config, 0, ...)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, input_ids, positions, forward_batch, ...):
        embeds = self.embed_tokens(input_ids)
        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)
        hidden_states, residual = self.midlayer(positions, embeds, hidden_states, forward_batch, None)
        hidden_states_to_logits, hidden_states_to_aux = self.norm(hidden_states, residual)
        return hidden_states_to_logits, [hidden_states_to_aux]

class LlamaForCausalLMEagle3(LlamaForCausalLM):
    """EAGLE3 Model Wrapper"""
    def __init__(self, config, ...):
        if config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")
        self.model = LlamaModel(config, ...)
        self.load_lm_head_from_target = False
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(config.draft_vocab_size, config.hidden_size)
        self.capture_aux_hidden_states = True
        self.hot_token_id = None
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """加载权重，处理合并的参数和特殊权重"""
        params_dict = dict(self.named_parameters())
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        for name, loaded_weight in weights:
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue
            if "t2d" in name:
                continue
            # 处理合并的参数映射
            # ...
```

---

## 六、训练注意事项

### 6.1 训练数据准备

**关键**: 需要 target model 的 hidden states 作为输入

**数据格式**:
```python
# 每个样本需要：
{
    "input_ids": torch.Tensor,           # Draft model 的输入 tokens
    "target_hidden_states": torch.Tensor, # Target model 的 hidden states
    "labels": torch.Tensor,              # 目标 tokens（用于计算 loss）
}
```

**获取 target hidden states**:
```python
# 1. 使用 target model forward
with torch.no_grad():
    target_output = target_model(input_ids)
    target_hidden_states = target_output.hidden_states[-1]  # 最后一层

# 2. 保存到训练数据
```

### 6.2 训练 Forward 实现

#### 6.2.1 EAGLE/EAGLE2 训练 Forward

```python
def forward(
    self,
    input_ids: torch.Tensor,
    target_hidden_states: torch.Tensor,  # 从训练数据加载
    labels: Optional[torch.Tensor] = None,
):
    # 1. Embedding
    hidden_states = self.embed_tokens(input_ids)
    
    # 2. 融合 target hidden states
    hidden_states = self.fc(torch.cat([hidden_states, target_hidden_states], dim=-1))
    
    # 3. 通过 Transformer layers
    for layer in self.layers:
        hidden_states, residual = layer(hidden_states, ...)
    
    # 4. LM Head
    logits = self.lm_head(hidden_states)
    
    # 5. 计算 loss
    if labels is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}
    return {"logits": logits}
```

#### 6.2.2 EAGLE3 训练 Forward

```python
def forward(
    self,
    input_ids: torch.Tensor,
    target_hidden_states: torch.Tensor,  # 从训练数据加载
    labels: Optional[torch.Tensor] = None,
):
    # 1. Embedding
    embeds = self.embed_tokens(input_ids)
    
    # 2. 如果 hidden_size 不同，投影
    if target_hidden_states.shape[-1] != embeds.shape[-1]:
        hidden_states = self.fc(target_hidden_states)
    else:
        hidden_states = target_hidden_states
    
    # 3. 通过 Decoder Layer（双输入）
    hidden_states, residual = self.midlayer(embeds, hidden_states, ...)
    
    # 4. Final Norm
    hidden_states_to_logits, hidden_states_to_aux = self.norm(hidden_states, residual)
    
    # 5. LM Head
    logits = self.lm_head(hidden_states_to_logits)
    
    # 6. 计算 loss
    if labels is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}
    return {"logits": logits}
```

### 6.3 权重保存代码

#### 6.3.1 保存 EAGLE/EAGLE2 模型

```python
def save_eagle_model(model, save_dir):
    """保存 EAGLE/EAGLE2 模型"""
    from safetensors.torch import save_file
    import json
    
    # 1. 收集权重
    state_dict = {}
    for name, param in model.named_parameters():
        state_dict[name] = param.detach().cpu()
    
    # 2. 保存权重
    save_file(state_dict, f"{save_dir}/model.safetensors")
    
    # 3. 保存配置
    config = {
        "architectures": ["LlamaForCausalLMEagle"],
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "intermediate_size": model.config.intermediate_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "tie_word_embeddings": model.config.tie_word_embeddings,
        "hot_vocab_size": getattr(model.config, "hot_vocab_size", None),
        # ... 其他配置
    }
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
```

#### 6.3.2 保存 EAGLE3 模型

```python
def save_eagle3_model(model, save_dir):
    """保存 EAGLE3 模型"""
    from safetensors.torch import save_file
    import json
    
    # 1. 收集权重
    state_dict = {}
    for name, param in model.named_parameters():
        # 处理合并的参数
        if "qkv_proj" in name or "gate_up_proj" in name:
            # 这些参数在训练时可能是分开的，需要合并
            # 这里假设已经合并好了
            state_dict[name] = param.detach().cpu()
        else:
            state_dict[name] = param.detach().cpu()
    
    # 2. 保存特殊权重（如果有）
    if hasattr(model, "hot_token_id") and model.hot_token_id is not None:
        d2t = model.hot_token_id - torch.arange(len(model.hot_token_id))
        state_dict["d2t"] = d2t.cpu()
    
    # 3. 保存权重
    save_file(state_dict, f"{save_dir}/model.safetensors")
    
    # 4. 保存配置
    config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "num_hidden_layers": 1,  # 必须为 1
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "draft_vocab_size": getattr(model.config, "draft_vocab_size", model.config.vocab_size),
        "target_hidden_size": getattr(model.config, "target_hidden_size", model.config.hidden_size),
        "tie_word_embeddings": model.config.tie_word_embeddings,
        "load_lm_head_from_target": model.load_lm_head_from_target,
        # ... 其他配置
    }
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
```

### 6.4 处理合并的参数（训练时分开，保存时合并）

**问题**: EAGLE3 使用合并的参数（`qkv_proj`, `gate_up_proj`），但训练时可能使用分开的参数

**解决方案**:

```python
def convert_separated_to_merged(state_dict):
    """将分开的参数转换为合并的参数"""
    new_state_dict = {}
    
    for name, weight in state_dict.items():
        # 处理 QKV
        if name.endswith(".q_proj.weight"):
            base_name = name.replace(".q_proj.weight", "")
            q_weight = weight
            k_weight = state_dict[base_name + ".k_proj.weight"]
            v_weight = state_dict[base_name + ".v_proj.weight"]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            new_state_dict[base_name + ".qkv_proj.weight"] = qkv_weight
            continue
        elif name.endswith(".k_proj.weight") or name.endswith(".v_proj.weight"):
            continue  # 已处理
        
        # 处理 Gate+Up
        elif name.endswith(".gate_proj.weight"):
            base_name = name.replace(".gate_proj.weight", "")
            gate_weight = weight
            up_weight = state_dict[base_name + ".up_proj.weight"]
            gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)
            new_state_dict[base_name + ".gate_up_proj.weight"] = gate_up_weight
            continue
        elif name.endswith(".up_proj.weight"):
            continue  # 已处理
        
        # 其他参数直接复制
        new_state_dict[name] = weight
    
    return new_state_dict
```

### 6.5 配置文件示例

#### 6.5.1 EAGLE/EAGLE2 config.json

```json
{
  "architectures": ["LlamaForCausalLMEagle"],
  "model_type": "llama",
  "num_hidden_layers": 1,
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 32000,
  "hot_vocab_size": 2048,
  "tie_word_embeddings": false,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "max_position_embeddings": 32768
}
```

#### 6.5.2 EAGLE3 config.json

```json
{
  "architectures": ["LlamaForCausalLMEagle3"],
  "model_type": "llama",
  "num_hidden_layers": 1,
  "hidden_size": 4096,
  "target_hidden_size": 4096,
  "intermediate_size": 11008,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 32000,
  "draft_vocab_size": 32000,
  "tie_word_embeddings": false,
  "load_lm_head_from_target": false,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "max_position_embeddings": 32768
}
```

---

## 七、训练流程总结

### 7.1 训练步骤

1. **准备训练数据**
   - 使用 target model 生成 hidden states
   - 保存为 `(input_ids, target_hidden_states, labels)` 格式

2. **初始化模型**
   - 根据架构选择 `LlamaForCausalLMEagle` 或 `LlamaForCausalLMEagle3`
   - 设置正确的配置参数

3. **训练循环**
   - Forward: 使用 target hidden states 作为输入
   - Loss: 计算 draft tokens 的 cross-entropy loss
   - Backward: 更新 draft model 参数

4. **保存模型**
   - 处理合并的参数（如果需要）
   - 保存为 HuggingFace 格式
   - 包含 `config.json` 和权重文件

### 7.2 关键检查点

- [ ] 模型结构正确（层数、FC 层、Decoder Layer）
- [ ] Forward 接收 target hidden states
- [ ] 权重命名符合规范
- [ ] 合并的参数正确处理（EAGLE3）
- [ ] 特殊权重（d2t）正确保存（如果使用）
- [ ] 配置文件完整且正确

---

## 八、参考资源

### 8.1 代码参考

- **EAGLE/EAGLE2**: `python/sglang/srt/models/llama_eagle.py`
- **EAGLE3**: `python/sglang/srt/models/llama_eagle3.py`
- **原始 EAGLE**: https://github.com/SafeAILab/EAGLE

### 8.2 论文参考

- **EAGLE**: [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- **EAGLE3**: [EAGLE-3: Faster Inference with Speculative Decoding](https://arxiv.org/abs/2412.08518)

---

**文档创建时间**: 2025-01-XX  
**最后更新**: 基于 SGLang 代码库分析



