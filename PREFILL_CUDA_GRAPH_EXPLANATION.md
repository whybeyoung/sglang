# Prefill CUDA Graph 的限制与 Piecewise CUDA Graph 解决方案

## 为什么 Prefill 一般不支持传统 CUDA Graph？

### 核心问题：序列长度变化

Prefill 阶段面临的主要挑战是**序列长度的高度变化性**：

1. **不同请求的序列长度不同**
   - 用户输入长度从几十到几万 token 不等
   - 每个请求的 prefill 长度都是动态的

2. **CUDA Graph 的固定形状要求**
   - CUDA Graph 在捕获（capture）时需要**固定的张量形状**（fixed tensor shapes）
   - 在重放（replay）时需要**固定的内存地址**（fixed memory addresses）
   - 一旦捕获，所有张量的形状和内存布局都不能改变

3. **传统 CUDA Graph 的限制**
   ```python
   # 传统 CUDA Graph 需要这样：
   # 捕获时：input_ids.shape = (batch_size, seq_len)
   # 重放时：必须完全相同的 shape
   
   # 但 prefill 中：
   # 请求1: seq_len = 100
   # 请求2: seq_len = 5000
   # 请求3: seq_len = 200
   # 无法使用同一个 graph！
   ```

### 为什么 Decode 可以使用 CUDA Graph？

Decode 阶段的特点：
- **固定长度**：每次只生成 1 个 token
- **固定形状**：`(batch_size, 1)` 的形状在同一个 batch 中保持不变
- **固定内存布局**：KV cache 的内存地址相对稳定

因此 decode 阶段可以轻松使用 CUDA Graph 加速。

## Piecewise CUDA Graph 的解决方案

### 核心思想：分块处理 + 动态编译

Piecewise CUDA Graph 通过以下策略解决了 prefill 的变长问题：

#### 1. **Chunked Prefill（分块预填充）**

将长序列分成多个固定大小的块：

```python
# 例如：chunked_prefill_size = 4096
# 序列长度 10000 会被分成：
# - Chunk 1: tokens 0-4095
# - Chunk 2: tokens 4096-8191  
# - Chunk 3: tokens 8192-10000 (剩余部分)
```

每个 chunk 都有固定的最大大小，可以单独捕获 CUDA Graph。

#### 2. **多尺寸 Graph 缓存**

为不同的 token 数量捕获多个 CUDA Graph：

```python
# 从 server_args.py:1027-1044
def _generate_piecewise_cuda_graph_tokens(self):
    capture_sizes = (
        list(range(4, 33, 4))      # 4, 8, 12, ..., 32
        + list(range(48, 257, 16))  # 48, 64, 80, ..., 256
        + list(range(288, 513, 32)) # 288, 320, ..., 512
        + list(range(640, 4096 + 1, 128))  # 640, 768, ..., 4096
        + list(range(4352, max_tokens + 1, 256))
    )
```

系统会为每个 token 数量捕获一个独立的 CUDA Graph，运行时选择最接近的 graph。

#### 3. **Torch Compile 动态形状支持**

结合 `torch.compile` 来处理动态形状：

```python
# 从 piecewise_cuda_graph_runner.py:242-252
with enable_piecewise_cuda_graph():
    with patch_model(model, compiler) as patched_model:
        install_torch_compiled(
            patched_model,
            fullgraph=True,
            dynamic_arg_dims=None,  # 可以处理动态维度
            compile_config=self.compile_config,
            graph_pool=get_global_graph_memory_pool(),
        )
```

`torch.compile` 可以在编译时处理动态形状，生成可以适应不同输入大小的优化代码。

#### 4. **固定内存池（Graph Memory Pool）**

使用全局内存池来保证内存地址的稳定性：

```python
# 从 piecewise_cuda_graph_runner.py:105-115
global_graph_memory_pool = None

def get_global_graph_memory_pool():
    return global_graph_memory_pool

# 在捕获时设置内存池
set_global_graph_memory_pool(self.device_module.graph_pool_handle())
```

这确保了在 graph replay 时，内存地址保持稳定。

### Piecewise CUDA Graph 的工作流程

1. **初始化阶段**：
   ```python
   # 为多个 token 数量捕获 graph
   for num_tokens in capture_sizes:
       # Warmup torch.compile
       warmup_torch_compile(num_tokens=num_tokens)
       # Capture CUDA Graph
       capture_graph(num_tokens)
   ```

2. **运行时**：
   ```python
   # 对于输入的 num_tokens，找到最接近的 graph
   graph_key = find_closest_graph(num_tokens)
   # 使用对应的 graph 重放
   result = graphs[graph_key].replay(input_ids, ...)
   ```

3. **分块处理长序列**：
   ```python
   # 如果序列长度 > chunked_prefill_size
   for chunk in split_into_chunks(sequence, chunk_size=4096):
       # 每个 chunk 使用对应的 graph
       output = piecewise_graph.replay(chunk)
   ```

## 技术优势

### 1. **覆盖所有 Prefill 长度**
- 通过多尺寸 graph 缓存，可以覆盖从 4 到 `chunked_prefill_size` 的所有长度
- 代码注释说明：
  ```python
  # 从 server_args.py:936-940
  # Capture piecewise cuda graph tokens up to the chunked prefill size. 
  # Two benefits:
  # 1. cuda graph acceleration for all prefill lengths.
  # 2. do not need more temporary memory for activations. Less fragmentation.
  ```

### 2. **内存效率**
- 不需要为每个可能的序列长度分配内存
- 使用固定大小的 chunk，减少内存碎片

### 3. **性能提升**
- 减少 kernel launch 开销
- 更紧凑的 GPU 调度
- 与 torch.compile 结合，进一步优化性能

## 限制和注意事项

### 1. **内存开销**
- 需要为每个 token 数量存储一个 graph
- 如果 `piecewise_cuda_graph_max_tokens` 很大，会占用较多显存

### 2. **编译时间**
- 初始化时需要为多个 token 数量编译和捕获 graph
- 首次启动时间较长

### 3. **兼容性要求**
- 需要支持 Standard GQA（Grouped Query Attention）
- 某些模型架构可能不支持（如 Non-Standard GQA）

### 4. **Pipeline Parallelism 限制**
- 当前不支持 Pipeline Parallelism（PP）
- 代码检查：
  ```python
  # 从 model_runner.py:1522
  if self.pp_size > 1:
      log_info_on_rank0(
          logger,
          "Disable piecewise CUDA graph because piecewise_cuda_graph does not support PP",
      )
      return False
  ```

## 使用示例

启用 piecewise CUDA Graph：

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/model \
    --enable-piecewise-cuda-graph \
    --piecewise-cuda-graph-max-tokens 4096 \
    --piecewise-cuda-graph-compiler eager \
    --chunked-prefill-size 4096
```

参数说明：
- `--enable-piecewise-cuda-graph`: 启用 piecewise CUDA Graph
- `--piecewise-cuda-graph-max-tokens`: 最大捕获的 token 数量（通常等于 `chunked_prefill_size`）
- `--piecewise-cuda-graph-compiler`: 编译器选择（`eager` 或 `inductor`）
- `--chunked-prefill-size`: Prefill 分块大小

## Attention 层为什么不走 Piecewise CUDA Graph？

### 核心原因：Attention 使用自己的 CUDA Graph 机制

虽然 attention 层**确实会使用 CUDA Graph**，但它**不走 piecewise CUDA Graph 的 torch.compile 路径**，而是使用**自己的专用 CUDA Graph 机制**。

#### 1. **Attention 层不是 MultiPlatformOp**

Attention 层（如 `FlashInferAttnBackend`, `TritonAttnBackend`）直接继承自 `AttentionBackend` 和 `nn.Module`，而不是 `MultiPlatformOp`：

```python
# Attention 层的结构
class FlashInferAttnBackend(AttentionBackend):
    # 不是 MultiPlatformOp，有自己的 CUDA Graph 实现
    def init_cuda_graph_state(self, ...):
        # 自己的 CUDA Graph 初始化
    def init_forward_metadata_capture_cuda_graph(self, ...):
        # 自己的 CUDA Graph 捕获逻辑
```

#### 2. **Attention 使用高度优化的原生 Kernel**

Attention 层使用的是**高度优化的 CUDA kernel**（如 FlashAttention, Triton kernels），这些 kernel：
- 已经是**高度优化的**，性能接近理论极限
- 使用**专门的 attention 算法**（如 FlashAttention, PagedAttention）
- 有**自己的内存管理**和**KV cache 机制**

```python
# 从 flashinfer_mla_backend.py:327
# Piecewise cuda graph should use paged prefill to be compatible with prefix cache
and not is_in_piecewise_cuda_graph()
```

#### 3. **Attention 有自己的 CUDA Graph 支持**

Attention backend 实现了自己的 CUDA Graph 机制：

```python
# 从 flashinfer_mla_backend.py:342-451
def init_cuda_graph_state(self, max_bs, max_num_tokens, ...):
    # 为 attention 初始化 CUDA Graph 状态
    self.cuda_graph_kv_indices = ...
    self.cuda_graph_qo_indptr = ...
    self.cuda_graph_kv_indptr = ...

def init_forward_metadata_capture_cuda_graph(self, ...):
    # 捕获 attention 的 CUDA Graph
    decode_wrapper = BatchMLAPagedAttentionWrapper(
        use_cuda_graph=True,  # 使用自己的 CUDA Graph
        ...
    )
```

#### 4. **Piecewise CUDA Graph 的作用范围**

Piecewise CUDA Graph 通过 `torch.compile` 优化以下层：

**被 torch.compile 编译的层**：

1. **Linear 层**（标准的 `nn.Linear`）
   - QKV projection layers
   - MLP 的 Linear layers（gate, up, down projections）
   - Embedding layers
   - Output projection layers

2. **MultiPlatformOp 的子类**（通过 `enter_torch_compile` 切换到 `forward_native`）：
   - **LayerNorm**：`RMSNorm`, `LayerNorm`, `GemmaRMSNorm` 等
   - **激活函数**：`SiluAndMul`, `GeluAndMul`, `NewGELU`, `QuickGELU`, `XIELU`
   - **RotaryEmbedding**：位置编码层
   - **TopK**：MoE 的路由选择层
   - **Elementwise operations**：元素级操作

3. **MoE 层**（特殊处理）：
   - 使用 `moe_forward_piecewise_cuda_graph_impl` 实现
   - 通过 `compile_config.add_split_op` 添加到编译配置中

**不被 torch.compile 编译的层**：

- **Attention 层**：
  - 在 piecewise CUDA Graph 中**被调用**，但**不被 torch.compile 编译**
  - 使用**自己的原生 kernel**（FlashInfer, Triton 等）
  - 这些 kernel 在 CUDA Graph 捕获时被**直接捕获**，而不是被重新编译

**总结**：Piecewise CUDA Graph **不仅仅作用于 MLP**，而是作用于**整个模型的前向传播**，包括：
- ✅ Linear 层（QKV, MLP 等）
- ✅ LayerNorm
- ✅ 激活函数
- ✅ RotaryEmbedding
- ✅ MoE 层
- ❌ Attention 层（使用自己的 CUDA Graph 机制）

## Linear 层具体做什么？

### 1. **QKV Projection（Query-Key-Value 投影）**

在 Attention 层中，QKV projection 将输入的 hidden states 转换为 Query、Key、Value 三个向量：

```python
# 从 llama.py:153-161, 188-199
self.qkv_proj = QKVParallelLinear(
    hidden_size,           # 输入维度（如 4096）
    head_dim,             # 每个 attention head 的维度（如 128）
    total_num_heads,      # Query heads 数量（如 32）
    total_num_kv_heads,   # Key/Value heads 数量（如 8，GQA）
)

# Forward 过程：
qkv, _ = self.qkv_proj(hidden_states)  # [batch, seq_len, hidden_size] -> [batch, seq_len, qkv_size]
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
```

**作用**：
- **Query (Q)**：用于"询问"信息，决定关注哪些位置
- **Key (K)**：用于"匹配"查询，提供被关注的内容
- **Value (V)**：实际的信息内容，被提取和聚合

**数学表示**：
```
Q = X × W_q  (Query 投影)
K = X × W_k  (Key 投影)
V = X × W_v  (Value 投影)
```

其中 `X` 是输入的 hidden states，`W_q`, `W_k`, `W_v` 是可学习的权重矩阵。

### 2. **Output Projection (o_proj)**

Attention 计算完成后，将多头 attention 的输出投影回原始 hidden size：

```python
# 从 llama.py:162-168
self.o_proj = RowParallelLinear(
    total_num_heads * head_dim,  # 输入：所有 heads 拼接后的维度
    hidden_size,                 # 输出：原始 hidden size
)

# Forward 过程：
attn_output = self.attn(q, k, v, forward_batch)  # Attention 计算
output, _ = self.o_proj(attn_output)             # 投影回 hidden_size
```

**作用**：
- 将多头 attention 的输出（多个 head 拼接）投影回原始维度
- 融合不同 attention head 的信息

### 3. **MLP 的 Linear Layers**

MLP（Multi-Layer Perceptron）通常包含 2-3 个 Linear 层：

#### **Gate-Projection 架构**（如 Llama, Qwen）：

```python
# 从 llama.py:250-280
self.gate_proj = ColumnParallelLinear(...)  # Gate projection
self.up_proj = ColumnParallelLinear(...)    # Up projection
self.down_proj = RowParallelLinear(...)     # Down projection

# Forward 过程：
gate, _ = self.gate_proj(x)      # [batch, seq_len, hidden_size] -> [batch, seq_len, intermediate_size]
up, _ = self.up_proj(x)          # [batch, seq_len, hidden_size] -> [batch, seq_len, intermediate_size]
x = self.act_fn(gate) * up       # 激活函数（如 SiLU）并做 element-wise 乘法
x, _ = self.down_proj(x)         # [batch, seq_len, intermediate_size] -> [batch, seq_len, hidden_size]
```

**作用**：
- **gate_proj**：生成"门控"信号，控制信息流
- **up_proj**：将 hidden states 扩展到更大的中间维度（通常是 4x hidden_size）
- **down_proj**：将扩展后的特征压缩回原始 hidden size

**数学表示**（SwiGLU 激活）：
```
MLP(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
```

其中 `⊙` 表示 element-wise 乘法，`SiLU` 是激活函数。

#### **简单 MLP 架构**（如 GPT-2）：

```python
# 从 gpt2.py:96-131
self.c_fc = ColumnParallelLinear(...)   # Feed-forward input projection
self.c_proj = RowParallelLinear(...)     # Feed-forward output projection

# Forward 过程：
x, _ = self.c_fc(x)           # [batch, seq_len, hidden_size] -> [batch, seq_len, intermediate_size]
x = self.act_fn(x)             # 激活函数（如 GELU）
x, _ = self.c_proj(x)          # [batch, seq_len, intermediate_size] -> [batch, seq_len, hidden_size]
```

### 4. **Embedding Layers**

```python
# Token Embedding：将 token IDs 转换为向量
self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)

# Position Embedding：位置编码（某些模型）
self.embed_positions = ...
```

**作用**：
- 将离散的 token IDs 映射到连续的向量空间
- 为模型提供位置信息

### 5. **Output Head（输出层）**

```python
# LM Head：将 hidden states 投影到词汇表大小
self.lm_head = ColumnParallelLinear(hidden_size, vocab_size)
```

**作用**：
- 将最后一个 hidden state 投影到词汇表大小
- 用于生成下一个 token 的概率分布

### Linear 层的计算特点

**计算量**：
- **矩阵乘法**：`Y = X × W + b`
- **计算复杂度**：O(batch_size × seq_len × hidden_size × output_size)
- **内存访问**：需要加载权重矩阵和输入张量

**为什么需要优化**：
1. **计算量大**：Linear 层占据了模型大部分的计算量（除了 Attention）
2. **频繁调用**：每个 token 都要经过多个 Linear 层
3. **内存带宽受限**：权重矩阵很大，内存访问是瓶颈
4. **Kernel Launch 开销**：频繁的 kernel launch 会带来显著开销

**Piecewise CUDA Graph 的优化**：
- 通过 `torch.compile` 优化 Linear 层的计算
- 减少 kernel launch 次数
- 更好的内存访问模式
- 融合多个操作（如 bias add）

### 总结

Linear 层在 Transformer 中的主要作用：

1. **QKV Projection**：将 hidden states 转换为 Query、Key、Value
2. **Output Projection**：将 attention 输出投影回原始维度
3. **MLP Layers**：
   - Gate/Up projection：扩展特征维度
   - Down projection：压缩回原始维度
4. **Embedding**：Token 和位置编码
5. **Output Head**：生成词汇表概率分布

这些 Linear 层占据了模型**大部分的计算量**（除了 Attention），因此通过 piecewise CUDA Graph 优化它们可以显著提升性能。

#### 5. **为什么这样设计？**

**性能考虑**：
- Attention kernel（如 FlashAttention）已经高度优化，torch.compile 很难进一步优化
- 专门的 attention kernel 通常比通用编译后的代码**更快**

**兼容性考虑**：
- Attention backend 需要处理**复杂的 KV cache 管理**
- 需要支持**PagedAttention**、**Prefix Cache** 等高级特性
- 这些特性在 attention backend 内部实现更合适

**内存管理**：
- Attention 需要**精确控制 KV cache 的内存布局**
- 使用自己的 CUDA Graph 可以更好地管理这些内存

### 总结：Attention 的 CUDA Graph 路径

```
Piecewise CUDA Graph 捕获流程：
├── Model Forward
│   ├── Embedding (被 torch.compile 编译)
│   ├── Layer 0
│   │   ├── Attention (使用自己的 CUDA Graph，不走 torch.compile)
│   │   │   └── FlashInfer/Triton Kernel (原生优化)
│   │   ├── MLP (被 torch.compile 编译)
│   │   └── LayerNorm (被 torch.compile 编译)
│   └── ...
└── Output (被 torch.compile 编译)
```

**Attention 层**：
- ✅ **会使用 CUDA Graph**（通过自己的机制）
- ❌ **不走 piecewise CUDA Graph 的 torch.compile 路径**
- ✅ **使用原生优化的 kernel**（FlashInfer, Triton 等）

## 总结

**传统 CUDA Graph 的限制**：
- ❌ 需要固定形状
- ❌ Prefill 序列长度变化大
- ❌ 无法处理动态长度

**Piecewise CUDA Graph 的解决方案**：
- ✅ 分块处理（chunked prefill）
- ✅ 多尺寸 graph 缓存
- ✅ 结合 torch.compile 处理动态形状
- ✅ 固定内存池保证地址稳定性

**Attention 层的 CUDA Graph**：
- ✅ 使用自己的专用 CUDA Graph 机制
- ✅ 使用高度优化的原生 kernel（FlashInfer, Triton）
- ✅ 不走 piecewise CUDA Graph 的 torch.compile 路径
- ✅ 在 piecewise CUDA Graph 中被调用，但保持原生实现

**Piecewise CUDA Graph 的作用范围**：
- ✅ **Linear 层**（QKV projection, MLP 的 Linear layers）
- ✅ **LayerNorm**（RMSNorm, LayerNorm 等）
- ✅ **激活函数**（SiluAndMul, GeluAndMul 等）
- ✅ **RotaryEmbedding**（位置编码）
- ✅ **MoE 层**（通过特殊实现）
- ❌ **Attention 层**（使用自己的 CUDA Graph）

这使得 prefill 阶段也能享受到 CUDA Graph 的性能优势，显著减少 kernel launch 开销，提升整体吞吐量。同时，attention 层通过自己的优化机制，确保获得最佳性能。

**注意**：Piecewise CUDA Graph **不仅仅作用于 MLP**，而是作用于**整个模型的大部分层**（除了 Attention）。MLP 只是其中一个重要的组成部分，但 LayerNorm、激活函数、RotaryEmbedding 等同样会被优化。

