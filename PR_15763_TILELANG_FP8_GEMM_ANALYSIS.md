# PR #15763: TileLang FP8 Blockwise GEMM Backend 详细分析

## PR 概述

**PR 链接**: https://github.com/sgl-project/sglang/pull/15763  
**作者**: @cscyuge  
**状态**: Open (等待 review)  
**变更规模**: +6,503 −1 (31 commits, 37 files changed)

## 核心目标

添加 **TileLang** 作为 SGLang 的新 FP8 blockwise GEMM backend，填补现有 FP8 GEMM backend 的空白，特别是对 Ada Lovelace (SM89) 架构的支持。

---

## 一、背景与动机

### 1.1 现有 FP8 GEMM Backend 的限制

根据 PR 描述和代码分析，SGLang 当前支持的 FP8 GEMM backend 存在以下限制：

| Backend | 支持的 GPU 架构 | 限制 |
|---------|---------------|------|
| **DeepGEMM/CUTLASS** | Hopper (SM90+) | 仅支持最新架构，不支持 Ada Lovelace (SM89) |
| **Triton** | 所有架构 | 可用但性能可能不是最优 |
| **FlashInfer TRTLLM** | Blackwell (SM100+) | 仅支持最新架构 |

### 1.2 问题场景

- **RTX 4090 (Ada Lovelace, SM89)** 用户无法使用 DeepGEMM/CUTLASS FP8 backend
- 小 batch size (M ≤ 32) 场景下，Triton 性能不够理想
- 需要更多针对不同 shape 的优化变体

### 1.3 TileLang 的优势

TileLang 是一个基于 TVM 的 DSL，用于编写高性能 GPU kernel，具有以下特点：
- 支持 Ada Lovelace (SM89) 和 Hopper (SM90+) GPUs
- 提供灵活的 kernel 变体优化
- 支持 swapAB 优化（针对小 batch size）

---

## 二、PR 实现内容

### 2.1 新增模块结构

根据 PR 描述，新增了以下模块：

```
python/sglang/srt/layers/tilelang_gemm_wrapper/
├── core/
│   ├── wrapper.py          # TileLang GEMM wrapper 主实现
│   ├── tuner.py            # Multi-GPU tuner 用于寻找最优配置
│   └── kernels/
│       ├── base.py          # 基础 kernel 实现
│       ├── swapab.py        # swapAB 优化 kernel (M ≤ 32)
│       └── split_k.py       # Split-K kernel 变体
├── __init__.py
└── configs/                 # 预编译的配置 (按设备/shape)
```

### 2.2 关键功能实现

#### 2.2.1 SwapAB 优化

**目的**: 针对小 batch size (M ≤ 32) 的性能优化

**原理**: 
- 当 M 很小时，传统的 A×B 矩阵乘法内存访问模式不高效
- SwapAB 通过交换 A 和 B 的位置，改变内存访问模式，提高 cache 命中率
- 参考实现：
  - TensorRT-LLM: [Feat: add deep_gemm swapab Kernel](https://github.com/NVIDIA/TensorRT-LLM/pull/4430)
  - DeepGEMM: [support swapAB for m_grouped_fp8_gemm_nt_masked](https://github.com/deepseek-ai/DeepGEMM/pull/192)

**性能提升**:
- 对于 shape `1×4096×1024`:
  - RTX 4090: **5.77x speedup** over Triton
  - H20: **1.42x speedup** over DeepGEMM

#### 2.2.2 Multi-GPU Tuner

**功能**: 为每个 (M, N, K) shape 寻找最优配置

**实现方式**:
- 自动测试不同的 kernel 变体（base, swapAB, split_k）
- 测试不同的 block shape 配置
- 记录最优配置到 JSON 文件
- 运行时根据 shape 自动选择最优 kernel

**配置文件格式**:
```json
{
  "M": 1,
  "N": 4096,
  "K": 1024,
  "device_name": "NVIDIA_RTX_4090",
  "dtype": "fp8_w8a8",
  "block_shape": [128, 128],
  "kernel_type": "swapAB",
  "tflops": 3.23
}
```

#### 2.2.3 Pre-compilation Script

**脚本**: `python/sglang/compile_tilelang_gemm.py`

**功能**:
- 为模型权重 shapes 预编译 TileLang kernels
- 生成最优配置 JSON 文件
- 支持批量编译多个 shapes

**使用场景**:
- 模型加载时预编译常用 shapes
- 减少首次运行时的 JIT 编译时间

### 2.3 集成到 SGLang

#### 2.3.1 添加 Backend 选项

**修改文件**: `python/sglang/srt/server_args.py`

```python
FP8_GEMM_RUNNER_BACKEND_CHOICES = [
    "auto",
    "deep_gemm",
    "flashinfer_trtllm",
    "cutlass",
    "triton",
    "aiter",
    "tilelang",  # 新增
]
```

#### 2.3.2 更新 Dispatch 逻辑

**修改文件**: `python/sglang/srt/layers/quantization/fp8_utils.py`

```python
def _dispatch_explicit_backend(backend: Fp8GemmRunnerBackend) -> Callable:
    # ... 现有代码 ...
    elif backend.is_tilelang():
        if not tilelang_gemm_wrapper.ENABLE_TILELANG_GEMM:
            raise RuntimeError(
                "TileLang backend requested via --fp8-gemm-backend=tilelang, "
                "but TileLang is not available."
            )
        return tilelang_w8a8_block_fp8_linear_with_fallback
    # ... 现有代码 ...
```

#### 2.3.3 Auto Backend 优先级

**更新优先级顺序**:
```python
def _dispatch_auto_backend() -> Callable:
    # Priority order:
    # 1. DeepGEMM (if enabled and available)
    # 2. FlashInfer TRTLLM (if Blackwell GPU)
    # 3. CUTLASS (if Hopper+ GPU)
    # 4. TileLang (if Ada/Hopper GPU)  # 新增
    # 5. AITER (if AMD GPU)
    # 6. Triton (fallback)
    
    if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return deepgemm_w8a8_block_fp8_linear_with_fallback
    elif is_blackwell_supported() and is_flashinfer_available():
        return flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
    elif _check_cutlass_block_fp8_hardware_support():
        return cutlass_w8a8_block_fp8_linear_with_fallback
    elif tilelang_gemm_wrapper.ENABLE_TILELANG_GEMM:  # 新增
        return tilelang_w8a8_block_fp8_linear_with_fallback
    elif _use_aiter:
        return aiter_w8a8_block_fp8_linear
    else:
        return triton_w8a8_block_fp8_linear
```

---

## 三、性能基准测试

### 3.1 RTX 4090 Benchmark 结果

**测试命令**: `python benchmark/kernels/tilelang_gemm/benchmark_tilelang_gemm.py --N 4096 --K 1024`

**关键结果** (部分):

| M | TileLang (ms) | Triton (ms) | Speedup | Kernel Type | Acc |
|---|---------------|-------------|---------|-------------|-----|
| 1 | 0.0026 | 0.0150 | **5.77x** | swapAB | ✓ |
| 2 | 0.0024 | 0.0107 | **4.43x** | swapAB | ✓ |
| 4 | 0.0024 | 0.0099 | **4.08x** | swapAB | ✓ |
| 8 | 0.0024 | 0.0095 | **3.90x** | swapAB | ✓ |
| 16 | 0.0027 | 0.0095 | **3.57x** | swapAB | ✓ |
| 32 | 0.0033 | 0.0095 | **2.87x** | swapAB | ✓ |
| 48 | 0.0042 | 0.0096 | **2.30x** | base | ✓ |
| 64 | 0.0046 | 0.0096 | **2.10x** | base | ✓ |

**关键观察**:
- **小 batch (M ≤ 32)**: swapAB kernel 带来显著性能提升 (2.87x - 5.77x)
- **中等 batch (M > 32)**: base kernel 仍然有 2x+ 的性能提升
- **准确性**: 所有测试通过 (MaxDiff = 0.000000 - 0.125000)

### 3.2 H20 Benchmark 结果

**对比 DeepGEMM**:
- Shape `1×4096×1024`: **1.42x speedup** over DeepGEMM
- 说明 TileLang 在 Hopper 架构上也有竞争力

### 3.3 端到端模型测试

**模型**: Qwen/Qwen3-8B-FP8  
**测试**: `python3 -m sglang.test.few_shot_gsm8k --num-questions 1319`

**结果对比**:

| Backend | Accuracy | Latency | Throughput |
|---------|----------|---------|------------|
| Default (DeepGEMM) | 0.904 | 55.820s | 2892.650 token/s |
| TileLang | 0.904 | 55.999s | 2859.664 token/s |

**结论**:
- **准确性完全一致**: 0.904 (无差异)
- **性能相当**: Latency 和 Throughput 几乎相同
- **验证了正确性**: TileLang 实现与 DeepGEMM 在端到端场景下等价

---

## 四、技术优势分析

### 4.1 架构支持优势

#### 4.1.1 Ada Lovelace (SM89) 支持

**现状**:
- DeepGEMM/CUTLASS: 不支持 SM89
- FlashInfer: 不支持 SM89
- Triton: 支持但性能不佳

**TileLang 优势**:
- ✅ **原生支持 SM89**
- ✅ 针对 SM89 优化的 kernel 变体
- ✅ RTX 4090 用户可以直接使用高性能 FP8 GEMM

#### 4.1.2 Hopper (SM90+) 支持

**现状**:
- DeepGEMM/CUTLASS: 支持但可能不是最优
- FlashInfer: 支持但需要特定条件

**TileLang 优势**:
- ✅ **与 DeepGEMM 性能相当或更好** (H20 上 1.42x)
- ✅ 提供更多 kernel 变体选择
- ✅ 灵活的调优机制

### 4.2 性能优势

#### 4.2.1 小 Batch Size 优化

**问题**: Decode 阶段通常 M=1，传统 GEMM kernel 效率低

**TileLang 解决方案**:
- **SwapAB kernel**: 专门优化 M ≤ 32 的场景
- **性能提升**: 2.87x - 5.77x over Triton
- **实际影响**: Decode 阶段吞吐量显著提升

#### 4.2.2 多 Kernel 变体

**优势**:
- **Base kernel**: 通用场景
- **SwapAB kernel**: 小 batch 优化
- **Split-K kernel**: 大 K 维度优化
- **自动选择**: 根据 shape 自动选择最优 kernel

### 4.3 开发与维护优势

#### 4.3.1 统一的 DSL

**TileLang DSL 优势**:
- **高级抽象**: 比 CUDA 更容易编写和维护
- **自动优化**: TileLang 编译器自动优化
- **可读性强**: 代码更清晰，易于理解

#### 4.3.2 灵活的调优机制

**Multi-GPU Tuner**:
- **自动化**: 无需手动调优
- **可扩展**: 支持新设备和新 shapes
- **可复现**: 配置 JSON 可版本控制

#### 4.3.3 Pre-compilation 支持

**优势**:
- **减少 JIT 延迟**: 预编译常用 shapes
- **生产就绪**: 避免运行时编译
- **可配置**: 支持自定义编译 shapes

### 4.4 生态系统优势

#### 4.4.1 与现有代码集成

**无缝集成**:
- 遵循 SGLang 的 backend 抽象
- 与现有 FP8 GEMM backend 接口一致
- 支持 auto backend 选择

#### 4.4.2 向后兼容

**兼容性保证**:
- 不影响现有 backend
- 可选启用 (`--fp8-gemm-backend=tilelang`)
- 默认 auto 模式会自动选择

---

## 五、实现细节分析

### 5.1 TileLang Kernel 实现

#### 5.1.1 Blockwise FP8 GEMM

**核心实现** (基于 PR 描述和 benchmark 代码):

```python
@tilelang.jit
def fp8_blockwise_gemm(
    A: T.Buffer[(M, K), "e4m3_float8"],
    scales_a: T.Buffer[(M, ceil_div(K, block_K)), "float32"],
    B: T.Buffer[(N, K), "e4m3_float8"],
    scales_b: T.Buffer[(ceil_div(N, block_N), ceil_div(K, block_K)), "float32"],
    C: T.Buffer[(M, N), "bfloat16"],
):
    # Block-wise computation
    # 1. Load A block with scale
    # 2. Load B block with scale
    # 3. Compute GEMM: C = (A * scale_a) @ (B * scale_b)
    # 4. Store result
```

**关键特性**:
- **Block-wise scaling**: 每个 block 独立的 scale factor
- **FP8 → BF16**: 输出转换为 bfloat16
- **TMA (Tensor Memory Accelerator)**: 利用 Hopper/Ada 的 TMA 特性

#### 5.1.2 SwapAB Kernel

**实现原理**:
```python
# 传统: C = A @ B^T  (A: M×K, B: N×K)
# SwapAB: C = B @ A^T  (B: N×K, A: M×K)
# 当 M << N 时，swapAB 的内存访问模式更高效
```

**适用场景**:
- M ≤ 32 (小 batch size)
- N 较大 (如 4096)
- K 中等 (如 1024)

### 5.2 Wrapper 实现

#### 5.2.1 接口设计

```python
def tilelang_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    TileLang FP8 blockwise linear layer.
    
    Args:
        input: (..., M, K) input tensor
        weight: (N, K) weight tensor (FP8)
        block_size: [block_M, block_N, block_K]
        weight_scale: (N // block_N, K // block_K) weight scales
        input_scale: (M // block_M, K // block_K) input scales (optional)
        bias: (N,) bias tensor (optional)
    
    Returns:
        output: (..., M, N) output tensor (BF16)
    """
    # 1. 根据 shape 选择最优 kernel
    # 2. 调用 TileLang kernel
    # 3. 处理 bias (如果有)
    # 4. 返回结果
```

#### 5.2.2 Fallback 机制

**实现策略**:
```python
try:
    # 尝试使用 TileLang kernel
    return tilelang_kernel(...)
except Exception as e:
    # Fallback 到 Triton
    logger.warning(f"TileLang kernel failed, falling back to Triton: {e}")
    return triton_w8a8_block_fp8_linear(...)
```

### 5.3 Tuner 实现

#### 5.3.1 调优流程

```python
def tune_kernel_config(M, N, K, device_name):
    """
    为给定 shape 寻找最优配置。
    """
    best_config = None
    best_time = float('inf')
    
    # 测试不同的 kernel 类型
    for kernel_type in ['base', 'swapab', 'split_k']:
        # 测试不同的 block shapes
        for block_shape in generate_block_shapes(M, N, K):
            # 编译并测试
            kernel = compile_kernel(kernel_type, block_shape)
            time = benchmark_kernel(kernel, M, N, K)
            
            if time < best_time:
                best_time = time
                best_config = {
                    'kernel_type': kernel_type,
                    'block_shape': block_shape,
                    'time': time
                }
    
    # 保存配置
    save_config(M, N, K, device_name, best_config)
    return best_config
```

#### 5.3.2 配置缓存

**存储格式**: JSON 文件
**位置**: `python/sglang/srt/layers/tilelang_gemm_wrapper/configs/`

**文件命名**: `N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{M},{N}].json`

---

## 六、使用方式

### 6.1 命令行使用

```bash
# 显式指定 TileLang backend
python -m sglang.rt.server \
    --model-path Qwen/Qwen3-8B-FP8 \
    --fp8-gemm-backend tilelang

# 使用 auto 模式 (会自动选择 TileLang 如果适合)
python -m sglang.rt.server \
    --model-path Qwen/Qwen3-8B-FP8 \
    --fp8-gemm-backend auto
```

### 6.2 Pre-compilation

```bash
# 为模型权重 shapes 预编译
python python/sglang/compile_tilelang_gemm.py \
    --model-path Qwen/Qwen3-8B-FP8 \
    --output-dir configs/
```

### 6.3 Benchmarking

```bash
# 基准测试特定 shape
python benchmark/kernels/tilelang_gemm/benchmark_tilelang_gemm.py \
    --N 4096 \
    --K 1024 \
    --M-range 1 256
```

---

## 七、潜在问题与限制

### 7.1 已知问题

根据 PR review comments，存在以下问题：

1. **Shape Mismatch in SwapAB Kernel**
   - Gemini Code Assist bot 发现 swapAB kernel 变体中存在 shape 不匹配问题
   - 可能导致不正确输出
   - **状态**: 已修复 (根据 PR 历史)

2. **Benchmark Script Issues**
   - Benchmark 脚本存在一些 minor issues
   - **状态**: 已修复

### 7.2 限制

1. **依赖 TileLang 库**
   - 需要安装 `tilelang` Python 包
   - 可能需要特定版本的 TileLang

2. **JIT 编译时间**
   - 首次运行需要 JIT 编译
   - 可以通过 pre-compilation 缓解

3. **配置管理**
   - 需要为每个设备/shape 组合生成配置
   - 配置文件可能较大

---

## 八、总结

### 8.1 核心贡献

1. **填补架构支持空白**: 为 Ada Lovelace (SM89) 提供高性能 FP8 GEMM
2. **性能优化**: 小 batch size 场景下显著性能提升 (2.87x - 5.77x)
3. **灵活的调优机制**: Multi-GPU tuner 自动寻找最优配置
4. **无缝集成**: 遵循 SGLang backend 抽象，易于使用

### 8.2 适用场景

**最适合**:
- ✅ RTX 4090 (Ada Lovelace) 用户
- ✅ Decode 阶段 (小 batch size)
- ✅ 需要灵活调优的场景

**也适合**:
- ✅ Hopper GPU 用户 (性能与 DeepGEMM 相当)
- ✅ 需要多 kernel 变体的场景

### 8.3 未来改进方向

1. **更多 Kernel 变体**: 支持更多优化场景
2. **自动调优**: 运行时自动调优 (无需预编译)
3. **更好的 Fallback**: 改进 fallback 机制
4. **文档完善**: 添加更多使用文档和示例

---

## 九、参考资料

1. **PR**: https://github.com/sgl-project/sglang/pull/15763
2. **TileLang**: https://github.com/tile-ai/tilelang
3. **TensorRT-LLM SwapAB**: https://github.com/NVIDIA/TensorRT-LLM/pull/4430
4. **DeepGEMM SwapAB**: https://github.com/deepseek-ai/DeepGEMM/pull/192
5. **SGLang FP8 GEMM**: `python/sglang/srt/layers/quantization/fp8_utils.py`

---

**文档创建时间**: 2025-01-XX  
**最后更新**: 基于 PR #15763 分析



