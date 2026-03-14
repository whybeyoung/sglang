# PR #15763 TileLang 代码结构分析

## 代码目录结构

```
python/sglang/srt/layers/tilelang_gemm_wrapper/
├── __init__.py                    # 模块导出
├── configurer.py                  # 配置和启用检查
├── entrypoint.py                  # 对外接口入口
└── core/
    ├── __init__.py
    ├── wrapper.py                 # 核心 wrapper 类
    ├── config_loader.py           # 配置加载器
    ├── kernel_registry.py         # Kernel 注册表
    ├── kernels/                   # Kernel 实现
    │   ├── __init__.py
    │   ├── base.py                # 基础 kernel
    │   ├── swap_ab.py             # SwapAB 优化 kernel
    │   ├── split_k.py              # Split-K kernel
    │   └── split_k_swap_ab.py     # Split-K + SwapAB 组合
    └── config/                     # 预调优配置 JSON 文件
        ├── README.md
        └── N={N},K={K},device_name={device},dtype=fp8_w8a8,block_shape=[128, 128].json

benchmark/kernels/tilelang_gemm/
├── benchmark_tilelang_gemm.py     # 性能基准测试脚本
└── tuning_tilelang_gemm.py        # 自动调优脚本
```

---

## 核心代码文件详解

### 1. `configurer.py` - 配置和启用检查

**功能**: 检查 TileLang GEMM 是否可用

**关键逻辑**:
```python
def _compute_enable_tilelang_gemm() -> bool:
    """启用条件:
    1. GPU SM version >= 89 (Ada Lovelace+)
    2. tilelang 包已安装
    """
```

**全局变量**:
- `ENABLE_TILELANG_GEMM`: 是否启用 TileLang GEMM
- `TILELANG_GEMM_CONFIG_DIR`: 配置文件目录路径

---

### 2. `entrypoint.py` - 对外接口入口

**主要函数**:

1. **`gemm_nt_f8f8bf16()`** - 核心 GEMM 接口
   ```python
   def gemm_nt_f8f8bf16(
       lhs: Tuple[torch.Tensor, torch.Tensor],  # (A_fp8, A_scale)
       rhs: Tuple[torch.Tensor, torch.Tensor],  # (B_fp8, B_scale)
       out: torch.Tensor,                        # (M, N) output
   ) -> None:
       """执行 FP8 GEMM: out = A @ B^T with blockwise scaling"""
   ```

2. **`warmup()`** - 预热编译
   ```python
   def warmup(shapes: List[Tuple[int, int, int]]) -> None:
       """预编译指定 shapes 的 kernels"""
   ```

3. **`warmup_common_shapes()`** - 预热常用 shapes
   ```python
   def warmup_common_shapes(
       m_values: Optional[List[int]] = None,
       nk_shapes: Optional[List[Tuple[int, int]]] = None,
   ) -> None:
   ```

4. **`get_kernel_info()`** - 获取 kernel 信息（调试用）
5. **`is_available()`** - 检查是否可用
6. **`list_available_configs()`** - 列出可用配置

---

### 3. `core/wrapper.py` - 核心 Wrapper 类

**类**: `TileLangGEMMWrapper`

**核心方法**:

#### 3.1 `__init__()`
- 初始化配置加载器
- 初始化 kernel 缓存
- 初始化 partial buffer 缓存（用于 split-K）

#### 3.2 `_get_kernel(M, N, K)` - 获取或编译 kernel
```python
def _get_kernel(self, M: int, N: int, K: int):
    """获取或编译指定维度的 kernel
    
    流程:
    1. 查找配置 (find_config)
    2. 获取 tuned_M (get_tuned_M)
    3. 检查缓存
    4. 如果未缓存，编译 kernel
    5. 缓存并返回
    """
```

**关键点**:
- M 是动态的 (`T.dynamic("m")`)，所以可以用 tuned_M 编译的 kernel 处理任意 M
- 缓存 key: `(tuned_M, N, K, kernel_type)`

#### 3.3 `gemm()` - 执行 GEMM
```python
def gemm(
    self,
    A_fp8: torch.Tensor,      # (M, K), float8_e4m3
    B_fp8: torch.Tensor,      # (N, K), float8_e4m3
    A_scale: torch.Tensor,    # A 的 scale
    B_scale: torch.Tensor,    # B 的 scale
    C: torch.Tensor,          # (M, N), bfloat16 output
) -> None:
    """执行 GEMM: C = A @ B^T with blockwise scaling"""
```

**处理逻辑**:
- 根据配置选择 kernel
- 如果是 swapAB，交换 A 和 B 的位置
- 如果有 split-K，使用 partial buffer

#### 3.4 `warmup_all_m()` - 预热所有 M
```python
def warmup_all_m(self, N: int, K: int, m_max: Optional[int] = None) -> None:
    """预编译所有 tuned M kernels for (N, K)
    
    特点:
    - 并行编译 (par_compile)
    - 只编译 tuned M 值，不是所有 M
    - 按 kernel_type 分组编译
    """
```

**全局函数**:

- **`tilelang_execution_hook(n, k)`** - 执行前预编译 hook
  ```python
  @contextmanager
  def tilelang_execution_hook(n: int, k: int):
      """在执行前预编译所有 M kernels for (N, K)"""
      _maybe_compile_tilelang_all(n, k)
      yield
  ```

---

### 4. `core/config_loader.py` - 配置加载器

**类**: `ConfigLoader`

**功能**:
- 加载 JSON 配置文件
- 查找最接近的 M 配置
- 提供默认配置（如果文件不存在）

**配置文件格式**:
```
文件名: N={N},K={K},device_name={device},dtype=fp8_w8a8,block_shape=[128, 128].json

内容:
{
    "1": {"kernel_type": "swapAB", "block_M": 64, ...},
    "32": {"kernel_type": "swapAB", "block_M": 64, ...},
    "128": {"kernel_type": "base", "block_M": 128, ...},
    ...
}
```

**关键方法**:

1. **`load_config(N, K)`** - 加载配置
   - 如果文件不存在，使用默认配置
   - 默认配置: M ≤ 32 用 swapAB，否则用 base

2. **`find_config(M, N, K)`** - 查找最接近的 M 配置
   ```python
   closest_M = min(configs.keys(), key=lambda x: abs(x - M))
   ```

3. **`get_tuned_M(M, N, K)`** - 获取最接近的 tuned M
   - 用于 kernel 编译（因为 M 是动态的）

---

### 5. `core/kernel_registry.py` - Kernel 注册表

**功能**: 注册和管理所有 kernel 类型

**注册的 Kernel 类型**:

| Kernel Type | Factory | has_split_k | is_swap_ab | scale_shm_key |
|------------|---------|-------------|------------|---------------|
| `base` | `base_kernel_factory` | False | False | `a_scale_shm` |
| `swapAB` | `swapAB_kernel_factory` | False | True | `b_scale_shm` |
| `splitK` | `splitK_kernel_factory` | True | False | `a_scale_shm` |
| `splitK_swapAB` | `splitK_swapAB_kernel_factory` | True | True | `b_scale_shm` |

**关键函数**:
- `get_kernel_factory(kernel_type)` - 获取 kernel factory 信息
- `is_registry_available()` - 检查注册表是否可用

---

### 6. `core/kernels/` - Kernel 实现

#### 6.1 `base.py` - 基础 Kernel

**功能**: 标准 FP8 Blockwise GEMM

**TileLang DSL 实现**:
```python
@tilelang.jit
def kernel_factory(M, N, K, block_M, block_N, block_K, ...):
    M = T.dynamic("m")  # M 是动态的
    
    @T.prim_func
    def tilelang_fp8_blockwise(A, B, C, a_scale, b_scale):
        # Block-wise computation
        # C = (A * a_scale) @ (B * b_scale)^T
```

**Scale 形状**:
- `A_scale`: `(M, K//128)` - per-token-group
- `B_scale`: `(N//128, K//128)` - per-block

#### 6.2 `swap_ab.py` - SwapAB Kernel

**功能**: 交换 A 和 B 的位置，优化小 batch size

**关键差异**:
- 输入: `A (M, K)`, `B (N, K)`
- 计算: `C(M, N) = A @ B^T`
- **输出**: `C^T (N, M)` - 转置输出！

**Scale 形状**:
- `A_scale`: `(M//128, K//128)` - per-block（交换后）
- `B_scale`: `(N, K//128)` - per-token-group（交换后）

**适用场景**: M ≤ 32（小 batch size）

#### 6.3 `split_k.py` - Split-K Kernel

**功能**: 将 K 维度分割，并行计算后合并

**两阶段实现**:
1. **`split_gemm`** - 分割计算 partial results
   ```python
   @T.macro
   def split_gemm(A, B, C_partial, a_scale, b_scale):
       # 计算 C_partial[split_k, M, N]
   ```

2. **`combine`** - 合并 partial results
   ```python
   @T.macro
   def combine(C_partial, C):
       # C = sum(C_partial, dim=0)
   ```

**适用场景**: 大 K 维度

#### 6.4 `split_k_swap_ab.py` - Split-K + SwapAB

**功能**: 组合 split-K 和 swapAB 优化

---

### 7. `core/config/` - 预调优配置

**配置文件命名规则**:
```
N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[128, 128].json
```

**示例文件**:
- `N=4096,K=1024,device_name=NVIDIA_GeForce_RTX_4090,dtype=fp8_w8a8,block_shape=[128, 128].json`
- `N=4096,K=1024,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`

**配置内容**:
```json
{
    "1": {
        "kernel_type": "swapAB",
        "block_M": 64,
        "block_N": 64,
        "block_K": 128,
        "num_stages": 2,
        "threads": 128,
        "latency_ms": 0.0026,
        "tflops": 3.23
    },
    "32": {...},
    "128": {...}
}
```

---

### 8. `benchmark/kernels/tilelang_gemm/` - 基准测试和调优

#### 8.1 `benchmark_tilelang_gemm.py`

**功能**: 性能基准测试

**对比基准**:
- Hopper (SM90+): DeepGEMM
- Ada (SM89): Triton

**输出格式**:
```
M | TileLang (ms) | Baseline (ms) | Speedup | Kernel Type | Acc
```

#### 8.2 `tuning_tilelang_gemm.py`

**功能**: 自动调优脚本

**流程**:
1. 测试不同的 kernel 类型（base, swapAB, splitK, splitK_swapAB）
2. 测试不同的 block shapes
3. 测试不同的 num_stages, threads 等参数
4. 选择最优配置
5. 保存到 JSON 文件

---

## 集成到 SGLang

### 1. `fp8_utils.py` 集成

**修改位置**: `python/sglang/srt/layers/quantization/fp8_utils.py`

**关键修改**:

1. **导入 TileLang wrapper**:
   ```python
   from sglang.srt.layers import tilelang_gemm_wrapper
   ```

2. **添加 TILELANG backend**:
   ```python
   class Fp8GemmRunnerBackend(Enum):
       TILELANG = "tilelang"
       
       def is_tilelang(self) -> bool:
           return self == Fp8GemmRunnerBackend.TILELANG
   ```

3. **Dispatch 逻辑**:
   ```python
   elif backend.is_tilelang():
       if not tilelang_gemm_wrapper.ENABLE_TILELANG_GEMM:
           raise RuntimeError(...)
       return tilelang_w8a8_block_fp8_linear_with_fallback
   ```

4. **实现函数**:
   ```python
   def tilelang_w8a8_block_fp8_linear_with_fallback(...):
       """TileLang FP8 block linear with fallback to Triton"""
       try:
           tilelang_gemm_wrapper.gemm_nt_f8f8bf16(...)
       except Exception:
           # Fallback to Triton
           return triton_w8a8_block_fp8_linear(...)
   ```

### 2. `server_args.py` 集成

**修改**: 添加 `tilelang` 到 `FP8_GEMM_RUNNER_BACKEND_CHOICES`

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

---

## 关键设计特点

### 1. 动态 M 维度

**设计**: M 使用 `T.dynamic("m")`，允许一个编译的 kernel 处理任意 M 值

**优势**:
- 减少编译次数
- 只需要为 tuned M 值编译 kernel
- 运行时自动适配实际 M

### 2. 配置驱动的 Kernel 选择

**流程**:
1. 根据 (M, N, K) 查找配置
2. 配置指定 kernel_type（base/swapAB/splitK/splitK_swapAB）
3. 配置指定 block shapes、num_stages 等参数
4. 自动选择最优 kernel

### 3. 并行编译

**实现**: `factory.par_compile(compile_configs, num_workers=128)`

**优势**: 加速 warmup 过程

### 4. 缓存机制

**三级缓存**:
1. **Kernel 缓存**: `_kernel_cache` - 已编译的 kernel
2. **配置缓存**: `_config_cache` - 已加载的配置
3. **Partial Buffer 缓存**: `_partial_buffer_cache` - Split-K 的中间结果 buffer

### 5. Fallback 机制

**实现**: 如果 TileLang kernel 失败，fallback 到 Triton

**代码**:
```python
try:
    tilelang_gemm_wrapper.gemm_nt_f8f8bf16(...)
except Exception:
    return triton_w8a8_block_fp8_linear(...)
```

---

## 使用流程

### 1. 初始化

```python
# 1. 检查是否启用
if tilelang_gemm_wrapper.ENABLE_TILELANG_GEMM:
    # 2. 初始化 wrapper
    wrapper = tilelang_gemm_wrapper.get_global_wrapper()
    
    # 3. 预热（可选）
    wrapper.warmup_all_m(N, K, m_max=1024)
```

### 2. 执行 GEMM

```python
# 方式 1: 通过 entrypoint
tilelang_gemm_wrapper.gemm_nt_f8f8bf16(
    lhs=(A_fp8, A_scale),
    rhs=(B_fp8, B_scale),
    out=C
)

# 方式 2: 直接使用 wrapper
wrapper.gemm(A_fp8, B_fp8, A_scale, B_scale, C)
```

### 3. 预编译（生产环境推荐）

```bash
python -m sglang.compile_tilelang_gemm \
    --model-path Qwen/Qwen3-8B-FP8 \
    --tp 1
```

---

## 总结

### 核心代码文件（按重要性排序）

1. **`core/wrapper.py`** - 核心 wrapper 类，管理 kernel 编译和调用
2. **`core/kernels/base.py`** - 基础 kernel 实现
3. **`core/kernels/swap_ab.py`** - SwapAB 优化 kernel（小 batch 关键）
4. **`entrypoint.py`** - 对外接口
5. **`core/config_loader.py`** - 配置管理
6. **`core/kernel_registry.py`** - Kernel 注册表
7. **`core/kernels/split_k.py`** - Split-K kernel（大 K 维度）

### 关键特性

- ✅ **动态 M**: 一个 kernel 处理任意 M
- ✅ **多 Kernel 变体**: base, swapAB, splitK, splitK_swapAB
- ✅ **配置驱动**: JSON 配置文件指定最优参数
- ✅ **并行编译**: 加速 warmup
- ✅ **Fallback**: 自动 fallback 到 Triton
- ✅ **缓存机制**: 多级缓存提升性能

### 代码质量

- ✅ **模块化设计**: 清晰的模块划分
- ✅ **类型注解**: 完整的类型提示
- ✅ **错误处理**: 完善的异常处理
- ✅ **日志记录**: 详细的日志输出
- ✅ **文档**: 良好的代码注释



