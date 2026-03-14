# TileLang 语言特性解析

## 一、TileLang 是什么？

**TileLang 不是独立的编程语言**，而是一个 **Python DSL（Domain Specific Language）**，类似于 Triton、TVM 等。

### 1.1 核心概念

- **DSL（领域特定语言）**: 嵌入在 Python 中的特殊语法，用于编写 GPU kernel
- **Python 宿主语言**: 使用 Python 作为宿主语言，通过装饰器和特殊 API 定义 kernel
- **JIT 编译**: 运行时编译成实际的 GPU 代码（CUDA PTX）

### 1.2 类比理解

| 特性 | TileLang | Triton | CUDA |
|------|----------|--------|------|
| **语言类型** | Python DSL | Python DSL | 独立语言（C++扩展） |
| **编写方式** | Python + `@tilelang.jit` | Python + `@triton.jit` | C++/CUDA |
| **编译时机** | JIT（运行时） | JIT（运行时） | AOT（提前编译） |
| **语法风格** | Python-like | Python-like | C++-like |
| **抽象级别** | 高（自动优化） | 高（自动优化） | 低（手动优化） |

---

## 二、为什么核心实现是 Python？

### 2.1 Python 作为宿主语言的优势

#### 2.1.1 易用性和开发效率

```python
# TileLang: 简洁的 Python 语法
@tilelang.jit
def kernel_factory(M, N, K, ...):
    @T.prim_func
    def tilelang_fp8_blockwise(A, B, C, ...):
        # 高级抽象，自动处理内存管理、线程调度等
        T.gemm(A_shared, B_shared, C_local)
```

对比 CUDA（需要手动管理）:
```cuda
// CUDA: 需要手动管理 shared memory、thread blocks 等
__global__ void cuda_gemm(float* A, float* B, float* C, ...) {
    __shared__ float A_shared[BLOCK_M][BLOCK_K];
    __shared__ float B_shared[BLOCK_N][BLOCK_K];
    // 大量样板代码...
}
```

#### 2.1.2 与 Python 生态系统无缝集成

```python
# 可以直接使用 Python 的所有功能
import numpy as np
import torch

# 在 Python 中准备数据
A = torch.randn(M, K).cuda()
B = torch.randn(N, K).cuda()

# 调用 TileLang kernel
kernel = kernel_factory(M, N, K, ...)
kernel(A, B, C)
```

#### 2.1.3 动态编译和元编程

```python
# Python 的灵活性：动态生成 kernel
@tilelang.jit
def kernel_factory(M, N, K, block_M, block_N, ...):
    # 运行时决定参数
    M = T.dynamic("m")  # 动态维度
    
    # 根据参数动态构建 kernel
    if block_M > 128:
        # 使用不同的优化策略
        ...
```

### 2.2 TileLang 的编译流程

```
Python 代码 (TileLang DSL)
    ↓
@tilelang.jit 装饰器捕获
    ↓
AST 解析和转换
    ↓
TVM IR (中间表示)
    ↓
优化 Pass
    ↓
GPU 代码生成 (CUDA PTX)
    ↓
运行时执行
```

**关键点**: Python 代码只是**描述**，实际执行的是编译后的 GPU 代码。

---

## 三、TileLang DSL 语法解析

### 3.1 基本结构

```python
import tilelang
import tilelang.language as T

# 1. 使用 @tilelang.jit 装饰器标记 kernel factory
@tilelang.jit
def kernel_factory(M, N, K, ...):
    # 2. 使用 T.prim_func 定义实际的 kernel
    @T.prim_func
    def tilelang_fp8_blockwise(
        A: T.Tensor((M, K), T.float8_e4m3),  # 类型注解使用 T.Tensor
        B: T.Tensor((N, K), T.float8_e4m3),
        C: T.Tensor((M, N), out_dtype),
        ...
    ):
        # 3. 使用 T.Kernel 定义线程块
        with T.Kernel(...) as (bx, by):
            # 4. 使用 T.alloc_shared 分配 shared memory
            A_shared = T.alloc_shared((block_M, block_K), T.float8_e4m3)
            
            # 5. 使用 T.gemm 等高级操作
            T.gemm(A_shared, B_shared, C_local)
    
    return tilelang_fp8_blockwise  # 返回编译后的 kernel
```

### 3.2 关键语法元素

#### 3.2.1 装饰器

```python
@tilelang.jit  # 标记为 TileLang kernel，触发编译
def kernel_factory(...):
    ...
```

#### 3.2.2 类型系统

```python
# TileLang 的类型系统（不是 Python 原生类型）
A: T.Tensor((M, K), T.float8_e4m3)  # Tensor 类型
B: T.Tensor((N, K), T.float8_e4m3)
C: T.Tensor((M, N), out_dtype)

# 数据类型
T.float8_e4m3    # FP8 E4M3
T.float32        # FP32
T.bfloat16       # BF16
T.int32          # Int32
```

#### 3.2.3 内存分配

```python
# Shared memory（线程块共享）
A_shared = T.alloc_shared((block_M, block_K), T.float8_e4m3)

# Fragment memory（线程私有，寄存器）
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
```

#### 3.2.4 控制流

```python
# 并行循环
for i in T.Parallel(block_M):
    A_scale_shared[i] = a_scale[by * block_M + i, k]

# 流水线循环（自动处理 prefetch）
for k in T.Pipelined(K_iters, num_stages=num_stages):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.gemm(A_shared, B_shared, C_local)
```

#### 3.2.5 高级操作

```python
# GEMM 操作（自动优化）
T.gemm(A_shared, B_shared, C_local, transpose_B=True)

# 内存拷贝（自动优化）
T.copy(A[by * block_M, k * block_K], A_shared)

# 清零
T.clear(C_local)
```

### 3.3 动态维度

```python
@tilelang.jit
def kernel_factory(M, N, K, ...):
    # M 是动态的，运行时决定
    M = T.dynamic("m")
    
    # 这样编译一次，可以处理任意 M 值
    @T.prim_func
    def kernel(A: T.Tensor((M, K), ...), ...):
        # M 在运行时确定
        ...
```

**优势**: 减少编译次数，一个 kernel 可以处理多种形状。

---

## 四、TileLang vs 纯 Python

### 4.1 关键区别

| 特性 | Python 代码 | TileLang DSL 代码 |
|------|------------|-------------------|
| **执行位置** | CPU | GPU |
| **执行时机** | 立即执行 | JIT 编译后执行 |
| **类型系统** | 动态类型 | 静态类型（编译时检查） |
| **内存管理** | Python GC | GPU 内存（手动/自动） |
| **性能** | 慢（解释执行） | 快（编译优化） |

### 4.2 代码示例对比

#### 4.2.1 Python 版本（CPU，慢）

```python
def python_gemm(A, B, C):
    """纯 Python GEMM（仅用于演示，实际很慢）"""
    M, K = A.shape
    N = B.shape[0]
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
```

#### 4.2.2 TileLang 版本（GPU，快）

```python
@tilelang.jit
def tilelang_gemm_factory(M, N, K, ...):
    @T.prim_func
    def tilelang_gemm(A, B, C):
        with T.Kernel(...) as (bx, by):
            # GPU 并行执行，自动优化
            T.gemm(A, B, C)
    return tilelang_gemm
```

**关键**: TileLang 代码**不是** Python 代码的执行，而是**描述**，会被编译成 GPU 代码。

---

## 五、TileLang 的底层实现

### 5.1 编译流程详解

```
1. Python 代码解析
   ↓
   @tilelang.jit 装饰器捕获函数 AST
   
2. AST 转换
   ↓
   转换为 TileLang IR（中间表示）
   
3. TVM 编译
   ↓
   转换为 TVM IR → 优化 → GPU 代码生成
   
4. PTX 生成
   ↓
   生成 CUDA PTX（GPU 汇编）
   
5. 运行时加载
   ↓
   通过 CUDA Driver API 加载并执行
```

### 5.2 为什么需要 Python？

#### 5.2.1 元编程能力

```python
# Python 的灵活性：动态生成 kernel
def generate_kernels(shapes):
    kernels = {}
    for M, N, K in shapes:
        @tilelang.jit
        def kernel_factory():
            # 使用闭包捕获 M, N, K
            @T.prim_func
            def kernel(A, B, C):
                # M, N, K 在编译时已知
                ...
        kernels[(M, N, K)] = kernel_factory()
    return kernels
```

#### 5.2.2 与 PyTorch 集成

```python
# 无缝集成 PyTorch
import torch

class TileLangGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        C = torch.empty(M, N, device=A.device)
        kernel = get_kernel(M, N, K)
        kernel(A, B, C)  # 调用 TileLang kernel
        return C
```

#### 5.2.3 配置和调优

```python
# Python 用于配置管理
config = {
    "block_M": 128,
    "block_N": 128,
    "block_K": 64,
    "num_stages": 3,
}

# 动态构建 kernel
kernel = kernel_factory(M, N, K, **config)
```

---

## 六、实际代码示例

### 6.1 完整的 TileLang Kernel

```python
"""Base FP8 Blockwise GEMM Kernel."""

import tilelang  # TileLang Python 包
import tilelang.language as T  # TileLang DSL 语法

@tilelang.jit  # 装饰器：标记为 TileLang kernel
def kernel_factory(
    M, N, K,
    block_M=None,
    block_N=None,
    block_K=None,
    num_stages=None,
    threads=None,
    out_dtype="bfloat16",
    accum_dtype="float32",
):
    # Python 代码：准备参数
    M = T.dynamic("m")  # M 是动态维度
    out_dtype = T.dtype(out_dtype)
    accum_dtype = T.dtype(accum_dtype)
    
    # TileLang DSL：定义 kernel
    @T.prim_func
    def tilelang_fp8_blockwise(
        A: T.Tensor((M, K), T.float8_e4m3),  # TileLang 类型注解
        B: T.Tensor((N, K), T.float8_e4m3),
        C: T.Tensor((M, N), out_dtype),
        a_scale: T.Tensor((M, T.ceildiv(K, 128)), T.float32),
        b_scale: T.Tensor((T.ceildiv(N, 128), T.ceildiv(K, 128)), T.float32),
    ):
        # TileLang DSL：GPU kernel 逻辑
        with T.Kernel(
            T.ceildiv(N, block_N), 
            T.ceildiv(M, block_M), 
            threads=threads
        ) as (bx, by):
            # 分配 shared memory
            A_shared = T.alloc_shared((block_M, block_K), T.float8_e4m3)
            B_shared = T.alloc_shared((block_N, block_K), T.float8_e4m3)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            # 流水线循环
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                
                # GEMM 操作
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            
            # 写回结果
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    # Python 代码：返回编译后的 kernel
    return tilelang_fp8_blockwise
```

### 6.2 使用方式

```python
# Python 代码：准备数据
import torch

A = torch.randn(M, K, dtype=torch.float8_e4m3).cuda()
B = torch.randn(N, K, dtype=torch.float8_e4m3).cuda()
C = torch.empty(M, N, dtype=torch.bfloat16).cuda()

# Python 代码：获取 kernel（首次调用会编译）
kernel = kernel_factory(
    M=M, N=N, K=K,
    block_M=128,
    block_N=128,
    block_K=64,
    num_stages=3,
)

# Python 代码：调用 kernel（实际执行 GPU 代码）
kernel(A, B, C, a_scale, b_scale)
```

---

## 七、总结

### 7.1 TileLang 的本质

1. **不是独立语言**: TileLang 是 Python 的 DSL，嵌入在 Python 中
2. **Python 作为宿主**: 使用 Python 的语法和生态系统
3. **DSL 语法**: 通过 `T.*` API 定义 GPU kernel
4. **JIT 编译**: 运行时编译成 GPU 代码
5. **高性能**: 编译后的代码性能接近手写 CUDA

### 7.2 为什么用 Python？

| 原因 | 说明 |
|------|------|
| **易用性** | Python 语法简洁，学习曲线低 |
| **生态集成** | 与 PyTorch、NumPy 等无缝集成 |
| **元编程** | 动态生成 kernel，灵活配置 |
| **开发效率** | 比手写 CUDA 快得多 |
| **维护性** | 代码更清晰，易于维护 |

### 7.3 类比理解

- **Python**: 宿主语言（就像 HTML 中的 JavaScript）
- **TileLang DSL**: 嵌入的 DSL（就像 HTML 中的 CSS）
- **编译后的 GPU 代码**: 实际执行（就像浏览器渲染的页面）

**关键**: Python 代码只是**描述**，实际执行的是编译后的 GPU 代码，所以性能不受 Python 解释器影响。

---

## 八、参考

- **TileLang**: 基于 TVM 的 GPU kernel DSL
- **类似项目**: Triton（也是 Python DSL）
- **底层**: TVM 编译器框架
- **目标**: 简化高性能 GPU kernel 开发

---

**文档创建时间**: 2025-01-XX  
**基于**: SGLang PR #15763 TileLang 实现分析



