# 基于 sgl-kernel 0.3.21 融合 PR #24 重新编译与 SGLang 适配

在保持 **sgl-kernel 版本 0.3.21** 的前提下，将 sgl-flash-attn 的 **PR #24**（SM90 Masked MHA / sparse mask）融合进 kernel，并完成 SGLang 侧适配。

**⚠️ PR #24 已在 sgl-flash-attn 上游被 revert。** 本仓库默认已回退到 sgl-attn（无 PR #24）。本文档适用于：你**自建了保留 PR #24 的 fork**（或对 sgl-attn 打了 PR #24 的 patch），并希望在 sgl-kernel 与 SGLang 中重新接入 masked MHA 时的参考。若仅用上游默认，无需按本文操作。

---

## 1. 变更摘要

### 1.1 sgl-kernel（仓库内 `sgl-kernel/`）

- **CMakeLists.txt**  
  - 将 flash-attention 的 FetchContent 改为使用 **sgl-project/sgl-flash-attn** 的 **sgl-kernel** 分支（含 PR #24），不再使用 sgl-attn 的固定 tag。
- **C++**  
  - `include/sgl_flash_kernel_ops.h`：`mha_fwd` 增加参数 `sparse_mask_fine_`；新增 `mha_get_tile_size()` 声明。  
  - `csrc/flash_extension.cc`：op 注册中为 `fwd` 增加 `sparse_mask_fine`；新增 `get_tile_size` op 并注册。
- **Python**  
  - `python/sgl_kernel/flash_attn.py`：  
    - `flash_attn_with_kvcache` / `flash_attn_varlen_func` 增加可选参数 `sparse_mask_fine=None`，并传入底层 op。  
    - 新增 `get_tile_size(headdim, headdim_v=None, qkv_dtype=..., ...)`，用于查询 (kBlockM, kBlockN)，供 SM90 稀疏 mask 构造使用。
- **版本**  
  - `pyproject.toml` 中版本仍为 **0.3.21**，无需改号。

### 1.2 SGLang（仓库内 `python/sglang/srt/`）

- **nsa_backend.py**  
  - 调用 `flash_attn_varlen_func` 时显式传入 `sparse_mask_fine=None`，与带 PR #24 的 kernel 接口一致，并为后续接入真实 sparse mask 预留位置。

---

## 2. 如何重新编译 sgl-kernel（0.3.21 + PR #24）

### 2.1 环境要求

- CMake ≥ 3.26（建议 3.31+）
- Python ≥ 3.10
- PyTorch（与当前 SGLang 一致，如 2.8+）
- CUDA 12.x（SM90 需 12.4+）
- 如需 SM90：Hopper 架构 GPU（如 H100）

### 2.2 在 SGLang 仓库内编译（推荐）

当前 sgl-kernel 作为子目录存在于 SGLang 仓库中。**若要用 PR #24**，需先将 sgl-kernel 的 CMake 指向你的 fork（保留 PR #24 的 sgl-flash-attn 或打过 patch 的源码），再编译：

```bash
cd /path/to/sglang/sgl-kernel

# 清理旧构建（可选，避免混用旧 flash-attn）
rm -rf _skbuild build dist *.egg-info

# 编译并安装到当前环境（需已改 CMake 指向带 PR #24 的 flash-attn 源）
pip install -e . -v
```

若使用 Makefile（仓库内若有）：

```bash
cd /path/to/sglang/sgl-kernel
make build
pip install dist/*.whl   # 或 pip install -e .
```

### 2.3 限制并行与资源（可选）

```bash
# 限制并行 job 数
make build MAX_JOBS=4

# 同时限制 NVCC 编译线程，降低内存占用
make build MAX_JOBS=4 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

### 2.4 验证安装

```bash
python -c "
from sgl_kernel.flash_attn import flash_attn_varlen_func, get_tile_size
# 新 kernel 支持 sparse_mask_fine 与 get_tile_size
kBlockM, kBlockN = get_tile_size(128, 128, __import__('torch').bfloat16, False, -1, -1, False)
print('get_tile_size:', kBlockM, kBlockN)
print('sgl-kernel (0.3.21 + PR24) OK')
"
```

---

## 3. 在 SGLang 中使用

- 重新编译并安装上述 kernel 后，**无需改 SGLang 启动参数**；现有 `flash_attn_varlen_func` / `flash_attn_with_kvcache` 调用会使用新 op（默认 `sparse_mask_fine=None`，行为与原先一致）。
- 若需在 SM90 上使用 **Masked MHA**（如 topK 稀疏 attention）：
  1. 使用 `get_tile_size` 得到 (kBlockM, kBlockN)。
  2. 使用 [sparse_mask_lib](https://github.com/leavelet/sparse_mask_lib) 等按 PR #24 约定构造 `sparse_mask_fine`（形状与 128-byte 对齐见 PR 说明）。
  3. 在对应 attention 路径（如 nsa_backend 中调用 `flash_attn_varlen_func` 处）传入 `sparse_mask_fine=...`；此时 causal 由 mask 表达，接口内部会关掉内置 causal。

---

## 4. 回退为仅用 sgl-attn（不含 PR #24）

**当前仓库默认已是 sgl-attn（无 PR #24）**；sgl-kernel 与 SGLang 已移除 sparse_mask_fine / get_tile_size 相关代码。若你曾在本地或 fork 中接入过 PR #24，想恢复为“仅用 sgl-attn”时可参考：

1. 在 **sgl-kernel/CMakeLists.txt** 中确保 flash-attention 的 FetchContent 为：
   - `GIT_REPOSITORY https://github.com/sgl-project/sgl-attn`
   - `GIT_TAG cc75c5c5979a607ad20a6828635646f9841acf01`
   - 无 `GIT_BRANCH`。
2. 在 **include/sgl_flash_kernel_ops.h** 与 **csrc/flash_extension.cc** 中无 `sparse_mask_fine` 与 `get_tile_size`。
3. 在 **python/sgl_kernel/flash_attn.py** 中无 `sparse_mask_fine` 参数与 `get_tile_size` 函数。
4. 在 **python/sglang/srt/layers/attention/nsa_backend.py** 的 `_forward_fa3` 中不传 `sparse_mask_fine`，使用 `causal=True`。
5. 重新执行 `pip install -e .` 或 `make build` 并安装。

---

## 5. 小结

| 项目 | 说明 |
|------|------|
| 版本 | sgl-kernel 保持 0.3.21 |
| 融合内容 | sgl-flash-attn PR #24（SM90 sparse_mask_fine + get_tile_size） |
| 编译 | 在 `sglang/sgl-kernel` 下 `pip install -e .` 或 `make build` 后安装 |
| SGLang | 已适配：nsa_backend 显式传 `sparse_mask_fine=None`，可与新 kernel 兼容并便于后续接 mask |

按上述步骤即可基于 0.3.21 重建带 PR #24 的 kernel，并在当前 SGLang 中直接使用。
