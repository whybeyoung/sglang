# sgl-flash-attn PR #24：SM90 Masked MHA 在 SGLang 中的使用

PR [sgl-flash-attn#24](https://github.com/sgl-project/sgl-flash-attn/pull/24) 在 sgl-flash-attn 中为 Hopper (SM90a) 增加了**细粒度稀疏 mask** 的 FlashAttention 支持，用于 topK 等稀疏 attention 模式。

**⚠️ PR #24 已在 sgl-flash-attn 上游被 revert。** 本仓库已配置为使用 fork [whybeyoung/sgl-flash-attn](https://github.com/whybeyoung/sgl-flash-attn)（commit `4df3035`）以支持 masked MHA；编译 sgl-kernel 后设置 **`SGLANG_USE_FA3_SPARSE_MASK=1`** 即可在 NSA fa3 路径（GLM5、DeepSeek-V3 等）上启用 SM90 的 masked kernel。

---

## 1. PR 内容摘要

- **仓库**：sgl-project/sgl-flash-attn（不是 SGLang 本体）。
- **能力**：在 `flash_attn_varlen_func` / `flash_attn_with_kvcache` 等接口中新增可选参数 `sparse_mask_fine`（3D tensor：`[total_q, max_k_blocks, num_int32_per_block]`），在 SM90 上走 TMA + 块级 bitmap 的 masked MHA 路径。
- **依赖**：mask 生成可配合 [sparse_mask_lib](https://github.com/leavelet/sparse_mask_lib)；PR 内新增 `get_tile_size()` 用于查询 (kBlockM, kBlockN)，便于构造正确形状的 mask。
- **限制**：仅 Hopper (SM90+)；启用 sparse mask 时 causal 由 mask 表达，接口层会关掉内置 causal。

---

## 2. Masked kernel 的优势

- **语义**：NSA 等模型用 topK 只对部分 K/V 做 attention，而不是全量 N。Masked kernel 在**不先做 gather** 的前提下，用一块块级 bitmap 告诉 kernel「每个 Q 只能 attend 到哪些 K 块」，在**同一套 FlashAttention 的 paged KV 布局**上做稀疏 attention。
- **算力/带宽**：只对 topK 对应的块算 attention，减少无效的 QK^T 与 softmax，降低算力和对 K/V 的读取量，长上下文、大 batch 时收益更明显。
- **SM90 上的实现**：
  - 用 **TMA** 加载 mask，和 K/V 加载**重叠**，减少额外延迟。
  - Mask 按 GMMA 友好方式排布，应用 mask 时**避免 bank conflict**，提高占用与吞吐。
  - 与现有 `flash_attn_with_kvcache` 接口一致，只是多传 `sparse_mask_fine`，便于在 fa3 路径上切换。
- **和别的稀疏路径的关系**：相对 TileLang sparse、FlashMLA sparse 等，这是在**同一条 fa3 + paged KV** 上多了一种「传 mask 而不是先 gather 再 dense」的选项，在 SM90 上可能更省带宽、更好利用 TMA/GMMA，具体需实测对比。

---

## 3. SGLang 与 sgl-flash-attn 的关系

- SGLang 的 FlashAttention 调用来自 **sgl-kernel**：`from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache`（见 `nsa_backend.py`、`flashattention_backend.py`）。
- sgl-kernel 默认拉取 **sgl-attn**（无 PR #24）。要“用上”PR #24，需要**自维护带 PR #24 的 fork** 并在构建 sgl-kernel 时指向该 fork（见下节）。

---

## 4. PR #24 被 revert 后如何继续用

上游 sgl-flash-attn 已 revert PR #24，因此：

- **默认行为**：sgl-kernel 使用 sgl-attn（`GIT_TAG cc75c5c...`），**不含** sparse_mask_fine / get_tile_size；SGLang 侧不传 mask，`SGLANG_USE_FA3_SPARSE_MASK` 无效果。
- **若仍需使用 masked kernel**，需自建“带 PR #24 的 flash-attn 源”，再让 sgl-kernel 指向它并恢复对应接口，二选一：
  1. **Fork sgl-flash-attn**：在 fork 里保留或重新应用 PR #24 的改动，得到分支或 tag（例如 `my-fork/masked-mha`）。
  2. **在 sgl-attn 上打 patch**：拉取 sgl-attn 后，在构建前用 PR #24 的 diff 打补丁（需自行保存该 diff）。

然后：

- 在 **sgl-kernel/CMakeLists.txt** 中，将 `repo-flash-attention` 的 `GIT_REPOSITORY` 指向上述 fork（或 patch 后的源码路径），并设置对应的 `GIT_BRANCH` 或 `GIT_TAG`。
- 在 **sgl-kernel** 中重新加上与 PR #24 配套的接口：在 `include/sgl_flash_kernel_ops.h` 和 `csrc/flash_extension.cc` 中恢复 `sparse_mask_fine` 参数与 `get_tile_size` 的声明/注册；在 `python/sgl_kernel/flash_attn.py` 中恢复 `sparse_mask_fine` 参数与 `get_tile_size()`。
- 在 **SGLang** 的 `nsa_backend.py` 中恢复 `_forward_fa3` 里根据 `page_table` 构造 `sparse_mask_fine` 并传入 `flash_attn_with_kvcache(..., sparse_mask_fine=..., causal=False)` 的逻辑（及对 `get_tile_size` 的调用）。辅助函数 `_prepare_sparse_mask_fine_from_page_table` 仍保留在仓库中，可直接复用。

具体改动的代码形态可参考本仓库在「接入 PR #24」时期的提交，或 **`docs/issues/SGL_KERNEL_0.3.21_REBUILD_WITH_PR24.md`** 中的说明（该文档描述的是“融合 PR #24 时的构建方式”，在 revert 后需配合 fork/patch 使用）。

---

## 5. 安装与构建（当前默认，无 PR #24）

- 当前 sgl-kernel 默认使用 **sgl-attn**，无需额外配置即可编译安装：
  ```bash
  cd /path/to/sglang/sgl-kernel
  rm -rf _skbuild build dist *.egg-info
  pip install -e . -v
  ```
- 此构建**不包含** masked MHA；NSA 仍走原有 fa3 / FlashMLA 等路径。

---

## 6. 小结

| 项目 | 说明 |
|------|------|
| PR 归属 | sgl-flash-attn，SM90 专用；**上游已 revert** |
| 当前默认 | sgl-kernel 用 sgl-attn，无 sparse_mask_fine / get_tile_size |
| SGLANG_USE_FA3_SPARSE_MASK | 当前无效果，需自建 fork/patch + 恢复接口后才生效 |
| **仍想用 masked MHA** | 见第 4 节：fork（或 patch）+ 改 CMake + 恢复 sgl-kernel 与 SGLang 的 PR #24 相关代码 |
