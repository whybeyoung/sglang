# Triton 预热与 cubin 体积分析

## 1. 为什么当前预热“没用”、没有真实模拟场景且没保存到预期目录

### 1.1 预热只覆盖 allocator 的 Triton，且与“真实请求”可能不同目录

- **当前 warmup（6966ce5c4）**只调用了 **allocator** 里的两个 Triton kernel：
  - `alloc_extend_kernel`（`mem_cache/allocator.py`）
  - `alloc_decode_kernel`（同上）
- 这两个是 **直接** `@triton.jit`，编译时写到哪里由 **`TRITON_CACHE_DIR`** 决定（未设时默认 `~/.triton/cache`）。

而进程里还有另一类 Triton 编译：

- **PyTorch Inductor**（`torch.compile`）会生成大量 Triton kernel，并用自己的缓存目录：
  - 默认是 **`/tmp/torchinductor_<user>`**（或文档里常写的 `/tmp/torchinductor_root`），其下会有 `triton/` 等子目录。
  - 由 **`TORCHINDUCTOR_CACHE_DIR`** 控制；Inductor 内部可能再设置 **`TRITON_CACHE_DIR`** 到该目录下的子路径，导致后续 Triton 编译（包括某些路径下再次用到的 kernel）写到 `/tmp/...` 而不是 `~/.triton`。

因此会出现：

1. **预热**：只跑了 alloc_extend/alloc_decode，写的是 **TRITON_CACHE_DIR**（若已按前面修改设成 `~/.triton/cache`，就写这里）。
2. **真实请求**：  
   - 第一次跑 **torch.compile / Inductor** 时，若此时 **TRITON_CACHE_DIR** 被改成 Inductor 的目录，则 **之后** 的 Triton 编译（包括 allocator 在真实请求里新触发的 **新 shape 变体**）会写到 **/tmp/torchinductor_root/** 之类路径。  
   - 所以会出现“预热没保存到 /tmp/torchroot_”的说法——因为预热写的是我们设的 TRITON_CACHE_DIR；而“真实场景”里 Inductor 占主导时，大量编译结果会进 `/tmp/torchinductor_*`。

结论：**预热并没有在“真实请求的同一套环境、同一缓存目录”下完整跑一遍**，也没有覆盖 Inductor 生成的那部分 Triton，所以会出现“没用、没真实模拟、没保存到 /tmp/torchroot_”的观感。

### 1.2 真实场景的 shape 与预热不一致

- allocator 的 Triton kernel 是按 **(bs_upper, page_size, max_num_extend_tokens)** 等 **constexpr** 做特化的，每个组合一个 cubin。
- 预热里用的是 **有限** 的 bs 列表（1,2,4,...,max_bs）和 **extend 上限 32k**；真实请求可能：
  - 更大的 batch、更长的 extend、或不同的 page_size 组合，
  - 第一次命中未预热过的 (bs, page_size, max_extend) 时，仍然会现场 JIT，并写到**当时进程里的 TRITON_CACHE_DIR**（若已被 Inductor 改成 /tmp/...，就写那里）。

所以即便 allocator 部分，预热也没有完全覆盖“真实场景”的所有 shape，未覆盖的会在首次请求时编译并可能写 /tmp。

---

## 2. 为何 GLM5 编译出约 270MB cubin，而 v3.2/v3 很小

这里 GLM5 泛指 GLM-4 系列 MoE（如 glm4_moe），v3.2/v3 泛指 DeepSeek v3/v3.2。

### 2.1 体积差异主要来自“编译的 kernel 变体数量”

Triton 会对每个 **不同的 constexpr 组合** 编译一个独立 cubin；Inductor 也会对每个编译过的子图/shape 生成大量 Triton 再编译。总体积 ≈ 单 kernel 体积 × 变体数。

可能原因包括：

1. **MoE 实现与 kernel 数量**
   - **GLM-4 MoE**（`glm4_moe.py`）使用 **FusedMoE**（`fused_moe_triton`），同一 kernel 有大量 constexpr：
     - `BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`top_k`、`compute_type`
     - `use_fp8_w8a8`、`use_int8_w8a8`、`use_int8_w8a16`、`per_channel_quant`、`even_Ks`、`c_sorted`、`filter_expert`、`swap_ab` 等。
   - 不同 (M, N, K)、不同量化、不同 top_k 都会生成新 cubin；层数多、专家多、intermediate 大时，组合数很大 → **容易到几百 MB**。
   - **DeepSeek v3/v3.2** 若走 **DeepEP / ep_moe** 或不同 backend（如 FlashInfer MoE），或 expert 数/中间维度更小，触发的 **FusedMoE Triton 变体** 更少，cubin 总体积就小。

2. **@torch.compile 与 Inductor**
   - 代码里 MoE 相关有大量 `@torch.compile(dynamic=True, backend=get_compiler_backend())`（如 `topk.py`、`fused_moe.py` 等），`get_compiler_backend()` 在 CUDA 上是 **"inductor"**。
   - Inductor 会为不同 **运行时 shape** 生成并编译不同 Triton；GLM5 若 batch/token/专家激活 pattern 更多样，编译的 graph 变体多 → 更多 cubin。
   - v3/v3.2 若请求 pattern 更单一或 backend 不同，Inductor 触发的编译少 → 体积小。

3. **模型结构差异**
   - **专家数、intermediate_size、层数**：GLM-4 系列若专家多、中间层大、层数多，FusedMoE 的 (E, N, K) 组合更多。
   - **量化**：GLM5 若启用 FP8/INT8 等，会打开 `use_fp8_w8a8`、`use_int8_*` 等分支，每个都是新 kernel 变体。
   - v3/v3.2 若更小或量化路径更少，变体数少 → 270MB vs 很小的差异。

### 2.2 如何验证

- 对比两边的 **cache 目录**（`TRITON_CACHE_DIR` 与 `TORCHINDUCTOR_CACHE_DIR`）里 **.cubin / 编译产物数量**。
- 看 **FusedMoE 是否为主要路径**：GLM5 用 FusedMoE + 多量化时，cubin 会明显多于主要用 DeepEP/FlashInfer MoE 的 v3/v3.2。

---

## 3. 建议

1. **统一缓存目录**（已做）：在进程最早处设置 `TRITON_CACHE_DIR` 和 `TORCHINDUCTOR_CACHE_DIR`（如 `~/.triton/cache` 与 `~/.cache/sglang/inductor`），减少写回 `/tmp/torchinductor_root`，便于排查和复用。
2. **预热与真实一致**：若希望“预热即真实”，需要：
   - 在**同一进程、同一 TRITON/INDUCTOR 目录**下，
   - 用与真实请求相近的 batch/seq/extend 跑一遍 **allocator + 主要 MoE/forward 路径**（至少触发一次 torch.compile 和 FusedMoE），这样首次真实请求的编译会少很多。
3. **GLM5 体积**：若要减小 270MB，只能减少“变体数”（例如限制 autotune、固定部分 block size、或改用更少分支的 MoE backend），而不是单次编译本身有问题。
