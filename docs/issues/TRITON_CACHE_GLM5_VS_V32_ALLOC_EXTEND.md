# Triton 缓存：GLM5 decode 远多于 v32/v3 的原因（尤其 alloc_extend_kernel）

## 现象

- **GLM5 decode**：`/tmp/torchinductor_root/triton` 下总文件约 **8065**，其中 `alloc_extend_kernel.*` 出现 **481 次**（每种扩展名 481 个，即 481 个不同编译产物）。
- **v32 / v3**：同一或类似缓存目录下文件很少，`alloc_extend_kernel` 几乎没有或很少。

## 根本原因概览

1. **alloc_extend_kernel 是按“形状/常量”特化的**  
   每次不同的 `(bs_upper, page_size, max_num_extend_tokens)` 都会触发一次新的 Triton JIT 编译，产生一组新文件（.source, .json, .ptx, .cubin 等），因此**特化组合越多，缓存里同名文件数越多**。

2. **GLM5 decode 会大量、多样地走 paged alloc_extend 路径**  
   - 使用 **page_size > 1** 的 paged KV，extend 分配走 `alloc_extend()` → `alloc_extend_kernel`。  
   - 请求的 **batch size、extend 长度** 组合很多，导致 `bs_upper` 和 `max_num_extend_tokens` 的取值组合非常多 → 产生大量特化 → 缓存里 481 个 `alloc_extend_kernel`。

3. **v32/v3 同样使用 paged KV（如 page_size=64）**  
   - 即 v32/v3 也会走 `alloc_extend_kernel`，并非因为 page_size=1 而不走。缓存差异来自**请求形状/负载差异**或**运行方式差异**（见下）。

下面按代码和参数说明。

---

## 1. alloc_extend_kernel 为何会产生大量缓存？

定义在 `python/sglang/srt/mem_cache/allocator.py`：

```python
@triton.jit
def alloc_extend_kernel(
    ...
    bs_upper: tl.constexpr,           # 编译期常量
    page_size: tl.constexpr,           # 编译期常量
    max_num_extend_tokens: tl.constexpr,  # 编译期常量
):
```

调用处（同文件）：

```python
# alloc_extend() 内
bs = len(prefix_lens)
...
alloc_extend_kernel[(bs,)](
    ...
    next_power_of_2(bs),                    # → bs_upper
    self.page_size,                         # → page_size
    self.seen_max_num_extend_tokens_next_power_of_2,  # → max_num_extend_tokens
)
```

- **Triton 的缓存 key** 会包含 kernel 签名 + 所有 **constexpr**（即 `bs_upper`、`page_size`、`max_num_extend_tokens`）；**不包含 grid**（即 `(bs,)` 是启动参数，同一编译产物可用不同 grid 多次 launch）。  
- 因此**每一组不同的 `(bs_upper, page_size, max_num_extend_tokens)` = 一次新编译 = 一组新缓存文件**。缓存 key 本身没问题，481 来自这三维组合过多，而非 key 设计错误。

关键点：

- **bs_upper** = `next_power_of_2(bs)`：bs=1,2,3,4 → 1,2,4,4；bs 从 1 到几百，会得到很多不同的 2 的幂。
- **seen_max_num_extend_tokens_next_power_of_2**：进程内**单调不降**，每次遇到更大的 `extend_num_tokens` 就更新为 `max(seen, next_power_of_2(extend_num_tokens))`，从 1 一直涨到 2^19 等（受 `TRITON_MAX_TENSOR_NUMEL` 限制）。  
  因此**不同请求的 extend 长度分布越广，运行时间越长，这个值变化越多**。
- **page_size**：由 server 配置/模型后端决定（如 1, 16, 32, 64, 128）。若同一环境跑过多种 page_size，也会乘上更多组合。

所以：  
**特化数 ≈ (不同 bs_upper) × (不同 page_size) × (不同 max_num_extend_tokens)**  
GLM5 decode 下请求形态多样、运行久，就会积累到 **481 个** 不同的 `alloc_extend_kernel` 特化。

### 缓存目录结构（你看到的那一长串目录名）

- **根目录**：即 `TRITON_CACHE_DIR` 或 `TORCHINDUCTOR_CACHE_DIR` 下的 `triton`（或 `torchinductor_root/triton`）。
- **一层子目录**：每个约 32～40 个字符的**大写字母+数字**串（如 `7XXXXXXXX...`），是 Triton 的 **cache key**：  
  `base32(sha256(version_hash + src_hash + signature + constants + backend/options))`。  
  即**一个 key = 一次 JIT 编译**，对应一组固定的 constexpr（对我们即一组 `(bs_upper, page_size, max_num_extend_tokens)`）。
- **每个 key 目录下**：同一 kernel 的多种产物，例如  
  `alloc_extend_kernel.source`、`alloc_extend_kernel.json`、`alloc_extend_kernel.ptx`、`alloc_extend_kernel.cubin`、`__grp__alloc_extend_kernel.json` 等。  
  所以「同名文件出现 481 次」= 有 **481 个这样的 key 目录**，每个目录里各有一份 `alloc_extend_kernel.*`。
- **为什么这么多 key**：每个 key 对应一种 `(bs_upper, page_size, max_num_extend_tokens)` 组合；旧逻辑下组合数多，所以会看到很多个这样的长串目录。  
  应用 `ALLOC_EXTEND_MAX_TOKENS_INIT` + `ALLOC_EXTEND_BS_UPPER_CAP` 后，新产生的 key 会少很多（个位数）。

### 为何 triton 下会有 0、1、2、…、7 这类数字子目录且内容相同？

- **0～7 是按 rank/进程分的缓存子目录**：在 8 卡或 8 个 worker（如 DP8、EP8）时，TorchInductor/缓存层会把 Triton 缓存按 rank 或 device 再挂一层，例如 `triton/0/`、`triton/1/`、…、`triton/7/`，多进程同时编译时**各写各的目录**。
- **为什么要按 rank 分？本质是避免多进程写同一目录的并发问题**：
  - 若多进程共享同一个缓存目录，会同时写同一批 cache 文件（如 autotune 的 `.json`、编译产物的 `.cubin`/.ptx），而 Triton/Inductor 的缓存写**不是多进程安全的**（已知问题如多进程下 JSONDecodeError、文件被写坏等）。
  - 用文件锁可以理论上解决，但会带来锁竞争、实现复杂；**按 rank 分目录**等于用“目录隔离”代替“文件锁”：每个进程只写自己的 `triton/<rank>/`，不碰别的进程的文件，从而避免并发写和锁竞争。
- **每个数字目录下内容“相同”**：8 个 rank 跑同一份模型、同一套 kernel，每个 rank 都会触发同一批 `(bs_upper, page_size, max)` 的编译，只是写到自己的那个数字目录里，所以 0/、1/、…、7/ 里看到的都是**同一批 kernel 缓存**，相当于 **8 份重复**。
- **总结**：区分 rank 主要是**避免多进程同时写同一目录导致的文件损坏/锁竞争**；不是 cache key 错，而是按 rank 分目录 + 每 rank 编译同一批 kernel，导致 8 份重复，总缓存体积 ≈ 单 rank 的 8 倍。

### 为何 EP16×DP16 时变体特别多？

- **EP16×DP16 = 256 个 worker**，每个 worker 有自己的一份 `PagedTokenToKVPoolAllocator`，各自维护 `seen_max_num_extend_tokens_next_power_of_2`，各自遇到不同的 bs、extend 序列。
- 缓存目录一般是**共享的**（同一 `TRITON_CACHE_DIR`），所以**任意一个 worker 第一次用到某个 (bs_upper, page_size, max) 就会编译一次**，之后其他 worker 命中缓存。
- 256 个 worker **合在一起**会很快覆盖很多 (bs_upper, max) 组合：有的 rank 先碰到 bs=1，有的先碰到 bs=8，有的先碰到 extend=256、有的 4096……于是短时间内就会出现十几档 bs_upper × 几十档 max，再乘上 page_size，轻松到几百个变体。
- **已做优化**：`seen_max` 初始化为 8192、`bs_upper` 在 ≤1024 时统一用 1024，这样无论 EP/DP 多少，变体数都收敛到**个位数**（同一 page_size 下通常 1～3 个 kernel）。

---

## 2. 为何 GLM5 decode 会触发这么多 alloc_extend？

- GLM5 使用 **paged KV**（`page_size > 1`），extend 阶段走 paged 分配逻辑。
- 在 `mem_cache/common.py` 的 `alloc_for_extend()` 里：
  - 若 `batch.tree_cache.page_size == 1` → 用 `alloc_token_slots`，**不经过 alloc_extend**。
  - 若 `page_size > 1` → 走 `alloc_paged_token_slots_extend()` → 最终调用 `alloc_extend()` → **会调用 alloc_extend_kernel**。
- Decode 时每次 extend（每个 batch 的 prefix_lens、seq_lens、extend_num_tokens）都可能不同，导致：
  - **bs** 变化大 → 很多不同的 `bs_upper`。
  - **extend_num_tokens** 变化大 → `seen_max_num_extend_tokens_next_power_of_2` 不断增大，产生很多不同的 2 的幂。
- 再加上若开启 **torch.compile**，Inductor 可能也会把部分逻辑编译进去，缓存目录与 Triton 共用（如 `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR`），所以你会看到在 `torchinductor_root/triton` 下大量 `alloc_extend_kernel.*`。

结论：**GLM5 decode + page_size>1 + 请求形态多样 + 长时间运行** → 大量 alloc_extend 特化 → 8065 总文件、481 个 alloc_extend_kernel 变体。

---

## 3. 为何 v32/v3（同样 page_size=64）的 Triton 缓存很少？

**前提**：v32、v3 均为 **page_size=64**，也会走 paged 的 `alloc_extend`，因此“因为 page_size=1 所以不触发 alloc_extend”不成立。差异更可能来自：

1. **形状/负载更集中**  
   - **bs 分布**：v32/v3 若常用固定或较窄的 batch（如多为 8、16），`bs_upper` 只有少数几个 2 的幂；GLM5 若从 1 到上百的 bs 都出现，`bs_upper` 会很多。  
   - **extend_num_tokens**：`seen_max_num_extend_tokens_next_power_of_2` 随运行中见过的最大 extend 单调上涨。若 v32/v3 的 prefill/chunk 更固定（如固定 chunked_prefill_size、少做大 extend），该值很快稳定在少数几个 2 的幂；GLM5 若 chunk 多样、大 extend 多，会经历 1→2→4→…→很大，产生大量特化。

2. **运行时长与缓存积累**  
   - GLM5 的 8065/481 可能来自**长时间或多轮运行**积累；v32/v3 若是短跑或新缓存目录，尚未触发那么多 (bs_upper, max_num_extend_tokens) 组合。

3. **是否开启 torch.compile / 缓存目录是否一致**  
   - 若 v32/v3 未开 `--enable-torch-compile`，则没有 Inductor 侧生成的大量 Triton 文件，总文件数会少；或使用不同的 `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR`，看到的目录本身就不包含 GLM5 的积累。

4. **调度/用法差异**  
   - 例如 GLM5 是否更多使用 chunked prefill、speculative decoding、或更频繁的 extend；v32/v3 若以“单步 decode + 固定 prefill”为主，extend 形态少，特化数自然少。

---

## 4. 小结与建议

| 项目 | 说明 |
|------|------|
| **为何 GLM5 有 481 个 alloc_extend_kernel？** | 使用 paged KV (page_size>1)，extend 频繁且 (bs, extend_num_tokens) 组合很多，导致 (bs_upper, page_size, max_num_extend_tokens) 特化极多，每个特化一组缓存文件。 |
| **为何 v32/v3 很少？** | v32/v3 也是 page_size=64，同样会走 alloc_extend。差异来自：**形状更集中**（bs、extend 种类少）、**运行时长/缓存积累更少**、**未开 torch.compile 或使用不同缓存目录**、或 **调度/用法**（如 chunked prefill、spec 等）不同导致 extend 形态更少。 |
| **想减少 GLM5 的 Triton 缓存体积？** | 1）**已做**：`allocator.py` 将 `seen_max_num_extend_tokens_next_power_of_2` 初始化为 `ALLOC_EXTEND_MAX_TOKENS_INIT`（8192），避免随请求单调增长；2）**已做**：`bs_upper` 在 ≤1024 时统一用 `ALLOC_EXTEND_BS_UPPER_CAP`（1024），多个 batch 共用同一 kernel；3）启动时 warmup；4）适当限制 max_bs、chunked_prefill。 |

相关代码与问题可进一步参考：

- `python/sglang/srt/mem_cache/allocator.py`：`alloc_extend_kernel`、`alloc_extend()`、`seen_max_num_extend_tokens_next_power_of_2`。
- `python/sglang/srt/mem_cache/common.py`：`alloc_for_extend()` 中 `page_size == 1` 分支。
- `docs/issues/PD_PREALLOC_ALLOC_EXTEND_TRITON_JIT_TTFT_DELAY.md`：PD 下首次 alloc_extend 导致 TTFT 延迟及 warmup 思路。
