# Decode 头部 ~15s 延迟根因分析（EP16 profile）

## 结论摘要

- **现象**：Decode 启动后存在约 **15–17s** 的头部延迟；预热后该延迟消失。
- **根因（两路可能并存）**：
  1. **EP-6 本机延迟**：从 EP-6 的 Perfetto trace 可见，前 ~15s 主要是 **Triton JIT 编译**（`alloc_extend` → Triton `alloc_extend_kernel` / `write_req_to_token_pool_triton` 等 → `triton/compiler` → `make_cubin` → subprocess），即 **KV 扩展/ token pool 相关 Triton kernel 首次运行时的编译成本**。
  2. **DP 集体等待**：其余 15 个 rank 在 ~0.35s 已进入 DP 的 `all_gather`，若 EP-6 因上述 JIT 或 recv 未就绪而晚到，会形成约 17s 的集体等待。
- **为何预热后消失**：Triton 编译是一次性的，kernel 缓存后不再编译；首包/建连也只发生一次，之后各 rank 同时就绪，不再出现长阻塞。

## 数据来源

- Decode 16 个 rank 的 Perfetto trace：  
  `/Users/yangyanbo/Downloads/chrome_down/glm5/1772299504.5988448/`  
  文件：`1772299504.6059935-TP-{0..15}-DP-{0..15}-EP-{0..15}.trace.json.gz`

## 1) 17s 阻塞发生位置

在所有 **除 EP-6 外** 的 15 个 rank 上，在 **trace 开始约 0.35s** 处出现同一组长时间事件（约 17.4s）：

| 事件 | 约开始时间(相对) | 持续时间 |
|------|------------------|----------|
| `scheduler_dp_attn_mixin.py(73): all_gather` | 0.348s | **17.44s** |
| `cudaStreamSynchronize` | 0.348s | **17.43s** |
| `nccl:_all_gather_base` | 0.348s | **17.43s** |

对应代码为 DP attention 的 MLP sync 路径中的集体通信：

- `get_next_disagg_decode_batch_to_run` → `maybe_prepare_mlp_sync_batch` → `prepare_mlp_sync_batch` → **`mlp_sync_info.all_gather(device, group)`**（`scheduler_dp_attn_mixin.py` 约 73 行，内部为 `torch.distributed.all_gather_into_tensor`）。

即：**15 个 rank 已进入这次 all_gather，并一直阻塞到第 16 个 rank 也进入并完成**。

## 2) 谁是 straggler：EP-6

- **EP-6 在前 25s 内没有出现上述“长 all_gather / 长 cudaStreamSynchronize”事件**，说明 EP-6 在这段时间内**根本没有进入**这次 DP all_gather。
- 因此：**EP-6 是本次 17s 阻塞的 straggler**——其他 rank 在等 EP-6 进入并完成 all_gather。

## 3) EP-6 在等什么：首包 recv_multipart

EP-6 的 trace 中，在 0~25s 内最突出的长阻塞事件为：

- **`zmq/sugar/socket.py(799): recv_multipart`**，持续时间约 **28.11s**（与整段 trace 同量级）。
- 调用栈对应 **mooncake decode 连接线程**：  
  `srt/disaggregation/mooncake/conn.py` 中 `start_decode_thread()` 内的循环：
  - `msg = self.server_socket.recv_multipart()`（约 951 行）。

含义：

- Decode 的 mooncake 端在**等待 prefill 通过 ZMQ 发来的第一条消息**（例如 bootstrap_room / status / prefill_rank 等）。
- **EP-6 收到这条消息的时间比其他 rank 晚约 17s**，因此一直卡在 `recv_multipart`，无法进入调度逻辑，也就无法参与 DP 的 `get_next_disagg_decode_batch_to_run` → `prepare_mlp_sync_batch` → `all_gather`。
- 其他 15 个 rank 已收到首包并进入 all_gather，集体等待 EP-6 → 表现为约 17s 的头部延迟。

## 4) 与 prefill 侧“15s”的关系

- 若 prefill 侧也观察到约 15s 的头部延迟，与 decode 侧是**同一协商阶段**的不同表现：
  - Prefill 可能在等与 decode 的建连、首包发送/确认，或首请求的调度。
  - Decode 侧则是**某一个 rank（如 EP-6）迟迟收不到 prefill 的首包**，导致该 rank 不参与 DP all_gather，进而拖住所有 decode rank 约 17s。
- 因此：**prefill 与 decode 在“协商阶段”的延迟是相互关联的**；decode 的 15s 来自“一个 decode rank 首包过晚”，而该首包由 prefill→decode 的发送/路由/建连决定。

## 5) 为何预热后延迟消失

- **首包/首请求**只发生一次：建连、首次 KV 传输、首次 batch 协商。
- 一旦 EP-6 也在 ~17s 后收到首包并开始参与 decode，后续 batch 的 prefill→decode 通信已稳定，**所有 16 个 rank 都会在相近时间收到数据并进入 all_gather**，不再出现“等一个 rank”的长阻塞。
- 因此表现为：**头部 15–17s 只在启动阶段出现，预热后消失**。

## 6) 建议的排查与优化方向

1. **Prefill→Decode 首包发送/路由**
   - 确认 prefill 向 16 个 decode rank 的**首包发送顺序与时机**（是否对某一 rank 明显滞后）。
   - 检查是否存在“先发 15 个 rank，再发第 16 个”或对某一 rank 重试/重连导致晚 15s。

2. **Decode 建连与绑定**
   - 检查 EP-6 对应的 ZMQ 地址、端口、绑定顺序是否与其他 rank 不同（例如晚绑定、不同网卡/路由）。
   - 确认 prefill 侧连接 16 个 decode 端点的顺序与延迟（是否有某一连接明显慢）。

3. **Scheduler / 协商逻辑**
   - 若有“等所有 decode rank 就绪再发首包”的逻辑，检查是否某个 rank 被判定就绪的时间晚（例如健康检查、注册顺序）。

4. **可选的代码级缓解**
   - 在 prefill 侧尽量**同时**向所有 decode rank 发送首包，或保证最慢的那路也在数秒内发出。
   - 在 decode 侧可考虑对“首包超时”打日志并区分 rank，便于定位总是晚到的 rank（是否固定为 EP-6 或随机的某一 rank）。

## 7) EP-6 上 JIT 在编译什么（Triton 首次编译）

从 EP-6 的 Perfetto 调用栈可以明确看到：

- **调用链**：`event_loop_normal_disagg_decode` → `allocator.py: alloc_extend` → **Triton** `jit.py: run` / `_do_compile` → `compiler.py: compile` → `nvidia/compiler.py: make_cubin` → `subprocess: run` / `wait` / `waitpid`。

即在 decode 事件循环里，**第一次做 KV 扩展分配**时会走到 `alloc_extend()`，进而触发 **Triton kernel 的首次 JIT 编译**。

**具体在编译的 kernel：**

1. **`alloc_extend_kernel`**（`python/sglang/srt/mem_cache/allocator.py` 约 235 行）  
   - 用 `@triton.jit` 写的 **分页 KV cache 扩展分配** kernel：根据 `prefix_lens` / `seq_lens` / `last_loc` / `free_pages` 计算并写出 `out_indices`（扩展段的 cache 位置）。  
   - 在 `alloc_extend()` 中当 `extend_num_tokens < TRITON_MAX_TENSOR_NUMEL` 时通过 `alloc_extend_kernel[(bs,)](...)` 调用（约 428 行），**第一次调用时** Triton 会编译该 kernel 并生成 CUBIN（subprocess 调用 nvcc），产生约 15s 延迟。

2. **可能还有 `write_req_to_token_pool_triton`**（`mem_cache/common.py`）  
   - 同样是 `@triton.jit`，用于把 **请求的 token 位置写进 req_to_token pool**（写 prefix + extend 的 cache loc）。  
   - 若 decode 路径上第一次 extend 时也走了 `write_cache_indices` 的 Triton 分支，则首次也会触发该 kernel 的编译。

因此，**EP-6 这 ~15s 主要是在编译上述与 KV 扩展 / token pool 分配相关的 Triton kernel**，属于一次性 JIT 成本，编译完成后 kernel 被缓存，后续请求不再重复编译，故“预热后消失”。

## 8) 复现与验证

- 使用本仓库脚本（在 trace 目录存在时）可复现上述结论：
  - `scripts/decode_trace_15s_analysis.py`：统计各 rank 首次长 all_gather 的进入时间，并发现 EP-6 无长 all_gather。
  - 对 EP-6 单独解析 trace，可看到 `recv_multipart` 持续约 28s，与 17s 集体等待一致（17s 后 EP-6 收到首包并参与 all_gather，整体才继续）。

以上分析基于你提供的 decode EP16 全 rank profile，结论为：**头部 15s 来自一个 decode rank（EP-6）在等 prefill 首包而阻塞在 recv_multipart，其余 15 个 rank 在 DP all_gather 中等待该 rank，预热后首包延迟不再出现，故延迟消失。**
