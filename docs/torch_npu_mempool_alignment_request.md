# 【特性请求】为 `torch_npu.npu.MemPool` 增加 sub-block 对齐参数(可选)

## 摘要

希望为 `torch_npu.npu.MemPool` / `NPUPluggableAllocator` 增加一个可选参数(例如 `sub_block_alignment`),使所有在 `with torch_npu.npu.use_mem_pool(pool):` 作用域内分配的 tensor,其 `data_ptr()` 都能严格对齐到调用方指定的字节边界(典型场景为 2 MB,即 CANN HCCL 的页粒度)。

当前 `MemPool` 接口只允许替换底层 `aclrtMalloc` / pluggable allocator,但**无法影响 NPUCachingAllocator 在其之上的子块切分策略**。caching allocator 会以 256B / 512B 等小粒度从已分配的大块中切片返回,导致单个 tensor 的 `data_ptr()` 几乎不可能自然对齐到 2 MB 等大边界。

## 背景与动机

### 业务场景:CANN HCCL IPC RMA 注册

CANN HCCL 在导出 IPC RMA 缓冲(`rtsIpcMemGetExportKey → halShmemCreateHandle`)时,要求每一个注册的 `(ptr, len)` 都必须以 2 MB 页边界(`page_size = 2097152`)起始,否则随后的 ADXL `Connect` 会以下面这类不易排查的错误失败:

```
[DRV] Invalid para. va=0x12d6371b8000 page_size=2097152
      Create_para_check fail.
[DRV] Ipc node attr pack fail. len=2479489024
[GE]  Failed to connect, ErrorNo: 503900
```

这一约束会影响所有使用 HCCL IPC RMA 做跨节点 KV 传输 / PD 分离推理的框架(sglang、vllm、基于 Mooncake 的部署等)。

### 仅靠底层 allocator 不能解决

我们尝试过通过 `torch_npu.npu.NPUPluggableAllocator` 绑定 `mooncake.allocator.AscendAllocator`(其底层调用 `aclrtMalloc(ACL_MEM_MALLOC_HUGE_ONLY)`)。**底层 raw block 的基址确实是 2 MB 对齐**,但在 `use_mem_pool` 作用域内单次 `torch.zeros(...)` 是由其上方的 NPUCachingAllocator 服务的,后者会把这块 2 MB raw block 切成任意偏移的子块。最终 tensor `data_ptr()` 几乎都不对齐,例如:

```
0x12d6371b8000 % 2MB = 0x1b8000   (不是 2MB 倍数)
```

调用链如下:

```
torch.zeros(...)                       ← 应用层
    ↓
torch_npu.NPUCachingAllocator         ← 负责切分 / 复用 / 返回 data_ptr()
    ↓
mooncake mc_ascend_malloc             ← 只提供 2MB 对齐的 raw block
    ↓
aclrtMalloc(ACL_MEM_MALLOC_HUGE_ONLY)
```

caching allocator 处于 `torch.zeros` 与 `NPUPluggableAllocator` 之间,**框架层目前没有任何公开 API 可以影响它的子块对齐策略**。

### 框架侧已有的几种 workaround 及其问题

| Workaround | 主要问题 |
|---|---|
| 在框架侧 over-allocate `nbytes + 2MB`,前移到下一个 2MB 边界返回对齐 view(sglang `_npu_align.zeros` 的做法) | 每个 tensor 浪费最多 2 MB;每个框架都得自己实现一遍;per-layer KV 池每层都要付 2 MB padding |
| 预分配一块 2MB padded 的大 buffer,所有字段做 interior view,只注册 base | 破坏了 Mooncake 等组件按 `(ptr, len)` per-field 注册的协议契约,需要改协议 |
| Mooncake 暴露 raw allocator API,框架用 `torch_npu.from_blob` 包装 raw ptr,完全绕过 caching allocator | `from_blob` / dlpack capsule 在 torch_npu 各版本兼容性不稳定;tensor 生命周期(custom deleter)实现脆弱 |
| 全局禁用 caching | 性能严重退化;失去 caching 本身的意义 |

如果 `MemPool` 能原生提供一个可选的对齐参数,以上所有 workaround 都可以下线,框架只需要在创建注册路径专用的 `MemPool` 时声明"这个池里出来的 alloc 都是对齐的",**既不影响默认全局 allocator,也不强制框架自己实现对齐逻辑**。

## 建议的 API

### 方案 1(推荐):为 `MemPool` 增加参数

```python
import torch_npu
from mooncake.allocator import AscendAllocator

allocator = AscendAllocator.get_allocator(torch.device("npu:0"))

# 新增参数:从该池服务的每一次子块分配,起始地址都对齐到 sub_block_alignment;
# caching 层在切分 / 复用 slab 时按对齐边界圆整。
pool = torch_npu.npu.MemPool(
    allocator.allocator(),
    sub_block_alignment=2 * 1024 * 1024,
)

with torch_npu.npu.use_mem_pool(pool):
    # data_ptr() 保证 2 MB 对齐,且与 size 无关
    x = torch.zeros((1024,), dtype=torch.int32, device="npu:0")
    assert x.data_ptr() % (2 * 1024 * 1024) == 0
```

### 方案 2:为底层 allocator 构造器增加参数

```python
allocator = torch_npu.npu.NPUPluggableAllocator(
    so_path,
    "mc_ascend_malloc",
    "mc_ascend_free",
    sub_block_alignment=2 * 1024 * 1024,
)
```

### 期望语义

- 所有从该池返回的子块都满足 `ptr % sub_block_alignment == 0`。
- caching 层只允许在某个 free block 自身也满足对齐时,从该 free block 中切出新分配。
- `free()` 后归还到对齐感知的 caching 池,后续仅供同样对齐要求的请求复用。
- 当底层 allocator 返回的 base 本身已经满足对齐(例如 `aclrtMalloc(HUGE_ONLY)`),无额外的过分配开销。
- **未传入新参数时,行为与今天完全一致**,不破坏现有用户。

## 验收标准

- 接受新参数,默认行为不变。
- 设置参数后,`with use_mem_pool(pool):` 作用域内分配的任何 tensor 都满足 `data_ptr() % sub_block_alignment == 0`。
- 未启用该参数的工作负载没有可测的性能回归。
- 文档中提供面向 CANN HCCL IPC RMA 场景的示例。

## 其它候选方案与权衡

- **在 `torch.zeros / torch.empty` 上提供 per-call 对齐 hint**:侵入大,每个调用点都要传参,容易遗漏。
- **全局环境变量强制 NPU alloc 全部对齐**:无差别浪费大量内存,影响无关分配。
- **由各框架自己 self-align**(当前现状):可以工作,但是多个框架重复实现、各自维护,且每个 tensor 多花 ≤ 2 MB padding。

## 参考资料

- CANN HCCL IPC RMA 注册路径:`rtsIpcMemGetExportKey` → `halShmemCreateHandle`,要求 `va % page_size == 0` 且 `len % page_size == 0`。
- Mooncake `AscendAllocator`(底层 raw block 通过 `aclrtMalloc(ACL_MEM_MALLOC_HUGE_ONLY)` 保证 2 MB 对齐):
  https://github.com/kvcache-ai/Mooncake/blob/fix_hccl_2m/mooncake-transfer-engine/ascend-allocator/ascend_allocator.cpp
- sglang 框架侧 workaround(本特性若落地可移除):
  `python/sglang/srt/disaggregation/_npu_align.py`
