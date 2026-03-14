# PR #15805 导致 Decode 节点 Health 探针失败分析

## 问题概述

PR #15805 ([HiCache] Fix deadlock when creating new group) 引入了一个新的函数 `create_custom_parallel_group` 来替代直接调用 `torch.distributed.new_group`，但这个改动在 PD（Prefill-Decode）分离模式下会导致 decode 节点的 health 探针失败。

## 根本原因

### 1. PR 变更内容

**文件**: `python/sglang/srt/distributed/parallel_state.py`

新增函数 `create_custom_parallel_group`:
```python
def create_custom_parallel_group(
    group_ranks: List[int], backend: str = "gloo"
) -> Optional[torch.distributed.ProcessGroup]:
    ...
    torch.distributed.all_gather_object(gathered_configs, local_config)  # 第1758行
    ...
```

**文件**: `python/sglang/srt/managers/cache_controller.py`

将原来的：
```python
self.prefetch_tp_group = torch.distributed.new_group(
    group_ranks, backend="gloo"
)
```

替换为：
```python
self.prefetch_tp_group = create_custom_parallel_group(
    group_ranks=group_ranks, backend="gloo"
)
```

### 2. 问题分析

#### 2.1 关键问题：`all_gather_object` 在默认 world group 上执行

在 `create_custom_parallel_group` 函数的第1758行：
```python
torch.distributed.all_gather_object(gathered_configs, local_config)
```

**这个调用没有指定 `group` 参数**，意味着它在**默认的 world group** 上执行。`all_gather_object` 是一个集体通信操作，需要**所有 world 进程**都参与。

#### 2.2 PD 分离模式下的问题场景

在 PD（Prefill-Decode）分离模式下：

1. **Prefill 节点**和 **Decode 节点**可能是分开的进程组，或者在不同的 world group 中
2. 当 **Decode 节点**初始化 `HiCacheController` 时（在 `DecodeKVCacheOffloadManager.__init__` 中），会调用 `create_custom_parallel_group`
3. 这个函数会调用 `all_gather_object`，需要**所有 world 进程**参与
4. 如果：
   - Decode 节点和 Prefill 节点在不同的 world group 中
   - Decode 节点启动时 Prefill 节点还没有准备好
   - Decode 节点调用时 Prefill 节点没有调用相同的函数
   
   就会导致**死锁或 hang**，因为 `all_gather_object` 会一直等待所有 world 进程参与。

#### 2.3 Health 探针失败的原因

1. Decode 节点在初始化过程中调用 `create_custom_parallel_group`
2. 函数在 `all_gather_object` 处 hang 住，等待所有 world 进程
3. 初始化过程无法完成，进程被阻塞
4. Health 探针（通常是 gRPC health check）超时失败
5. Kubernetes 认为节点不健康，可能导致重启或流量切换

### 3. 原始实现 vs PR 后的实现

#### 原始实现（PR 前）
```python
self.prefetch_tp_group = torch.distributed.new_group(
    group_ranks, backend="gloo"
)
```

**特点**：
- 只需要 `group_ranks` 中的进程参与
- 不需要所有 world 进程参与
- 在 PD 分离模式下，Decode 节点可以独立创建自己的 group，不需要 Prefill 节点参与

#### PR 后的实现
```python
self.prefetch_tp_group = create_custom_parallel_group(
    group_ranks=group_ranks, backend="gloo"
)
```

**特点**：
- `create_custom_parallel_group` 内部调用 `all_gather_object`（无 group 参数）
- 需要在**默认 world group** 上的所有进程参与
- 在 PD 分离模式下，如果 Prefill 和 Decode 节点不在同一个 world group，会导致死锁

### 4. 调用链分析

```
DecodeKVCacheOffloadManager.__init__
  └─> HiCacheController.__init__
      └─> create_custom_parallel_group (line 321 in cache_controller.py)
          └─> torch.distributed.all_gather_object (line 1758 in parallel_state.py)
              └─> [HANG] 等待所有 world 进程参与
```

### 5. 为什么 PR 要引入这个函数？

根据 PR 描述 "[HiCache] Fix deadlock when creating new group"，这个 PR 的目的是修复创建新 group 时的死锁问题。

**可能的原始问题**：
- 当多个进程同时调用 `torch.distributed.new_group` 创建不同的 group 时，可能会因为 group 创建的顺序或同步问题导致死锁
- `create_custom_parallel_group` 通过先收集所有进程的配置，然后统一创建 group 来避免这个问题

**但引入的新问题**：
- `all_gather_object` 在默认 world group 上执行，需要所有 world 进程参与
- 在 PD 分离模式下，这会导致 Decode 节点等待 Prefill 节点，造成死锁

## 解决方案建议

### 方案 1：在 `all_gather_object` 中指定 group 参数

修改 `create_custom_parallel_group` 函数，使用传入的 `group_ranks` 对应的 group 来执行 `all_gather_object`：

```python
def create_custom_parallel_group(
    group_ranks: List[int], backend: str = "gloo"
) -> Optional[torch.distributed.ProcessGroup]:
    ...
    # 获取当前 rank 所在的 group（如果存在）
    # 或者使用一个更小的 group 来执行 all_gather_object
    # 需要确保所有调用此函数的进程都在同一个 group 中
    ...
```

**问题**：需要确定使用哪个 group 来执行 `all_gather_object`。

### 方案 2：回退到原始实现，但添加同步机制

如果原始实现的问题不是致命的，可以考虑回退，但添加适当的同步机制来避免死锁。

### 方案 3：条件性使用新函数（推荐）

只在特定条件下使用 `create_custom_parallel_group`，在 PD 分离模式下使用原始实现。

**修改 `cache_controller.py`**:

```python
# 在 HiCacheController.__init__ 中
if self.tp_world_size > 1:
    from sglang.srt.distributed.parallel_state import (
        create_custom_parallel_group,
    )
    from sglang.srt.utils import get_global_server_args

    group_ranks = torch.distributed.get_process_group_ranks(tp_group)
    
    # 检查是否是 PD 分离模式下的 decode 节点
    server_args = get_global_server_args()
    if server_args and server_args.disaggregation_mode == "decode":
        # PD 分离模式下，decode 节点使用原始实现，避免依赖 world group
        self.prefetch_tp_group = torch.distributed.new_group(
            group_ranks, backend="gloo"
        )
    else:
        # 其他情况使用新函数，避免死锁
        self.prefetch_tp_group = create_custom_parallel_group(
            group_ranks=group_ranks, backend="gloo"
        )
```

**优点**：
- 最小化改动
- 保持 PR 的原始目的（修复非 PD 模式下的死锁）
- 在 PD 模式下使用原始实现，避免新问题

### 方案 4：修复 `create_custom_parallel_group` 以支持 PD 模式

修改 `create_custom_parallel_group` 函数，使其能够正确处理 PD 分离模式。

**方案 4a：使用传入的 group_ranks 对应的 group**

需要修改函数签名，传入一个用于同步的 group：

```python
def create_custom_parallel_group(
    group_ranks: List[int], 
    backend: str = "gloo",
    sync_group: Optional[torch.distributed.ProcessGroup] = None
) -> Optional[torch.distributed.ProcessGroup]:
    """
    Create a custom parallel group based on the provided ranks.
    
    Args:
        group_ranks: The list of ranks that the CURRENT process wants to join.
        backend: The communication backend (default: "gloo").
        sync_group: Optional group to use for synchronization. If None, uses default world group.
    """
    assert torch.distributed.is_initialized()
    
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    local_config = sorted(list(set(group_ranks)))
    gathered_configs = [None for _ in range(world_size)]
    
    # 使用 sync_group 或默认 world group
    if sync_group is not None:
        # 使用指定的 group 进行同步
        sync_world_size = torch.distributed.get_world_size(group=sync_group)
        gathered_configs = [None for _ in range(sync_world_size)]
        torch.distributed.all_gather_object(
            gathered_configs, local_config, group=sync_group
        )
    else:
        # 使用默认 world group（原始行为）
        torch.distributed.all_gather_object(gathered_configs, local_config)
    
    # ... 后续逻辑相同
```

**方案 4b：检测 PD 模式并自动选择同步 group**

在函数内部检测 PD 模式：

```python
def create_custom_parallel_group(
    group_ranks: List[int], backend: str = "gloo"
) -> Optional[torch.distributed.ProcessGroup]:
    ...
    # 检测是否是 PD 分离模式
    try:
        from sglang.srt.utils import get_global_server_args
        server_args = get_global_server_args()
        is_pd_decode = (
            server_args is not None 
            and server_args.disaggregation_mode == "decode"
        )
    except:
        is_pd_decode = False
    
    if is_pd_decode:
        # PD 模式下，使用 tp_group 进行同步（如果可用）
        # 或者直接使用原始实现
        # 这里需要传入 tp_group，可能需要修改函数签名
        ...
    else:
        # 非 PD 模式，使用 world group
        torch.distributed.all_gather_object(gathered_configs, local_config)
    ...
```

**问题**：方案 4 需要修改函数签名或增加参数，可能影响其他调用者。

## 验证方法

1. **复现问题**：
   - 在 PD 分离模式下启动 Decode 节点
   - 观察初始化过程是否 hang 住
   - 检查 health 探针是否失败

2. **验证修复**：
   - 应用修复后，在 PD 分离模式下启动 Decode 节点
   - 确认初始化能够正常完成
   - 确认 health 探针能够成功

## 相关代码位置

- `python/sglang/srt/distributed/parallel_state.py`:1736-1782 (create_custom_parallel_group)
- `python/sglang/srt/managers/cache_controller.py`:313-323 (HiCacheController.__init__)
- `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py`:70-80 (DecodeKVCacheOffloadManager.__init__)
- `python/sglang/srt/grpc/health_servicer.py`:78-161 (Health check implementation)

## 总结

PR #15805 引入的 `create_custom_parallel_group` 函数在 PD 分离模式下会导致 Decode 节点 hang 住，原因是 `all_gather_object` 在默认 world group 上执行，需要所有 world 进程参与，但在 PD 分离模式下，Decode 节点和 Prefill 节点可能不在同一个 world group 中，导致死锁和 health 探针失败。

建议的修复方向是修改 `create_custom_parallel_group` 函数，使其能够在 PD 分离模式下正确工作，或者条件性地使用原始实现。

