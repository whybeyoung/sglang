# 双机 PP + CP Prefill 配置指南

基于 PR #13959 的优化，结合 Pipeline Parallelism (PP) 和 Context Parallelism (CP) 实现双机 prefill。

## 架构概览

### 配置示例：PP2 TP8 CP8（双机）

- **总 GPU 数**: 16 (每机 8 GPU)
- **PP size**: 2 (两个 pipeline stages)
- **TP size**: 8 (每个 PP stage 内 8 个 TP ranks)
- **CP size**: 8 (每个 PP stage 内 8 个 CP ranks，等于 atten_tp_size)

### 进程分布

**Node 0 (PP Stage 0)**:
- PP0 TP0-7: GPU 0-7
- 处理序列的前半部分

**Node 1 (PP Stage 1)**:
- PP1 TP0-7: GPU 0-7
- 处理序列的后半部分

## 启动命令

### Node 0 (PP Stage 0)

```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --tp-size 8 \
    --pp-size 2 \
    --dist-init-addr <node0_ip>:5000 \
    --nnodes 2 \
    --node-rank 0 \
    --enable-dp-attention \
    --enable-nsa-prefill-context-parallel \
    --nsa-prefill-context-parallel-size 8 \
    --moe-a2a-backend fused_moe \
    --moe-dense-tp-size 1 \
    --dp-size 1 \
    --chunked-prefill-size 16384 \
    --kv-cache-dtype bf16 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code
```

### Node 1 (PP Stage 1)

```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --tp-size 8 \
    --pp-size 2 \
    --dist-init-addr <node0_ip>:5000 \
    --nnodes 2 \
    --node-rank 1 \
    --enable-dp-attention \
    --enable-nsa-prefill-context-parallel \
    --nsa-prefill-context-parallel-size 8 \
    --moe-a2a-backend fused_moe \
    --moe-dense-tp-size 1 \
    --dp-size 1 \
    --chunked-prefill-size 16384 \
    --kv-cache-dtype bf16 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code
```

## 关键配置说明

### 1. CP 配置

```bash
--enable-nsa-prefill-context-parallel          # 启用 CP
--nsa-prefill-context-parallel-size 8          # CP size，等于 atten_tp_size (tp_size / dp_size)
```

**注意**:
- CP size 必须等于 `atten_tp_size = tp_size / dp_size`
- 当前代码中 CP size 默认使用 `atten_tp_size`，如果设置 `--nsa-prefill-context-parallel-size`，需要确保它等于 `atten_tp_size`

### 2. PP 配置

```bash
--pp-size 2                                    # Pipeline stages 数量
--nnodes 2                                     # 节点数量
--node-rank 0/1                                # 节点 rank
--dist-init-addr <node0_ip>:5000              # 主节点地址
```

### 3. MoE 配置（基于 PR #13959）

```bash
--moe-a2a-backend fused_moe                    # 使用 fused MoE（需要 dp-size=1）
--moe-dense-tp-size 1                          # Dense layer TP size
--dp-size 1                                    # 必须为 1（fused MoE 要求）
```

### 4. 其他优化参数

```bash
--chunked-prefill-size 16384                  # Chunked prefill 大小
--kv-cache-dtype bf16                          # KV cache 数据类型（或 fp8_e4m3）
--max-running-requests 128                     # 最大并发请求数
```

## PR #13959 的关键优化

### 1. 新的 Token Splitting Scheme (Mode 1)

PR 引入了新的 token splitting 方法，使用 `token_idx % cp_size` 来分配 tokens，相比原来的 zigzag 方式：
- ✅ 支持 multi-batch prefill
- ✅ 更好的负载均衡
- ✅ 兼容 fused MoE

**注意**: PR 中提到通过 `--nsa-prefill-cp-mode 1` 启用，但当前代码库中可能还未完全合并。如果未合并，可以使用现有的 CP 实现。

### 2. Fused MoE 支持

- 需要 `dp-size=1`
- 性能比 DeepEP 更好（PR 显示 TTFT 减少 8.9%-32%）

### 3. FP8 KV Cache 支持

```bash
--kv-cache-dtype fp8_e4m3                      # 使用 FP8 KV cache
```

## 工作流程

### Prefill 阶段

1. **PP Stage 0 (Node 0)**:
   - 接收请求
   - 将序列按 CP 方式分割（zigzag 或 mode 1）
   - 在 8 个 TP ranks 上并行处理前半部分序列
   - 通过 CP allgather 收集结果
   - 发送中间结果到 PP Stage 1

2. **PP Stage 1 (Node 1)**:
   - 接收来自 PP Stage 0 的中间结果
   - 继续处理后半部分序列
   - 通过 CP allgather 收集结果
   - 返回最终输出

### CP 通信模式

在每个 PP stage 内部：
- CP ranks 之间通过 allgather 通信
- 使用 `cp_all_gather_rerange_output` 重组输出
- CP group 独立于 TP group，但共享相同的设备

## 性能优化建议

### 1. 网络配置

```bash
# 设置 NCCL 环境变量优化跨节点通信
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网络接口调整
export NCCL_DEBUG=INFO          # 调试时启用
```

### 2. Chunked Prefill 配置

```bash
--chunked-prefill-size 16384    # 根据序列长度调整
```

对于长序列（>16K），chunked prefill 可以：
- 减少 PP bubbles
- 提高 GPU 利用率
- 降低内存峰值

### 3. 动态 Chunking（如果支持）

根据 PR #11852，SGLang 支持动态 chunking 来减少 PP bubbles：
- 使用二次函数拟合运行时间
- 动态调整 chunk size

## 验证和测试

### 1. 检查 CP 是否启用

```python
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
print(f"CP enabled: {is_nsa_enable_prefill_cp()}")
```

### 2. 检查 CP Group 初始化

```python
from sglang.srt.layers.attention.nsa.cp_group import get_cp_size, get_cp_rank
print(f"CP size: {get_cp_size()}, CP rank: {get_cp_rank()}")
```

### 3. 性能测试

```bash
python -m sglang.bench_serving \
    --model <model_path> \
    --base-url http://<node0_ip>:8000 \
    --dataset-name random \
    --num-prompts 100 \
    --random-input-len 16384 \
    --random-output-len 1 \
    --max-concurrency 1
```

## 已知限制

1. **Multi-batch prefill**: 
   - 原 CP 实现（zigzag mode）不支持 multi-batch
   - PR #13959 的 mode 1 支持 multi-batch，但需要确认是否已合并

2. **Cross-machine CP**: 
   - 文档提到目前主要测试在单机（TP=8, EP=8）
   - 双机部署需要验证 CP 通信是否正常工作

3. **DP size**: 
   - Fused MoE 要求 `dp-size=1`
   - 如果需要更大的 DP，需要使用 DeepEP backend

## 故障排除

### 1. CP Group 初始化失败

检查：
- `--enable-nsa-prefill-context-parallel` 是否设置
- `--nsa-prefill-context-parallel-size` 是否等于 `tp_size / dp_size`
- 分布式环境是否正确初始化

### 2. PP 通信失败

检查：
- `--dist-init-addr` 是否正确
- 防火墙是否开放相应端口
- NCCL 环境变量是否正确设置

### 3. 性能不佳

- 检查网络带宽和延迟
- 调整 `--chunked-prefill-size`
- 考虑使用 FP8 KV cache 减少通信量

## 实现 PR #13959 Mode 1 Token Splitting

如果 PR #13959 的 mode 1 还未完全合并，可以基于现有代码实现新的 token splitting scheme。

### Mode 1 实现思路

PR #13959 引入的新 token splitting 使用 `token_idx % cp_size` 来分配 tokens，相比 zigzag 方式：
- 更简单的实现
- 支持 multi-batch prefill
- 更好的负载均衡

### 实现步骤

1. **添加配置参数**（如果 PR 未合并）:

在 `server_args.py` 中添加：
```python
nsa_prefill_cp_mode: int = 0  # 0: zigzag (original), 1: token_idx % cp_size (new)
```

2. **实现 Mode 1 Token Splitting**:

在 `utils.py` 中添加新函数：
```python
def prepare_input_dp_with_cp_dsa_mode1(
    kv_len,
    cp_rank,
    cp_size,
    seqs_len,
    atten_tp_size=None,
    atten_tp_rank=None,
):
    """Mode 1: Simple token splitting using token_idx % cp_size
    
    This mode evenly distributes tokens across CP ranks:
    - Rank 0: tokens 0, cp_size, 2*cp_size, ...
    - Rank 1: tokens 1, cp_size+1, 2*cp_size+1, ...
    - ...
    - Rank cp_size-1: tokens cp_size-1, 2*cp_size-1, ...
    
    This supports multi-batch prefill naturally.
    """
    kv_len_origin = kv_len
    
    # Calculate tokens per rank (evenly distributed)
    tokens_per_rank = (kv_len + cp_size - 1) // cp_size
    max_rank_len = [tokens_per_rank] * cp_size
    
    # Adjust for remainder
    remainder = kv_len % cp_size
    if remainder > 0:
        # Distribute remainder tokens to first 'remainder' ranks
        for i in range(remainder):
            max_rank_len[i] = tokens_per_rank
        for i in range(remainder, cp_size):
            max_rank_len[i] = tokens_per_rank - 1
    
    # Calculate actual tokens for this rank
    per_rank_actual_token = [max_rank_len[i] for i in range(cp_size)]
    
    # For mode 1, split_list is simply the tokens assigned to each rank
    split_list = per_rank_actual_token.copy()
    
    # zigzag_index is not needed for mode 1 (sequential assignment)
    # But we keep it for compatibility with existing code
    zigzag_index = list(range(cp_size))
    
    # reverse_split_len is same as split_list for mode 1
    reverse_split_len = split_list.copy()
    cp_reverse_index = list(range(cp_size))
    
    # Calculate kv_len for this rank
    prefix_sum = sum(split_list[:cp_rank])
    kv_len_prev = prefix_sum
    kv_len_next = prefix_sum + split_list[cp_rank]
    actual_seq_q_prev = split_list[cp_rank]
    actual_seq_q_next = 0  # Not used in mode 1
    
    # Create metadata
    nsa_cp_metadata = NSAContextParallelMetadata(
        split_list=split_list,
        max_rank_len=max_rank_len,
        zigzag_index=zigzag_index,
        per_rank_actual_token=per_rank_actual_token,
        reverse_split_len=reverse_split_len,
        cp_reverse_index=cp_reverse_index,
        kv_len_prev=kv_len_prev,
        kv_len_next=kv_len_next,
        actual_seq_q_prev=actual_seq_q_prev,
        actual_seq_q_next=actual_seq_q_next,
        kv_len_prev_tensor=torch.tensor(kv_len_prev, device="cuda", dtype=torch.int32),
        kv_len_next_tensor=torch.tensor(kv_len_next, device="cuda", dtype=torch.int32),
        actual_seq_q_prev_tensor=torch.tensor(actual_seq_q_prev, device="cuda", dtype=torch.int32),
        actual_seq_q_next_tensor=torch.tensor(actual_seq_q_next, device="cuda", dtype=torch.int32),
        total_seq_lens=kv_len_origin,
    )
    return nsa_cp_metadata
```

3. **修改 deepseek_v2.py 使用 Mode 1**:

在 `forward` 方法中根据配置选择使用哪个函数：
```python
if can_split:
    if get_global_server_args().nsa_prefill_cp_mode == 1:
        forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa_mode1(
            torch.tensor(len(input_ids)),
            self.cp_rank,
            self.cp_size,
            forward_batch.seq_lens_cpu.tolist(),
            atten_tp_size=atten_tp_size if (cp_size_config is not None and cp_size_config != atten_tp_size) else None,
            atten_tp_rank=atten_tp_rank if (cp_size_config is not None and cp_size_config != atten_tp_size) else None,
        )
    else:
        forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(
            # ... existing code ...
        )
```

## 参考

- PR #13959: [DeepSeek v3.2] opt Context Parallelism
- PR #11852: Pipeline Parallelism with async communication
- PR #12065: Context Parallelism design reference




