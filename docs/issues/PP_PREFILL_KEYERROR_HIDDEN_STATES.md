# PP Prefill KeyError: 'hidden_states'

## 现象

在 **PP (Pipeline Parallelism) + PD (Prefill-Decode) disaggregation** 的 prefill 阶段，非首 rank 报错：

```text
KeyError: 'hidden_states'
```

调用栈涉及：
- `scheduler_pp_mixin.py`：`event_loop_pp_disagg_prefill`
- `deepseek_v2.py`：`forward` 中 `hidden_states = pp_proxy_tensors["hidden_states"]`
- `forward_batch_info.py`：`PPProxyTensors.__getitem__` 中 `return self.tensors[key]`

即：当前 rank 从上一 PP 阶段 **recv** 得到的 `pp_proxy_tensors` 里没有 `"hidden_states"` 这个 key。

## 根因（通信顺序/配对错误，与近期改动相关）

**相关 commit**：**c6306a81c**（fix: support PP2+CP8+TP8）中，将 PP 的 pyobj 收发自仅 **TP0** 收窄为 **TP0+CP0**（`_pp_send_pyobj_to_next_stage` / `_pp_recv_pyobj_from_prev_stage` 仅在 `attn_tp_rank==0 and attn_cp_rank==0` 时执行，再对 TP/CP 做 broadcast）。

在 **recv_requests()** 里：
- 非首 PP 时，只有 TP0+CP0 通过 `point_to_point_pyobj` 拿到 `recv_reqs`，其余 rank 为 `None`。
- **enable_dp_attention 为 True** 时，会对 TP 和 CP 都做 broadcast，各 rank 的 `recv_reqs` 一致。
- **enable_dp_attention 为 False** 时，原先只有 `elif self.tp_size != 1` 对 **TP** 做 broadcast，**没有对 CP 做 broadcast**。因此非 CP0 的 rank（如 CP1..CP7）的 `recv_reqs` 一直为 `None`，`get_new_batch_prefill()` 与 CP0 不一致，**mbs\[mb_id\] / cur_batch 在不同 CP rank 上不同**。
- 后果：上一 PP 阶段只有“有 batch 的 rank”会 send proxy tensor，下一阶段每个 rank 都按自己的 `cur_batch` 去 recv，导致 **proxy tensor 的 send 次数与 recv 次数、顺序错配**，某次 recv 拿到的是别的 mb 或别的类型的 dict，从而缺少 `"hidden_states"` → KeyError。

**其它可能原因**：上一阶段发送的 dict 结构异常；非 CP 场景下 mbs 因其它原因不同步导致配对错乱。

## 已做修改

### 1. 修复 PP+CP 时 recv_reqs 未向 CP 广播（根因修复）

**`scheduler.py` — `recv_requests()`**  
在 **未** 使用 `enable_dp_attention` 且 `attn_cp_size > 1` 时，对 `recv_reqs` 增加 **CP broadcast**（与 enable_dp_attention 路径中已有 TP+CP broadcast 一致），保证所有 CP rank 拿到相同 `recv_reqs`，从而 `get_new_batch_prefill()` 与 proxy tensor 的 send/recv 顺序一致，避免 KeyError。

### 2. 防御性检查与错误信息（便于后续排查）

- **`forward_batch_info.py` — `PPProxyTensors.__getitem__`**  
  当 key 不存在时，抛出带**当前实际 keys** 的 KeyError，并提示可能是 PP 各阶段 microbatch 不同步导致。

- **`deepseek_v2.py` — 非首 rank 使用 `pp_proxy_tensors` 前**  
  若 `"hidden_states"` 不在 `pp_proxy_tensors.tensors` 中，先抛出一个明确 KeyError，并列出当前收到的 keys。

下次再出现同类错误时，日志里会看到 **Available keys: [...]**，便于区分是“发错结构”还是“recv 错批/错序”。

## 建议排查步骤

1. **确认复现条件**：是否仅在 PP + PD prefill、特定 batch 或特定 mb 上出现；是否与 `pp_async_batch_depth`、`pp_loop_size` 等配置相关。
2. **看新日志中的 Available keys**：若 keys 为空或为其它名字（如 `logits`），可判断是“上一阶段发了别的结构”或“recv 到了别的消息”。
3. **核对 PP 各 rank 的 mbs 是否一致**：在 prefill 调度里，各 rank 的 `mbs[mb_id]` 是否在同一轮、同一 mb 上一致有 batch；若上一 rank 某 mb 为 None 而当前 rank 同 mb 有 batch，易出现“当前 rank recv 到非预期消息”。
4. **检查上一阶段是否有异常/提前 return**：确认上一 rank 在对应 mb 上是否正常执行到 `return PPProxyTensors({"hidden_states": ..., "residual": ...})`，且该返回值被正确用于 `_pp_send_dict_to_next_stage`。

## 相关代码

- 非首 rank 取数：`python/sglang/srt/models/deepseek_v2.py`（forward 中 `pp_proxy_tensors["hidden_states"]`）。
- PP proxy 容器：`python/sglang/srt/model_executor/forward_batch_info.py`（`PPProxyTensors`）。
- PP prefill 事件循环与 recv/send：`python/sglang/srt/managers/scheduler_pp_mixin.py`（`event_loop_pp_disagg_prefill`、`_pp_recv_proxy_tensors`、`_pp_send_dict_to_next_stage`）。
