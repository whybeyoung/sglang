# Eagle draft extend 融合优化方向

eagle_draft_extend 在 decode profile 中约占 10%，可与下面几类融合配合做优化。

---

## 1. Replay 前多段 copy 融合（Python 侧 → buffer）

**位置**: `eagle_draft_extend_cuda_graph_runner.py` 的 `replay()`（约 443–460、475 行）

**现状**: 每次 replay 前对 graph buffer 做多次 `.copy_()`：

- `input_ids`, `seq_lens`, `extend_seq_lens`, `out_cache_loc`, `positions`, `hidden_states`, `accept_length`, `req_pool_indices`, `seq_lens_cpu` 等，每项一次 kernel launch。

**融合思路**:

- **方案 A**：若这些 tensor 在 `forward_batch` 里连续或可拼成少量大块，先在一个 kernel 里 pack 到一块 buffer，再 1～2 次 `copy_` 进 graph buffer，减少 launch 次数。
- **方案 B**：写一个 Triton/CUDA 的 “multi-copy” 或 “pack_forward_batch” kernel：按固定 layout 从 `forward_batch` 的多个字段一次读、一次写进 `buffers` 的对应位置，替代多段 `.copy_()`。
- **方案 C**：若部分字段可指向同一块显存（alias），用 `view`/`narrow` 避免重复 copy。

目标：把 replay 前的多次 copy 收敛成 1～2 次 kernel 或 1 次 fused pack。

---

## 2. Graph 内 softmax + topk 融合

**位置**: 同一文件里 `capture_one_batch_size` 的 `run_once()`（约 398–404 行）：

```python
ret = self.model_runner.model.forward(...)
probs = torch.softmax(ret.next_token_logits, dim=-1)
ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)
```

**现状**: 先对 logits 做 softmax 得到完整 `probs`，再对 `probs` 做 topk；两次 kernel，且 probs 整块写回显存。

**融合思路**:

- 实现 **fused_softmax_topk**：输入 logits，输出 (topk_p, topk_index)，在 kernel 内做 softmax 后只写出 top-k 的值和下标，不写完整 probs。
- 若已有 `sgl_kernel.fast_topk`，可在此基础上加 “from_logits” 路径，或单独加 `fused_softmax_topk(logits, topk)`，在 graph 里用其替代 `softmax` + `fast_topk`。
- 这样 graph 内少一次 kernel、少一次大 tensor 的显存读写，有利于延迟和带宽。

---

## 3. init_forward_metadata_replay_cuda_graph 已用 Triton

**位置**: `nsa_backend.py` 的 `init_forward_metadata_replay_cuda_graph`（约 988、1020 行）

**现状**: 已使用 `seqlens_expand_triton`，与 init_forward_meta 的优化一致，此处无需再为 seqlens 做融合。

若 profile 里该函数仍占比较高，可再查是否还有其它 Python 循环、`.tolist()` 或小 tensor 构造，能挪到 GPU 的用 Triton 写成一两个 kernel。

---

## 4. 实施顺序建议

1. **先做 (2) 图内 fused_softmax_topk**：改 graph 内逻辑即可，收益直接、风险相对可控。
2. **再做 (1) replay 前 copy 融合**：需要理清 `forward_batch` 与 `buffers` 的 layout，再实现 pack 或 multi-copy kernel。
3. **(3)** 已优化，仅当新 profile 显示该路径仍占主导时再深挖。

---

## 5. 参考

- NSA 元数据优化（seqlens）：[PR #19536](https://github.com/sgl-project/sglang/pull/19536)（commit 80a6b32），`seqlens_expand_triton` 在 `layers/attention/utils.py`。
- Eagle draft extend 入口：`eagle_worker.py` / `eagle_worker_v2.py` 的 `forward_draft_extend`、`forward_draft_extend_after_decode`；图逻辑在 `eagle_draft_extend_cuda_graph_runner.py`。
