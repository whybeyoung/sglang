---
name: analyze-decode-profile
description: Step-by-step guide for analyzing decode-stage profiles (PyTorch profiler trace). Use when the user provides a directory of trace files (e.g. .trace.json.gz), wants to find decode hotspots, or asks for EP/DP/TP decode optimization suggestions.
---

# 分析 Decode 阶段 Profile

当用户提供 decode 的 profile 目录（如 PyTorch profiler 导出的 `.trace.json.gz`），或询问 decode/EP/DP 的算子优化空间时，按本 skill 执行：解析 trace、按算子汇总耗时、给出优化优先级与建议。

## 1. 确认输入

- **目录**：用户给出的路径，例如 `.../decode_youhua/1772362500.6322927/`
- **文件**：通常为按 rank 命名的 `*TP-*-DP-*-EP-*.trace.json.gz` 或类似
- **格式**：Chrome trace JSON（PyTorch profiler 导出），`traceEvents` 中 `ph=="X"` 为区间事件，含 `name`、`dur`（微秒）

先列出目录与单个文件大小，确认是否有多个 rank 的 trace：

```bash
ls -la <PROFILE_DIR>/
find <PROFILE_DIR> -name "*.trace.json.gz" -type f
```

## 2. 解析并汇总（按事件 name 聚合）

对**单个 rank** 的 trace 做解析，按 `name` 聚合 `dur` 与 `count`，按 `dur` 降序得到热点。

可用项目内脚本：

```bash
python scripts/analyze_decode_profile_trace.py <PROFILE_DIR> [--rank 0] [--top 80] [--gpu-only]
# 汇总所有 rank 的算子耗时，得到全局各算子时间占比：
python scripts/analyze_decode_profile_trace.py <PROFILE_DIR> --all-ranks [--top 100] [--gpu-only]
```

## 2.1 分析各个算子时间占比

- **含义**：对每个算子（事件 `name`）汇总其 `dur` 和 `count`，用 **总 dur** 作分母算 **占比(%)**，按占比降序排列。
- **输出表格**：至少包含列「占比%」「耗时(s)」「调用次数」「算子/模块名」；可选「类型」（拷贝/GEMM/通信/attention 等）。
- **单 rank**：只解析一个 trace 文件，总 dur = 该文件内所有事件 dur 之和（注意事件可能并发，总和 > wall-clock）。
- **多 rank 汇总**：使用 `--all-ranks` 时，脚本会对目录下所有 `*.trace.json.gz` 按 name 累加 dur 和 count，再算全局占比，便于看「整次 run 中哪些算子最吃时间」。
- **GPU/通信子集**：加 `--gpu-only` 时，只保留 GPU kernel 与通信相关事件，总 dur 仅含这部分，占比为「占 GPU/通信时间的比例」。

或内联 Python 一次性跑（替换 `TRACE_FILE` 为实际路径）：

```bash
zcat TRACE_FILE | python3 -c "
import json, sys
from collections import defaultdict
d = json.load(sys.stdin)
events = d.get('traceEvents', [])
by_name = defaultdict(lambda: {'dur': 0, 'count': 0})
for e in events:
    if e.get('ph') != 'X': continue
    name, dur = e.get('name', ''), e.get('dur', 0)
    if not name or dur <= 0: continue
    by_name[name]['dur'] += dur
    by_name[name]['count'] += 1
sorted_list = sorted(by_name.items(), key=lambda x: -x[1]['dur'])
for name, v in sorted_list[:80]:
    print(f\"{v['dur']/1e6:.3f}s  cnt={v['count']}  {name[:90]}\")
"
```

- **dur 单位**：trace 里一般为**微秒**，汇总后除以 1e6 得到秒。
- **注意**：总 duration 是各事件 duration 之和，事件可能并发，故总和大于 wall-clock；**占比和排序**仍可用来找热点。

## 3. 仅看 GPU/通信（可选）

若需排除纯 Python/CPU，只保留 GPU kernel 与通信，可在聚合后按名称过滤，或在上面的脚本中加 `--gpu-only`。典型规则：

- 保留：`void `、`cuda`、`aten::`、`deep_gemm`、`deep_ep`、`flash_`、`nvjet`、`nccl`、`FusedAddRMSNorm`、`elementwise`、`sbtopk`、`quant_`、`_kernel` 等
- 排除：`threading`、`socket`、`watchdog`、`_bootstrap` 等

然后按 `dur` 排序，并计算占「GPU/通信总 dur」的百分比，便于写报告。

## 4. 写分析报告（Markdown）

在用户指定的目录或项目 `docs/` 下生成一份分析文档，建议结构：

1. **Profile 数据说明**：路径、设备、trace 格式、rank 数、场景（如 Eagle + MoE decode）
2. **各算子时间占比表**：算子/模块名、**占比(%)**、耗时(约)、调用次数、类型（拷贝/GEMM/通信/attention 等）；按占比降序
3. **优化方向与算子级建议**：按优先级分块（高/中/低），每块包含：
   - 现象（哪些算子、占比）
   - 建议（减拷贝、重叠通信、复用元数据、调 GEMM shape、减 sync 等）
4. **建议落地顺序**：按收益与改动量排序的 3–5 条
5. **备注**：多 rank 时建议对比各 rank 是否均衡；若需更准的 GPU 时间可建议用 nsys 再抓

可参考项目内已有示例：`docs/` 或用户提供的 `GLM5_Decode_EP16_DP16_Profile_Analysis.md` 结构。

## 5. 常见热点与优化对应（速查）

| 热点 | 可能原因 | 优化方向 |
|------|----------|----------|
| `aten::copy_` / `aten::to` / `aten::_to_copy` | 多余设备/类型转换、拷贝 | 减少 to(device)/contiguous，buffer 复用，热路径避免 .cpu()/.tolist() |
| `tensor.tolist()` / `.numpy()` | 热路径 D2H | 移出热路径或异步化 |
| `nsa_backend.init_forward_meta` | 每步重建元数据 | 复用/增量更新或移入 graph |
| Eagle draft extend / cuda graph 相关 | 图回放或 Python 准备 | 区分捕获 vs 回放，减 Python、合并 step |
| `cudaMemcpyAsync` | D2H/H2D/D2D | nsys 看方向，消除或与计算重叠 |
| `cudaStreamSynchronize` | 多余同步 | 减少 sync 点，通信与计算重叠 |
| `deep_ep::dispatch` / `combine` | MoE EP 通信与重排 | 与 GEMM 重叠，检查负载均衡 |
| `deep_gemm::sm90_fp8_gemm_*` | FP8 GEMM 多种 shape | shape 调优、与 quant 融合 |
| `nccl:all_gather` 等 | DP/EP collective | 与计算重叠，减少调用次数 |
| `flash_fwd_*` / `FusedAddRMSNorm` | attention、norm | 占比低时可后置，优先上面几类 |

## 6. 多 rank 对比（可选）

若目录中有多个 rank 的 trace，可对 2–3 个 rank 各跑一遍上述聚合，对比同一算子在不同 rank 上的 `dur` 与 `count`，判断是否存在负载不均（例如某 rank 的 dispatch/GEMM 明显更长）。

---

**使用本 skill 时**：先确认用户提供的 profile 路径与格式，再执行步骤 2–3 得到热点列表，最后按步骤 4–5 生成分析报告并给出算子级优化建议。
