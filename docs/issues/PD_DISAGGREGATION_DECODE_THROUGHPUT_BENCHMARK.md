# PD 分离后单独测 Decode 吞吐（含 100k 输入）

PD（prefill-decode 分离）下，prefill 和 decode 在不同节点。要评估「输入 100k context 时 decode 节点的吞吐」，可以按下面方式测。

---

## 1. 用 bench_serving 打 PD 路由，再从结果里拆出 decode 阶段

请求走**完整链路**（router → prefill → 转交 → decode），用固定 100k 输入 + 固定生成长度，从返回的 **TTFT** 和 **ITL** 反推 decode 阶段耗时与吞吐。

### 1.1 启动 PD 环境

- 已起好 prefill 服务、decode 服务、以及 PD router（或 LB）。
- 记下对外入口 URL，例如 `http://<router_host>:8000`。

### 1.2 跑 bench_serving（100k 输入）

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://<router_host>:8000 \
  --dataset-name random \
  --num-prompts 200 \
  --random-input-len 100000 \
  --random-output-len 512 \
  --random-range-ratio 0 \
  --request-rate inf \
  --output-details
```

- `--random-input-len 100000`：每个请求输入约 100k tokens（走 prefill 再转到 decode）。
- `--random-output-len 512`：每个请求生成 512 tokens，decode 阶段占主导。
- `--output-details`：会写 jsonl，里面带 `ttfts`、`itls`、`output_lens` 等，便于算 decode 吞吐。

若模型最大长度是 128k，可把 100000 改成 128000。

### 1.3 从结果里算「decode 阶段吞吐」

两种算法（二选一或都报）：

**方法 A：用 ITL（inter-token latency）**

- 每个请求的 decode 阶段：从第 2 个 token 到最后一个 token，每两个相邻 token 之间有一个 itl。
- 对单请求：`decode_throughput_req = (len(itl)) / sum(itl)`（tokens/s）。
- 对所有请求取平均：**decode 吞吐 ≈ mean( (len(itl))/sum(itl) )**，或等价地 **1 / mean(per-request mean(itl))**。

**方法 B：用 TTFT 和总 latency**

- 单请求：decode 时间 ≈ `latency - ttft`，decode 生成 token 数 ≈ `output_len - 1`。
- 单请求 decode 吞吐 = `(output_len - 1) / (latency - ttft)`。
- 对所有请求取平均：**decode 吞吐 ≈ mean( (output_len-1) / (latency - ttft) )**。

注意：`output_throughput` 是**整条链路**（从请求发出到收完最后 token）的总输出 token 吞吐，包含 prefill+传输+decode；上面算的是**仅 decode 阶段**的吞吐。

---

## 2. 用脚本从 jsonl 直接算 decode 吞吐

bench_serving 加上 `--output-details` 会写 jsonl，每行包含 `itls`、`ttfts`、`latencies`、`output_lens`。用仓库自带脚本即可算出 decode 阶段吞吐：

```bash
python scripts/pd_decode_throughput_from_bench.py sglang_xxx_200_100000_512.jsonl
```

- 默认同时输出两种算法：**来自 ITL**（1/mean(itl)）和 **来自 latency**（(output_len-1)/(latency-ttft)）。
- 可选：`--method itl` 或 `--method latency` 只输出一种。

---

## 3. 单独压 prefill 节点（可选）

若还要单独看 prefill 吞吐（例如 100k 输入的 prefill tokens/s）：

- 用同一 `--random-input-len 100000`，**短输出**（如 `--random-output-len 1`），这样总时间里 prefill 占比大。
- 报表里的 **input_throughput**（或总 input tokens / 总时间）可近似看作 prefill 侧吞吐；TTFT 可看作 prefill+传输 的延迟。

---

## 4. 小结

| 目标                     | 做法 |
|--------------------------|------|
| 输入 100k 时的 decode 吞吐 | bench_serving 用 `--random-input-len 100000`、`--random-output-len 512` 打 PD 路由，用返回的 **ITL** 或 **latency - TTFT** 按上式计算 decode 阶段 tokens/s。 |
| 写结果到文件便于后处理   | 加 `--output-details`，从生成的 jsonl 用脚本算 decode 吞吐。 |
| 整体 e2e 输出吞吐        | 直接看 bench_serving 打印的 `Output token throughput (tok/s)`。 |

这样即可在 PD 分离、输入 100k 的场景下，得到「decode 阶段」的吞吐指标。
