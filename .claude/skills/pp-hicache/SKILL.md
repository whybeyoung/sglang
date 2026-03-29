---
name: pp-hicache-pd-debug
description: >-
  Debug PP2+CP8+TP8 disaggregated prefill with hierarchical cache (HiCache) and Mooncake on two SSH nodes.
  Covers local contract_v2 changes, dual-node launch commands, env vars, deploy via git push/pull, cleaning residual sglang processes before restart, and when to use logs or py-spy.
  Use when working on pp prefill + hicache + mooncake stability, PD disaggregation on nodes 36.138.60.54:30239/30243, or contract_v2 PD debugging.
---

# PP + HiCache + PD 联调（双节点）

## 环境

| 节点   | 主机           | SSH 端口 | 说明        |
|--------|----------------|----------|-------------|
| node-1 | `36.138.60.54` | `30239`  | 直接可登录  |
| node-2 | `36.138.60.54` | `30243`  | 直接可登录  |

**远端路径（两台 node 一致）**

- **启动命令的工作目录**：`/home/aiges`（`nohup` 前先 `cd /home/aiges`）。
- **SGLang 源码仓库**：`/usr/local/src/sglang`（在此目录执行 `git pull` 更新代码；若进程跑的是该检出而非已安装包，启动前需 `export PYTHONPATH=/usr/local/src/sglang/python:$PYTHONPATH` 或使用能 import 到该树内 `sglang` 的解释器）。

用户与认证：按你方惯例（与 `spy-pp0-stuck-threads` 技能一致时可用 `root` + 密钥）。

## 目标

1. 在本地 **`contract_v2`** 分支上改代码，使 **PP2 + CP8 + TP8** 的 PD 预填 + **HiCache（Mooncake）** 能稳定推理。
2. 在 node-1 / node-2 按下方命令拉起服务；约 **5 分钟** 后服务就绪。其它机器上可能已有压测流量，就绪后流量会**立刻打满**。
3. 观察是否 **卡死、异常、无法正常推理**；必要时看日志或抓 **py-spy** 堆栈，再迭代本地代码。
4. **退出条件**：双节点长时间处于稳定推理状态，且无上述异常。

架构背景可参考仓库内 `docs/pp_mooncake_architecture.md`（若存在）。

## 调试环境变量（两节点启动前 export）

```bash
export SGLANG_DEBUG_HICACHE_MATCH=1
export SGLANG_DEBUG_PP_PREFILL_SHAPE=1
export SGLANG_DEBUG_HICACHE_MATCH_CHAIN=1
```

## node-1 启动示例

（业务二进制与配置路径以现场为准；二进制与 `*.toml` 应在 `/home/aiges` 下或此处使用相对路径能找到。）

```bash
cd /home/aiges
export SGLANG_DEBUG_HICACHE_MATCH=1
export SGLANG_DEBUG_PP_PREFILL_SHAPE=1
export SGLANG_DEBUG_HICACHE_MATCH_CHAIN=1
nohup ./AIservice -m=1 -c=xdeepseekv3testbo.toml -s=xdeepseekv3testbo \
  -u=http://companion-dx.xfyun.iflytek:6868 -p=sparkv2 -g=pddev &
```

## node-2 启动示例（sglang prefill + HiCache + Mooncake）

```bash
cd /home/aiges
export SGLANG_DEBUG_HICACHE_MATCH=1
export SGLANG_DEBUG_PP_PREFILL_SHAPE=1
export SGLANG_DEBUG_HICACHE_MATCH_CHAIN=1
# 若使用 /usr/local/src/sglang 源码而非 site-packages，取消下一行注释：
# export PYTHONPATH=/usr/local/src/sglang/python:$PYTHONPATH
nohup python3 -m sglang.launch_server \
  --model-path /work/models \
  --log-level debug \
  --enable-cache-report \
  --tokenizer-worker-num 8 \
  --page-size 64 \
  --disaggregation-mode prefill \
  --mem-fraction-static 0.88 \
  --context-length 131072 \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
  --chunked-prefill-size 16384 \
  --max-running-requests 512 \
  --tp 8 --pp-size 2 --dp-size 1 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --moe-dense-tp-size 1 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --allow-auto-truncate \
  --max-total-tokens 540000 \
  --dist-init-addr test-glm5-deploy-prefill-0.test-glm5-deploy-prefill.aiservice:20102 \
  --nnodes 2 --node-rank 1 \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend mooncake \
  --hicache-storage-prefetch-policy wait_complete &
```

**说明**：`--dist-init-addr`、`--model-path`（示例为 `/work/models`，若模型在其它路径请改）、IB 设备名等需与集群实际一致；`node-rank` 与 node-1 角色需匹配分布式约定。

## 迭代代码后重启：先清理残留 SGLang

每次 **`git pull` 之后、再次 `nohup` 启动之前**，在对应机器上清掉旧进程，避免占 GPU / 端口 / 分布式 rank，或子进程残留导致新版本行为异常。

**node-2（直接 `python3 -m sglang.launch_server`）**

1. **先看 PID**（避免误杀其它任务）：

   ```bash
   pgrep -af sglang || true
   ps aux | grep -E '[p]ython.*sglang|[s]glang\.launch_server' || true
   ```

2. **结束旧实例**（先温和退出；仍存活再 `kill -9` 具体 PID）：

   ```bash
   pkill -f 'sglang\.launch_server' 2>/dev/null || true
   pkill -f 'sglang::scheduler' 2>/dev/null || true
   sleep 2
   pgrep -af sglang || echo "no sglang pids"
   ```

3. **`nvidia-smi`**：确认无残留 `python`/推理进程占显存；若仍有，按表中 PID `kill`，必要时 `kill -9`。

**node-1（`AIservice`）**

- 若由 `AIservice` 拉起 SGLang worker：重启前按现场方式**停干净旧 `AIservice`/守护进程**，再启动新实例，避免两套服务并存。
- 若该机上也有独立 `sglang.launch_server` 或 `sglang::scheduler_*` 进程，同样先 `pgrep`/`pkill` 再启。

完成清理后再执行上文 **node-1 / node-2 启动示例**。

## 观察与排障流程

1. **就绪后**：关注吞吐、延迟、错误日志、是否无响应（卡死）。
2. **先看日志**：HiCache / PP 相关行（已开 debug 变量时信息更全）。
3. **疑似死锁或调度挂起**：使用技能 **`spy-pp0-stuck-threads`**（同主机 `36.138.60.54`，端口按 node 选 `30239` 或 `30243`）对 PP0 等 scheduler 进程做 `py-spy dump`。
4. **代码迭代**：重点文件通常包括 `python/sglang/srt/managers/scheduler_pp_mixin.py`、`scheduler.py`、HiCache 相关模块（如 `hiradix_cache.py` 等），以实际栈与日志为准。

## 推荐迭代闭环（Mac 上 push，节点上 HTTPS pull + 停服 + 启服 + 看日志）

**约定**：代码从 **Mac** 推到 **`why`（whybeyoung/sglang）**；两台机器上 **只用 HTTPS 拉取**，不要求节点具备 `git@github.com` 的 SSH key。

1. **Mac**：提交后推送（仅此一步在本地用 SSH/HTTPS 连 GitHub）：

   ```bash
   git push why contract_v2
   ```

2. **Mac**：一键在两台 node 上把 `origin` 设为 HTTPS 并拉取 `contract_v2`。脚本在**远端执行 `git fetch` / `git pull` 前**会默认设置：

   `export HTTPS_PROXY=http://10.104.102.203:7890`（同时设置 `https_proxy`）。

   ```bash
   ./scripts/pp_hicache_sync_cluster.sh
   ```

   改用其它代理：`PP_HICACHE_HTTPS_PROXY=http://host:port ./scripts/pp_hicache_sync_cluster.sh`；不走代理：`PP_HICACHE_HTTPS_PROXY= ./scripts/pp_hicache_sync_cluster.sh`。

   可选：`PP_HICACHE_ORIGIN_URL`（默认 `https://github.com/whybeyoung/sglang.git`）、`PP_HICACHE_BRANCH`、`PP_HICACHE_REPO`。

3. **Mac**：停残留推理进程（两台都执行）：

   ```bash
   ./scripts/pp_hicache_stop_cluster.sh
   ```

   若 node-1 由 **AIservice** 拉起、需要一并停掉：

   ```bash
   PP_HICACHE_STOP_AISERVICE=1 ./scripts/pp_hicache_stop_cluster.sh
   ```

   仅查看、不杀进程：`PP_HICACHE_DRY_RUN=1 ./scripts/pp_hicache_stop_cluster.sh`

4. **两台 node**：按上文 **node-1 / node-2 启动示例** 在 `/home/aiges` 下 `nohup` 启新版本（确认 `PYTHONPATH` 指向 `/usr/local/src/sglang/python` 若跑源码树）。

5. **观察与 debug**：日志里搜 `PPContract`、`HiCache`、`PP recv`；卡死则 **`spy-pp0-stuck-threads`**。回到步骤 1 改代码再推、再跑 2–5。

---

## 手动在节点上拉取（等价于脚本）

```bash
cd /usr/local/src/sglang
git remote set-url origin https://github.com/whybeyoung/sglang.git
export HTTPS_PROXY=http://10.104.102.203:7890   # 按需
git fetch origin && git checkout contract_v2 && git pull --ff-only origin contract_v2
```

**私有仓库**：HTTPS 需 token（`https://<token>@github.com/...` 或 credential helper）；公开库无需。

拉取后：**先停干净旧进程**（脚本或上文「迭代代码后重启」），再启动；若运行依赖 `PYTHONPATH` 指向该仓库，启动脚本需包含 `export PYTHONPATH=/usr/local/src/sglang/python:...`。

## 相关技能与文档

- **PP0 卡死 / py-spy**：`.claude/skills/spy-pp0-stuck-threads/SKILL.md`
- **架构说明**（若需）：`docs/pp_mooncake_architecture.md`
