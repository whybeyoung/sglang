#!/usr/bin/env bash
# Start PP+HiCache cluster exactly as in .claude/skills/pp-hicache/SKILL.md — do not edit
# the remote commands here unless the skill is updated first.
#
# - Port 30239 (node0): ONLY AIservice (it spawns sglang children). No direct launch_server.
# - Port 30243 (node1): ONLY the skill's nohup python3 -m sglang.launch_server ... block.
#
# Prerequisites: run pp_hicache_stop_cluster.sh first (with PP_HICACHE_STOP_AISERVICE=1 on node0).
#
# Optional:
#   PP_HICACHE_SLEEP_BEFORE_RANK1=30   # seconds after AIservice before starting 30243

set -euo pipefail

HOST="${PP_HICACHE_HOST:-36.138.60.54}"
PORT0="${PP_HICACHE_PORT_NODE0:-30239}"
PORT1="${PP_HICACHE_PORT_NODE1:-30243}"
SLEEP="${PP_HICACHE_SLEEP_BEFORE_RANK1:-30}"

ssh_start() {
  local port="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=30 -p "$port" "root@${HOST}" bash -s
}

echo "=== [node0 / ${PORT0}] AIservice only (skill node-1 启动示例) ==="
ssh_start "${PORT0}" <<'REMOTE'
set -euo pipefail
cd /home/aiges
export SGLANG_DEBUG_HICACHE_MATCH=1
export SGLANG_DEBUG_PP_PREFILL_SHAPE=1
export SGLANG_DEBUG_HICACHE_MATCH_CHAIN=1
nohup ./AIservice -m=1 -c=xdeepseekv3testbo.toml -s=xdeepseekv3testbo \
  -u=http://companion-dx.xfyun.iflytek:6868 -p=sparkv2 -g=pddev &
echo "AIservice nohup started, pid=$!"
REMOTE

echo "=== sleep ${SLEEP}s before node1 launch_server ==="
sleep "${SLEEP}"

echo "=== [node1 / ${PORT1}] launch_server (skill node-2 启动示例, unchanged) ==="
ssh_start "${PORT1}" <<'REMOTE'
set -euo pipefail
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
echo "launch_server nohup started, pid=$!"
REMOTE

echo "Done. Use ./scripts/pp_hicache_observe_cluster.sh after workers settle."
