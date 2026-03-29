#!/usr/bin/env bash
# Stop residual SGLang / optional AIservice on both PP+HiCache nodes.
# See .claude/skills/pp-hicache/SKILL.md
#
# Usage:
#   ./scripts/pp_hicache_stop_cluster.sh
# Optional:
#   PP_HICACHE_STOP_AISERVICE=1   # also pkill AIservice (node-1 entrypoint)
#   PP_HICACHE_DRY_RUN=1          # only print pgrep, no pkill

set -euo pipefail

HOST="${PP_HICACHE_HOST:-36.138.60.54}"
PORTS="${PP_HICACHE_PORTS:-30239 30243}"
STOP_AI="${PP_HICACHE_STOP_AISERVICE:-0}"
DRY="${PP_HICACHE_DRY_RUN:-0}"

for p in ${PORTS}; do
  echo "--- SSH port ${p} ---"
  ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$p" "root@${HOST}" bash -s <<REMOTE
DRY="${DRY}"
STOP_AI="${STOP_AI}"
set -euo pipefail
echo "=== \$(hostname) ==="
echo "--- before ---"
pgrep -af sglang 2>/dev/null || true
ps aux | grep -E '[p]ython.*sglang|[s]glang\.launch_server' 2>/dev/null || true
pgrep -af AIservice 2>/dev/null || true

if [ "\$DRY" = "1" ]; then
  echo "DRY_RUN=1, skipping pkill"
  exit 0
fi

pkill -f 'sglang\.launch_server' 2>/dev/null || true
pkill -f 'sglang::scheduler' 2>/dev/null || true
if [ "\$STOP_AI" = "1" ]; then
  pkill -f 'AIservice' 2>/dev/null || true
fi
sleep 2
echo "--- after pkill ---"
pgrep -af sglang 2>/dev/null || echo "no sglang pids"
pgrep -af AIservice 2>/dev/null || echo "no AIservice pids"
echo "--- nvidia-smi (compute apps) ---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || nvidia-smi -L || true
REMOTE
  echo
done
