#!/usr/bin/env bash
# One-shot: push contract_v2 → dual-node HTTPS pull (with proxy) → stop sglang → observe.
# Run from repo root on your Mac (SSH keys to 36.138.60.54:30239/30243).
#
# Usage:
#   ./scripts/pp_hicache_deploy_cycle.sh
#
# Optional:
#   PP_HICACHE_SKIP_PUSH=1          # only sync + stop + observe (no git push)
#   PP_HICACHE_STOP_AISERVICE=1     # also kill AIservice on nodes (node-1)
#   PP_HICACHE_SKIP_STOP=1        # push + sync only
#   PP_HICACHE_REMOTE=why         # git remote for push (default: why)
#   PP_HICACHE_BRANCH=contract_v2
#
# After this script: restart launch_server / AIservice manually with:
#   export PYTHONPATH=/usr/local/src/sglang/python:\$PYTHONPATH
# Then run ./scripts/pp_hicache_observe_cluster.sh again under load.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REMOTE="${PP_HICACHE_REMOTE:-why}"
BRANCH="${PP_HICACHE_BRANCH:-contract_v2}"

echo "=== [1/4] Local: branch + short status ==="
git rev-parse --abbrev-ref HEAD
git status -sb | head -5
git log -1 --oneline

if [ "${PP_HICACHE_SKIP_PUSH:-0}" != "1" ]; then
  echo "=== [2/4] git push ${REMOTE} ${BRANCH} ==="
  git push "${REMOTE}" "${BRANCH}"
else
  echo "=== [2/4] SKIP push (PP_HICACHE_SKIP_PUSH=1) ==="
fi

echo "=== [3/4] Cluster: HTTPS pull (scripts/pp_hicache_sync_cluster.sh) ==="
"$ROOT/scripts/pp_hicache_sync_cluster.sh"

if [ "${PP_HICACHE_SKIP_STOP:-0}" != "1" ]; then
  echo "=== [4a/4] Cluster: stop (scripts/pp_hicache_stop_cluster.sh) ==="
  export PP_HICACHE_STOP_AISERVICE="${PP_HICACHE_STOP_AISERVICE:-0}"
  "$ROOT/scripts/pp_hicache_stop_cluster.sh"
else
  echo "=== [4a/4] SKIP stop (PP_HICACHE_SKIP_STOP=1) ==="
fi

echo "=== [4b/4] Cluster: observe snapshot ==="
"$ROOT/scripts/pp_hicache_observe_cluster.sh"

cat <<EOF

--- Next (manual restart) ---
  node-1 (30239): cd /home/aiges && export PYTHONPATH=/usr/local/src/sglang/python:\\\$PYTHONPATH
                  + debug env + nohup ./AIservice ... (see skill)
  node-2 (30243): same PYTHONPATH + nohup python3 -m sglang.launch_server ... (see skill)

Then:  ./scripts/pp_hicache_observe_cluster.sh
EOF
