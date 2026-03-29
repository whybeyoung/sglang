#!/usr/bin/env bash
# One-shot: push contract_v2 → dual-node HTTPS pull (with proxy) → stop sglang → observe.
# Run from repo root on your Mac (SSH keys to 36.138.60.54:30239/30243).
#
# Usage:
#   ./scripts/pp_hicache_deploy_cycle.sh
#
# Optional:
#   PP_HICACHE_SKIP_PUSH=1          # only sync + stop + observe (no git push)
#   PP_HICACHE_SKIP_STOP=1        # push + sync only
#   PP_HICACHE_SKIP_START=1       # no auto-start after stop (default: auto-start)
#   PP_HICACHE_STOP_AISERVICE=0   # if set before run: do not kill AIservice on 30239 when stopping
#   PP_HICACHE_REMOTE=why
#   PP_HICACHE_BRANCH=contract_v2
#
# Auto-start (unless SKIP_START=1): scripts/pp_hicache_start_cluster.sh
#   30239 = node0: AIservice only (skill). 30243 = skill launch_server verbatim.

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
  echo "=== [4a] Cluster: stop (scripts/pp_hicache_stop_cluster.sh) ==="
  # Full cycle must stop AIservice on 30239 (node0); override only if explicitly set.
  if [ "${PP_HICACHE_SKIP_START:-0}" != "1" ]; then
    export PP_HICACHE_STOP_AISERVICE="${PP_HICACHE_STOP_AISERVICE:-1}"
  else
    export PP_HICACHE_STOP_AISERVICE="${PP_HICACHE_STOP_AISERVICE:-0}"
  fi
  "$ROOT/scripts/pp_hicache_stop_cluster.sh"
else
  echo "=== [4a] SKIP stop (PP_HICACHE_SKIP_STOP=1) ==="
fi

if [ "${PP_HICACHE_SKIP_START:-0}" != "1" ] && [ "${PP_HICACHE_SKIP_STOP:-0}" != "1" ]; then
  echo "=== [4b] Cluster: start (scripts/pp_hicache_start_cluster.sh) ==="
  "$ROOT/scripts/pp_hicache_start_cluster.sh"
  WAIT="${PP_HICACHE_POST_START_WAIT:-45}"
  echo "=== sleep ${WAIT}s before observe ==="
  sleep "${WAIT}"
else
  echo "=== [4b] SKIP start (SKIP_START=1 or SKIP_STOP=1) ==="
fi

echo "=== [4c] Cluster: observe snapshot ==="
"$ROOT/scripts/pp_hicache_observe_cluster.sh"
