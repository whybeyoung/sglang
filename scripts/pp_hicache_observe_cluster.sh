#!/usr/bin/env bash
# Snapshot PP+HiCache cluster: git HEAD, processes, recent errors in nohup.out.
# See .claude/skills/pp-hicache/SKILL.md
#
# Usage:
#   ./scripts/pp_hicache_observe_cluster.sh
# Optional:
#   PP_HICACHE_LOG_LINES=80   # tail lines per host
#   PP_HICACHE_LOG=/home/aiges/nohup.out

set -euo pipefail

HOST="${PP_HICACHE_HOST:-36.138.60.54}"
PORTS="${PP_HICACHE_PORTS:-30239 30243}"
REPO="${PP_HICACHE_REPO:-/usr/local/src/sglang}"
LOG="${PP_HICACHE_LOG:-/home/aiges/nohup.out}"
NLINE="${PP_HICACHE_LOG_LINES:-80}"

for p in ${PORTS}; do
  echo "############################################"
  echo "### port ${p} root@${HOST}"
  echo "############################################"
  ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$p" "root@${HOST}" bash -s <<REMOTE
set -euo pipefail
echo "--- git HEAD (${REPO}) ---"
cd "${REPO}" 2>/dev/null && git rev-parse HEAD 2>/dev/null && git log -1 --oneline 2>/dev/null || echo "(no repo)"
echo "--- sglang / AIservice (pgrep) ---"
pgrep -af 'sglang|launch_server' 2>/dev/null | head -20 || echo "(none)"
pgrep -af AIservice 2>/dev/null | head -5 || echo "(no AIservice)"
echo "--- grep hot errors (last ${NLINE} lines of ${LOG}) ---"
if [ -f "${LOG}" ]; then
  tail -n "${NLINE}" "${LOG}" | grep -E 'PPContract|PP recv:|Traceback|ERROR:|RuntimeError|fatal|CUDA error' || echo "(no matches in tail)"
else
  echo "(no ${LOG})"
fi
echo "--- tail ${LOG} (last 15 lines) ---"
tail -15 "${LOG}" 2>/dev/null || true
REMOTE
  echo
done
