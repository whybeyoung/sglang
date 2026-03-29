#!/usr/bin/env bash
# Sync SGLang on PP+HiCache cluster nodes via HTTPS (no GitHub SSH key on nodes).
# See .claude/skills/pp-hicache/SKILL.md
#
# On your Mac: push first, then run this.
#   git push why contract_v2
#   ./scripts/pp_hicache_sync_cluster.sh
#
# Optional:
#   PP_HICACHE_HTTPS_PROXY=  (empty = no proxy; default is cluster proxy below)
#   PP_HICACHE_ORIGIN_URL=https://github.com/whybeyoung/sglang.git
#   PP_HICACHE_BRANCH=contract_v2

set -euo pipefail

HOST="${PP_HICACHE_HOST:-36.138.60.54}"
PORTS="${PP_HICACHE_PORTS:-30239 30243}"
REPO="${PP_HICACHE_REPO:-/usr/local/src/sglang}"
BRANCH="${PP_HICACHE_BRANCH:-contract_v2}"
ORIGIN_URL="${PP_HICACHE_ORIGIN_URL:-https://github.com/whybeyoung/sglang.git}"
# Default: same as pp-hicache skill (nodes pull GitHub via proxy).
PROXY="${PP_HICACHE_HTTPS_PROXY:-http://10.104.102.203:7890}"

remote_cmd() {
  local port="$1"
  # shellcheck disable=SC2087
  ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$port" "root@${HOST}" bash -s <<EOF
set -euo pipefail
cd "${REPO}"
echo "=== \$(hostname) port ${port} ==="
if [ -n "${PROXY}" ]; then
  export HTTPS_PROXY="${PROXY}"
  export https_proxy="${PROXY}"
  echo "Using HTTPS_PROXY=${PROXY}"
fi
git remote set-url origin "${ORIGIN_URL}"
git remote -v | head -2
git fetch origin
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}"
git rev-parse HEAD
EOF
}

for p in ${PORTS}; do
  echo "--- SSH port ${p} ---"
  remote_cmd "$p"
  echo
done
echo "Done. Next: ./scripts/pp_hicache_stop_cluster.sh then start services from /home/aiges (see skill)."
