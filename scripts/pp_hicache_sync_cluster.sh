#!/usr/bin/env bash
# Sync SGLang contract_v2 on PP+HiCache cluster nodes (see .claude/skills/pp-hicache/SKILL.md).
# Run from your laptop; requires SSH key access to root@36.138.60.54 on ports 30239 and 30243.
#
# Fixes common "Host key verification failed" when nodes fetch github.com over SSH.
#
# Usage:
#   ./scripts/pp_hicache_sync_cluster.sh
# Optional:
#   PP_HICACHE_BRANCH=contract_v2 ./scripts/pp_hicache_sync_cluster.sh
#   PP_HICACHE_REPO=/usr/local/src/sglang ./scripts/pp_hicache_sync_cluster.sh

set -euo pipefail

HOST="${PP_HICACHE_HOST:-36.138.60.54}"
PORTS="${PP_HICACHE_PORTS:-30239 30243}"
REPO="${PP_HICACHE_REPO:-/usr/local/src/sglang}"
BRANCH="${PP_HICACHE_BRANCH:-contract_v2}"

remote_cmd() {
  local port="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=15 -p "$port" "root@${HOST}" bash -s <<EOF
set -euo pipefail
umask 077
mkdir -p "\$HOME/.ssh"
chmod 700 "\$HOME/.ssh"
# Idempotent: append GitHub host keys if missing (fixes fetch failures on fresh roots).
if ! grep -q 'github.com' "\$HOME/.ssh/known_hosts" 2>/dev/null; then
  ssh-keyscan -t ed25519,rsa github.com >> "\$HOME/.ssh/known_hosts" 2>/dev/null || true
fi
chmod 600 "\$HOME/.ssh/known_hosts" 2>/dev/null || true
cd "${REPO}"
echo "=== \$(hostname) port ${port} ==="
git remote -v | head -4
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes"
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
echo "Done. Restart services only after cleaning old sglang processes (see skill)."
