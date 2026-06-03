#!/usr/bin/env bash
set -euo pipefail

# Runtime-only cluster sync.
#
# Run from your host machine, not from inside the Docker/devcontainer image:
#
#   cd ~/Documents/VAE-Textures/vae-textures-dev
#   scripts/sync_runtime_to_cluster.sh          # dry-run preview, default
#   scripts/sync_runtime_to_cluster.sh dry-run  # same preview, explicit
#   scripts/sync_runtime_to_cluster.sh sync     # apply runtime-only sync
#   scripts/sync_runtime_to_cluster.sh --help   # show usage
#
# This script is intentionally stricter than sync_to_cluster.sh: it sends only
# the code/scripts needed to run the POP909 cluster smoke and representative
# training path. It does not sync notebooks, GSD/planning files, artifacts,
# local notes, generated outputs, checkpoints, or datasets. It never passes
# --delete, so remote datasets and outputs are preserved.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INCLUDE_FILE="${RSYNC_INCLUDE_FILE:-${PROJECT_ROOT}/.rsyncinclude-cluster-runtime}"
CLUSTER_USER="${CLUSTER_USER:-micael.antunes}"
CLUSTER_LOGIN_HOST="${CLUSTER_LOGIN_HOST:-139.124.22.4}"
CLUSTER_ALIAS="${CLUSTER_ALIAS:-sms}"
REMOTE_DIR="${REMOTE_DIR:-/home/${CLUSTER_USER}/vae-textures-dev}"
MODE="${1:-dry-run}"

usage() {
  cat <<'MSG'
Usage: scripts/sync_runtime_to_cluster.sh [dry-run|sync|no-delete|--help]

Modes:
  dry-run    Preview the runtime-only sync. This is the default.
  sync       Apply the runtime-only sync. Does not delete remote files.
  no-delete  Alias for sync; kept to make the no-delete behavior explicit.
  --help     Show this help.

Run from your host machine:
  cd ~/Documents/VAE-Textures/vae-textures-dev
  scripts/sync_runtime_to_cluster.sh

This sync only manages runtime code/scripts. It does not manage datasets,
notebooks, _artefatos, .planning/GSD files, local notes, checkpoints, W&B
outputs, or generated training results.
MSG
}

if [[ "${MODE}" == "--help" || "${MODE}" == "-h" || "${MODE}" == "help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${INCLUDE_FILE}" ]]; then
  echo "[runtime-sync][erro] Include file not found: ${INCLUDE_FILE}" >&2
  exit 2
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "[runtime-sync][erro] rsync is not installed in this environment." >&2
  echo "[runtime-sync][hint] Run this from your host machine, or install rsync before retrying." >&2
  exit 127
fi
if ! command -v ssh >/dev/null 2>&1; then
  echo "[runtime-sync][erro] ssh is not installed in this environment." >&2
  exit 127
fi

case "${MODE}" in
  dry-run)
    DRY_RUN_FLAG="--dry-run"
    ;;
  sync|no-delete)
    DRY_RUN_FLAG=""
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

cat <<MSG
[runtime-sync] Source: ${PROJECT_ROOT}/
[runtime-sync] Target: ${CLUSTER_USER}@${CLUSTER_ALIAS}:${REMOTE_DIR}
[runtime-sync] Includes: ${INCLUDE_FILE}
[runtime-sync] Mode: ${MODE}
[runtime-sync] Delete enabled: no
[runtime-sync] Remote datasets, checkpoints, notebooks, artifacts, and local/GSD files are not managed by this sync.
MSG

RSYNC_CMD=(
  rsync -avz --progress --human-readable --itemize-changes
  -e "ssh -J ${CLUSTER_USER}@${CLUSTER_LOGIN_HOST}"
  --include-from "${INCLUDE_FILE}"
)

if [[ -n "${DRY_RUN_FLAG}" ]]; then
  RSYNC_CMD+=("${DRY_RUN_FLAG}")
fi

RSYNC_CMD+=("${PROJECT_ROOT}/" "${CLUSTER_USER}@${CLUSTER_ALIAS}:${REMOTE_DIR}")

printf '[runtime-sync] Command: '
printf '%q ' "${RSYNC_CMD[@]}"
printf '\n'

"${RSYNC_CMD[@]}"
