#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXCLUDE_FILE="${RSYNC_EXCLUDE_FILE:-${PROJECT_ROOT}/.rsyncignore-cluster}"
CLUSTER_USER="${CLUSTER_USER:-micael.antunes}"
CLUSTER_LOGIN_HOST="${CLUSTER_LOGIN_HOST:-139.124.22.4}"
CLUSTER_ALIAS="${CLUSTER_ALIAS:-sms}"
REMOTE_DIR="${REMOTE_DIR:-/home/${CLUSTER_USER}/vae-textures-dev}"
MODE="${1:-dry-run}"

if [[ ! -f "${EXCLUDE_FILE}" ]]; then
  echo "[sync][erro] Exclude file not found: ${EXCLUDE_FILE}" >&2
  exit 2
fi

case "${MODE}" in
  dry-run)
    DELETE_FLAG="--delete"
    DRY_RUN_FLAG="--dry-run"
    ;;
  sync)
    DELETE_FLAG="--delete"
    DRY_RUN_FLAG=""
    ;;
  no-delete)
    DELETE_FLAG=""
    DRY_RUN_FLAG=""
    ;;
  *)
    echo "Usage: scripts/sync_to_cluster.sh [dry-run|sync|no-delete]" >&2
    exit 2
    ;;
esac

cat <<MSG
[sync] Source: ${PROJECT_ROOT}/
[sync] Target: ${CLUSTER_USER}@${CLUSTER_ALIAS}:${REMOTE_DIR}
[sync] Excludes: ${EXCLUDE_FILE}
[sync] Mode: ${MODE}
[sync] Delete enabled: ${DELETE_FLAG:+yes}
[sync] Excluded paths are not deleted on the cluster unless you add --delete-excluded (this script does not).
MSG

RSYNC_CMD=(
  rsync -avz --progress --human-readable --itemize-changes
  -e "ssh -J ${CLUSTER_USER}@${CLUSTER_LOGIN_HOST}"
  --exclude-from "${EXCLUDE_FILE}"
)

if [[ -n "${DELETE_FLAG}" ]]; then
  RSYNC_CMD+=("${DELETE_FLAG}")
fi
if [[ -n "${DRY_RUN_FLAG}" ]]; then
  RSYNC_CMD+=("${DRY_RUN_FLAG}")
fi

RSYNC_CMD+=("${PROJECT_ROOT}/" "${CLUSTER_USER}@${CLUSTER_ALIAS}:${REMOTE_DIR}")

printf '[sync] Command: '
printf '%q ' "${RSYNC_CMD[@]}"
printf '\n'

"${RSYNC_CMD[@]}"
