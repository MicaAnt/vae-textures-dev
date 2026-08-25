#!/usr/bin/env bash
set -euo pipefail

# Download Phase 9 authors-vs-ours validation evidence from the cluster.
# Run locally after the smoke or full validation Slurm job finishes.
# It downloads evidence only. It never deletes remote or local files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLUSTER_USER="${CLUSTER_USER:-micael.antunes}"
CLUSTER_LOGIN_HOST="${CLUSTER_LOGIN_HOST:-139.124.22.4}"
CLUSTER_ALIAS="${CLUSTER_ALIAS:-sms}"
REMOTE_DIR="${REMOTE_DIR:-/home/${CLUSTER_USER}/vae-textures-dev}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-${REMOTE_DIR}/logs/pop909-phase9-validation}"
LOCAL_ARTIFACT_ROOT="${LOCAL_ARTIFACT_ROOT:-${PROJECT_ROOT}/_artefatos}"

usage() {
  cat <<'MSG'
Usage:
  scripts/download_pop909_phase9_artifacts.sh JOB_ID REMOTE_RUN_DIR [REMOTE_RUN_DIR ...]

Required:
  JOB_ID          Slurm job id for the Phase 9 validation job.
  REMOTE_RUN_DIR  Remote _artefatos/pop909-conditioned-reconstruction/<run_id> directory.

Optional environment overrides:
  CLUSTER_USER        Default: micael.antunes
  CLUSTER_LOGIN_HOST  Default: 139.124.22.4
  CLUSTER_ALIAS       Default: sms
  REMOTE_DIR          Default: /home/${CLUSTER_USER}/vae-textures-dev
  REMOTE_LOG_DIR      Default: ${REMOTE_DIR}/logs/pop909-phase9-validation
  LOCAL_ARTIFACT_ROOT Default: ./_artefatos

Expected run artifacts:
  config/resolved_config.json
  tables/comparison_wide.csv
  manifests/comparison_manifest.jsonl
  rankings/ranking_strata.json
  summaries/summary.json
  summaries/summary.md

Human gate:
  Downloading is not acceptance. Inspect logs and summaries before deciding
  accept / rerun / block.
MSG
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || "$#" -lt 2 ]]; then
  usage
  exit $([[ "$#" -lt 2 && "${1:-}" != "--help" && "${1:-}" != "-h" ]] && echo 2 || echo 0)
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[phase9-download][erro] rsync is not installed." >&2
  exit 127
fi
if ! command -v ssh >/dev/null 2>&1; then
  echo "[phase9-download][erro] ssh is not installed." >&2
  exit 127
fi

JOB_ID="$1"
shift

LOCAL_DIR="${LOCAL_ARTIFACT_ROOT}/cluster-pop909-phase9-${JOB_ID}"
mkdir -p "${LOCAL_DIR}/logs" "${LOCAL_DIR}/runs" "${LOCAL_DIR}/remote"

SSH_CMD=(ssh -J "${CLUSTER_USER}@${CLUSTER_LOGIN_HOST}" "${CLUSTER_USER}@${CLUSTER_ALIAS}")
RSYNC_RSH="ssh -J ${CLUSTER_USER}@${CLUSTER_LOGIN_HOST}"

cat <<MSG
[phase9-download] Job id: ${JOB_ID}
[phase9-download] Remote logs: ${CLUSTER_USER}@${CLUSTER_ALIAS}:${REMOTE_LOG_DIR}
[phase9-download] Local target: ${LOCAL_DIR}
[phase9-download] RUN_DIRs:
MSG
for run_dir in "$@"; do
  echo "  - ${run_dir}"
done

echo "[phase9-download] Capturing remote sacct/listing metadata..."
"${SSH_CMD[@]}" "sacct -j '${JOB_ID}' --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES%60,ExitCode 2>/dev/null || true"   > "${LOCAL_DIR}/remote/sacct-${JOB_ID}.txt"
"${SSH_CMD[@]}" "ls -lah '${REMOTE_LOG_DIR}'/*'${JOB_ID}'* 2>/dev/null || true"   > "${LOCAL_DIR}/remote/log-files-${JOB_ID}.txt"

echo "[phase9-download] Downloading stdout/stderr logs..."
rsync -avz --human-readable --itemize-changes   -e "${RSYNC_RSH}"   --include="*${JOB_ID}*"   --exclude="*"   "${CLUSTER_USER}@${CLUSTER_ALIAS}:${REMOTE_LOG_DIR}/"   "${LOCAL_DIR}/logs/"

for run_dir in "$@"; do
  run_name="$(basename "${run_dir}")"
  echo "[phase9-download] Capturing listing for ${run_dir}..."
  "${SSH_CMD[@]}" "find '${run_dir}' -maxdepth 5 -type f | sort 2>/dev/null || true"     > "${LOCAL_DIR}/remote/${run_name}-files.txt"

  echo "[phase9-download] Downloading ${run_dir}..."
  rsync -avz --human-readable --itemize-changes     -e "${RSYNC_RSH}"     "${CLUSTER_USER}@${CLUSTER_ALIAS}:${run_dir}/"     "${LOCAL_DIR}/runs/${run_name}/"
done

cat > "${LOCAL_DIR}/README.md" <<MSG
# POP909 Phase 9 Artifacts - Job ${JOB_ID}

Downloaded from cluster on $(date -Is).

## Contents

- \`logs/\` - stdout/stderr files matching Slurm job ${JOB_ID}
- \`runs/\` - downloaded Phase 9 RUN_DIR directories
- \`remote/sacct-${JOB_ID}.txt\` - Slurm accounting snapshot
- \`remote/log-files-${JOB_ID}.txt\` - remote log files that matched this job id
- \`remote/*-files.txt\` - remote RUN_DIR file listings captured before download

## Manual Validation Gate

Downloading is not acceptance. Before advancing:

- Check Slurm state and exit code.
- Read stdout/stderr for CUDA, config, checkpoint loading, row count, and errors.
- Inspect \`runs/*/summaries/summary.json\` and \`summary.md\`.
- Confirm \`tables/comparison_wide.csv\`, \`manifests/comparison_manifest.jsonl\`, and \`rankings/ranking_strata.json\` exist.
- For smoke: accept/rerun/block before launching full validation.
- For full validation: accept/rerun/block before 24-case selection and report writing.
MSG

echo "[phase9-download] Done: ${LOCAL_DIR}"
