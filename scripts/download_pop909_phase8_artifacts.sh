#!/usr/bin/env bash
set -euo pipefail

# Download Phase 8 representative POP909 training evidence from the cluster.
#
# Run this on your local machine after a staged training job finishes:
#
#   cd ~/Documents/VAE-Textures/vae-textures-dev
#   scripts/download_pop909_phase8_artifacts.sh JOB_ID REMOTE_RESULT_DIR [REMOTE_RESULT_DIR ...]
#
# Example:
#   scripts/download_pop909_phase8_artifacts.sh 333001 \
#     /home/micael.antunes/vae-textures-dev/base_model/result_2026-06-10_210501
#
# The script intentionally downloads evidence only. It does not delete, resume,
# submit jobs, or decide whether the next staged epoch should run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLUSTER_USER="${CLUSTER_USER:-micael.antunes}"
CLUSTER_LOGIN_HOST="${CLUSTER_LOGIN_HOST:-139.124.22.4}"
CLUSTER_ALIAS="${CLUSTER_ALIAS:-sms}"
REMOTE_DIR="${REMOTE_DIR:-/home/${CLUSTER_USER}/vae-textures-dev}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-${REMOTE_DIR}/logs/pop909-representative}"
LOCAL_ARTIFACT_ROOT="${LOCAL_ARTIFACT_ROOT:-${PROJECT_ROOT}/_artefatos}"

usage() {
  cat <<'MSG'
Usage:
  scripts/download_pop909_phase8_artifacts.sh JOB_ID REMOTE_RESULT_DIR [REMOTE_RESULT_DIR ...]

Required:
  JOB_ID             Slurm job id for the staged Phase 8 session.
  REMOTE_RESULT_DIR  Remote base_model/result_* directory produced by train.py.

Optional environment overrides:
  CLUSTER_USER       Default: micael.antunes
  CLUSTER_LOGIN_HOST Default: 139.124.22.4
  CLUSTER_ALIAS      Default: sms
  REMOTE_DIR         Default: /home/${CLUSTER_USER}/vae-textures-dev
  REMOTE_LOG_DIR     Default: ${REMOTE_DIR}/logs/pop909-representative
  LOCAL_ARTIFACT_ROOT Default: ./_artefatos

This downloads:
  - stdout/stderr logs matching the job id from REMOTE_LOG_DIR
  - the supplied result directory/directories, including checkpoints and W&B files
  - remote manifest text files with sacct and result file listing when available

It never deletes remote or local files.
MSG
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || "$#" -lt 2 ]]; then
  usage
  exit $([[ "$#" -lt 2 && "${1:-}" != "--help" && "${1:-}" != "-h" ]] && echo 2 || echo 0)
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[phase8-download][erro] rsync is not installed." >&2
  exit 127
fi
if ! command -v ssh >/dev/null 2>&1; then
  echo "[phase8-download][erro] ssh is not installed." >&2
  exit 127
fi

JOB_ID="$1"
shift

LOCAL_DIR="${LOCAL_ARTIFACT_ROOT}/cluster-pop909-phase8-${JOB_ID}"
mkdir -p "${LOCAL_DIR}/logs" "${LOCAL_DIR}/results" "${LOCAL_DIR}/remote"

SSH_CMD=(ssh -J "${CLUSTER_USER}@${CLUSTER_LOGIN_HOST}" "${CLUSTER_USER}@${CLUSTER_ALIAS}")
RSYNC_RSH="ssh -J ${CLUSTER_USER}@${CLUSTER_LOGIN_HOST}"

cat <<MSG
[phase8-download] Job id: ${JOB_ID}
[phase8-download] Remote logs: ${CLUSTER_USER}@${CLUSTER_ALIAS}:${REMOTE_LOG_DIR}
[phase8-download] Local target: ${LOCAL_DIR}
[phase8-download] Result dirs:
MSG
for result_dir in "$@"; do
  echo "  - ${result_dir}"
done

echo "[phase8-download] Capturing remote sacct/listing metadata..."
"${SSH_CMD[@]}" "sacct -j '${JOB_ID}' --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES%60,ExitCode 2>/dev/null || true" \
  > "${LOCAL_DIR}/remote/sacct-${JOB_ID}.txt"
"${SSH_CMD[@]}" "ls -lah '${REMOTE_LOG_DIR}'/*'${JOB_ID}'* 2>/dev/null || true" \
  > "${LOCAL_DIR}/remote/log-files-${JOB_ID}.txt"

echo "[phase8-download] Downloading stdout/stderr logs..."
rsync -avz --human-readable --itemize-changes \
  -e "${RSYNC_RSH}" \
  --include="*${JOB_ID}*" \
  --exclude="*" \
  "${CLUSTER_USER}@${CLUSTER_ALIAS}:${REMOTE_LOG_DIR}/" \
  "${LOCAL_DIR}/logs/"

for result_dir in "$@"; do
  result_name="$(basename "${result_dir}")"
  echo "[phase8-download] Capturing listing for ${result_dir}..."
  "${SSH_CMD[@]}" "find '${result_dir}' -maxdepth 4 -type f | sort 2>/dev/null || true" \
    > "${LOCAL_DIR}/remote/${result_name}-files.txt"

  echo "[phase8-download] Downloading ${result_dir}..."
  rsync -avz --human-readable --itemize-changes \
    -e "${RSYNC_RSH}" \
    "${CLUSTER_USER}@${CLUSTER_ALIAS}:${result_dir}/" \
    "${LOCAL_DIR}/results/${result_name}/"
done

cat > "${LOCAL_DIR}/README.md" <<MSG
# POP909 Phase 8 Artifacts - Job ${JOB_ID}

Downloaded from cluster on $(date -Is).

## Contents

- \`logs/\` - stdout/stderr files matching Slurm job ${JOB_ID}
- \`results/\` - downloaded \`base_model/result_*\` training output directories
- \`remote/sacct-${JOB_ID}.txt\` - Slurm accounting snapshot
- \`remote/log-files-${JOB_ID}.txt\` - remote log files that matched this job id
- \`remote/*-files.txt\` - remote result file listings captured before download

## Manual Validation Gate

Do not launch the next Phase 8 staged session until these are inspected:

- Slurm state and exit code
- stdout/stderr tail
- W&B run URL, config, metrics, and artifact/checkpoint evidence
- checkpoint files in \`results/*/models/\`
- train/validation loss behavior
- epoch duration and checkpoint save timing
MSG

echo "[phase8-download] Done: ${LOCAL_DIR}"
