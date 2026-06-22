#!/usr/bin/env bash
set -euo pipefail

# Submit one Phase 8 representative POP909 session through Slurm sbatch.
# Modes:
#   preflight  Build the canonical loaders and print dataset/batch evidence.
#   train      Run exactly one staged epoch through base_model/train.py by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-train}"

IMAGE_PATH="${CLUSTER_IMAGE:-/home/${USER}/devcontainer_images/dcli_fidle_tuto.squashfs}"
WORKSPACE_HOST_PATH="${WORKSPACE_HOST_PATH:-/home/${USER}/vae-textures-dev}"
PARTITION="${SLURM_PARTITION:-gpu}"
GPUS="${SLURM_GPUS:-1}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"
NTASKS="${SLURM_NTASKS:-1}"
TIME_LIMIT="${SLURM_TIME:-04:00:00}"
JOB_NAME="${SLURM_JOB_NAME:-pop909-phase8}"
LOG_DIR="${PHASE8_LOG_DIR:-${PROJECT_ROOT}/logs/pop909-representative}"

export VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-128}"
export VAE_N_EPOCH="${VAE_N_EPOCH:-6}"
export VAE_RUN_EPOCHS_THIS_JOB="${VAE_RUN_EPOCHS_THIS_JOB:-1}"
export VAE_SEED="${VAE_SEED:-3345}"
export VAE_RUN_NAME="${VAE_RUN_NAME:-pop909-phase8-representative-$(date +%Y%m%d-%H%M%S)}"
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-pop909-reproduction}"
export WANDB_GROUP="${WANDB_GROUP:-${VAE_RUN_NAME}}"
export WANDB_TAGS="${WANDB_TAGS:-pop909,phase8,representative,released-code-faithful}"
export WANDB_CHECKPOINT_POLICY="${WANDB_CHECKPOINT_POLICY:-valid,final,epoch-state,last-state,final-state}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb-config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/tmp/wandb-data}"

if [[ "${VAE_LIMIT_TRAIN_SAMPLES:-0}" =~ ^[0-9]+$ && "${VAE_LIMIT_TRAIN_SAMPLES:-0}" -gt 0 ]]; then
  echo "[phase8-submit][error] VAE_LIMIT_TRAIN_SAMPLES must be unset or 0 for Phase 8 representative sessions." >&2
  exit 2
fi
if [[ "${VAE_LIMIT_VAL_SAMPLES:-0}" =~ ^[0-9]+$ && "${VAE_LIMIT_VAL_SAMPLES:-0}" -gt 0 ]]; then
  echo "[phase8-submit][error] VAE_LIMIT_VAL_SAMPLES must be unset or 0 for Phase 8 representative sessions." >&2
  exit 2
fi

# Source private W&B metadata if present. This may define WANDB_API_KEY.
# The script must never echo the API key value.
if [[ -f "${HOME}/.config/wandb/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.config/wandb/env.sh"
fi

mkdir -p "${LOG_DIR}"

CONTAINER_MOUNTS="${WORKSPACE_HOST_PATH}:/workspace,/lib64/libcuda.so.1:/nvidia/libcuda.so.1,/lib64/libnvidia-ml.so.1:/nvidia/libnvidia-ml.so.1,/lib64/libnvidia-ptxjitcompiler.so.1:/nvidia/libnvidia-ptxjitcompiler.so.1"

build_inner_cmd() {
  local command="$1"
  cat <<INNER
set -euo pipefail
export LD_LIBRARY_PATH=/nvidia\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export WANDB_CACHE_DIR=\${WANDB_CACHE_DIR:-/tmp/wandb-cache}
export WANDB_CONFIG_DIR=\${WANDB_CONFIG_DIR:-/tmp/wandb-config}
export WANDB_DATA_DIR=\${WANDB_DATA_DIR:-/tmp/wandb-data}
export WANDB_DIR=/workspace/base_model
mkdir -p "\$WANDB_CACHE_DIR" "\$WANDB_CONFIG_DIR" "\$WANDB_DATA_DIR"
cd /workspace/base_model
echo "[phase8-submit] mode=${MODE}"
echo "[phase8-submit] VAE_RUN_NAME=\${VAE_RUN_NAME}"
echo "[phase8-submit] VAE_SEED=\${VAE_SEED}"
echo "[phase8-submit] VAE_BATCH_SIZE=\${VAE_BATCH_SIZE}"
echo "[phase8-submit] VAE_N_EPOCH=\${VAE_N_EPOCH}"
echo "[phase8-submit] VAE_RUN_EPOCHS_THIS_JOB=\${VAE_RUN_EPOCHS_THIS_JOB}"
echo "[phase8-submit] VAE_RESUME_FROM=\${VAE_RESUME_FROM:-}"
echo "[phase8-submit] VAE_LIMIT_TRAIN_SAMPLES=\${VAE_LIMIT_TRAIN_SAMPLES:-0}"
echo "[phase8-submit] VAE_LIMIT_VAL_SAMPLES=\${VAE_LIMIT_VAL_SAMPLES:-0}"
echo "[phase8-submit] WANDB_PROJECT=\${WANDB_PROJECT}"
echo "[phase8-submit] WANDB_GROUP=\${WANDB_GROUP:-}"
echo "[phase8-submit] WANDB_RUN_ID=\${WANDB_RUN_ID:-}"
echo "[phase8-submit] WANDB_RESUME=\${WANDB_RESUME:-}"
${command}
echo "[phase8-submit] result directories:"
find /workspace/base_model -maxdepth 1 -type d -name 'result_*' -printf '%T@ %p\n' | sort -nr | head -5
echo "[phase8-submit] latest full-state checkpoints:"
find /workspace/base_model/result_* -path '*/models/*_state.pt' -type f 2>/dev/null | sort | tail -20 || true
INNER
}

case "${MODE}" in
  preflight)
    INNER_CMD="$(build_inner_cmd 'python -u phase8_preflight.py')"
    ;;
  train)
    INNER_CMD="$(build_inner_cmd 'python -u train.py')"
    ;;
  *)
    echo "Usage: $0 [preflight|train]" >&2
    exit 2
    ;;
esac

printf -v WRAPPED_CMD 'srun --container-image=%q --container-mounts=%q --container-workdir=/workspace bash -lc %q' "${IMAGE_PATH}" "${CONTAINER_MOUNTS}" "${INNER_CMD}"

cat <<MSG
[phase8-submit] Submitting POP909 Phase 8 session
[phase8-submit] mode: ${MODE}
[phase8-submit] image: ${IMAGE_PATH}
[phase8-submit] workspace mount: ${WORKSPACE_HOST_PATH} -> /workspace
[phase8-submit] partition: ${PARTITION}
[phase8-submit] gpus: ${GPUS}
[phase8-submit] cpus-per-task: ${CPUS}
[phase8-submit] time limit: ${TIME_LIMIT}
[phase8-submit] log dir: ${LOG_DIR}
[phase8-submit] run name: ${VAE_RUN_NAME}
[phase8-submit] seed: ${VAE_SEED}
[phase8-submit] batch size: ${VAE_BATCH_SIZE}
[phase8-submit] target epochs: ${VAE_N_EPOCH}
[phase8-submit] epochs this job: ${VAE_RUN_EPOCHS_THIS_JOB}
[phase8-submit] resume from: ${VAE_RESUME_FROM:-}
[phase8-submit] W&B project: ${WANDB_PROJECT}
[phase8-submit] W&B entity: ${WANDB_ENTITY:-}
[phase8-submit] W&B group: ${WANDB_GROUP:-}
[phase8-submit] W&B run id: ${WANDB_RUN_ID:-}
[phase8-submit] W&B resume: ${WANDB_RESUME:-}
[phase8-submit] sample limits: train=${VAE_LIMIT_TRAIN_SAMPLES:-0}, val=${VAE_LIMIT_VAL_SAMPLES:-0}
[phase8-submit] execution: sbatch allocation -> non-interactive srun container -> /workspace/base_model
MSG

SBATCH_ARGS=(
  --parsable
  --job-name="${JOB_NAME}"
  --partition="${PARTITION}"
  --gres="gpu:${GPUS}"
  --ntasks="${NTASKS}"
  --cpus-per-task="${CPUS}"
  --time="${TIME_LIMIT}"
  --output="${LOG_DIR}/%x-%j.out"
  --error="${LOG_DIR}/%x-%j.err"
  --export=ALL
)

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" --wrap "${WRAPPED_CMD}")"
JOB_ID="${JOB_ID%%;*}"

cat <<MSG
[phase8-submit] Submitted job: ${JOB_ID}

[phase8-submit] Monitor:
  squeue -j ${JOB_ID}
  tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out
  tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err

[phase8-submit] After completion:
  sacct -j ${JOB_ID} --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES%60,ExitCode

[phase8-submit] Expected remote evidence:
  logs: ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out and ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err
  results: /home/${USER}/vae-textures-dev/base_model/result_*

[phase8-submit] Local download template after completion:
  scripts/download_pop909_phase8_artifacts.sh ${JOB_ID} REMOTE_RESULT_DIR

[phase8-submit] Human gate:
  Inspect Slurm, stdout/stderr, W&B config/metrics/artifacts, losses, epoch timing, and checkpoints.
  Record accept / rerun / block before launching the next Phase 8 session.
MSG
