#!/usr/bin/env bash
set -euo pipefail

# Submit the POP909 checkpoint/resume probe through Slurm sbatch.
# Run from the cluster login node after syncing runtime files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_PATH="${CLUSTER_IMAGE:-/home/${USER}/devcontainer_images/dcli_fidle_tuto.squashfs}"
WORKSPACE_HOST_PATH="${WORKSPACE_HOST_PATH:-/home/${USER}/vae-textures-dev}"
PARTITION="${SLURM_PARTITION:-gpu}"
GPUS="${SLURM_GPUS:-1}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"
NTASKS="${SLURM_NTASKS:-1}"
TIME_LIMIT="${SLURM_TIME:-00:45:00}"
JOB_NAME="${SLURM_JOB_NAME:-pop909-resume-probe}"
LOG_DIR="${RESUME_PROBE_LOG_DIR:-${PROJECT_ROOT}/logs/pop909-resume-probe}"

export VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-128}"
export VAE_LIMIT_TRAIN_SAMPLES="${VAE_LIMIT_TRAIN_SAMPLES:-256}"
export VAE_LIMIT_VAL_SAMPLES="${VAE_LIMIT_VAL_SAMPLES:-128}"
export VAE_RUN_EPOCHS_THIS_JOB="${VAE_RUN_EPOCHS_THIS_JOB:-1}"
export VAE_RUN_NAME="${VAE_RUN_NAME:-pop909-resume-probe-bs${VAE_BATCH_SIZE}-train${VAE_LIMIT_TRAIN_SAMPLES}-$(date +%Y%m%d-%H%M%S)}"
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-pop909-reproduction}"
export WANDB_GROUP="${WANDB_GROUP:-${VAE_RUN_NAME}}"
export WANDB_TAGS="${WANDB_TAGS:-pop909,resume-probe,phase7}"
export WANDB_CHECKPOINT_POLICY="${WANDB_CHECKPOINT_POLICY:-valid,final,epoch-state,last-state,final-state}"

# Source private W&B metadata if present. WANDB_API_KEY may be defined here, but the key is never printed.
if [[ -f "${HOME}/.config/wandb/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.config/wandb/env.sh"
fi

mkdir -p "${LOG_DIR}"

CONTAINER_MOUNTS="${WORKSPACE_HOST_PATH}:/workspace,/lib64/libcuda.so.1:/nvidia/libcuda.so.1,/lib64/libnvidia-ml.so.1:/nvidia/libnvidia-ml.so.1,/lib64/libnvidia-ptxjitcompiler.so.1:/nvidia/libnvidia-ptxjitcompiler.so.1"
INNER_CMD='export LD_LIBRARY_PATH=/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/tmp/wandb-cache}; export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-/tmp/wandb-config}; export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/tmp/wandb-data}; export WANDB_DIR=/workspace/base_model; mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"; cd /workspace/base_model; ./run_pop909_resume_probe.sh both'
printf -v WRAPPED_CMD 'srun --container-image=%q --container-mounts=%q --container-workdir=/workspace bash -lc %q' "${IMAGE_PATH}" "${CONTAINER_MOUNTS}" "${INNER_CMD}"

cat <<MSG
[resume-submit] Submitting POP909 checkpoint/resume probe
[resume-submit] image: ${IMAGE_PATH}
[resume-submit] workspace mount: ${WORKSPACE_HOST_PATH} -> /workspace
[resume-submit] partition: ${PARTITION}
[resume-submit] gpus: ${GPUS}
[resume-submit] cpus-per-task: ${CPUS}
[resume-submit] time limit: ${TIME_LIMIT}
[resume-submit] log dir: ${LOG_DIR}
[resume-submit] run name base: ${VAE_RUN_NAME}
[resume-submit] batch size: ${VAE_BATCH_SIZE}
[resume-submit] train sample limit: ${VAE_LIMIT_TRAIN_SAMPLES}
[resume-submit] val sample limit: ${VAE_LIMIT_VAL_SAMPLES}
[resume-submit] W&B project: ${WANDB_PROJECT}
[resume-submit] W&B entity: ${WANDB_ENTITY:-}
[resume-submit] W&B group: ${WANDB_GROUP}
[resume-submit] execution: sbatch allocation -> non-interactive srun container -> base_model/run_pop909_resume_probe.sh -> python -u train.py
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
[resume-submit] Submitted job: ${JOB_ID}

[resume-submit] Monitor:
  squeue -j ${JOB_ID}
  tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out
  tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err

[resume-submit] After completion:
  sacct -j ${JOB_ID} --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES%60,ExitCode

[resume-submit] Send Codex:
  - job id: ${JOB_ID}
  - stdout tail: ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out
  - stderr tail: ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err
  - sacct output
  - W&B run URLs/config/metrics/artifacts for group ${WANDB_GROUP}
  - initial last-state checkpoint path printed by the logs
  - resume VAE_RESUME_FROM path printed by the logs
  - resumed last-state checkpoint path printed by the logs
MSG
