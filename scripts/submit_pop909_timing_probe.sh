#!/usr/bin/env bash
set -euo pipefail

# Submit a representative POP909 timing probe through Slurm sbatch.
# Run this on the cluster login node from ~/vae-textures-dev after syncing the
# runtime package. The batch job then uses non-interactive srun to enter the
# same GPU container style already validated by scripts/cluster_gpu_shell.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_PATH="${CLUSTER_IMAGE:-/home/${USER}/devcontainer_images/dcli_fidle_tuto.squashfs}"
WORKSPACE_HOST_PATH="${WORKSPACE_HOST_PATH:-/home/${USER}/vae-textures-dev}"
PARTITION="${SLURM_PARTITION:-gpu}"
GPUS="${SLURM_GPUS:-1}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"
NTASKS="${SLURM_NTASKS:-1}"
TIME_LIMIT="${SLURM_TIME:-02:00:00}"
JOB_NAME="${SLURM_JOB_NAME:-pop909-timing}"
LOG_DIR="${TIMING_LOG_DIR:-${PROJECT_ROOT}/logs/pop909-timing}"

# Probe defaults. They are exported so the batch job, container command, and
# train.py see them, but callers can override any value before submission.
export VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-128}"
export VAE_N_EPOCH="${VAE_N_EPOCH:-1}"
export VAE_LIMIT_TRAIN_SAMPLES="${VAE_LIMIT_TRAIN_SAMPLES:-4096}"
export VAE_LIMIT_VAL_SAMPLES="${VAE_LIMIT_VAL_SAMPLES:-512}"
export VAE_LR="${VAE_LR:-1e-3}"
export VAE_RUN_NAME="${VAE_RUN_NAME:-pop909-timing-bs${VAE_BATCH_SIZE}-train${VAE_LIMIT_TRAIN_SAMPLES}-$(date +%Y%m%d-%H%M%S)}"
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-pop909-reproduction}"

# Source private W&B metadata if present. This may define WANDB_API_KEY,
# WANDB_ENTITY, and cache paths. The key is never printed.
if [[ -f "${HOME}/.config/wandb/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.config/wandb/env.sh"
fi

mkdir -p "${LOG_DIR}"

CONTAINER_MOUNTS="${WORKSPACE_HOST_PATH}:/workspace,/lib64/libcuda.so.1:/nvidia/libcuda.so.1,/lib64/libnvidia-ml.so.1:/nvidia/libnvidia-ml.so.1,/lib64/libnvidia-ptxjitcompiler.so.1:/nvidia/libnvidia-ptxjitcompiler.so.1"
INNER_CMD='export LD_LIBRARY_PATH=/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/tmp/wandb-cache}; export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-/tmp/wandb-config}; export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/tmp/wandb-data}; export WANDB_DIR=/workspace/base_model; mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"; cd /workspace/base_model; ./run_pop909_timing_probe.sh'
printf -v WRAPPED_CMD 'srun --container-image=%q --container-mounts=%q --container-workdir=/workspace bash -lc %q' "${IMAGE_PATH}" "${CONTAINER_MOUNTS}" "${INNER_CMD}"

cat <<MSG
[timing-submit] Submitting POP909 representative timing probe
[timing-submit] image: ${IMAGE_PATH}
[timing-submit] workspace mount: ${WORKSPACE_HOST_PATH} -> /workspace
[timing-submit] partition: ${PARTITION}
[timing-submit] gpus: ${GPUS}
[timing-submit] cpus-per-task: ${CPUS}
[timing-submit] time limit: ${TIME_LIMIT}
[timing-submit] log dir: ${LOG_DIR}
[timing-submit] run name: ${VAE_RUN_NAME}
[timing-submit] batch size: ${VAE_BATCH_SIZE}
[timing-submit] train sample limit: ${VAE_LIMIT_TRAIN_SAMPLES}
[timing-submit] val sample limit: ${VAE_LIMIT_VAL_SAMPLES}
[timing-submit] W&B project: ${WANDB_PROJECT}
[timing-submit] W&B entity: ${WANDB_ENTITY:-}
[timing-submit] execution: sbatch allocation -> non-interactive srun container -> base_model/train.py
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
[timing-submit] Submitted job: ${JOB_ID}

[timing-submit] Monitor:
  squeue -j ${JOB_ID}
  tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out
  tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err

[timing-submit] After completion:
  sacct -j ${JOB_ID} --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES%60,ExitCode

[timing-submit] Send Codex:
  - job id: ${JOB_ID}
  - stdout tail: ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out
  - stderr tail: ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err
  - sacct output
  - W&B run URL/config/metrics for ${VAE_RUN_NAME}
MSG
