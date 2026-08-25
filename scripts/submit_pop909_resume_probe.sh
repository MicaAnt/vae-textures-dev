#!/usr/bin/env bash
set -euo pipefail

# Submit POP909 checkpoint/resume validation through Slurm sbatch.
#
# What this script does, in plain language:
#   1. Requests ONE short GPU Slurm job.
#   2. Inside that one job, runs the same numbered A/B continuity proof used on CPU.
#   3. Turns W&B on for the three mini training legs.
#   4. Gives each mini training leg its own W&B run name.
#   5. Keeps the three W&B runs grouped together with WANDB_GROUP.
#   6. Prints exactly which files/logs to inspect before accepting the result.
#
# This script does not launch the representative long training run.
# It only runs the bounded Phase 7 validation gate.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-${RESUME_PROBE_SUBMIT_MODE:-exact}}"

IMAGE_PATH="${CLUSTER_IMAGE:-/home/${USER}/devcontainer_images/dcli_fidle_tuto.squashfs}"
WORKSPACE_HOST_PATH="${WORKSPACE_HOST_PATH:-/home/${USER}/vae-textures-dev}"
PARTITION="${SLURM_PARTITION:-gpu}"
GPUS="${SLURM_GPUS:-1}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"
NTASKS="${SLURM_NTASKS:-1}"
TIME_LIMIT="${SLURM_TIME:-00:45:00}"
JOB_NAME="${SLURM_JOB_NAME:-pop909-resume-probe}"
LOG_DIR="${RESUME_PROBE_LOG_DIR:-${PROJECT_ROOT}/logs/pop909-resume-probe}"

# These limits make the validation short. They are intentionally tiny: this is
# a pre-launch correctness/observability check, not the real representative run.
export RESUME_CONTINUITY_USE_GPU="${RESUME_CONTINUITY_USE_GPU:-1}"
export RESUME_CONTINUITY_BATCH_SIZE="${RESUME_CONTINUITY_BATCH_SIZE:-2}"
export RESUME_CONTINUITY_LIMIT_TRAIN_SAMPLES="${RESUME_CONTINUITY_LIMIT_TRAIN_SAMPLES:-4}"
export RESUME_CONTINUITY_LIMIT_VAL_SAMPLES="${RESUME_CONTINUITY_LIMIT_VAL_SAMPLES:-2}"
export RESUME_CONTINUITY_LIMIT_TRAIN_SHUFFLE="${RESUME_CONTINUITY_LIMIT_TRAIN_SHUFFLE:-1}"

# W&B is ON by default for this cluster validation because Phase 8 will use W&B.
# The three mini training legs become three separate W&B runs in one group.
export RESUME_CONTINUITY_WANDB="${RESUME_CONTINUITY_WANDB:-1}"
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-pop909-reproduction}"
export VAE_RUN_NAME="${VAE_RUN_NAME:-pop909-resume-probe-$(date +%Y%m%d-%H%M%S)}"
export WANDB_GROUP="${WANDB_GROUP:-${VAE_RUN_NAME}}"
export WANDB_TAGS="${WANDB_TAGS:-pop909,resume-continuity,phase7}"
export WANDB_CHECKPOINT_POLICY="${WANDB_CHECKPOINT_POLICY:-valid,final,epoch-state,last-state,final-state}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb-config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/tmp/wandb-data}"

# These env vars are used only by the older wandb-probe mode. They stay here so
# that mode remains available, but exact mode uses the RESUME_CONTINUITY_* vars.
export VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-128}"
export VAE_LIMIT_TRAIN_SAMPLES="${VAE_LIMIT_TRAIN_SAMPLES:-256}"
export VAE_LIMIT_VAL_SAMPLES="${VAE_LIMIT_VAL_SAMPLES:-128}"
export VAE_RUN_EPOCHS_THIS_JOB="${VAE_RUN_EPOCHS_THIS_JOB:-1}"

# Source private W&B metadata if present. WANDB_API_KEY may be defined here.
# The script must never echo the API key value.
if [[ -f "${HOME}/.config/wandb/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.config/wandb/env.sh"
fi

mkdir -p "${LOG_DIR}"

CONTAINER_MOUNTS="${WORKSPACE_HOST_PATH}:/workspace,/lib64/libcuda.so.1:/nvidia/libcuda.so.1,/lib64/libnvidia-ml.so.1:/nvidia/libnvidia-ml.so.1,/lib64/libnvidia-ptxjitcompiler.so.1:/nvidia/libnvidia-ptxjitcompiler.so.1"

# This is the command that will run INSIDE the container, INSIDE the single
# Slurm allocation. It is deliberately written as a readable script instead of
# a one-line wrapper so you can inspect what will happen before launching.
build_exact_inner_cmd() {
  cat <<'INNER'
set -euo pipefail

# Make the host GPU driver libraries visible inside the container.
export LD_LIBRARY_PATH=/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Keep W&B runtime/cache files in temporary container-local directories.
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/tmp/wandb-cache}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-/tmp/wandb-config}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/tmp/wandb-data}
export WANDB_DIR=/workspace/base_model
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

cd /workspace

# These options are read by base_model/resume_continuity_test/test_common.py.
# They make the numbered CPU proof run on GPU, with W&B enabled.
export RESUME_CONTINUITY_USE_GPU=${RESUME_CONTINUITY_USE_GPU:-1}
export RESUME_CONTINUITY_WANDB=${RESUME_CONTINUITY_WANDB:-1}
export RESUME_CONTINUITY_BATCH_SIZE=${RESUME_CONTINUITY_BATCH_SIZE:-2}
export RESUME_CONTINUITY_LIMIT_TRAIN_SAMPLES=${RESUME_CONTINUITY_LIMIT_TRAIN_SAMPLES:-4}
export RESUME_CONTINUITY_LIMIT_VAL_SAMPLES=${RESUME_CONTINUITY_LIMIT_VAL_SAMPLES:-2}
export RESUME_CONTINUITY_LIMIT_TRAIN_SHUFFLE=${RESUME_CONTINUITY_LIMIT_TRAIN_SHUFFLE:-1}

# STEP 01: Reference path.
# This launches mini training run #1 in W&B.
# W&B run name shape: resume-test-direct-<run_id>
# Meaning: train epoch 1, then epoch 2 without interruption.
echo "[resume-submit] STEP 01 direct uninterrupted 2 epochs"
export RESUME_CONTINUITY_NEW_RUN=1
python base_model/resume_continuity_test/01_train_direct_2_epochs.py
unset RESUME_CONTINUITY_NEW_RUN

# STEP 02: Interrupted path, first leg.
# This launches mini training run #2 in W&B.
# W&B run name shape: resume-test-initial-<same_run_id>
# Meaning: train epoch 1 only, save a full-state last-state checkpoint.
echo "[resume-submit] STEP 02 initial 1 epoch + checkpoint"
python base_model/resume_continuity_test/02_train_one_epoch_checkpoint.py

# STEP 03: Interrupted path, resumed leg.
# This launches mini training run #3 in W&B.
# W&B run name shape: resume-test-resumed-<same_run_id>
# Meaning: load the STEP 02 checkpoint, then run epoch 2.
echo "[resume-submit] STEP 03 resume from checkpoint and run epoch 2"
python base_model/resume_continuity_test/03_resume_second_epoch.py

# STEP 04: Compare final state from STEP 01 versus STEP 03.
# This is not a training run. It writes state_comparison.txt and reports whether
# model/training state matched or differed.
echo "[resume-submit] STEP 04 compare direct final state vs resumed final state"
python base_model/resume_continuity_test/04_compare_final_states.py

# STEP 05: Produce tensor-by-tensor inspection CSV for human review.
# This is also not a training run.
echo "[resume-submit] STEP 05 inspect weights"
python base_model/resume_continuity_test/05_inspect_weights.py

# Print report locations so the stdout log itself tells us what to fetch.
echo "[resume-submit] Reports:"
ls -l base_model/resume_continuity_test/outputs \
      base_model/resume_continuity_test/outputs/reports
INNER
}

# Older mode kept as a fallback. It runs one initial+resume probe through the
# wrapper script, but it is less faithful to the accepted numbered A/B proof.
build_wandb_probe_inner_cmd() {
  cat <<'INNER'
set -euo pipefail
export LD_LIBRARY_PATH=/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/tmp/wandb-cache}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-/tmp/wandb-config}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/tmp/wandb-data}
export WANDB_DIR=/workspace/base_model
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
cd /workspace/base_model
./run_pop909_resume_probe.sh both
INNER
}

case "${MODE}" in
  exact)
    INNER_CMD="$(build_exact_inner_cmd)"
    ;;
  wandb-probe)
    INNER_CMD="$(build_wandb_probe_inner_cmd)"
    ;;
  *)
    echo "Usage: $0 [exact|wandb-probe]" >&2
    exit 2
    ;;
esac

printf -v WRAPPED_CMD 'srun --container-image=%q --container-mounts=%q --container-workdir=/workspace bash -lc %q' "${IMAGE_PATH}" "${CONTAINER_MOUNTS}" "${INNER_CMD}"

cat <<MSG
[resume-submit] Submitting POP909 checkpoint/resume validation
[resume-submit] mode: ${MODE}
[resume-submit] image: ${IMAGE_PATH}
[resume-submit] workspace mount: ${WORKSPACE_HOST_PATH} -> /workspace
[resume-submit] partition: ${PARTITION}
[resume-submit] gpus: ${GPUS}
[resume-submit] cpus-per-task: ${CPUS}
[resume-submit] time limit: ${TIME_LIMIT}
[resume-submit] log dir: ${LOG_DIR}
[resume-submit] execution: one sbatch allocation -> one non-interactive srun container -> numbered A/B train.py proof
[resume-submit] exact mode sample limits: batch=${RESUME_CONTINUITY_BATCH_SIZE}, train=${RESUME_CONTINUITY_LIMIT_TRAIN_SAMPLES}, val=${RESUME_CONTINUITY_LIMIT_VAL_SAMPLES}, train_shuffle=${RESUME_CONTINUITY_LIMIT_TRAIN_SHUFFLE}
[resume-submit] exact mode W&B enabled: ${RESUME_CONTINUITY_WANDB}
[resume-submit] W&B project: ${WANDB_PROJECT}
[resume-submit] W&B entity: ${WANDB_ENTITY:-}
[resume-submit] W&B group: ${WANDB_GROUP}
[resume-submit] W&B run names will be:
  - resume-test-direct-<run_id>
  - resume-test-initial-<run_id>
  - resume-test-resumed-<run_id>
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
  - mode: ${MODE}
  - job id: ${JOB_ID}
  - stdout tail: ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out
  - stderr tail: ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err
  - sacct output
  - manifest: base_model/resume_continuity_test/outputs/manifest.json
  - comparison report: base_model/resume_continuity_test/outputs/reports/state_comparison.txt
  - weight reports: base_model/resume_continuity_test/outputs/reports/weight_diff_report.csv and weight_inspection.csv
  - timing lines containing: Epoch train/eval seconds, Saved model weights in, Saved training state
  - W&B group: ${WANDB_GROUP}
  - W&B runs for legs: direct, initial, resumed
  - W&B config/metrics/artifacts for those three runs
MSG
