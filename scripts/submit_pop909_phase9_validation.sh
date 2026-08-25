#!/usr/bin/env bash
set -euo pipefail

# Submit POP909 Phase 9 authors-vs-ours validation through Slurm.
# Modes:
#   smoke  Two validation segments, hard gate before full validation.
#   full   Full POP909 validation split, only after smoke acceptance.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-smoke}"

IMAGE_PATH="${CLUSTER_IMAGE:-/home/${USER}/devcontainer_images/dcli_fidle_tuto.squashfs}"
WORKSPACE_HOST_PATH="${WORKSPACE_HOST_PATH:-/home/${USER}/vae-textures-dev}"
PARTITION="${SLURM_PARTITION:-gpu}"
GPUS="${SLURM_GPUS:-1}"
CPUS="${SLURM_CPUS_PER_TASK:-4}"
NTASKS="${SLURM_NTASKS:-1}"
LOG_DIR="${PHASE9_LOG_DIR:-${PROJECT_ROOT}/logs/pop909-phase9-validation}"
JOB_NAME="${SLURM_JOB_NAME:-pop909-phase9-${MODE}}"

case "${MODE}" in
  smoke)
    CONFIG_PATH="/workspace/configs/pop909_conditioned_reconstruction_cluster_smoke.json"
    TIME_LIMIT="${SLURM_TIME:-00:30:00}"
    EXPECTED_RUN_DIR="/workspace/_artefatos/pop909-conditioned-reconstruction/cluster-smoke-four-way-2case"
    EXPECTED_REMOTE_RUN_DIR="${WORKSPACE_HOST_PATH}/_artefatos/pop909-conditioned-reconstruction/cluster-smoke-four-way-2case"
    ;;
  full)
    CONFIG_PATH="/workspace/configs/pop909_conditioned_reconstruction_cluster_full_validation.json"
    TIME_LIMIT="${SLURM_TIME:-08:00:00}"
    EXPECTED_RUN_DIR="/workspace/_artefatos/pop909-conditioned-reconstruction/cluster-full-validation-four-way"
    EXPECTED_REMOTE_RUN_DIR="${WORKSPACE_HOST_PATH}/_artefatos/pop909-conditioned-reconstruction/cluster-full-validation-four-way"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}"

CONTAINER_MOUNTS="${WORKSPACE_HOST_PATH}:/workspace,/lib64/libcuda.so.1:/nvidia/libcuda.so.1,/lib64/libnvidia-ml.so.1:/nvidia/libnvidia-ml.so.1,/lib64/libnvidia-ptxjitcompiler.so.1:/nvidia/libnvidia-ptxjitcompiler.so.1"

build_inner_cmd() {
  cat <<INNER
set -euo pipefail
export LD_LIBRARY_PATH=/nvidia\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
cd /workspace
echo "[phase9-submit] mode=${MODE}"
echo "[phase9-submit] config=${CONFIG_PATH}"
echo "[phase9-submit] expected_run_dir=${EXPECTED_RUN_DIR}"
echo "[phase9-submit] cwd=\$(pwd)"
echo "[phase9-submit] python=\$(python -V 2>&1)"
python - <<'PY'
import json
from pathlib import Path
import torch
cfg_path = Path("${CONFIG_PATH}")
cfg = json.loads(cfg_path.read_text())
print("[phase9-submit] cuda_available=", torch.cuda.is_available())
print("[phase9-submit] cuda_device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[phase9-submit] cuda_device_0=", torch.cuda.get_device_name(0))
print("[phase9-submit] run_id=", cfg.get("run_id"))
print("[phase9-submit] run_role=", cfg.get("run_role"))
print("[phase9-submit] sample_count=", cfg.get("sample_count"))
print("[phase9-submit] full_split_target=", cfg.get("full_split_target"))
print("[phase9-submit] fallback_used=", cfg.get("fallback_used"))
for label, ref in cfg.get("checkpoints", {}).items():
    path = Path(ref["path"])
    resolved = path if path.is_absolute() else Path("/workspace") / path
    print(f"[phase9-submit] checkpoint {label}: role={ref.get('role')} epoch={ref.get('epoch')} path={resolved} exists={resolved.exists()}")
PY
python -u pop909_conditioned_reconstruction.py --config "${CONFIG_PATH}" --validate-config --check-files
python -u pop909_conditioned_reconstruction.py --config "${CONFIG_PATH}"
echo "[phase9-submit] produced Phase 9 run directories:"
find /workspace/_artefatos/pop909-conditioned-reconstruction -maxdepth 2 -type f \( -name 'summary.json' -o -name 'comparison_wide.csv' -o -name 'comparison_manifest.jsonl' -o -name 'ranking_strata.json' \) -printf '%p
' 2>/dev/null | sort || true
INNER
}

INNER_CMD="$(build_inner_cmd)"
printf -v WRAPPED_CMD 'srun --container-image=%q --container-mounts=%q --container-workdir=/workspace bash -lc %q' "${IMAGE_PATH}" "${CONTAINER_MOUNTS}" "${INNER_CMD}"

cat <<MSG
[phase9-submit] Submitting POP909 Phase 9 validation
[phase9-submit] mode: ${MODE}
[phase9-submit] image: ${IMAGE_PATH}
[phase9-submit] workspace mount: ${WORKSPACE_HOST_PATH} -> /workspace
[phase9-submit] partition: ${PARTITION}
[phase9-submit] gpus: ${GPUS}
[phase9-submit] cpus-per-task: ${CPUS}
[phase9-submit] time limit: ${TIME_LIMIT}
[phase9-submit] log dir: ${LOG_DIR}
[phase9-submit] job name: ${JOB_NAME}
[phase9-submit] config: ${CONFIG_PATH}
[phase9-submit] expected container RUN_DIR: ${EXPECTED_RUN_DIR}
[phase9-submit] expected remote RUN_DIR: ${EXPECTED_REMOTE_RUN_DIR}
[phase9-submit] execution: sbatch allocation -> non-interactive srun container -> /workspace
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
[phase9-submit] Submitted job: ${JOB_ID}

[phase9-submit] Monitor:
  squeue -j ${JOB_ID} -o "%.18i %.9P %.30j %.8u %.2t %.10M %.10l %.6D %R"
  tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out
  tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err

[phase9-submit] After completion:
  sacct -j ${JOB_ID} --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES%60,ExitCode

[phase9-submit] Expected remote evidence:
  logs: ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out and ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err
  container RUN_DIR: ${EXPECTED_RUN_DIR}
  remote RUN_DIR: ${EXPECTED_REMOTE_RUN_DIR}

[phase9-submit] Local download template after completion:
  scripts/download_pop909_phase9_artifacts.sh ${JOB_ID} ${EXPECTED_REMOTE_RUN_DIR}

[phase9-submit] Human gate:
  Smoke mode: inspect logs, CUDA, checkpoint exists=true, row_count=2, CSV/JSONL/rankings/summary, then record accept/rerun/block.
  Full mode: run only after accepted smoke; inspect row_count/manifest/summary/rankings before 24-case selection.
MSG
