#!/usr/bin/env bash
set -euo pipefail

# Small POP909 checkpoint/resume probe for the canonical base_model/train.py path.
# Default mode runs two legs in one shell: initial checkpoint leg, then resume leg.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODE="${1:-${VAE_RESUME_PROBE_MODE:-both}}"
BASE_RUN_NAME="${VAE_RUN_NAME:-pop909-resume-probe-$(date +%Y%m%d-%H%M%S)}"

export VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-128}"
export VAE_LIMIT_TRAIN_SAMPLES="${VAE_LIMIT_TRAIN_SAMPLES:-256}"
export VAE_LIMIT_VAL_SAMPLES="${VAE_LIMIT_VAL_SAMPLES:-128}"
export VAE_LR="${VAE_LR:-1e-3}"
export VAE_RUN_EPOCHS_THIS_JOB="${VAE_RUN_EPOCHS_THIS_JOB:-1}"
export VAE_FULL_CHECKPOINT_POLICY="${VAE_FULL_CHECKPOINT_POLICY:-epoch-state,last-state,final-state}"

if [[ "${WANDB_ENABLED:-1}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  export WANDB_ENABLED="${WANDB_ENABLED:-1}"
  export WANDB_PROJECT="${WANDB_PROJECT:-pop909-reproduction}"
  export WANDB_GROUP="${WANDB_GROUP:-${BASE_RUN_NAME}}"
  export WANDB_TAGS="${WANDB_TAGS:-pop909,resume-probe,phase7}"
  export WANDB_CHECKPOINT_POLICY="${WANDB_CHECKPOINT_POLICY:-valid,final,epoch-state,last-state,final-state}"
  if [[ -z "${WANDB_PROJECT:-}" ]]; then
    echo "[resume-probe][erro] WANDB_ENABLED is set, but WANDB_PROJECT is missing." >&2
    exit 2
  fi
  if [[ "${WANDB_MODE:-}" != "offline" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "[resume-probe][erro] WANDB_ENABLED is set, but WANDB_API_KEY is missing." >&2
    echo "[resume-probe][erro] Source ~/.config/wandb/env.sh or set WANDB_MODE=offline." >&2
    exit 2
  fi
fi

print_common_config() {
  cat <<MSG
[resume-probe] mode=${MODE}
[resume-probe] base run name=${BASE_RUN_NAME}
[resume-probe] VAE_BATCH_SIZE=${VAE_BATCH_SIZE}
[resume-probe] VAE_LIMIT_TRAIN_SAMPLES=${VAE_LIMIT_TRAIN_SAMPLES}
[resume-probe] VAE_LIMIT_VAL_SAMPLES=${VAE_LIMIT_VAL_SAMPLES}
[resume-probe] VAE_RUN_EPOCHS_THIS_JOB=${VAE_RUN_EPOCHS_THIS_JOB}
[resume-probe] VAE_FULL_CHECKPOINT_POLICY=${VAE_FULL_CHECKPOINT_POLICY}
[resume-probe] WANDB_ENABLED=${WANDB_ENABLED:-0}
[resume-probe] WANDB_PROJECT=${WANDB_PROJECT:-}
[resume-probe] WANDB_ENTITY=${WANDB_ENTITY:-}
[resume-probe] WANDB_GROUP=${WANDB_GROUP:-}
[resume-probe] execution: canonical python -u train.py
MSG
}

find_last_state() {
  local run_name="$1"
  find . -path "./result_*/models/${run_name}_last-state_state.pt" -print | sort | tail -n 1
}

run_initial() {
  export VAE_N_EPOCH="${VAE_INITIAL_N_EPOCH:-1}"
  export VAE_RUN_NAME="${INITIAL_RUN_NAME:-${BASE_RUN_NAME}-initial}"
  unset VAE_RESUME_FROM
  echo "[resume-probe] Starting initial leg: VAE_RUN_NAME=${VAE_RUN_NAME}"
  python -u train.py
  local state_path
  state_path="$(find_last_state "${VAE_RUN_NAME}")"
  if [[ -z "${state_path}" || ! -f "${state_path}" ]]; then
    echo "[resume-probe][erro] Could not find last-state checkpoint for ${VAE_RUN_NAME}" >&2
    exit 3
  fi
  echo "[resume-probe] Initial leg last-state checkpoint: ${state_path}"
  printf '%s
' "${state_path}" > .resume_probe_last_state_path
}

run_resume() {
  export VAE_RESUME_FROM="${VAE_RESUME_FROM:-$(cat .resume_probe_last_state_path 2>/dev/null || true)}"
  if [[ -z "${VAE_RESUME_FROM}" || ! -f "${VAE_RESUME_FROM}" ]]; then
    echo "[resume-probe][erro] VAE_RESUME_FROM is missing or not a file: ${VAE_RESUME_FROM:-}" >&2
    exit 4
  fi
  export VAE_N_EPOCH="${VAE_RESUME_TARGET_N_EPOCH:-2}"
  export VAE_RUN_NAME="${RESUME_RUN_NAME:-${BASE_RUN_NAME}-resume}"
  echo "[resume-probe] Starting resume leg: VAE_RUN_NAME=${VAE_RUN_NAME}"
  echo "[resume-probe] VAE_RESUME_FROM=${VAE_RESUME_FROM}"
  python -u train.py
  local state_path
  state_path="$(find_last_state "${VAE_RUN_NAME}")"
  if [[ -z "${state_path}" || ! -f "${state_path}" ]]; then
    echo "[resume-probe][erro] Could not find resumed last-state checkpoint for ${VAE_RUN_NAME}" >&2
    exit 5
  fi
  echo "[resume-probe] Resume leg last-state checkpoint: ${state_path}"
}

print_common_config
case "${MODE}" in
  initial)
    run_initial
    ;;
  resume)
    run_resume
    ;;
  both)
    run_initial
    run_resume
    ;;
  *)
    echo "Usage: $0 [initial|resume|both]" >&2
    exit 2
    ;;
esac

echo "[resume-probe] Completed mode=${MODE}"
