#!/usr/bin/env bash
set -euo pipefail

# Representative POP909 timing probe wrapper.
#
# This script intentionally keeps base_model/train.py as the canonical training
# implementation. It only supplies environment defaults that make the run useful
# for estimating representative training time before the full Phase 7 launch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Representative defaults. Override any value from the submit script or shell.
# Batch size 128 matches the current canonical train.py default. Sample limits
# bound the probe while preserving the same batch shape as the real run.
export VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-128}"
export VAE_N_EPOCH="${VAE_N_EPOCH:-1}"
export VAE_LIMIT_TRAIN_SAMPLES="${VAE_LIMIT_TRAIN_SAMPLES:-4096}"
export VAE_LIMIT_VAL_SAMPLES="${VAE_LIMIT_VAL_SAMPLES:-512}"
export VAE_LR="${VAE_LR:-1e-3}"
export VAE_RUN_NAME="${VAE_RUN_NAME:-pop909-representative-timing-bs${VAE_BATCH_SIZE}-train${VAE_LIMIT_TRAIN_SAMPLES}-val${VAE_LIMIT_VAL_SAMPLES}}"

if [[ "${WANDB_ENABLED:-1}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  export WANDB_ENABLED="${WANDB_ENABLED:-1}"
  export WANDB_PROJECT="${WANDB_PROJECT:-pop909-reproduction}"
  if [[ -z "${WANDB_PROJECT:-}" ]]; then
    echo "[timing-probe][erro] WANDB_ENABLED is set, but WANDB_PROJECT is missing." >&2
    exit 2
  fi
  if [[ "${WANDB_MODE:-}" != "offline" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "[timing-probe][erro] WANDB_ENABLED is set, but WANDB_API_KEY is missing." >&2
    echo "[timing-probe][erro] Source ~/.config/wandb/env.sh or set WANDB_MODE=offline for a non-uploading test." >&2
    exit 2
  fi
fi

cat <<MSG
[timing-probe] Running canonical train.py with representative timing settings:
[timing-probe]   VAE_BATCH_SIZE=${VAE_BATCH_SIZE}
[timing-probe]   VAE_N_EPOCH=${VAE_N_EPOCH}
[timing-probe]   VAE_LIMIT_TRAIN_SAMPLES=${VAE_LIMIT_TRAIN_SAMPLES}
[timing-probe]   VAE_LIMIT_VAL_SAMPLES=${VAE_LIMIT_VAL_SAMPLES}
[timing-probe]   VAE_LR=${VAE_LR}
[timing-probe]   VAE_RUN_NAME=${VAE_RUN_NAME}
[timing-probe]   WANDB_ENABLED=${WANDB_ENABLED:-0}
[timing-probe]   WANDB_PROJECT=${WANDB_PROJECT:-}
[timing-probe]   WANDB_ENTITY=${WANDB_ENTITY:-}
[timing-probe]   WANDB_MODE=${WANDB_MODE:-}
[timing-probe] Notes:
[timing-probe]   - This is a bounded timing probe, not the full representative training run.
[timing-probe]   - The probe keeps train.py, POP909 loader defaults, model, beta, weights, and optimizer path unchanged.
[timing-probe]   - Estimate the full epoch from observed train steps/sec and the full train step count.
MSG

python -u train.py
