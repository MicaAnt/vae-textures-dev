#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-2}"
export VAE_N_EPOCH="${VAE_N_EPOCH:-1}"
export VAE_LIMIT_TRAIN_SAMPLES="${VAE_LIMIT_TRAIN_SAMPLES:-4}"
export VAE_LIMIT_VAL_SAMPLES="${VAE_LIMIT_VAL_SAMPLES:-2}"
export VAE_RUN_NAME="${VAE_RUN_NAME:-phase3-cluster-proof}"

if [[ "${WANDB_ENABLED:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  if [[ -z "${WANDB_PROJECT:-}" ]]; then
    echo "[cluster-proof][erro] WANDB_ENABLED is set, but WANDB_PROJECT is missing." >&2
    exit 2
  fi
  if [[ "${WANDB_MODE:-}" != "offline" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "[cluster-proof][erro] WANDB_ENABLED is set, but WANDB_API_KEY is missing." >&2
    echo "[cluster-proof][erro] Export WANDB_API_KEY or set WANDB_MODE=offline for a smoke test." >&2
    exit 2
  fi
fi

cat <<MSG
[cluster-proof] Running train.py with:
[cluster-proof]   VAE_BATCH_SIZE=${VAE_BATCH_SIZE}
[cluster-proof]   VAE_N_EPOCH=${VAE_N_EPOCH}
[cluster-proof]   VAE_LIMIT_TRAIN_SAMPLES=${VAE_LIMIT_TRAIN_SAMPLES}
[cluster-proof]   VAE_LIMIT_VAL_SAMPLES=${VAE_LIMIT_VAL_SAMPLES}
[cluster-proof]   VAE_RUN_NAME=${VAE_RUN_NAME}
[cluster-proof]   WANDB_ENABLED=${WANDB_ENABLED:-0}
[cluster-proof]   WANDB_PROJECT=${WANDB_PROJECT:-}
[cluster-proof]   WANDB_MODE=${WANDB_MODE:-}
MSG

python -u train.py
