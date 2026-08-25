#!/usr/bin/env bash
set -euo pipefail

# Always run from base_model so relative paths inside train.py behave the same
# whether the wrapper is launched directly or from another directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Tiny defaults for an operational smoke. Callers may override any value from
# the terminal, for example: VAE_RUN_NAME=my-smoke ./run_cluster_proof.sh
export VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-2}"
export VAE_N_EPOCH="${VAE_N_EPOCH:-1}"
export VAE_LIMIT_TRAIN_SAMPLES="${VAE_LIMIT_TRAIN_SAMPLES:-4}"
export VAE_LIMIT_VAL_SAMPLES="${VAE_LIMIT_VAL_SAMPLES:-2}"
export VAE_RUN_NAME="${VAE_RUN_NAME:-phase3-cluster-proof}"

# W&B is opt-in. If enabled, fail before training when required metadata or
# credentials are missing. The key is checked for presence only; it is never
# printed by this wrapper.
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

# Print the non-secret execution contract before handing off to train.py. This
# is the evidence we inspect to confirm the smoke is bounded and W&B is enabled.
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

# This is the canonical training path. The wrapper only sets environment knobs;
# it does not bypass or replace train.py.
python -u train.py
