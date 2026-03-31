#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-2}"
export VAE_N_EPOCH="${VAE_N_EPOCH:-1}"
export VAE_LIMIT_TRAIN_SAMPLES="${VAE_LIMIT_TRAIN_SAMPLES:-4}"
export VAE_LIMIT_VAL_SAMPLES="${VAE_LIMIT_VAL_SAMPLES:-2}"
export VAE_RUN_NAME="${VAE_RUN_NAME:-phase3-cluster-proof}"

cat <<MSG
[cluster-proof] Running train.py with:
[cluster-proof]   VAE_BATCH_SIZE=${VAE_BATCH_SIZE}
[cluster-proof]   VAE_N_EPOCH=${VAE_N_EPOCH}
[cluster-proof]   VAE_LIMIT_TRAIN_SAMPLES=${VAE_LIMIT_TRAIN_SAMPLES}
[cluster-proof]   VAE_LIMIT_VAL_SAMPLES=${VAE_LIMIT_VAL_SAMPLES}
[cluster-proof]   VAE_RUN_NAME=${VAE_RUN_NAME}
MSG

python -u train.py
