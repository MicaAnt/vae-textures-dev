#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run_paper_objective_eval.sh path/to/checkpoint.pt [extra args...]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift
DEVICE="${EVAL_DEVICE:-cpu}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
OUTPUT="${EVAL_OUTPUT:-}"

CMD=(
  python3 evaluate_paper_objective_metrics.py
  --checkpoint "$CHECKPOINT"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --max-batches "$MAX_BATCHES"
)

if [[ -n "$OUTPUT" ]]; then
  CMD+=(--output "$OUTPUT")
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf '[paper-eval] Command: '
printf '%q ' "${CMD[@]}"
printf '
'

"${CMD[@]}"
