
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/vae-textures-dev}"
BATCH_FILE="${1:-}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/COMMUDataset/losses_enriched}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ -z "$BATCH_FILE" ]]; then
  echo "Usage: scripts/run_commu_enriched_batch.sh /absolute/path/to/batch_XXX.txt"
  exit 1
fi

cd "$REPO_ROOT/NotebooksVAESymTex"

ARGS=(
  python3 generate_commu_enriched_loss_dataset.py
  --batch-file "$BATCH_FILE"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
)

if [[ "$SKIP_EXISTING" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

echo "[commu-batch] repo=$REPO_ROOT"
echo "[commu-batch] batch_file=$BATCH_FILE"
echo "[commu-batch] device=$DEVICE"
echo "[commu-batch] output_dir=$OUTPUT_DIR"
"${ARGS[@]}"
