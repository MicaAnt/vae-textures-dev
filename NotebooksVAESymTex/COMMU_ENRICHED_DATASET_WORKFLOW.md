
# COMMU Enriched Dataset Workflow

This workflow creates a **new** per-segment dataset for study notebooks without overwriting the existing `COMMUDataset/losses/` directory.

## Purpose

The original COMMU segment cache already stores:

- `z_chd`
- `z_txt`
- `final_loss`
- `kl_loss`, `kl_chd`, `kl_rhy`
- selected COMMU metadata

For the notebook-driven latent-space study, we also want explicit reconstruction and didactic harmony/texture loss components:

- `recon_loss`
- `pitch_loss`
- `duration_loss`
- `chord_loss`
- `root_loss`
- `chroma_loss`
- `bass_loss`

Those enriched fields are written into a new dataset directory:

- `COMMUDataset/losses_enriched/`

## Function lineage

The enriched dataset generator intentionally follows the same function lineage used in the earlier notebook/script workflow:

1. `wrap_dataset(...)` from `base_model.dataset`
2. `prepare_tensors(...)`
3. `run_with_latents(...)`
4. `DisentangleVAE.loss_function(...)`

This keeps the enriched dataset semantically close to the original pipeline.

## Main generator

Script:

- `NotebooksVAESymTex/generate_commu_enriched_loss_dataset.py`

Example: small test run

```bash
cd /workspace/vae-textures-dev/NotebooksVAESymTex
python3 generate_commu_enriched_loss_dataset.py --max-files 20 --output-dir /workspace/vae-textures-dev/COMMUDataset/losses_enriched --device cpu
```

Example: resume-safe run on a prepared batch

```bash
cd /workspace/vae-textures-dev/NotebooksVAESymTex
python3 generate_commu_enriched_loss_dataset.py   --batch-file /workspace/vae-textures-dev/COMMUDataset/enriched_loss_batches/batch_000.txt   --output-dir /workspace/vae-textures-dev/COMMUDataset/losses_enriched   --device cpu   --skip-existing
```

## Batch creation

Script:

- `NotebooksVAESymTex/create_commu_enriched_batches.py`

By default it creates overnight-friendly batches from a runtime estimate.

Example: derive batches for roughly 8-hour runs

```bash
cd /workspace/vae-textures-dev/NotebooksVAESymTex
python3 create_commu_enriched_batches.py --target-hours 8
```

Example: fixed-size batches

```bash
cd /workspace/vae-textures-dev/NotebooksVAESymTex
python3 create_commu_enriched_batches.py --batch-size 2500
```

Output directory:

- `COMMUDataset/enriched_loss_batches/`

Each batch file contains one COMMU track id per line.

## Simple runner for cluster or local CPU

Shell script:

- `scripts/run_commu_enriched_batch.sh`

Example:

```bash
cd /workspace/vae-textures-dev
scripts/run_commu_enriched_batch.sh /workspace/vae-textures-dev/COMMUDataset/enriched_loss_batches/batch_000.txt
```

Optional environment variables:

- `DEVICE=cpu` or `DEVICE=cuda`
- `OUTPUT_DIR=/custom/output/path`
- `SKIP_EXISTING=1`

Example:

```bash
cd /workspace/vae-textures-dev
DEVICE=cpu SKIP_EXISTING=1 scripts/run_commu_enriched_batch.sh /workspace/vae-textures-dev/COMMUDataset/enriched_loss_batches/batch_000.txt
```

## Runtime note

A small CPU benchmark on this machine processed 5 COMMU source files in about 41.3 seconds.
That corresponds to roughly:

- `8.264 seconds / source file`
- `~20.5 hours` for all 8,924 COMMU source files on this CPU baseline

That estimate is only a planning baseline. Real runtime depends on:

- CPU speed
- storage speed
- average segment count per file
- whether you run on CPU or GPU

## Notebook integration

The mature notebook:

- `NotebooksVAESymTex/COMMU_Latent_UMAP_Explorer.ipynb`

loads `COMMUDataset/losses/` first and merges `COMMUDataset/losses_enriched/` automatically when it exists.

So the notebook does not need to be rewritten after enriched batches finish.
