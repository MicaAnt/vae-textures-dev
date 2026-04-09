
# Notebook Reorganization Plan

## Goal

Make `NotebooksVAESymTex/` easier to navigate by separating mature analysis assets from exploratory drafts.

## Proposed structure

### 1. Keep one clearly mature notebook at the top level

For now, keep the conference-ready notebook visible at the root:

- `COMMU_Latent_UMAP_Explorer.ipynb`

### 2. Group support code explicitly

Keep reusable notebook support files together:

- `commu_umap_support.py`
- `generate_commu_enriched_loss_dataset.py`
- `create_commu_enriched_batches.py`
- `build_commu_umap_cache.py`
- `COMMU_ENRICHED_DATASET_WORKFLOW.md`

Optional future folder:

- `NotebooksVAESymTex/support/`

### 3. Move draft notebooks into a drafts archive

Suggested future destination:

- `NotebooksVAESymTex/drafts/`

Candidates:

- `commuMetaData.ipynb`
- `compute_losses.ipynb`
- `criandoFuncaoPerdas+Latentes.ipynb`
- `estudandoPerdas.ipynb`
- `process_features.ipynb`
- `recons_umap.ipynb`
- `reconstructSamples.ipynb`
- `FilterCommuDataset.ipynb`
- ad-hoc `Untitled*.ipynb`

### 4. Keep generated outputs separate from notebooks

Generated media and exports should live outside the root notebook area.

Suggested future destinations:

- `NotebooksVAESymTex/exports/`
- `NotebooksVAESymTex/figures/`
- `NotebooksVAESymTex/_cache/`

### 5. Add one short README at the notebook root

Suggested future file:

- `NotebooksVAESymTex/README.md`

It should answer:

- which notebook is the mature entry point
- which scripts generate supporting datasets
- where drafts live
- where generated figures go

## Operational note

With every new dataset variant, create a dedicated mature notebook rather than overloading a single exploratory notebook with too many modes.

That means the current COMMU notebook can stay focused, while future POP909 or alternative-loss analyses can become separate, clearly named notebooks.
