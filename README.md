# VAE Representation of Symbolic Musical Textures

This repository is an academic workspace for studying and extending the Poly-Dis VAE pipeline proposed by Wang et al. It includes MIDI preprocessing, latent-space analysis, POP909-based reproduction work, and COMMU-based evaluation and experiment preparation.

## Repository Overview

The repository currently combines two connected lines of work:

- reproduction and validation of the original Poly-Dis training and evaluation path on `POP909`
- preprocessing, latent analysis, and experiment preparation for `COMMU` as the target dataset for further work

The POP909 path is important because it provides the closest available baseline to the original model. The COMMU path is important because it is the intended target of downstream experiments and evaluation.

## Dataset

Many of the dataset-processing, latent-analysis, and current exploratory evaluation workflows in this repository are based on the **COMMU Dataset**. At the same time, the repository also contains a restored `POP909`-based path used for reproduction and validation work around the original Poly-Dis setup.

The COMMU dataset includes:

- A collection of `.mid` files with symbolic musical data.
- A corresponding metadata file (`commu_meta.csv`) containing annotations such as chord progressions.

The metadata CSV must contain a column named `chord_progressions`, where each row is a string-encoded chord sequence. Each entry is indexed by a unique track ID that matches the name of the corresponding MIDI file.

## Preprocessing MIDI Files

All MIDI preprocessing functions are implemented in `utilProcessing.py`.

To convert a folder of `.mid` files and associated metadata into `.npz` files (NumPy archive format), use the script `processMidiPath.py`.

The resulting `.npz` files contain the following arrays:

- `beat` — shape `(n, 6)`, dtype `int32`
- `chord` — shape `(n, 14)`, dtype `float64`
- `melody` — shape `(n, 8)`, dtype `int32`
- `bridge` — shape `(n, 8)`, dtype `int32`
- `piano` — shape `(n, 8)`, dtype `int32`

These preprocessing steps are an essential part of the project workflow. Without them, neither latent-space analysis nor downstream dataset-specific experimentation can proceed in a reliable way.

## Analysing VAE Representations

To analyse the quality of the latent representation learned by the model, the repository includes workflows that compute loss components and latent vectors from `.npz` segments.

The existing `calc_latent_loss.py` workflow produces:

- `z_chd` – latent vector associated with the **chord** encoding
- `z_txt` – latent vector associated with the **texture** encoding
- `kl_loss` – total Kullback–Leibler divergence between posterior and prior
- `kl_chd` – KL divergence for the **chord** latent variable
- `kl_rhy` – KL divergence for the **texture** latent variable
- `final_loss` – total reconstruction loss of the VAE

These outputs can be used to:

- study how well the model captures harmonic and textural features
- visualise latent spaces using dimensionality reduction techniques such as UMAP
- compare reconstruction quality across different musical textures

The repository also includes a newer richer recomputation workflow for COMMU-based study notebooks:

- `NotebooksVAESymTex/recompute_commu_loss_components.py`

This newer script is designed to recompute per-segment outputs from COMMU `.npz` files using the authors' `polydis-v1.pt` checkpoint, generating a richer set of quantities for exploratory notebook analysis. In addition to `z_chd`, `z_txt`, and `final_loss`, it also extracts:

- `recon_loss`
- `pitch_loss`
- `duration_loss`
- `kl_loss`
- `kl_chd`
- `kl_rhy`
- `chord_loss`
- `root_loss`
- `chroma_loss`
- `bass_loss`

It also preserves selected metadata fields from the original COMMU `.npz` files, including:

- `audio_key`
- `chord_progressions`
- `pitch_range`
- `num_measures`
- `bpm`
- `genre`
- `track_role`
- `inst`
- `sample_rhythm`
- `time_signature`

At the moment, this richer COMMU recomputation workflow should be understood as an available study and analysis path, not yet as a completed experimental result.

## Training And Smoke-Test Quickstart

### Canonical POP909 training entrypoint

The canonical reproduction training entrypoint is:

- `base_model/train.py`

Important defaults in the current script are close to the original paper setup by Wang et al.:

- `batch_size = 128`
- `n_epoch = 6`
- latent sizes `256 / 256`
- transposition range `shift_low = -6`, `shift_high = 5`
- `num_bar = 2`, corresponding to 8-beat segments in the restored code path

These align with important elements described in the original Poly-Dis paper.

One important caveat is that the current restored code uses `portion=8` in the loader path, which should not be presented as a perfect one-to-one match with the paper's stated 90/10 song-level split.

### Quick smoke test: local

Use this when you want a minimal proof that `train.py` still runs locally.

```bash
cd /workspace/vae-textures-dev/base_model

VAE_BATCH_SIZE=2 \
VAE_N_EPOCH=1 \
VAE_LIMIT_TRAIN_SAMPLES=4 \
VAE_LIMIT_VAL_SAMPLES=2 \
VAE_RUN_NAME=local-smoke \
python -u train.py
```

Expected signs of success:

- POP909 data loads correctly
- training reaches `Epoch: 01`
- `Train Loss` and `Valid. Loss` appear
- the run ends with `Model saved.`

### Quick smoke test: local with W&B offline

Use this when you want to validate logging without depending on live connectivity.

```bash
cd /workspace/vae-textures-dev/base_model

WANDB_ENABLED=1 \
WANDB_MODE=offline \
WANDB_PROJECT=pop909-reproduction \
VAE_BATCH_SIZE=2 \
VAE_N_EPOCH=1 \
VAE_LIMIT_TRAIN_SAMPLES=4 \
VAE_LIMIT_VAL_SAMPLES=2 \
VAE_RUN_NAME=wandb-local-smoke \
python -u train.py
```

### Quick smoke test: cluster

The cluster proof path uses:

- `base_model/run_cluster_proof.sh`
- `scripts/cluster_gpu_shell.sh`

See also:

- `_artefatos/Smoke run guide - local and cluster.md`

## Training And Evaluation Paths

### POP909 reproduction path

The main files for the POP909 reproduction path are:

- `base_model/train.py`
- `base_model/model.py`
- `base_model/evaluate_paper_objective_metrics.py`
- `interface.py`
- `compute_single_loss.py`

This path is used to study whether the restored training/evaluation flow remains compatible with the original Poly-Dis setup.

### COMMU evaluation and preparation path

The repository also contains active work for COMMU-oriented evaluation and preparation, including:

- `COMMUDataset/`
- `NotebooksVAESymTex/recompute_commu_loss_components.py`
- `NotebooksVAESymTex/`

This side of the project supports dataset preparation, latent analysis, and planning for future COMMU-oriented experiments.

## Latent-Space And Evaluation Studies

A major part of the repository is dedicated to studying the latent representations learned by the model, especially:

- `z_chd`
- `z_txt`
- loss components
- latent-space structure under dimensionality reduction and grouping analysis

Examples of study material currently present in the repository include:

- `NotebooksVAESymTex/POP909_Training_Study.ipynb`
- `NotebooksVAESymTex/`
- `_artefatos/testesTreinamento.md`

These materials are useful for understanding what has been inspected, what has been tested, and how latent behavior is being interpreted in practice.

## Current Experimental Direction

The current direction of the repository is not limited to reproducing POP909 results.

The broader goal is to use the restored original path as a reliable baseline, while extending the project toward:

- COMMU preprocessing
- COMMU latent-space evaluation
- COMMU-oriented experiment planning and future retraining

In that sense:

- `POP909` is the closest baseline to the original model proposed by Wang et al.
- `COMMU` is the target dataset for the next experimental stage

## References

Original model and paper:

- Wang et al., Poly-Dis / `polyphonic-chord-texture-disentanglement`
- https://github.com/ZZWaang/polyphonic-chord-texture-disentanglement
- https://arxiv.org/abs/2008.07122

Checkpoint/tutorial repository:

- https://github.com/ZZWaang/icm-deep-music-generation

## COMMU Phase 10 Pipeline Readiness

Phase 10 adds a compact, reusable COMMU readiness surface around the existing data treatment code. The practical boundary is `MIDI -> NPZ -> modelo`: raw COMMU MIDI plus metadata are converted through `utilProcessing.GenDataSet`, audited as NPZ data, and smoke-tested through model/loss compatibility.

Clean entrypoint examples:

```bash
python3 -m commu_pipeline.preprocess --midi-dir midiDataTest --metadata-csv midiDataTest/commu_meta.csv --output-dir _artefatos/commu_phase10/regenerated_npz --track-id commu00001
python3 -m commu_pipeline.harmony_trace --metadata-csv COMMUDataset/CommuVAEDataset.csv --npz COMMUDataset/npzFiles/commu00001.npz --track-id commu00001 --output-dir _artefatos/commu_phase10
python3 -m commu_pipeline.audit_dataset --npz-dir COMMUDataset/npzFiles --metadata-csv COMMUDataset/CommuVAEDataset.csv --output-dir _artefatos/commu_phase10 --max-files 200 --sample-per-role 2
python3 -m commu_pipeline.forward_loss_probe --input-dir COMMUDataset/npzFiles --max-files 3 --device cpu --checkpoint model_param/polydis-v1.pt --output _artefatos/commu_phase10/commu_forward_loss_probe.json
```

Review artifacts live in `_artefatos/commu_phase10`, with the human-facing notebook at `NotebooksVAESymTex/COMMU_Phase10_Pipeline_Readiness.ipynb`. Phase 10 is a readiness and organization phase; benchmark comparison belongs to Phase 11.
