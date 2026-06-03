# POP909 losses and W&B metrics guide

Purpose: explain what the POP909 VAE training losses mean, where they are calculated, how they appear in Weights & Biases, and what to monitor during the representative run.

This guide is about the canonical path `base_model/train.py`. It does not change the training code.

## One-line mental model

The model is trained to reconstruct a piano-tree texture, keep two latent spaces close to a normal prior, and reconstruct chord information:

```text
loss = recon_loss + beta * kl_loss + chord_loss
```

In the current `train.py`, `beta` is configured with high value `0.1`, and the reconstruction weights are `[1, 0.5]`.

## Where the metric names come from

`base_model/train.py` defines the metric order in `writer_names`:

```text
loss, recon_loss, pl, dl, kl_loss, kl_chd, kl_rhy,
chord_loss, root_loss, chroma_loss, bass_loss
```

That order must match the tuple returned by `DisentangleVAE.loss_function`. The training loop zips those names to returned tensors before writing TensorBoard and W&B metrics.

Source references:

- `base_model/train.py:91` defines `writer_names`.
- `base_model/model.py:57` defines `loss_function`.
- `base_model/amc_dl/torch_plus/module.py:114` and `module.py:120` map returned loss tensors to those names.

## Loss table

| Metric | What it means | Formula / source | W&B names |
|---|---|---|---|
| `loss` | Total objective used for backpropagation. This is the main optimization loss. | `recon_loss + beta * kl_loss + chord_loss` in `model.py:66`. | `train/loss`, `val/loss`, plus epoch sums as `epoch/train_loss`, `epoch/valid_loss`. |
| `recon_loss` | Piano-tree reconstruction loss: how well the decoder reconstructs notes and durations. | `weights[0] * pitch_loss + weights[1] * dur_loss` in `ptvae.py:528`; current weights are `[1, 0.5]` from `train.py:36`. | `train/recon_loss`, `val/recon_loss`. |
| `pl` | Pitch loss. Cross-entropy over predicted pitch symbols. | `pitch_loss` in `ptvae.py:500-504`, ignoring `pitch_pad`. | `train/pl`, `val/pl`. |
| `dl` | Duration loss. Cross-entropy over the duration bit outputs. | `dur_loss` in `ptvae.py:506-511`, ignoring `dur_pad`. | `train/dl`, `val/dl`. |
| `kl_loss` | Total latent regularization. It keeps both latent distributions near a standard normal prior. | `kl_chd + kl_rhy` in `model.py:85-90`. | `train/kl_loss`, `val/kl_loss`. |
| `kl_chd` | KL term for the chord latent distribution from `chd_encoder(c)`. | `kl_with_normal(dist_chd)` in `model.py:87`; helper in `train_utils.py:45-49`. | `train/kl_chd`, `val/kl_chd`. |
| `kl_rhy` | KL term for the rhythm/texture latent distribution from `rhy_encoder(pr_mat)`. | `kl_with_normal(dist_rhy)` in `model.py:88`; helper in `train_utils.py:45-49`. | `train/kl_rhy`, `val/kl_rhy`. |
| `chord_loss` | Total chord reconstruction loss. It asks the chord decoder to reconstruct root, chroma, and bass. | `root_loss + chroma_loss + bass_loss` in `model.py:82`. | `train/chord_loss`, `val/chord_loss`. |
| `root_loss` | Cross-entropy for chord root class. | Target is `c[:, :, 0:12].max(-1)` in `model.py:72`; loss in `model.py:79`. | `train/root_loss`, `val/root_loss`. |
| `chroma_loss` | Cross-entropy for chord chroma presence/state. | Target is `c[:, :, 12:24]` in `model.py:73`; loss in `model.py:80`. | `train/chroma_loss`, `val/chroma_loss`. |
| `bass_loss` | Cross-entropy for bass class. | Target is `c[:, :, 24:].max(-1)` in `model.py:74`; loss in `model.py:81`. | `train/bass_loss`, `val/bass_loss`. |

## Important detail: step values vs epoch values

W&B has two kinds of loss values here.

Per-step metrics:

- `train/loss`, `train/recon_loss`, etc. are logged once per train batch.
- `val/loss`, `val/recon_loss`, etc. are logged once per validation batch.
- These are the direct values returned by the model for that batch.

Epoch metrics:

- `epoch/train_loss`
- `epoch/valid_loss`
- `epoch/duration_seconds`

The epoch losses are accumulated sums over all batches, not averages. In `module.py`, each batch value is added into `epoch_loss_dic`; then `training.run()` reads `self.train()['loss']` and `self.eval()['loss']` and logs those totals. This means a representative run with many batches will have much larger epoch losses than a tiny smoke run, even if per-batch behavior is similar.

For comparing training health, prefer the per-step curves and compare epoch sums only within runs that use the same dataset size, batch size, and sample limits.

## What W&B records

When `WANDB_ENABLED=1`, `WandbRunLogger.from_env()` initializes a W&B run with:

- project from `WANDB_PROJECT`
- optional entity from `WANDB_ENTITY`
- run name from `VAE_RUN_NAME`
- config from `wandb_config` in `train.py`

Useful config keys to inspect in W&B:

- `batch_size`
- `n_epoch`
- `learning_rate`
- `beta`
- `weights`
- `tf_rates`
- `limit_train_samples`
- `limit_val_samples`
- `shift_low`, `shift_high`
- `num_bar`
- `contain_chord`
- `train_portion`
- `writer_names`
- `log_path`, `writer_path`, `model_path`

Metric namespace setup:

- `train/*` uses `train/step`.
- `val/*` uses `val/step`.
- `epoch/*` uses `epoch`.

Checkpoint artifact policy:

- Default `WANDB_CHECKPOINT_POLICY` is `valid,final`.
- `valid` checkpoints get aliases `valid` and `best`.
- `final` checkpoints get alias `final`.
- Epoch checkpoints are saved on disk every epoch, but are not logged as W&B artifacts unless `WANDB_CHECKPOINT_POLICY` includes `epoch`.

Source references:

- W&B config in `train.py:106-126`.
- W&B initialization and config update in `wandb_helper.py:19-80`.
- Per-step logging in `wandb_helper.py:82-85`.
- Epoch logging in `wandb_helper.py:87-96`.
- Artifact logging in `wandb_helper.py:98-117`.
- Checkpoint calls in `module.py:213-226`.

## What to monitor in the representative run

### First: confirm the run is the intended run

Before interpreting curves, inspect W&B config:

- `limit_train_samples` and `limit_val_samples` should be `0` or absent for a full representative run. If they are tiny values like `4` and `2`, you are still in smoke mode.
- `batch_size`, `n_epoch`, `learning_rate`, `weights`, and `beta` should match the approved launch config.
- `device` should be `cuda`.
- `writer_names` should include all eleven metrics listed above.

### Primary training-health metrics

Watch these first:

- `train/loss` and `val/loss`: overall batch-level objective.
- `train/recon_loss` and `val/recon_loss`: whether the model is learning to reconstruct piano-tree material.
- `train/pl` / `val/pl`: pitch reconstruction behavior.
- `train/dl` / `val/dl`: duration reconstruction behavior.
- `train/chord_loss` / `val/chord_loss`: chord decoder behavior.
- `epoch/duration_seconds`: operational runtime, useful for ETA and cluster planning.

### Latent-health metrics

Watch:

- `train/kl_loss`, `val/kl_loss`
- `train/kl_chd`, `val/kl_chd`
- `train/kl_rhy`, `val/kl_rhy`

Interpretation:

- If KL terms collapse near zero while reconstruction does not improve, the latent variables may not be carrying useful information.
- If KL terms explode or dominate while reconstruction is unstable, the training dynamics may be poor.
- `kl_chd` and `kl_rhy` do not need to be identical, but a strong imbalance is worth noting because the model has separate chord and rhythm/texture latents.

### Chord-family metrics

Watch:

- `root_loss`
- `chroma_loss`
- `bass_loss`

Interpretation:

- These are classification losses for different chord targets.
- If only one of them behaves badly, the issue may be localized: root prediction, chroma/state prediction, or bass prediction.
- If all chord terms are flat or rising while `recon_loss` improves, the texture decoder may be learning while the chord decoder is not.

## Pause / rerun / block cues

Pause and inspect if:

- W&B config still shows smoke sample limits.
- `train/loss` changes but `val/loss` is missing.
- `epoch/duration_seconds` suggests the job will exceed the requested Slurm time.
- No `valid` or `final` artifact appears after checkpoint save points.
- `train/loss` or `val/loss` becomes NaN/inf.
- `val/loss` is consistently much worse while `train/loss` falls sharply; this may be overfitting or a train/val split/config issue.

Rerun may be appropriate if:

- The wrong batch size, epoch count, sample limits, or W&B run name was used.
- The run was launched interactively and died before useful checkpoint evidence.
- The W&B run is missing key config or checkpoint evidence due to environment setup.

Block and diagnose before Phase 7 if:

- CUDA is not visible.
- W&B requires credentials but `WANDB_API_KEY` is absent.
- The code path is not `base_model/train.py`.
- Losses are NaN/inf early in a reproducible way.

## Practical reading order in W&B

1. Open the run config and verify this is not smoke mode.
2. Open `train/loss`, `val/loss`, `train/recon_loss`, and `val/recon_loss`.
3. Add `pl` and `dl` to see whether pitch or duration is the main reconstruction problem.
4. Add `kl_chd` and `kl_rhy` to see whether both latent spaces are active.
5. Add `root_loss`, `chroma_loss`, and `bass_loss` to inspect chord-decoder behavior.
6. Check artifacts for `valid`/`best` and `final`.
7. Check `epoch/duration_seconds` for runtime planning.

## Bottom line

For the representative run, the most useful dashboard is:

- Batch curves: `train/loss`, `val/loss`, `train/recon_loss`, `val/recon_loss`, `train/pl`, `val/pl`, `train/dl`, `val/dl`.
- Latent curves: `train/kl_chd`, `val/kl_chd`, `train/kl_rhy`, `val/kl_rhy`.
- Chord curves: `train/chord_loss`, `val/chord_loss`, plus root/chroma/bass if chord behavior looks suspicious.
- Operational curves/artifacts: `epoch/duration_seconds`, `epoch/train_loss`, `epoch/valid_loss`, and model artifacts.

Use per-step metrics to understand learning behavior. Use epoch metrics for high-level reporting only after remembering that epoch losses are sums in the current code.
