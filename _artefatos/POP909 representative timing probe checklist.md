# POP909 representative timing probe checklist

Purpose: estimate the runtime of the representative POP909 training run before launching Phase 7, using a persistent `sbatch` job instead of an interactive shell.

This timing probe is operational evidence, not scientific validation. It keeps the canonical `base_model/train.py` path and changes only environment variables that bound the timing job.

## Scripts created

- `base_model/run_pop909_timing_probe.sh`: runs `python -u train.py` with representative timing defaults.
- `scripts/submit_pop909_timing_probe.sh`: submits the wrapper with `sbatch`, GPU container options, persistent logs, W&B env, and monitor commands.

## Default probe profile

The default probe is intentionally representative but bounded:

- `VAE_BATCH_SIZE=128`: matches the current canonical `train.py` default.
- `VAE_N_EPOCH=1`: one bounded timing epoch over the sample-limited subset.
- `VAE_LIMIT_TRAIN_SAMPLES=4096`: enough batches to smooth startup noise while staying much smaller than full training.
- `VAE_LIMIT_VAL_SAMPLES=512`: enough validation work to estimate validation overhead.
- `VAE_LR=1e-3`: matches `train.py` default.
- W&B enabled by default in project `pop909-reproduction`.

If this is too slow or too fast in practice, rerun with adjusted sample limits. Keep `VAE_BATCH_SIZE=128` unless the representative training config changes.

## Before launch

On your host/local machine, sync runtime files to the cluster:

`scripts/sync_runtime_to_cluster.sh`

Inspect the dry-run. If it only includes runtime code/scripts, apply:

`scripts/sync_runtime_to_cluster.sh sync`

On the cluster login node:

`cd ~/vae-textures-dev`

Confirm the private W&B env exists without printing the key:

`test -f ~/.config/wandb/env.sh && echo "W&B env file exists"`

Do not print `WANDB_API_KEY`.

## Launch

From the cluster login node:

`scripts/submit_pop909_timing_probe.sh`

Optional overrides before launch:

`VAE_LIMIT_TRAIN_SAMPLES=8192 SLURM_TIME=04:00:00 scripts/submit_pop909_timing_probe.sh`

`VAE_RUN_NAME=pop909-timing-bs128-train4096-v2 scripts/submit_pop909_timing_probe.sh`

The submitter prints a job id and monitor commands.

## Monitor commands

Replace `<JOB_ID>` and log paths with the values printed by the submitter.

`squeue -j <JOB_ID>`

`tail -f ~/vae-textures-dev/logs/pop909-timing/pop909-timing-<JOB_ID>.out`

`tail -f ~/vae-textures-dev/logs/pop909-timing/pop909-timing-<JOB_ID>.err`

After completion:

`sacct -j <JOB_ID> --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES%60,ExitCode`

## What to send back to Codex

Send these items so I can calculate the estimate and help decide accept/rerun/block:

1. Job id.
2. The submitter terminal output.
3. Last 80-120 lines of stdout log.
4. Last 40-80 lines of stderr log, even if empty.
5. `sacct` output.
6. W&B run URL.
7. W&B config values for batch size, epoch count, train/val sample limits, device, and run name.
8. W&B metric evidence: final `train/step`, `val/step`, `epoch/duration_seconds`, `epoch/train_loss`, `epoch/valid_loss` if present.
9. Any checkpoint/artifact evidence.

## Objective estimation strategy

We will estimate runtime from observed work, not from hope.

Minimum useful calculations:

- Train batches in probe: observed final `train/step` delta.
- Validation batches in probe: observed final `val/step` delta.
- Probe elapsed wall time: from `sacct Elapsed` and/or `epoch/duration_seconds`.
- Seconds per train batch: probe train time divided by train steps when separable, otherwise wall time divided by combined train/val work with a caveat.
- Full epoch estimate: estimated full train batches times seconds per train batch, plus validation overhead.

Important caveat: current `epoch/train_loss` and `epoch/valid_loss` are sums over batches, not normalized averages. Use them for run evidence, not direct quality comparison across different sample limits.

## Accept / rerun / block criteria

Accept the timing probe if all are true:

- Job ran through `sbatch`, not an interactive shell.
- W&B run config matches the intended probe: batch size 128, sample limits as expected, canonical `train.py` path.
- Logs show CUDA-visible execution and `Epoch: 01` completion.
- W&B has train/val/epoch metrics.
- `sacct` shows a completed job with acceptable exit code.
- The estimate is stable enough to choose Slurm time/memory/log policy for the real run.

Rerun if:

- Sample limit was too small to smooth startup overhead.
- The job hit time limit before enough batches completed.
- W&B/log evidence is incomplete.
- Batch size or run name was wrong.

Block if:

- CUDA is not visible.
- W&B secret/config is missing and online logging is required.
- The job does not use `base_model/train.py`.
- Losses become NaN/inf in a reproducible way.
- Slurm/container options fail before training starts.

## Preparing the real Phase 7 training run

After accepting the timing probe, use the estimate to choose:

- representative `SLURM_TIME` with buffer;
- log directory and run name;
- W&B group/tags/notes;
- checkpoint artifact policy;
- whether the real run should use the same wrapper style with `VAE_LIMIT_TRAIN_SAMPLES=0` and `VAE_LIMIT_VAL_SAMPLES=0`.

Do not launch the real representative run until the timing estimate and W&B/loss interpretation are accepted.
