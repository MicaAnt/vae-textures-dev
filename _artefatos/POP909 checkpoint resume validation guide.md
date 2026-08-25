# POP909 checkpoint resume validation guide

Date: 2026-06-03
Phase: 7 - checkpoint/resume gate

Purpose: explain the canonical `base_model/train.py` checkpoint/resume support added before the representative POP909 training run.

## Canonical entrypoint

Resume support stays on the canonical POP909 path:

```bash
cd base_model
python -u train.py
```

Wrappers may set environment variables, but they must still call `python -u train.py`.

## Environment variables

- `VAE_RESUME_FROM`: optional path to a full-state training checkpoint. If unset, training starts from scratch.
- `VAE_RUN_EPOCHS_THIS_JOB`: optional per-job epoch budget. If `0` or unset, run until `VAE_N_EPOCH` total target is reached. If set to `1`, a job resumes/runs for one epoch and stops after saving state.
- `VAE_FULL_CHECKPOINT_POLICY`: documents full-state checkpoint kinds. Default: `epoch-state,last-state,final-state`.
- `WANDB_CHECKPOINT_POLICY`: controls W&B artifact upload. Default now includes `valid,final,epoch-state,last-state,final-state`.

Existing controls remain valid:

- `VAE_BATCH_SIZE`
- `VAE_N_EPOCH`
- `VAE_LIMIT_TRAIN_SAMPLES`
- `VAE_LIMIT_VAL_SAMPLES`
- `VAE_LIMIT_TRAIN_SHUFFLE`: optional; when set to `1`, bounded subset tests keep train-batch shuffling enabled so the DataLoader consumes RNG like the representative training path.
- `VAE_LR`
- `VAE_RUN_NAME`
- `WANDB_ENABLED`
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `WANDB_MODE`

## Full-state checkpoint payload

A resume-capable checkpoint must contain these keys:

- `model_state_dict`
- `optimizer_state_dict`
- `lr_scheduler_state_dict`
- `optimizer_scheduler_step`
- `param_scheduler_steps`
- `epoch`
- `train_step`
- `val_step`
- `best_valid_loss`
- `config`
- `rng_state`

These are intentionally broader than model-only checkpoints. A model-only checkpoint can reproduce weights but cannot faithfully resume training because optimizer, scheduler, step state, and RNG state would be reset.

## Files created during training

Model-only files continue to be saved in the run `models/` directory:

- `<run_name>_epoch.pt`
- `<run_name>_valid.pt`
- `<run_name>_final.pt`

Full-state files are saved beside them:

- `<run_name>_epoch-state_state.pt`
- `<run_name>_last-state_state.pt`
- `<run_name>_final-state_state.pt`

Use `last-state` for normal resume after an epoch boundary.

## Resume command shape

Example for a staged epoch-boundary resume:

```bash
cd base_model
VAE_RESUME_FROM=result_YYYY-MM-DD_HHMMSS/models/pop909-resume-probe_last-state_state.pt VAE_N_EPOCH=2 VAE_RUN_EPOCHS_THIS_JOB=1 python -u train.py
```

Interpretation:

- The checkpoint contains `epoch=1` after the first leg.
- `VAE_N_EPOCH=2` means the total target is two epochs.
- `VAE_RUN_EPOCHS_THIS_JOB=1` means this job should only run one additional epoch.

## W&B evidence

Expected W&B config fields include:

- `resume_from`
- `run_epochs_this_job`
- `full_checkpoint_policy`
- `batch_size`
- `n_epoch`
- `limit_train_samples`
- `limit_val_samples`
- `train_portion`
- `writer_names`

Expected W&B metric evidence includes:

- `train/step` continuing above the initial run's value when using the same run context or clearly documented resumed leg naming.
- `val/step` continuing or being documented per resumed leg.
- `epoch/duration_seconds` for each completed epoch.
- `checkpoint/<kind>_save_seconds` and `checkpoint/last_save_seconds` when W&B is enabled.
- `epoch/train_loss` and `epoch/valid_loss`, with the existing caveat that they are sums over batches.

Expected artifact evidence, if `WANDB_CHECKPOINT_POLICY` includes state kinds:

- `epoch-state`
- `last-state`
- `final-state`
- `valid`
- `final`

## Objective resume evidence checks

Accept resume continuity only if at least these checks pass:

1. The first leg saves a full-state checkpoint with all required keys listed above.
2. The resumed leg prints `[resume] Loaded training state from ...` and reports nonzero `epoch`, `train_step`, or `val_step` from the checkpoint.
3. The resumed leg reaches a later epoch/step than the initial leg and saves a new `last-state` or `final-state` checkpoint.
4. W&B config/logs identify whether the evidence is one resumed run or clearly named initial/resumed legs in the same group.
5. `sacct` for the sbatch probe reports `COMPLETED` with exit code `0:0`, or the guide records a concrete rerun/block reason.

## Local smoke evidence

A bounded local smoke without W&B verified default no-resume behavior after adding full-state checkpoints:

```text
VAE_BATCH_SIZE=2 VAE_N_EPOCH=1 VAE_RUN_EPOCHS_THIS_JOB=1 VAE_LIMIT_TRAIN_SAMPLES=4 VAE_LIMIT_VAL_SAMPLES=2 VAE_RUN_NAME=resume-default-smoke WANDB_ENABLED=0 python -u train.py
```

Observed evidence:

- reached `Epoch: 01`;
- saved `resume-default-smoke_epoch-state_state.pt`;
- saved `resume-default-smoke_last-state_state.pt`;
- saved `resume-default-smoke_final-state_state.pt`.

The `last-state` payload was inspected and contained the required full-state keys:

```text
best_valid_loss,config,epoch,lr_scheduler_state_dict,model_state_dict,
optimizer_scheduler_step,optimizer_state_dict,param_scheduler_steps,rng_state,train_step,val_step
```

Payload values included:

```text
epoch=1
train_step=2
val_step=1
best_valid_loss=10.120491981506348
param_scheduler_steps={'tfr1': 2, 'tfr2': 2, 'tfr3': 2, 'beta': 2, 'weights': 2.0}
```

A second bounded local run resumed from that state:

```text
VAE_RESUME_FROM=./result_2026-06-03_185738/models/resume-default-smoke_last-state_state.pt VAE_BATCH_SIZE=2 VAE_N_EPOCH=2 VAE_RUN_EPOCHS_THIS_JOB=1 VAE_LIMIT_TRAIN_SAMPLES=4 VAE_LIMIT_VAL_SAMPLES=2 VAE_RUN_NAME=resume-second-leg-smoke WANDB_ENABLED=0 python -u train.py
```

Observed resume evidence:

```text
[resume] Loaded training state from ./result_2026-06-03_185738/models/resume-default-smoke_last-state_state.pt | epoch=1 train_step=2 val_step=1 best_valid_loss=10.120491981506348
Epoch: 02 | Time: 0m 5s
```

This proves local epoch-boundary resume continuity before the cluster `sbatch` resume probe.

## Human tutorial gate

Local exact-continuity proof decision: `accept` on 2026-06-04. The user followed the numbered `base_model/resume_continuity_test/README.md` flow enough to report being convinced by the results. Agent-run evidence alone was not used as acceptance.

## Exact continuity evidence

A deterministic local A/B continuity test now proves epoch-boundary resume continuity beyond the earlier "Epoch: 02" smoke evidence. The numbered continuity test runs three bounded CPU jobs with `WANDB_ENABLED=0`, `CUDA_VISIBLE_DEVICES=`, `VAE_SEED=3345`, `VAE_BATCH_SIZE=2`, `VAE_LIMIT_TRAIN_SAMPLES=4`, `VAE_LIMIT_VAL_SAMPLES=2`, and `VAE_LIMIT_TRAIN_SHUFFLE=1`:

1. Two epochs uninterrupted.
2. One epoch initial leg, saving `last-state`.
3. Resume from that `last-state` and run epoch 2.

Command:

```bash
python vae-textures-dev/base_model/resume_continuity_test/01_train_direct_2_epochs.py
python vae-textures-dev/base_model/resume_continuity_test/02_train_one_epoch_checkpoint.py
python vae-textures-dev/base_model/resume_continuity_test/03_resume_second_epoch.py
python vae-textures-dev/base_model/resume_continuity_test/04_compare_final_states.py
python vae-textures-dev/base_model/resume_continuity_test/05_inspect_weights.py
```

Expected result shape after the user runs the numbered test:

```text
[test] STEP 03: resume from epoch-1 checkpoint and run epoch 2
[resume] Loaded training state from ... | epoch=1 train_step=... val_step=... best_valid_loss=...
Epoch: 02 | Time: ...
[test] STEP 04: compare final training states
[test] RESULT=PASSED
```

Acceptance still requires inspecting:

```text
base_model/resume_continuity_test/outputs/manifest.json
base_model/resume_continuity_test/outputs/reports/state_comparison.txt
base_model/resume_continuity_test/outputs/reports/weight_diff_report.csv
base_model/resume_continuity_test/outputs/reports/weight_inspection.csv
```

The comparison step checks final full-state checkpoints for `epoch`, `train_step`, `val_step`, `best_valid_loss`, model tensors, optimizer state, learning-rate scheduler state, parameter-scheduler steps, and RNG state. The report now separates those checks into model, optimizer, scheduler, counters, and RNG categories before emitting the strict overall `RESULT=PASSED`. This means the resumed epoch 2 is not a fake relabeling: under deterministic CPU conditions, it reaches the same final training state as an uninterrupted two-epoch run.

The bounded test now sets `VAE_LIMIT_TRAIN_SHUFFLE=1` through the continuity helper/Slurm submitter so even the tiny subset path exercises shuffled train-batch iteration and the associated RNG consumption.

Remaining caveat: exact equality on GPU can still depend on deterministic CUDA kernel behavior. The training script sets CuDNN deterministic mode and disables benchmarking, but it does not globally force `torch.use_deterministic_algorithms(True)` or require `CUBLAS_WORKSPACE_CONFIG`. The cluster `sbatch` probe should therefore be read as operational GPU/W&B/checkpoint evidence; the local CPU continuity test remains the strongest exact-determinism proof.

The continuity helper now refuses ambiguous checkpoint matches instead of selecting by filesystem modification time. If a rerun reuses an old manifest/run id and leaves stale matching result directories behind, clean the stale outputs or start a fresh manifest before interpreting the result.

Hardening rerun on 2026-06-10 passed with `limit_train_shuffle=true` and run id `1781106640-47483`. The category summary was: model PASS, optimizer PASS, scheduler PASS, counters PASS, RNG PASS, with `non_equal_weight_tensors=0`.

## GPU checkpoint timing evidence

The final Phase 7 validation requested by supervisor feedback was accepted after
the hardened cluster GPU A/B run was inspected. Use the default exact mode from the
Slurm submitter after syncing runtime files:

```bash
scripts/sync_runtime_to_cluster.sh sync
# on the cluster login node, inside /home/<user>/vae-textures-dev:
scripts/submit_pop909_resume_probe.sh exact
```

Evidence to collect from the job:

- Slurm job id.
- stdout/stderr tails.
- `sacct -j <job_id> --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES%60,ExitCode`.
- `base_model/resume_continuity_test/outputs/manifest.json`.
- `base_model/resume_continuity_test/outputs/reports/state_comparison.txt`, including category summary for model, optimizer, scheduler, counters, and RNG.
- `base_model/resume_continuity_test/outputs/reports/weight_diff_report.csv`.
- `base_model/resume_continuity_test/outputs/reports/weight_inspection.csv`.
- Timing lines: `Epoch train/eval seconds`, `Saved model weights in`, and `Saved training state`.
- W&B group containing separate direct, initial, and resumed mini-training runs.
- W&B config/metrics/artifacts for those three mini-runs.

Decision rule:

```text
checkpoint overhead percent = checkpoint save seconds / epoch train+eval seconds
```

If W&B auth/artifact upload and checkpoint overhead are both acceptable, keep checkpointing every epoch for Phase 8.
If overhead is large, choose checkpoint every X epochs and record the chosen X.

GPU/timing decision: accept. Checkpointing every epoch is acceptable for Phase 8 based on the small probe timing; include the measured overhead caveat in the supervisor package.

## Resume probe decision

Local exact-continuity test: `accept` on 2026-06-04.

Cluster sbatch resume probe: `accept` on 2026-06-10.

Accepted evidence:

- Slurm job: `332828`.
- Slurm state: `COMPLETED`, exit code `0:0`, elapsed `00:04:44`, GPU allocation `gres/gpu:a40-48=1`.
- W&B group: `pop909-resume-probe-20260610-183406`.
- W&B runs:
  - `resume-test-direct-1781109342-1868350` at `https://wandb.ai/micael-antunes-lis-cnrs/pop909-reproduction/runs/wtx7va8t`.
  - `resume-test-initial-1781109342-1868350` at `https://wandb.ai/micael-antunes-lis-cnrs/pop909-reproduction/runs/9d2mr1rr`.
  - `resume-test-resumed-1781109342-1868350` at `https://wandb.ai/micael-antunes-lis-cnrs/pop909-reproduction/runs/o5egtenn`.
- Downloaded artifacts: `vae-textures-dev/_artefatos/cluster-pop909-resume-probe-332828/`.
- Manifest: GPU mode, W&B enabled, `limit_train_shuffle=true`, run id `1781109342-1868350`.
- Comparison report: model PASS, optimizer PASS, scheduler PASS, counters PASS, RNG PASS, `RESULT=PASSED`.
- Stdout: `non_equal_weight_tensors=0`, resumed leg used `resume_from=/workspace/base_model/result_2026-06-10_183731/models/resume-test-initial-1781109342-1868350_last-state_state.pt`.
- W&B stderr: metrics, checkpoint timing, and checkpoint artifacts synced for direct, initial, and resumed mini-runs.

Resume probe decision: accept.

Remaining caveat: this is still a small bounded validation run, not the representative Phase 8 training run. Phase 8 can now use the accepted checkpoint/resume and W&B path, while the supervisor-facing package should state the CUDA determinism caveat and the measured checkpoint overhead from this probe.
