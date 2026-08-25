# POP909 Phase 8 staged representative training guide

Status: Phase 8 staged launch tooling ready for pre-flight

## What "no sample limits" means

`base_model/train.py` has optional environment variables for bounded proof runs:

- `VAE_LIMIT_TRAIN_SAMPLES`
- `VAE_LIMIT_VAL_SAMPLES`

When either value is greater than `0`, `train.py` replaces the canonical loader
with a small `Subset(...)`. That is useful for smoke tests and the Phase 7
checkpoint/resume proof.

For the representative run, these limits should be unset or set to `0`. That
means `train.py` uses the full POP909 loader created by:

```text
MusicDataLoaders.get_loaders(
  SEED,
  bs_train=128,
  bs_val=128,
  portion=8,
  shift_low=-6,
  shift_high=5,
  num_bar=2,
  contain_chord=True
)
```

This is the faithful path for the authors-style run in this restored codebase.
The full training loader uses `random_train=True` by default, so the
representative run keeps the normal shuffled training behavior. The sample
limits were added only to make short validation jobs practical.

## Phase 8 operating rule

Training proceeds in staged sessions. Nothing advances without human validation.

Before epoch 1:

1. Confirm the POP909 dataset pre-flight evidence.
2. Confirm or run a representative timing estimate.
3. Confirm the initial seed.
4. Confirm the W&B continuity strategy.

For each training session:

1. Submit one sbatch job that runs the canonical `base_model/train.py`.
2. Let it run one epoch.
3. Inspect Slurm, logs, W&B, losses, runtime, and checkpoints.
4. Download the evidence locally.
5. Decide `accept`, `rerun`, or `block`.
6. Only after `accept`, submit the next staged session.

## Recommended run shape

- Use the canonical `base_model/train.py`.
- Use released-code-faithful configuration accepted in Phase 7.
- Use the full POP909 dataset: do not set positive sample limits.
- Keep W&B enabled in `pop909-reproduction`.
- Keep full-state checkpointing active:
  - `epoch-state`
  - `last-state`
  - `final-state`
- Run one epoch per sbatch session:
  - `VAE_N_EPOCH=6`
  - `VAE_RUN_EPOCHS_THIS_JOB=1`
- Resume sessions from the latest accepted `*_last-state_state.pt`.

## W&B continuity decision

Preferred behavior: the six staged jobs should appear as one continuous W&B run.
That means the Phase 8 submit path should preserve a stable W&B run id across
sessions, for example with `WANDB_RUN_ID`, and use W&B resume behavior when a
subsequent job continues from `VAE_RESUME_FROM`.

Implemented Phase 8 behavior: `base_model/wandb_helper.py` maps
`WANDB_RUN_ID` to `wandb.init(id=...)` and `WANDB_RESUME` to
`wandb.init(resume=...)`. `base_model/train.py` also records `wandb_run_id` and
`wandb_resume` in the W&B config so the continuity strategy is visible in the
evidence.

Recommended first-session values:

```bash
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
export WANDB_RESUME=allow
export WANDB_GROUP=pop909-phase8-representative-ATTEMPT_ID
```

Recommended resumed-session values:

```bash
export WANDB_RUN_ID=THE_SAME_ID_FROM_EPOCH_1
export WANDB_RESUME=must
export WANDB_GROUP=THE_SAME_GROUP_FROM_EPOCH_1
```

Fallback if continuous W&B resume is not robust enough: keep separate W&B runs
under one group and maintain a manifest that joins session -> epoch ->
checkpoint -> W&B URL. This fallback must be called out to the supervisor.

## Initial seed decision

The default seed is `3345`, imported from `base_model/dataset.py` and exposed in
`train.py` as `VAE_SEED`. For a repeatable representative run, record the seed
for session 1 in logs and W&B config. Later sessions should continue from
checkpoint-restored RNG state rather than re-seeding as a fresh experiment.

If a different seed is chosen, set it explicitly before epoch 1:

```bash
export VAE_SEED=3345
```

## Dataset pre-flight before epoch 1

Do not launch epoch 1 until the cluster has shown dataset evidence. The Phase 8
pre-flight helper is:

```text
base_model/phase8_preflight.py
```

Run it on the cluster through the staged submitter so it uses the same sbatch and
container path as training:

```bash
cd ~/vae-textures-dev
scripts/submit_pop909_phase8_session.sh preflight
```

The helper prints stable `[phase8-preflight]` lines and exits nonzero if
`VAE_LIMIT_TRAIN_SAMPLES` or `VAE_LIMIT_VAL_SAMPLES` is positive. Capture:

- number of POP909 `.npz` files visible;
- number of selected duple-meter files;
- train/validation dataset lengths;
- train/validation batch counts with `VAE_BATCH_SIZE=128`;
- confirmation that `VAE_LIMIT_TRAIN_SAMPLES` and `VAE_LIMIT_VAL_SAMPLES` are
  unset or `0`.

The existing dataset path prints some of this during loader construction:

```text
The folder contains ... .npz files.
Selected ... files, all are in duple meter.
<train_dataset_len> <val_dataset_len>
```

If counts are missing, unexpectedly small, or inconsistent with prior evidence,
block launch instead of training on an accidental subset.

## Timing before epoch 1

Do not choose Slurm `--time` from the tiny Phase 7 four-sample probe. Use an
accepted representative timing estimate first, then choose a conservative buffer
for the full first epoch. If epoch 1 reveals a materially different runtime,
adjust later session time limits only after human review.

## Staged submitter

The Phase 8 submitter is:

```text
scripts/submit_pop909_phase8_session.sh
```

It supports two modes:

```bash
scripts/submit_pop909_phase8_session.sh preflight
scripts/submit_pop909_phase8_session.sh train
```

Defaults for representative sessions:

```bash
export VAE_SEED=3345
export VAE_BATCH_SIZE=128
export VAE_N_EPOCH=6
export VAE_RUN_EPOCHS_THIS_JOB=1
export WANDB_PROJECT=pop909-reproduction
export WANDB_TAGS=pop909,phase8,representative,released-code-faithful
export WANDB_CHECKPOINT_POLICY=valid,final,epoch-state,last-state,final-state
```

The submitter does not set positive sample limits and refuses to submit if
`VAE_LIMIT_TRAIN_SAMPLES` or `VAE_LIMIT_VAL_SAMPLES` is positive.

## Local sync before a session

Run from the local host:

```bash
cd ~/Documents/VAE-Textures/vae-textures-dev
scripts/sync_runtime_to_cluster.sh sync
```

## Manual evidence download after a session

Run from the local host after the cluster job completes:

```bash
cd ~/Documents/VAE-Textures/vae-textures-dev
scripts/download_pop909_phase8_artifacts.sh JOB_ID REMOTE_RESULT_DIR
```

The download helper is download-only: it does not submit jobs, delete files,
resume training, or make the accept/rerun/block decision.

```text
scripts/download_pop909_phase8_artifacts.sh JOB_ID REMOTE_RESULT_DIR
```

Example:

```bash
scripts/download_pop909_phase8_artifacts.sh 333001 \
  /home/micael.antunes/vae-textures-dev/base_model/result_2026-06-10_210501
```

The local evidence will be placed under:

```text
_artefatos/cluster-pop909-phase8-JOB_ID/
```

## What to inspect before accepting a staged session

Slurm:

```bash
sacct -j JOB_ID --format=JobID,JobName%30,State,Elapsed,Timelimit,AllocTRES%60,ExitCode
```

Logs:

```bash
tail -n 80 logs/pop909-representative/*.out
tail -n 80 logs/pop909-representative/*.err
```

Key log patterns:

```bash
grep -E "Epoch:|Epoch train/eval seconds|Saved model weights in|Saved training state|resume_from=|wandb" LOG_FILE
```

Checkpoints:

```bash
find base_model/result_*/models -maxdepth 1 -type f | sort
```

W&B:

- Confirm the run is in project `pop909-reproduction`.
- Confirm config shows the full representative settings.
- Confirm metrics are being logged.
- Confirm checkpoint/artifact evidence exists.
- Capture the run URL in the Phase 8 notes.

## Accept/rerun/block decision

Accept the session only if:

- Slurm completed or stopped for an understood recoverable reason.
- stdout/stderr do not show unexplained training/runtime failures.
- W&B has the expected run, config, metrics, and artifact/checkpoint evidence.
- W&B continuity matches the accepted strategy: one resumed run, or the explicit grouped-run fallback.
- At least one usable full-state checkpoint exists.
- Loss behavior is plausible enough to continue.
- No positive sample limits are active for the representative training session.
- The user has inspected the evidence and explicitly approves the next session.

Rerun if:

- The job failed before producing useful checkpoints.
- W&B/logging was misconfigured.
- The run used wrong config or sample limits.

Block if:

- The failure is not understood.
- Checkpoint/resume evidence contradicts Phase 7 assumptions.
- Loss behavior or outputs suggest the representative setup is invalid.

## Resume rule

If a session is accepted, the next session resumes from the latest accepted:

```text
*_last-state_state.pt
```

Do not resume from a checkpoint that has not been inspected and accepted.

The next launch should set:

```bash
export VAE_RESUME_FROM=/workspace/base_model/result_.../models/..._last-state_state.pt
export VAE_N_EPOCH=6
export VAE_RUN_EPOCHS_THIS_JOB=1
export WANDB_RUN_ID=THE_SAME_ID_FROM_EPOCH_1
export WANDB_RESUME=must
```

`VAE_RESUME_FROM` uses the path as seen inside the container (`/workspace/...`),
not the host path (`/home/...`).

## Phase 8 pre-flight/timing decision - 2026-06-22

Pre-flight/timing decision: accept

Evidence source: Slurm pre-flight job `336626`, submitted with `scripts/submit_pop909_phase8_session.sh preflight`.

Pre-flight result:

- Slurm state: `COMPLETED`, exit code `0:0`
- `VAE_SEED=3345`
- `VAE_BATCH_SIZE=128`
- `VAE_LIMIT_TRAIN_SAMPLES=0`
- `VAE_LIMIT_VAL_SAMPLES=0`
- POP909 `.npz` count: `886`
- Selected duple-meter files: `858`
- Train dataset length: `702756`
- Validation dataset length: `7718`
- Train batch count: `5491`
- Validation batch count: `61`
- Pre-flight status: `ok`

Timing decision:

- Epoch 1 will be used as the default full-data timing calibration run.
- Selected epoch-1 `SLURM_TIME=12:00:00`.
- Rationale: Phase 7 four-sample timing is correctness/checkpoint evidence, not full-epoch timing evidence. The pre-flight shows a full epoch will cover `5491` train batches plus `61` validation batches, so epoch 1 should use a deliberately conservative time limit and later sessions should be adjusted only after human review of epoch-1 duration.

W&B continuity strategy for epoch 1:

- Generate a stable `WANDB_RUN_ID` before launch.
- Use `WANDB_RESUME=allow` for the first session.
- Reuse the same `WANDB_RUN_ID` with `WANDB_RESUME=must` for resumed sessions after an accepted checkpoint.

## Phase 8 epoch 1 decision - 2026-06-23

Epoch 1 decision: accept

Evidence source: Slurm training job `336628`, downloaded locally to `_artefatos/cluster-pop909-phase8-336628/`.

Run identity:

- `VAE_RUN_NAME=pop909-phase8-representative-20260622-155059`
- `WANDB_RUN_ID=phase8-1782136259-24729`
- `WANDB_RESUME=allow`
- W&B URL: `https://wandb.ai/micael-antunes-lis-cnrs/pop909-reproduction/runs/phase8-1782136259-24729`

Configuration evidence:

- Slurm state: `COMPLETED`, exit code `0:0`
- Slurm elapsed: `07:21:18` against `12:00:00` time limit
- Epoch train/eval seconds: `26405.490` (`7h20m05s`)
- `VAE_SEED=3345`
- `VAE_BATCH_SIZE=128`
- `VAE_N_EPOCH=6`
- `VAE_RUN_EPOCHS_THIS_JOB=1`
- `VAE_LIMIT_TRAIN_SAMPLES=0`
- `VAE_LIMIT_VAL_SAMPLES=0`
- POP909 `.npz` count: `886`
- Selected duple-meter files: `858`
- Train dataset length: `702756`
- Validation dataset length: `7718`

Loss evidence:

- Epoch train loss sum: `7329.875`
- Epoch validation loss sum: `52.351`
- Approximate train loss per batch: `7329.875 / 5491 = 1.335`
- Approximate validation loss per batch: `52.351 / 61 = 0.858`
- Interpretation: acceptable for continuation; epoch sums are not directly comparable because train and validation use very different batch counts. No NaN/inf/crash evidence was observed.

Checkpoint evidence:

- Downloaded model/result directory: `_artefatos/cluster-pop909-phase8-336628/results/result_2026-06-22_155143/`
- Downloaded checkpoint files include `epoch`, `valid`, `epoch-state`, `last-state`, `final`, and `final-state`.
- W&B synced `5` files and `10` artifact files.

Accepted resume checkpoint for epoch 2:

```bash
export VAE_RESUME_FROM=/workspace/base_model/result_2026-06-22_155143/models/pop909-phase8-representative-20260622-155059_last-state_state.pt
export WANDB_RUN_ID=phase8-1782136259-24729
export WANDB_RESUME=must
```

Next session should keep:

```bash
export VAE_N_EPOCH=6
export VAE_RUN_EPOCHS_THIS_JOB=1
export VAE_BATCH_SIZE=128
export VAE_SEED=3345
```

## Phase 8 epoch 2 decision - 2026-06-24

Epoch 2 decision: accept

Evidence source: Slurm training job `337113`, downloaded locally to `_artefatos/cluster-pop909-phase8-337113/`.

Run identity:

- `VAE_RUN_NAME=pop909-phase8-representative-20260623-163458`
- `WANDB_RUN_ID=phase8-1782136259-24729`
- `WANDB_RESUME=must`
- W&B URL: `https://wandb.ai/micael-antunes-lis-cnrs/pop909-reproduction/runs/phase8-1782136259-24729`

Configuration evidence:

- Slurm state: `COMPLETED`, exit code `0:0`
- Slurm elapsed: `07:05:15` against `12:00:00` time limit
- Epoch train/eval seconds: `25416.847` (`7h03m37s`)
- `VAE_SEED=3345`
- `VAE_BATCH_SIZE=128`
- `VAE_N_EPOCH=6`
- `VAE_RUN_EPOCHS_THIS_JOB=1`
- `VAE_RESUME_FROM=/workspace/base_model/result_2026-06-22_155143/models/pop909-phase8-representative-20260622-155059_last-state_state.pt`
- `VAE_LIMIT_TRAIN_SAMPLES=0`
- `VAE_LIMIT_VAL_SAMPLES=0`
- POP909 `.npz` count: `886`
- Selected duple-meter files: `858`
- Train dataset length: `702756`
- Validation dataset length: `7718`

Loss evidence:

- Epoch train loss sum: `3683.561`
- Epoch validation loss sum: `41.810`
- Previous best validation loss restored from epoch 1: `52.350839018821716`
- Approximate train loss per batch: `3683.561 / 5491 = 0.671`
- Approximate validation loss per batch: `41.810 / 61 = 0.685`
- Interpretation: acceptable for continuation; validation improved from epoch 1, full-data settings were preserved, and no training/runtime failure was observed.

Checkpoint evidence:

- Downloaded model/result directory: `_artefatos/cluster-pop909-phase8-337113/results/result_2026-06-23_230321/`
- Downloaded checkpoint files include `epoch`, `valid`, `epoch-state`, `last-state`, `final`, and `final-state`.
- W&B resumed the existing run and synced `5` W&B files and `10` artifact files.

Accepted resume checkpoint for epoch 3:

```bash
export VAE_RESUME_FROM=/workspace/base_model/result_2026-06-23_230321/models/pop909-phase8-representative-20260623-163458_last-state_state.pt
export WANDB_RUN_ID=phase8-1782136259-24729
export WANDB_RESUME=must
```

Next session should keep:

```bash
export VAE_N_EPOCH=6
export VAE_RUN_EPOCHS_THIS_JOB=1
export VAE_BATCH_SIZE=128
export VAE_SEED=3345
```

## Phase 8 epoch 3 decision - 2026-06-24

Epoch 3 decision: accept

Evidence source: Slurm training job `338187`, downloaded locally to `_artefatos/cluster-pop909-phase8-338187/`.

Run identity:

- `VAE_RUN_NAME=pop909-phase8-representative-20260624-084349`
- `WANDB_RUN_ID=phase8-1782136259-24729`
- `WANDB_RESUME=must`
- W&B URL: `https://wandb.ai/micael-antunes-lis-cnrs/pop909-reproduction/runs/phase8-1782136259-24729`

Configuration evidence:

- Slurm state: `COMPLETED`, exit code `0:0`
- Slurm elapsed: `07:38:15` against `12:00:00` time limit
- Epoch train/eval seconds: `27407.870` (`7h36m47s`)
- `VAE_SEED=3345`
- `VAE_BATCH_SIZE=128`
- `VAE_N_EPOCH=6`
- `VAE_RUN_EPOCHS_THIS_JOB=1`
- `VAE_RESUME_FROM=/workspace/base_model/result_2026-06-23_230321/models/pop909-phase8-representative-20260623-163458_last-state_state.pt`
- `VAE_LIMIT_TRAIN_SAMPLES=0`
- `VAE_LIMIT_VAL_SAMPLES=0`
- POP909 `.npz` count: `886`
- Selected duple-meter files: `858`
- Train dataset length: `702756`
- Validation dataset length: `7718`

Loss evidence:

- Epoch train loss sum: `2758.509`
- Epoch validation loss sum: `37.309`
- Previous best validation loss restored from epoch 2: `41.8102271258831`
- Approximate train loss per batch: `2758.509 / 5491 = 0.502`
- Approximate validation loss per batch: `37.309 / 61 = 0.612`
- Interpretation: acceptable for continuation; validation improved from epoch 2, full-data settings were preserved, and no training/runtime failure was observed.

Checkpoint evidence:

- Downloaded model/result directory: `_artefatos/cluster-pop909-phase8-338187/results/result_2026-06-24_084446/`
- Downloaded checkpoint files include `epoch`, `valid`, `epoch-state`, `last-state`, `final`, and `final-state`.
- W&B resumed the existing run and synced `5` W&B files and `10` artifact files.

Accepted resume checkpoint for epoch 4:

```bash
export VAE_RESUME_FROM=/workspace/base_model/result_2026-06-24_084446/models/pop909-phase8-representative-20260624-084349_last-state_state.pt
export WANDB_RUN_ID=phase8-1782136259-24729
export WANDB_RESUME=must
```

Next session should keep:

```bash
export VAE_N_EPOCH=6
export VAE_RUN_EPOCHS_THIS_JOB=1
export VAE_BATCH_SIZE=128
export VAE_SEED=3345
```

## Epoch 4 validation

Epoch 4 decision: accept

Evidence source: Slurm training job `338855`, downloaded locally to `_artefatos/cluster-pop909-phase8-338855/`.

Run identity:

- `VAE_RUN_NAME=pop909-phase8-representative-20260624-191423`
- `WANDB_RUN_ID=phase8-1782136259-24729`
- `WANDB_RESUME=must`
- W&B URL: `https://wandb.ai/micael-antunes-lis-cnrs/pop909-reproduction/runs/phase8-1782136259-24729`

Configuration evidence:

- Slurm state: `COMPLETED`, exit code `0:0`
- Slurm elapsed: `05:42:02` against `12:00:00` time limit
- Epoch train/eval seconds: `20439.444` (`5h40m39s`)
- `VAE_SEED=3345`
- `VAE_BATCH_SIZE=128`
- `VAE_N_EPOCH=6`
- `VAE_RUN_EPOCHS_THIS_JOB=1`
- `VAE_RESUME_FROM=/workspace/base_model/result_2026-06-24_084446/models/pop909-phase8-representative-20260624-084349_last-state_state.pt`
- `VAE_LIMIT_TRAIN_SAMPLES=0`
- `VAE_LIMIT_VAL_SAMPLES=0`
- POP909 `.npz` count: `886`
- Selected duple-meter files: `858`
- Train dataset length: `702756`
- Validation dataset length: `7718`

Loss evidence:

- Epoch train loss sum: `2276.080`
- Epoch validation loss sum: `37.175`
- Previous best validation loss restored from epoch 3: `37.30945956707001`
- Approximate train loss per batch: `2276.080 / 5491 = 0.415`
- Approximate validation loss per batch: `37.175 / 61 = 0.609`
- Validation audit note: epoch 4 shows a plateau rather than clear deterioration. `val/loss` mean improved only slightly from epoch 3 (`0.6116` to `0.6094`), `val/recon_loss` mean was essentially flat/slightly higher (`0.4709` to `0.4728`), and `val/kl_rhy` remained active but stable (`0.8230` to `0.8327`). Continue to epoch 5 with stricter validation-curve monitoring.

Checkpoint evidence:

- Downloaded model/result directory: `_artefatos/cluster-pop909-phase8-338855/results/result_2026-06-24_191514/`
- Downloaded checkpoint files include `epoch`, `valid`, `epoch-state`, `last-state`, `final`, and `final-state`.
- W&B resumed the existing run and synced `5` W&B files and `10` artifact files.

Accepted resume checkpoint for epoch 5:

```bash
export VAE_RESUME_FROM=/workspace/base_model/result_2026-06-24_191514/models/pop909-phase8-representative-20260624-191423_last-state_state.pt
export WANDB_RUN_ID=phase8-1782136259-24729
export WANDB_RESUME=must
```

Next session should keep:

```bash
export VAE_N_EPOCH=6
export VAE_RUN_EPOCHS_THIS_JOB=1
export VAE_BATCH_SIZE=128
export VAE_SEED=3345
```


## Epoch 5 validation

Epoch 5 decision: accept

Evidence source: Slurm training job `340811`, downloaded locally to `_artefatos/cluster-pop909-phase8-340811/`.

Run identity:

- `VAE_RUN_NAME=pop909-phase8-representative-20260625-113904`
- `WANDB_RUN_ID=phase8-1782136259-24729`
- `WANDB_RESUME=must`
- W&B URL: `https://wandb.ai/micael-antunes-lis-cnrs/pop909-reproduction/runs/phase8-1782136259-24729`

Configuration evidence:

- Slurm state: `COMPLETED`, exit code `0:0`
- Slurm elapsed: `05:59:02` against `12:00:00` time limit
- Epoch train/eval seconds: `21462.660` (`5h57m43s`)
- `VAE_SEED=3345`
- `VAE_BATCH_SIZE=128`
- `VAE_N_EPOCH=6`
- `VAE_RUN_EPOCHS_THIS_JOB=1`
- `VAE_RESUME_FROM=/workspace/base_model/result_2026-06-24_191514/models/pop909-phase8-representative-20260624-191423_last-state_state.pt`
- `VAE_LIMIT_TRAIN_SAMPLES=0`
- `VAE_LIMIT_VAL_SAMPLES=0`
- POP909 `.npz` count: `886`
- Selected duple-meter files: `858`
- Train dataset length: `702756`
- Validation dataset length: `7718`

Loss evidence:

- Epoch train loss sum: `2002.946`
- Epoch validation loss sum: `37.572`
- Previous best validation loss restored from epoch 4: `37.17481905221939`
- Approximate train loss per batch: `2002.946 / 5491 = 0.365`
- Approximate validation loss per batch: `37.572 / 61 = 0.616`
- Validation audit note: epoch 5 shows mild validation plateau/regression rather than a breakdown. Mean `val/loss` worsened slightly from `0.609423` to `0.615938`, mean `val/recon_loss` worsened from `0.472804` to `0.483319`, mean `val/kl_rhy` stayed stable in-band from `0.832675` to `0.840709`, and mean `val/chord_loss` improved from `0.018384` to `0.014981`. The stricter pause condition was not met.

Checkpoint evidence:

- Downloaded model/result directory: `_artefatos/cluster-pop909-phase8-340811/results/result_2026-06-25_114018/`
- Downloaded checkpoint files include `epoch`, `epoch-state`, `last-state`, `final`, and `final-state`.
- W&B resumed the existing run and synced `5` W&B files and `8` artifact files.
- Detailed audit: `_artefatos/POP909 phase 8 epoch 5 validation audit.md`

Accepted resume checkpoint for epoch 6:

```bash
export VAE_RESUME_FROM=/workspace/base_model/result_2026-06-25_114018/models/pop909-phase8-representative-20260625-113904_last-state_state.pt
export WANDB_RUN_ID=phase8-1782136259-24729
export WANDB_RESUME=must
```

Final staged epoch should keep:

```bash
export VAE_N_EPOCH=6
export VAE_RUN_EPOCHS_THIS_JOB=1
export VAE_BATCH_SIZE=128
export VAE_SEED=3345
```

## Epoch 6 validation

Epoch 6 decision: accept

Recommendation: accept epoch 6 as protocol-completion checkpoint, with validation-regression caveat

Human decision: accepted by user on 2026-06-26.

Evidence source: Slurm training job `341594`, downloaded locally to `_artefatos/cluster-pop909-phase8-341594/`.

Operational evidence:

- Slurm state: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `06:28:00` against `12:00:00`
- GPU allocation: `gres/gpu:a40-48=1`
- `WANDB_RUN_ID=phase8-1782136259-24729`
- `WANDB_RESUME=must`
- W&B URL: `https://wandb.ai/micael-antunes-lis-cnrs/pop909-reproduction/runs/phase8-1782136259-24729`

Resume evidence:

```text
[resume] Loaded training state from /workspace/base_model/result_2026-06-25_114018/models/pop909-phase8-representative-20260625-113904_last-state_state.pt | epoch=5 train_step=27455 val_step=305 best_valid_loss=37.17481905221939
```

Loss evidence:

- Epoch train/eval seconds: `23196.110` (`6h26m36s`)
- Epoch train loss sum: `1844.998`
- Epoch validation loss sum: `38.082`
- Previous epoch validation loss: `37.572`
- Best restored validation loss: `37.17481905221939`
- Approximate train loss per batch: `1844.998 / 5491 = 0.336`
- Approximate validation loss per batch: `38.082 / 61 = 0.624`
- Validation audit note: epoch 6 shows a small but real monotonic validation regression after the epoch 4 best point (`37.175 -> 37.572 -> 38.082`). Training loss continued improving, so this may reflect late-epoch overfitting/generalization drift rather than runtime failure. W&B summary `val/*` values are last-batch metrics, not epoch means; epoch 6 TensorBoard validation tags are missing locally, so the validation trend rests on stdout/W&B epoch metrics. Treat `val/chord_loss` improvement as a small absolute change, not decisive evidence.

Checkpoint evidence:

- Downloaded model/result directory: `_artefatos/cluster-pop909-phase8-341594/results/result_2026-06-25_212850/`
- Downloaded checkpoint files include `epoch`, `epoch-state`, `last-state`, `final`, and `final-state`.
- W&B resumed the existing run and synced `5` W&B files and `8` artifact files.
- Detailed audit: `_artefatos/POP909 phase 8 epoch 6 validation audit.md`

Recommended final full-state checkpoint for Phase 9:

```text
/workspace/vae-textures-dev/_artefatos/cluster-pop909-phase8-341594/results/result_2026-06-25_212850/models/pop909-phase8-representative-20260625-212752_final-state_state.pt
```

Recommended final model-weights checkpoint for Phase 9 diagnostics:

```text
/workspace/vae-textures-dev/_artefatos/cluster-pop909-phase8-341594/results/result_2026-06-25_212850/models/pop909-phase8-representative-20260625-212752_final.pt
```


## Final Phase 8 decision

Phase 8 decision: accept

Human decision: accepted by user on 2026-06-26.

Accepted scope:

- Representative POP909 training completed through the planned 6 staged epochs.
- All staged jobs completed with Slurm exit code `0:0`.
- Full POP909 data settings were preserved throughout (`VAE_LIMIT_TRAIN_SAMPLES=0`, `VAE_LIMIT_VAL_SAMPLES=0`).
- W&B continuity preserved run id `phase8-1782136259-24729`.
- Final protocol-completion checkpoint is available for Phase 9.
- Epoch 4 best-validation checkpoint is retained as a diagnostic comparator for Phase 9.

Caveat:

Phase 8 acceptance is operational/protocol acceptance, not scientific equivalence to the authors. Validation loss worsened monotonically after epoch 4 (`37.175 -> 37.572 -> 38.082`), so Phase 9 must compare authors, our epoch 4, and our epoch 6 outputs before making the final reproduction claim.
