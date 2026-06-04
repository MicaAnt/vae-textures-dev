# POP909 exact checkpoint/resume tutorial

Date: 2026-06-04
Purpose: give the human operator a complete, manual path to understand, run, inspect, and decide whether the exact checkpoint/resume proof is acceptable.

This document is not a success certificate. It is a tutorial. The proof only counts after you run it, inspect the outputs, and choose one of: `accept`, `rerun`, or `block`.

## What you are proving

You are proving this specific claim:

> If training runs for epoch 1, saves a full-state checkpoint, then resumes and runs epoch 2, the final training state matches a normal uninterrupted two-epoch run under deterministic local CPU conditions.

This is narrower than the cluster `sbatch` proof. It proves technical state continuity locally. The cluster probe still proves operational continuity on Slurm/W&B.

## Mental model

A fake resume would do something like this:

1. Load only model weights.
2. Start a new optimizer/scheduler/RNG state.
3. Print `Epoch: 02` because we told it to.

A real epoch-boundary resume must restore all training context that affects the next epoch:

- model weights;
- Adam optimizer state;
- learning-rate scheduler state;
- internal parameter scheduler steps;
- epoch counter;
- train/validation step counters;
- best validation loss so far;
- random-number generator state.

The exact proof compares two paths:

```text
Path A: train epoch 1 -> train epoch 2 -> final state
Path B: train epoch 1 -> save last-state -> load last-state -> train epoch 2 -> final state
```

If final states match, the resumed epoch 2 is not fake.

## Part 1 - Understand the code before running

Run these commands from `/workspace` and read the listed lines.

### 1. Check the checkpoint contract

```bash
nl -ba vae-textures-dev/base_model/canonical_checkpoint.py | sed -n '1,110p'
```

Look for `REQUIRED_TRAINING_STATE_KEYS`. You should see:

```text
model_state_dict
optimizer_state_dict
lr_scheduler_state_dict
optimizer_scheduler_step
param_scheduler_steps
epoch
train_step
val_step
best_valid_loss
config
rng_state
```

Decision being made here:

- A resume-capable checkpoint is not just weights.
- If any required key is missing, loading/saving must fail.
- `rng_state` is required because randomness affects epoch 2.

Now look for `capture_rng_state()` and `restore_rng_state()`.

You should see that we capture/restore:

```text
python_random_state
numpy_random_state
torch_rng_state
torch_cuda_rng_state_all
```

What this means:

- Python `random` matters because teacher forcing uses `random.random()`.
- NumPy matters because dataset split/setup can use NumPy randomness.
- PyTorch RNG matters because latent sampling uses PyTorch distributions.
- CUDA RNG is included for GPU state, though exact GPU equality can still depend on deterministic kernels.

### 2. Check where the training state is created

```bash
nl -ba vae-textures-dev/base_model/amc_dl/torch_plus/module.py | sed -n '216,275p'
```

Look at `_training_state_payload()`.

You should see the payload include model, optimizer, scheduler, counters, best loss, config, and `rng_state`.

Then look at `load_training_state_checkpoint()`.

You should see the order:

1. load payload;
2. restore model weights;
3. restore optimizer;
4. restore learning-rate scheduler;
5. restore scheduler step counters;
6. restore `epoch`, `train_step`, `val_step`;
7. restore RNG state;
8. print `[resume] Loaded training state from ...`.

Important point: the printed `epoch=1 train_step=... val_step=...` comes from the checkpoint, not from wishful thinking.

### 3. Check when epoch state is saved

```bash
nl -ba vae-textures-dev/base_model/amc_dl/torch_plus/module.py | sed -n '288,330p'
```

Read the loop:

- while `self.epoch < self.n_epoch`;
- train;
- validate;
- save model-only `epoch.pt`;
- increment `self.epoch`;
- save `epoch-state`;
- save `last-state`;
- after the loop, save `final-state`.

The normal resume file is `last-state`, because it is saved after a complete epoch boundary.

### 4. Check how `train.py` controls deterministic runs and resume

```bash
nl -ba vae-textures-dev/base_model/train.py | sed -n '23,65p'
nl -ba vae-textures-dev/base_model/train.py | sed -n '126,165p'
```

Look for:

```text
VAE_SEED
VAE_RESUME_FROM
VAE_RUN_EPOCHS_THIS_JOB
```

Interpretation:

- `VAE_SEED` makes the initial run deterministic.
- `VAE_RESUME_FROM` points at a full-state checkpoint.
- `VAE_N_EPOCH=2` means the total target is epoch 2.
- `VAE_RUN_EPOCHS_THIS_JOB=1` means this job should run only one epoch and stop.

For a resumed job, if the checkpoint says `epoch=1` and total target is `VAE_N_EPOCH=2`, the loop should run exactly the next epoch.

### 5. Check what the verifier actually does

```bash
nl -ba vae-textures-dev/base_model/verify_exact_resume_continuity.py | sed -n '1,220p'
```

Read it in this order:

1. `run_train(...)` sets environment variables and calls `python -u train.py`.
2. First call runs direct two-epoch training.
3. Second call runs one epoch and saves `last-state`.
4. Third call resumes from that `last-state` and targets epoch 2.
5. `compare_training_states(...)` compares the final direct checkpoint against the final resumed checkpoint.

The verifier is not just checking that the log says `Epoch: 02`. It compares the saved state contents.

## Part 2 - Run the proof manually

From `/workspace`, run:

```bash
python vae-textures-dev/base_model/verify_exact_resume_continuity.py
```

Expected shape of the output:

```text
[verify] run=exact-resume-direct-... n_epoch=2 resume_from=
...
Epoch: 01
...
Epoch: 02
...
[verify] run=exact-resume-initial-... n_epoch=1 resume_from=
...
Epoch: 01
...
[verify] run=exact-resume-resumed-... n_epoch=2 resume_from=/workspace/...last-state_state.pt
...
[resume] Loaded training state from ... | epoch=1 train_step=2 val_step=1 best_valid_loss=...
Epoch: 02
...
[verify] direct_final=...
[verify] initial_last=...
[verify] resumed_final=...
[verify] exact resume continuity PASSED
[verify] resumed epoch 2 matches uninterrupted two-epoch training state
```

Do not accept the proof just because the final line says `PASSED`. Copy the three paths printed at the end:

```text
direct_final=...
initial_last=...
resumed_final=...
```

You will inspect them next.

## Part 3 - Inspect the checkpoint files yourself

Set shell variables using the paths printed by your run. Example only; replace with your actual paths:

```bash
export DIRECT="/workspace/vae-textures-dev/base_model/result_YYYY-MM-DD_HHMMSS/models/exact-resume-direct-..._final-state_state.pt"
export INITIAL="/workspace/vae-textures-dev/base_model/result_YYYY-MM-DD_HHMMSS/models/exact-resume-initial-..._last-state_state.pt"
export RESUMED="/workspace/vae-textures-dev/base_model/result_YYYY-MM-DD_HHMMSS/models/exact-resume-resumed-..._final-state_state.pt"
```

### 1. Inspect required keys

```bash
python - <<'PY'
import os
import torch
for label, path in [('direct', os.environ['DIRECT']), ('initial', os.environ['INITIAL']), ('resumed', os.environ['RESUMED'])]:
    state = torch.load(path, map_location='cpu', weights_only=False)
    print('\n==', label, '==')
    print('\n'.join(sorted(state.keys())))
PY
```

You must see `rng_state` in all three.

If `rng_state` is missing: `block`.

### 2. Inspect epoch and step counters

```bash
python - <<'PY'
import os
import torch
for label, path in [('direct', os.environ['DIRECT']), ('initial', os.environ['INITIAL']), ('resumed', os.environ['RESUMED'])]:
    state = torch.load(path, map_location='cpu', weights_only=False)
    print(label, {
        'epoch': state['epoch'],
        'train_step': state['train_step'],
        'val_step': state['val_step'],
        'best_valid_loss': state['best_valid_loss'],
        'optimizer_scheduler_step': state['optimizer_scheduler_step'],
        'param_scheduler_steps': state['param_scheduler_steps'],
    })
PY
```

Expected interpretation:

- `initial` should have `epoch=1`.
- `direct` and `resumed` should have `epoch=2`.
- `direct` and `resumed` should have the same `train_step`, `val_step`, scheduler steps, and best loss.

If direct and resumed counters differ: `block` or investigate.

### 3. Inspect RNG state presence

```bash
python - <<'PY'
import os
import torch
for label, path in [('direct', os.environ['DIRECT']), ('initial', os.environ['INITIAL']), ('resumed', os.environ['RESUMED'])]:
    rng = torch.load(path, map_location='cpu', weights_only=False)['rng_state']
    print('\n==', label, 'rng ==')
    for key, value in rng.items():
        if hasattr(value, 'shape'):
            print(key, tuple(value.shape), value.dtype)
        elif isinstance(value, list):
            print(key, 'list length', len(value))
        else:
            print(key, type(value).__name__)
PY
```

Expected interpretation:

- `python_random_state`: tuple
- `numpy_random_state`: tuple
- `torch_rng_state`: tensor
- `torch_cuda_rng_state_all`: list, possibly empty on CPU

If these are absent or malformed: `block`.

### 4. Compare direct vs resumed manually

```bash
python - <<'PY'
import os
import torch

direct = torch.load(os.environ['DIRECT'], map_location='cpu', weights_only=False)
resumed = torch.load(os.environ['RESUMED'], map_location='cpu', weights_only=False)

print('epoch equal:', direct['epoch'] == resumed['epoch'])
print('train_step equal:', direct['train_step'] == resumed['train_step'])
print('val_step equal:', direct['val_step'] == resumed['val_step'])
print('best_valid_loss equal:', direct['best_valid_loss'] == resumed['best_valid_loss'])
print('param_scheduler_steps equal:', direct['param_scheduler_steps'] == resumed['param_scheduler_steps'])

model_diffs = []
for key in direct['model_state_dict']:
    if not torch.equal(direct['model_state_dict'][key].cpu(), resumed['model_state_dict'][key].cpu()):
        model_diffs.append(key)
print('model tensor diffs:', len(model_diffs))
print('first model diffs:', model_diffs[:5])
PY
```

Expected interpretation:

```text
epoch equal: True
train_step equal: True
val_step equal: True
best_valid_loss equal: True
param_scheduler_steps equal: True
model tensor diffs: 0
```

This is the human-inspectable core of the proof.

## Part 4 - Decide

Use this decision table.

### Accept

Accept the local exact-continuity proof if all are true:

- You read the code paths above and can explain what is saved/restored.
- Your manual run prints `[resume] Loaded training state from ... epoch=1 ...` before `Epoch: 02`.
- You inspected the checkpoint keys and saw `rng_state`.
- `direct` and `resumed` both have `epoch=2`.
- Direct and resumed counters/schedulers/best loss match.
- `model tensor diffs: 0`.
- You understand the caveat that this is local deterministic CPU proof, not yet cluster operational proof.

If accepted, record:

```text
Exact local resume-continuity tutorial decision: accept
Date:
Your notes:
```

### Rerun

Choose rerun if:

- output is confusing;
- paths were lost;
- you want a larger sample limit;
- you want to run twice to see repeatability.

Possible rerun with slightly larger local sample limits:

```bash
python vae-textures-dev/base_model/verify_exact_resume_continuity.py --limit-train-samples 8 --limit-val-samples 4 --batch-size 2
```

### Block

Choose block if:

- resume does not print `epoch=1` before `Epoch: 02`;
- required keys are missing;
- direct/resumed counters differ;
- `model tensor diffs` is not zero;
- you cannot explain the result well enough to defend it.

If blocked, write the exact command and first surprising output line. That becomes the next debugging task.

## Part 5 - What this does not prove

This does not prove yet that the cluster job setup is correct. The cluster probe must still show:

- `sbatch` job id;
- stdout/stderr logs;
- `sacct` completed with exit code `0:0`;
- W&B run/config/metrics/artifacts if W&B is enabled;
- checkpoint paths on the cluster filesystem;
- human decision: `accept`, `rerun`, or `block`.

The correct sequence is:

1. Understand and accept/reject this local exact-continuity proof.
2. Only then continue to the Phase 7 cluster `sbatch` checkpoint/resume probe.
