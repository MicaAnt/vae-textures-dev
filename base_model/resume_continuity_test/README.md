# Checkpoint/resume continuity test

This directory is the human-facing test for epoch-boundary resume continuity.

The proof compares two paths:

```text
Path A: epoch 1 -> epoch 2 -> final state
Path B: epoch 1 -> checkpoint -> resume -> epoch 2 -> final state
```

If Path A and Path B finish with the same model weights, optimizer state,
scheduler state, counters, best validation loss, and RNG state, then epoch 2 is
a real continuation, not a fake restart with an epoch label. The comparison
report separates these into model, optimizer, scheduler, counters, and RNG
categories so a failure says what kind of continuity broke.

## Where to run from

If your terminal says:

```bash
pwd
```

and the output is:

```text
/workspace/vae-textures-dev
```

run commands like this:

```bash
python base_model/resume_continuity_test/01_train_direct_2_epochs.py
```

If your terminal is instead at:

```text
/workspace
```

prefix the project directory:

```bash
python vae-textures-dev/base_model/resume_continuity_test/01_train_direct_2_epochs.py
```

## Run the test

From `/workspace/vae-textures-dev`:

```bash
python base_model/resume_continuity_test/01_train_direct_2_epochs.py
python base_model/resume_continuity_test/02_train_one_epoch_checkpoint.py
python base_model/resume_continuity_test/03_resume_second_epoch.py
python base_model/resume_continuity_test/04_compare_final_states.py
python base_model/resume_continuity_test/05_inspect_weights.py
```


## GPU/cluster timing validation

CPU is still the default for the strict exact-equality proof. For the final
Phase 7 cluster validation, run the same numbered A/B flow on a GPU by using
the Slurm submitter from the cluster login node after the runtime sync:

```bash
scripts/submit_pop909_resume_probe.sh exact
```

The submitter sets `RESUME_CONTINUITY_USE_GPU=1`,
`RESUME_CONTINUITY_WANDB=1`, and `RESUME_CONTINUITY_LIMIT_TRAIN_SHUFFLE=1`,
then runs the five numbered steps in one `sbatch` job. The first step creates a fresh manifest, and the later steps
reuse it. The three mini training legs appear in W&B as separate runs named
`resume-test-direct-...`, `resume-test-initial-...`, and
`resume-test-resumed-...` under one shared `WANDB_GROUP`.

Collect the W&B group/runs plus these timing lines from the stdout log:

```text
Epoch train/eval seconds: ...
[checkpoint] Saved model weights in ...s: ...
[checkpoint] Saved training state (epoch-state) in ...s: ...
[checkpoint] Saved training state (last-state) in ...s: ...
[checkpoint] Saved training state (final-state) in ...s: ...
```

Decision rule:

```text
checkpoint overhead percent = checkpoint save seconds / epoch train+eval seconds
```

If checkpoint save time is small compared with one epoch, keep checkpointing
every epoch. If it is large, choose checkpoint every X epochs and record the
measured ratio.

GPU exact equality can depend on deterministic CUDA kernel behavior. The
training script sets CuDNN deterministic mode and disables benchmarking, but it
does not globally force `torch.use_deterministic_algorithms(True)` or require a
`CUBLAS_WORKSPACE_CONFIG`. If the GPU comparison differs bit-for-bit while the
resume leg loads the correct state, continues to epoch 2, saves checkpoints,
and the timing/W&B evidence is clear, record that as an operational GPU result
with a determinism caveat rather than pretending it is the same claim as the
CPU proof.

## What each step does

`01_train_direct_2_epochs.py`

Runs normal uninterrupted training:

```text
epoch 1 -> epoch 2 -> direct_final
```

This is the reference answer.

`02_train_one_epoch_checkpoint.py`

Runs only the first epoch:

```text
epoch 1 -> initial_last
```

`initial_last` is the checkpoint that should contain the full training state.

`03_resume_second_epoch.py`

Loads `initial_last` and runs one more job with total target `n_epoch=2`:

```text
load checkpoint at epoch=1 -> epoch 2 -> resumed_final
```

This is the resumed answer.

`04_compare_final_states.py`

Compares:

```text
direct_final vs resumed_final
```

It checks model weights, optimizer, LR scheduler, parameter scheduler steps,
epoch counter, train/validation step counters, best validation loss, and RNG.

It writes:

```text
base_model/resume_continuity_test/outputs/reports/state_comparison.txt
base_model/resume_continuity_test/outputs/reports/weight_diff_report.csv
```

`state_comparison.txt` includes a category summary for model, optimizer,
scheduler, counters, and RNG state. Overall `RESULT=PASSED` remains strict: all
categories must pass.

`05_inspect_weights.py`

Prints and saves a tensor-by-tensor weight inspection table:

```text
tensor, shape, exact_equal, direct_mean, resumed_mean, max_abs_diff
```

It writes:

```text
base_model/resume_continuity_test/outputs/reports/weight_inspection.csv
```

## Where paths are stored

Each step records the actual checkpoint paths in:

```text
base_model/resume_continuity_test/outputs/manifest.json
```

Open this file after running the first three steps. It should contain:

```text
direct_final
initial_last
resumed_final
```

These are real file paths, not placeholders. The helper refuses ambiguous
checkpoint matches instead of choosing by filesystem modification time, so stale
outputs should be cleaned or a fresh manifest should be started before reruns.

## What to show a supervisor

The short version:

```text
We compared uninterrupted two-epoch training against one epoch plus full-state
checkpoint resume plus the second epoch. The final model weights and all
training-continuity state match exactly under deterministic CPU conditions.
```

The concrete artifacts:

```text
outputs/manifest.json
outputs/reports/state_comparison.txt
outputs/reports/weight_diff_report.csv
outputs/reports/weight_inspection.csv
```

Do not accept the proof from the console message alone. Inspect the report files.
