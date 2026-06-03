# POP909 paper-faithful training parameters

Purpose: collect the training parameters reported in the paper, compare them rigorously against the current `base_model/train.py` path, and define a paper-faithful strategy for reproducing the published VAE experiment on the cluster.

Paper source: Ziyu Wang, Dingsu Wang, Yixiao Zhang, Gus Xia, **Learning Interpretable Representation for Controllable Polyphonic Music Generation**, ISMIR 2020, arXiv:2008.07122.

Paper access used for this document:

- arXiv abstract page: https://arxiv.org/abs/2008.07122
- ar5iv HTML rendering of the paper: https://ar5iv.labs.arxiv.org/html/2008.07122

I did not find a local PDF copy of `2008.07122` in the repository. I found previous local paper-comparison artifacts, but not the paper PDF itself.

## Executive answer

The current restored `base_model/train.py` is mostly aligned with the paper's VAE training recipe:

- POP909 path: yes, current code uses POP909 piano-roll data.
- 8-beat / 2-bar samples: yes, current code uses `num_bar=2` and produces 32 time steps.
- Training augmentation to 12 transpositions: yes for the training set, through `shift_low=-6`, `shift_high=5`.
- Validation/test no transposition augmentation: yes, current validation uses `shift_low=0`, `shift_high=0`.
- Batch size 128: yes.
- 6 epochs: yes.
- Adam optimizer: yes.
- Learning rate from `1e-3` down to minimum `1e-5`: yes, via exponential scheduler.
- KL annealing from 0 to 0.1: yes.
- Chord encoder hidden dimension 1024 in code, not the 256 stated in the paper. This is the biggest architecture mismatch to investigate.
- Chord decoder hidden dimension 512: yes.
- Texture encoder GRU hidden dimension 1024 in code, not the 512 stated in the paper. This is another architecture mismatch to investigate.
- Latent dimensions `z_chd=256` and `z_rhy=256`: yes.
- Dataset split: close but not exact. Paper says 90/10 song-level split; current code uses `portion=8`, i.e. approximately 8/9 train and 1/9 validation at song level.

Bottom line: the current path is a strong author-code-faithful reproduction path, but a strictly paper-faithful reproduction should explicitly address the hidden-dimension and split mismatches before making a scientific claim.

## Paper parameters

The paper states the following training/data setup in Section 5.1, Dataset and Training:

| Category | Paper parameter |
|---|---|
| Dataset | POP909, about 1K MIDI files of pop songs with paired vocal melody and piano accompaniment. |
| Meter filtering | Keep pieces with 2/4 and 4/4 meters. |
| Segment length | 8-beat music segments. |
| Time resolution | 32 time steps under 16th-note resolution. |
| Total base samples | About 66K samples. |
| Split | Random song-level train/test split: 90% train, 10% test. |
| Augmentation | All training samples transposed to all 12 keys. |
| Chord encoder GRU hidden dimension | 256, as reported in the paper. |
| Chord decoder GRU hidden dimension | 512. |
| Texture encoder GRU hidden dimension | 512. |
| Chord latent dimension | 256. |
| Texture latent dimension | 256. |
| PianoTree decoder size | Same as original PianoTree VAE implementation. |
| Optimizer | Adam. |
| Learning rate | Scheduled from `1e-3` to `1e-5`. |
| KL annealing | KL weight starts at 0 and rises to 0.1. |
| Batch size | 128. |
| VAE convergence | Within 6 epochs. |

The paper also describes the model architecture:

- Chord progression representation is a 36 x 8 matrix: root, bass, and chroma for 8 beats.
- Texture input is an image-like 128 x 32 matrix: 128 MIDI pitches by 32 time steps.
- Texture encoder uses convolution with 1 input channel and 10 output channels, followed by ReLU and max-pooling.
- PianoTree decoder reconstructs frame-wise hidden states and note-level pitch/duration outputs.

## Current code parameters

Canonical current path: `vae-textures-dev/base_model/train.py`.

| Category | Current code value | Source |
|---|---|---|
| Device | `cuda` if available else `cpu` | `train.py:30` |
| Batch size | `VAE_BATCH_SIZE` default `128` | `train.py:32` |
| Epochs | `VAE_N_EPOCH` default `6` | `train.py:33` |
| Gradient clip | `1` | `train.py:34` |
| Reconstruction weights | `[1, 0.5]` | `train.py:36` |
| KL high beta | `0.1` | `train.py:37` |
| Teacher forcing rates | `[(0.6, 0), (0.5, 0), (0.5, 0)]` | `train.py:38` |
| Learning rate | `VAE_LR` default `1e-3` | `train.py:39` |
| Run name | `VAE_RUN_NAME`, default `disvae-nozoth` | `train.py:40` |
| Optional train sample limit | `VAE_LIMIT_TRAIN_SAMPLES`, default `0` | `train.py:41` |
| Optional val sample limit | `VAE_LIMIT_VAL_SAMPLES`, default `0` | `train.py:42` |
| Chord encoder | `RnnEncoder(36, 1024, 256)` | `train.py:48` |
| Texture encoder | `TextureEncoder(256, 1024, 256)` | `train.py:49` |
| Chord decoder | `RnnDecoder(z_dim=256)` | `train.py:52`; defaults in `ptvae.py:34-35` |
| PianoTree decoder | `PtvaeDecoder(dec_dur_hid_size=64, z_size=512)` | `train.py:53-54` |
| Dataset split portion | `portion=8` | `train.py:60-61`; `dataset.py:242-246` |
| Transposition range | train `-6..5`, validation `0..0` | `train.py:61`; `dataset.py:274-277` |
| Segment bars | `num_bar=2` | `train.py:62` |
| Chord included | `contain_chord=True` | `train.py:63` |
| LR scheduler | `MinExponentialLR(gamma=0.9999, minimum=1e-5)` | `train.py:88`; `example.py:4-12` |
| KL annealing scheduler | `TeacherForcingScheduler(beta, 0., f=kl_anealing)` | `train.py:100`; `train_utils.py:24-30` |
| Optimizer | Adam | `train.py:87` |

## Paper vs code: rigorous comparison

| Parameter | Paper | Current code | Status | Impact |
|---|---|---|---|---|
| Dataset | POP909 | POP909 piano-roll quantized data path | Match | Good. |
| Base sample count | About 66K | Observed base count from recent run: `58563 + 7718 = 66281` before train augmentation | Match | Strong evidence dataset scale is paper-consistent. |
| Train/test split | 90/10 at song level | `portion=8`, roughly 8/9 train and 1/9 validation at song level | Approximate | This is close but not exact. A paper-faithful claim should call it an approximation unless changed to exact 90/10. |
| Training augmentation | Train samples transposed to all 12 keys | Train dataset length multiplied by 12 via shifts `-6..5` | Match | Strong. |
| Validation/test augmentation | Not explicitly augmented in paper | Validation uses shift `0..0`, no augmentation | Likely match | Reasonable. |
| Segment length | 8 beats | `num_bar=2`, 4 time steps per beat -> 8 beats / 32 time steps | Match | Strong. |
| Time steps | 32 under 16th-note resolution | PianoTree tensors use 32 steps | Match | Strong. |
| Batch size | 128 | default 128 | Match | Strong. |
| Epochs | Converges within 6 | default 6 | Match | Strong. |
| Optimizer | Adam | Adam | Match | Strong. |
| LR schedule | 1e-3 to 1e-5 | `1e-3`, exponential decay floor `1e-5` | Match in endpoints | Schedule shape not fully specified in paper; endpoint matches. |
| KL annealing | 0 to 0.1 | `kl_anealing(... high=0.1, low=0)` | Match | Strong. |
| Chord latent dimension | 256 | 256 | Match | Strong. |
| Texture latent dimension | 256 | 256 | Match | Strong. |
| Chord encoder hidden dimension | 256 | code uses 1024 | Mismatch | Important. Could be paper shorthand, implementation discrepancy, or restored-code mismatch. Needs decision before strict paper-faithful launch. |
| Chord decoder hidden dimension | 512 | default 512 | Match | Strong. |
| Texture encoder hidden dimension | 512 | code uses 1024 | Mismatch | Important. Same concern as chord encoder. |
| Texture CNN channels | 10 output channels | default `num_channel=10` | Match | Strong. |
| Texture CNN kernel/stride/pool | Paper describes image-like CNN with 10 channels; exact rendered formulas are unclear in ar5iv text | code uses Conv2d kernel `(4,12)`, stride `(4,1)`, max pool `(1,4)` | Likely match, but verify PDF/formula if needed | The code likely reflects the original implementation. |
| Loss objective | VAE reconstruction + KL + chord reconstruction | `loss = recon_loss + beta * kl_loss + chord_loss` | Match | Strong. |
| Chord loss | Paper describes root, bass, chroma losses | code has root/chroma/bass cross entropy | Mostly match | Paper wording says chroma as independent Bernoulli; current code uses cross entropy over binary chroma entries. Need qualify. |
| Checkpoint policy | Not specified | saves epoch, best valid, final | Extra implementation detail | Useful, not a paper mismatch. |
| W&B logging | Not in paper | added observability | Extra, non-scientific | Should not affect model if logging-only. |

## Dataset and batch math

Recent timing probe output showed:

```text
The folder contains 886 .npz files.
Selected 858 files, all are in duple meter.
702756 7718
```

Interpretation:

- `702756` is the augmented training dataset length.
- Training uses 12 transpositions, so base train segments are `702756 / 12 = 58563`.
- Validation has no transposition augmentation and has `7718` samples.
- Base segments in this split: `58563 + 7718 = 66281`, which matches the paper's “about 66K samples”.

Full training batches with batch size 128:

| Split | Samples | Batch size | Batches per epoch |
|---|---:|---:|---:|
| Train augmented | 702756 | 128 | ceil(702756 / 128) = 5491 |
| Validation | 7718 | 128 | ceil(7718 / 128) = 61 |
| Combined per epoch | 710474 | 128 | about 5552 batch iterations |

For 6 epochs:

- Train steps: `5491 * 6 = 32946`.
- Validation steps: `61 * 6 = 366`.
- Total train+validation batch iterations: about `33312`.

## Timing estimate from cluster probe

Timing probe run:

- Job id: `330497`.
- Node/GPU from `sacct`: `gres/gpu:a40-48=1`.
- State: `COMPLETED`.
- Slurm elapsed: `00:03:46`.
- W&B run: https://wandb.ai/micael-antunes-lis-cnrs/pop909-reproduction/runs/dpkn36fh
- Probe train sample limit: `4096`.
- Probe validation sample limit: `512`.
- Probe batch size: `128`.
- Probe train batches: `4096 / 128 = 32`.
- Probe validation batches: `512 / 128 = 4`.
- W&B epoch duration: `145` seconds.

Two useful estimates:

### Estimate A: train-dominant conservative estimate

Use `145s / 32 train batches = 4.53s/train batch`.

Full train epoch:

```text
5491 train batches * 4.53s = 24878s = 6.9h
```

Add validation, W&B, checkpoint, and cluster variance. Practical estimate: **8-10 hours per full epoch**.

For 6 epochs: **48-60 hours**.

### Estimate B: combined batch estimate

Use `145s / 36 combined train+val batches = 4.03s/batch`.

Full combined epoch:

```text
5552 combined batches * 4.03s = 22375s = 6.2h
```

With overhead and safety margin: **7-9 hours per full epoch**.

For 6 epochs: **42-54 hours**.

### Operational recommendation

For a paper-faithful 6-epoch run on the same class of GPU, request **at least 60 hours** if cluster policy permits. If the cluster allows 72 hours, use **72 hours** for safety.

If the cluster time limit is lower than this, do not launch the full 6-epoch run until we either:

1. implement safe resume-from-checkpoint for `train.py`, or
2. split the experiment into paper-equivalent chunks while preserving optimizer/scheduler state.

Without preserving optimizer, scheduler, epoch, and step state, splitting into separate jobs would no longer be strictly faithful to the original continuous 6-epoch training run.

## Paper-faithful strategy

Your requested strategy is **paper-faithful**, not merely current-code-faithful. Therefore:

### Gate 1: Decide hidden-dimension mismatch

The current `train.py` uses:

```python
chd_encoder = RnnEncoder(36, 1024, 256)
rhy_encoder = TextureEncoder(256, 1024, 256)
```

The paper reports:

```text
chord encoder GRU hidden dimension = 256
texture encoder GRU hidden dimension = 512
chord decoder GRU hidden dimension = 512
latent dimensions = 256 and 256
```

For a strict paper-faithful run, we should not silently ignore this mismatch.

Options:

1. **Paper-literal architecture**: modify or parameterize `train.py` to use chord encoder hidden `256` and texture encoder hidden `512`. This is closer to the paper text but may diverge from the released authors' implementation if the code intentionally used 1024.
2. **Released-code architecture with paper caveat**: keep current code because it likely reflects the authors' public implementation, but document that hidden dimensions differ from the paper text. This is code-faithful, not strictly paper-literal.
3. **Run both**: paper-literal run for claims against the paper, current-code run for claims against released implementation/checkpoint. This is strongest scientifically but costs more cluster time.

Because you asked for **Plano fiel ao Paper**, my recommendation is: create a parameterized training wrapper or minimal code path that can run the paper-literal hidden sizes, while preserving all other `train.py` behavior. Before doing that, we should verify whether the original GitHub code associated with the paper also used 1024. If it did, then the mismatch is likely paper-text imprecision, not local drift.

### Gate 2: Decide exact split

The paper says 90/10 song-level split. Current code uses `portion=8`, approximately 8/9.

For paper-faithful reproduction:

- either change the loader to use an exact 90/10 song-level split;
- or document current split as an approximation and avoid claiming exact paper split.

Because the current split already yields about 66K base samples and is close to the paper, this is less severe than hidden dimensions, but it still matters for rigorous reproduction.

### Gate 3: Launch config for a paper-faithful full run

Assuming we accept the current code as the effective author implementation except for no sample limits, the full run should use:

```bash
VAE_BATCH_SIZE=128 \
VAE_N_EPOCH=6 \
VAE_LIMIT_TRAIN_SAMPLES=0 \
VAE_LIMIT_VAL_SAMPLES=0 \
VAE_LR=1e-3 \
VAE_RUN_NAME=pop909-paper-faithful-bs128-6epoch \
WANDB_ENABLED=1 \
WANDB_PROJECT=pop909-reproduction \
WANDB_TAGS=paper-faithful,pop909,vae,phase7 \
WANDB_NOTES='Paper-faithful POP909 VAE run: batch 128, 6 epochs, full train/val, no sample limits.' \
SLURM_TIME=72:00:00 \
scripts/submit_pop909_representative_train.sh
```

That final submit script does not exist yet as a dedicated real-training launcher. We can adapt the timing-probe submitter, but the real launcher should make the following impossible to miss:

- no train/val sample limits;
- 6 epochs;
- paper/code parameter table printed before submit;
- Slurm time chosen from accepted timing evidence;
- W&B run name and tags explicitly paper-faithful;
- checkpoint artifact policy clear.

## Monitoring plan for the real run

During the full run, monitor:

- `train/loss`, `val/loss`;
- `train/recon_loss`, `val/recon_loss`;
- `train/pl`, `val/pl`;
- `train/dl`, `val/dl`;
- `train/kl_chd`, `val/kl_chd`;
- `train/kl_rhy`, `val/kl_rhy`;
- `train/chord_loss`, `val/chord_loss`;
- `epoch/duration_seconds`;
- `epoch/train_loss`, `epoch/valid_loss` with the caveat that they are sums, not averages.

Expected rough time:

- one full epoch: about 7-10 hours on the probed A40-style GPU;
- six epochs: about 42-60 hours;
- recommended Slurm request: 60-72 hours, depending on cluster limits.

Acceptance evidence for a completed paper-faithful run:

- W&B config shows batch size `128`, epochs `6`, sample limits `0`, LR `1e-3`, full writer names.
- Logs show six epoch reports.
- `sacct` shows `COMPLETED` and exit code `0:0`.
- W&B artifacts include `valid`/`best` and `final` checkpoints.
- Result directory contains epoch, valid, and final model files.
- Documentation explicitly records whether architecture was paper-literal or released-code-faithful.

## Critical unresolved questions

1. **Did the original public GitHub implementation use hidden dimensions 1024/1024 despite the paper saying 256/512?**

   This decides whether current `train.py` is an implementation-faithful path or a local mismatch.

2. **Do we need exact 90/10 split for the supervisor's claim?**

   If yes, implement exact song-level split. If no, call it an approximation: current code uses an 8/9 split and reproduces the same order of dataset size.

3. **Can the cluster grant a 60-72 hour GPU job?**

   If no, we need checkpoint resume before launching the paper-faithful 6-epoch run.

## My recommended next step

Before launching Phase 7, do one short research/verification task:

```text
Verify original authors' GitHub training hyperparameters for arXiv:2008.07122, especially chord encoder hidden dim, texture encoder hidden dim, exact split logic, and any training command defaults.
```

Then choose one of:

- **Paper-literal run**: hidden dims exactly as paper text, exact 90/10 split if needed.
- **Released-code-faithful run**: current `train.py`, full data, 6 epochs, batch 128, with documented paper-text divergences.

Given your stated priority, I would only call the next experiment “paper-faithful” after resolving the hidden-dimension mismatch.
