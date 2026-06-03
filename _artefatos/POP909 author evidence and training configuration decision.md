# POP909 author evidence and training configuration decision

Date: 2026-06-03
Phase: 7 - POP909 Pre-launch Validation and Checkpoint Resume

Purpose: decide, before the representative POP909 training run, whether the run should be described and configured as `released-code-faithful`, `paper-literal`, `run-both`, or `blocked`.

## Official sources inspected

Primary paper/source references:

- Paper: Ziyu Wang, Dingsu Wang, Yixiao Zhang, Gus Xia, "Learning Interpretable Representation for Controllable Polyphonic Music Generation", ISMIR 2020.
- arXiv: https://arxiv.org/abs/2008.07122
- ISMIR PDF: https://archives.ismir.net/ismir2020/paper/000094.pdf
- Official paper repository: https://github.com/ZZWaang/polyphonic-chord-texture-disentanglement
- Official tutorial/weights repository: https://github.com/ZZWaang/icm-deep-music-generation

Local inspection snapshots used for this decision:

- `ZZWaang/polyphonic-chord-texture-disentanglement` clone commit: `224b81e437dc0ef413c8920760672c28976bfb7e`
- `ZZWaang/icm-deep-music-generation` clone commit: `bcb9de482584d6a0c834316f8d404a8b5b9522e2`

The official paper repository README describes it as the repository of the ISMIR 2020 paper, but also warns that it contains selected files and may be incomplete/messy. The ICM tutorial repository says it provides architecture code and one version of pretrained model parameters, while all training code is removed.

## Paper-literal parameters

From the paper comparison already recorded in `_artefatos/POP909 paper-faithful training parameters.md`, the strict paper-literal interpretation includes:

| Parameter | Paper-literal value |
|---|---|
| Dataset | POP909 |
| Segment length | 8 beats / 2 bars |
| Time resolution | 32 time steps |
| Base samples | about 66K |
| Split | 90/10 random song-level split |
| Training augmentation | transpose all training samples to all 12 keys |
| Chord encoder hidden dimension | 256 | ok
| Texture encoder hidden dimension | 512 | ok
| Chord decoder hidden dimension | 512 | ok
| Chord latent dimension | 256 | ok
| Texture latent dimension | 256 | ok
| Optimizer | Adam | ok
| Learning rate | `1e-3` scheduled to `1e-5` | ok
| KL beta | anneal from 0 to 0.1 | ok
| Batch size | 128 | ok
| Epochs | 6 | ok

The strict paper-literal path would require changing or parameterizing the current local `train.py` hidden sizes and possibly the split logic.

## Released-code evidence

The official `ZZWaang/polyphonic-chord-texture-disentanglement/train.py` contains the following released training values:

| Parameter | Released-code evidence |
|---|---|
| Batch size | `batch_size = 128` | ok
| Epochs | `n_epoch = 6` | ok
| Learning rate | `lr = 1e-3` | ok
| Optimizer | `optim.Adam(model.parameters(), lr=lr)` | ok
| Chord encoder | `RnnEncoder(36, 1024, 256)` | ok
| Texture/Piano encoder in train.py | `PtvaeEncoder(device=device, z_size=256, max_pitch=39 - 8, min_pitch=0)` | active in official `train.py`, but incompatible with the released `polydis-v1.pt` key shapes inspected locally |
| Texture encoder line in train.py | `TextureEncoder(256, 1024, 256)` exists but is commented | ok
| Dataset split | `portion=8` | ok
| Transposition | `shift_low=-6`, `shift_high=5` | ok
| Segment bars | `num_bar=2` | ok
| Chord flag | `contain_chord=True` | ok
| Writer names | `loss`, `recon_loss`, `pl`, `dl`, `kl_loss`, `kl_chd`, `kl_rhy`, `chord_loss`, `root_loss`, `chroma_loss`, `bass_loss` |

The official repository `model.py` initialization path contains:

- `chd_encoder = RnnEncoder(36, 1024, chd_size)`
- `rhy_encoder = TextureEncoder(256, 1024, txt_size, num_channel)`

The official ICM tutorial `poly_dis/model.py` initialization path contains:

- `chd_encoder = ChordEncoder(36, 1024, chd_size)`
- `txt_encoder = TextureEncoder(256, 1024, txt_size, num_channel)`

Interpretation: official released code strongly supports hidden dimension `1024` for the chord/text encoder internals, despite the smaller hidden dimensions described in the paper text. Official released code also supports `portion=8` rather than an exact 90/10 split implementation.

One caveat: the official paper repository has two architecture paths. Its `train.py` actively uses `PtvaeEncoder(...)` for `rhy_encoder` and leaves `TextureEncoder(256, 1024, 256)` commented, so that script can run because `PtvaeEncoder` is a valid encoder that returns a texture/rhythm latent distribution. However, the official repository `model.py`, the ICM tutorial architecture, and the local released checkpoint `model_param/polydis-v1.pt` all point to the `TextureEncoder` architecture. Therefore the best interpretation is not that `TextureEncoder` being commented makes the official repo broken; rather, the released materials expose multiple/partially inconsistent code paths. For comparison against the released authors' weights, `TextureEncoder` is the relevant path.

## Released weights evidence

The official ICM tutorial repository states that all training code is removed and that one version of pretrained model parameters is provided through Google Drive links inside `model_param` folders.

The inspected file path is:

- `poly_dis/model_param/gdrive_link.txt`

The inspected link is:

- `https://drive.google.com/file/d/1cwycGbbivs4EOfqG2euLs56o1g8LncGE/view?usp=sharing`

A local copy of the released weights is already present at:

- `model_param/polydis-v1.pt`

### Checkpoint shape inspection

Checkpoint shape inspection: conclusive for the local released weights copy `model_param/polydis-v1.pt`.

The checkpoint is an `OrderedDict` with keys matching the `TextureEncoder` path, including:

| Tensor | Shape | Interpretation |
|---|---:|---|
| `chd_encoder.gru.weight_ih_l0` | `(3072, 36)` | GRU gate matrix = `3 * 1024`, so chord encoder hidden size is 1024. |
| `chd_encoder.gru.weight_hh_l0` | `(3072, 1024)` | recurrent chord encoder hidden size 1024. |
| `chd_encoder.linear_mu.weight` | `(256, 2048)` | bidirectional hidden output `2 * 1024`, latent dimension 256. |
| `rhy_encoder.cnn.0.weight` | `(10, 1, 4, 12)` | `TextureEncoder` CNN front-end is present. |
| `rhy_encoder.fc1.weight` | `(1000, 290)` | `TextureEncoder` dense layer is present. |
| `rhy_encoder.fc2.weight` | `(256, 1000)` | `TextureEncoder` embedding size 256. |
| `rhy_encoder.gru.weight_ih_l0` | `(3072, 256)` | GRU gate matrix = `3 * 1024`, so texture encoder hidden size is 1024. |
| `rhy_encoder.gru.weight_hh_l0` | `(3072, 1024)` | recurrent texture encoder hidden size 1024. |
| `rhy_encoder.linear_mu.weight` | `(256, 2048)` | bidirectional hidden output `2 * 1024`, latent dimension 256. |

This makes the authors' released weights incompatible with a paper-literal `TextureEncoder` hidden size 512 and incompatible with the active `PtvaeEncoder(...)` line from the official paper repository `train.py`. It strongly supports using `TextureEncoder(256, 1024, 256)` for any run we want to compare directly against `polydis-v1.pt`.

## Decision options

### Option 1: released-code-faithful

Use the current released-code-supported setup as the Phase 8 default:

- batch size 128;
- 6 epochs;
- LR `1e-3` with scheduler floor `1e-5`;
- KL beta annealing to 0.1;
- 8-beat / 2-bar / 32-step segments;
- train transposition `-6..5`;
- validation untransposed;
- `portion=8` split;
- encoder hidden size evidence from official code: 1024;
- explicitly document that this is not paper-literal for hidden dimensions or exact split wording.

This is the most defensible path if the goal is to reproduce the authors' released implementation behavior and later compare our samples/checkpoint behavior against `model_param/polydis-v1.pt`. If our samples differ substantially from the authors' samples in Phase 9, the documented `PtvaeEncoder` vs `TextureEncoder` inconsistency becomes one explicit hypothesis to test.

### Option 2: paper-literal

Change or parameterize the current code to match the paper text more strictly:

- chord encoder hidden dimension 256;
- texture encoder hidden dimension 512;
- exact or better-documented 90/10 song-level split.

This is closer to the paper prose, but it may diverge from the authors' released implementation and released checkpoint architecture.

### Option 3: run-both

Run a released-code-faithful experiment and a paper-literal experiment. This is strongest for resolving ambiguity but costs substantially more GPU time.

### Option 4: blocked

Block Phase 8 launch if the evidence is still insufficient for the claim we need to make to the supervisor.

## Human decision

Decision: `released-code-faithful`

Validated by the user on 2026-06-03.

Rationale: checkpoint shape inspection of the local released authors' weights `model_param/polydis-v1.pt` is conclusive for the `TextureEncoder` path with 1024 hidden size. Therefore Phase 8 should keep the current canonical code as close as possible to the released authors' weights/code path, while documenting the paper-text mismatch instead of silently calling it paper-literal.

## Phase 8 configuration consequence

Phase 8 should use the current canonical `base_model/train.py` architecture as a released-code-faithful run:

- keep `TextureEncoder(256, 1024, 256)`;
- keep `RnnEncoder(36, 1024, 256)`;
- keep batch size 128;
- keep 6 epochs;
- keep `portion=8`;
- keep train transposition `shift_low=-6`, `shift_high=5`;
- keep `num_bar=2` and `contain_chord=True`;
- do not describe the result as strictly paper-literal without caveats.

The caveat to carry forward: the released weights and official architecture paths support `TextureEncoder` with 1024 hidden size and `portion=8`, while paper prose reports smaller hidden dimensions and a 90/10 split. If Phase 9 sample comparison shows large divergence from the authors' samples, the `PtvaeEncoder` vs `TextureEncoder` inconsistency should be listed as a concrete hypothesis, not ignored.

released-code-faithful -> se na hora da avaliacao comparando os samples dos pesos dos autores e nossos a gente ver que os samples estao muito diferentes, a gente ja coloca isso como hipotese.