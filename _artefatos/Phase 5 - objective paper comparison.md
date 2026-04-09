# Phase 5 - objective paper comparison

## Scope

This report compares the reproduced project only against the objective parts of the ISMIR 2020 paper by Wang et al. Subjective listening evaluation is explicitly out of scope.

Paper reference:
- `vae-textures-dev/base_model/README.md`
- ISMIR 2020 paper: *Learning Interpretable Representation for Controllable Polyphonic Music Generation*

## Objective targets extracted from the paper

### Training/setup targets from section 5.1
- dataset: POP909
- keep only 2/4 and 4/4 pieces
- cut into 8-beat segments
- about 66K samples
- song-level split: 90% train / 10% test
- transpose training samples to all 12 keys
- latent dimensions: `zchd = 256`, `ztxt = 256`
- batch size: `128`
- VAE convergence within `6 epochs`

### Objective disentanglement targets from section 5.2
The paper defines three augmentation-based objective checks:
- `Fi`: transpose all notes by `i` semitones, `i in [1, 12]`
- `Pi`: randomly transpose all notes in one beat up/down one semitone under probability `i`, `i in [0.1, 1.0]`
- `Ri`: randomly reduce note duration by half

Expected paper behavior:
- under `Fi`, small pitch changes should affect `zchd` more than `ztxt`
- under `Fi`, `zchd` should be highly sensitive near a tritone and least sensitive at an octave
- under `Pi`, pitch perturbation should affect `zchd` more than `ztxt`
- under `Ri`, rhythm perturbation should affect `ztxt` more than `zchd`

## Reproduction assets added in this phase
- `base_model/evaluate_paper_objective_metrics.py`
- `base_model/run_paper_objective_eval.sh`

The new evaluation script computes latent delta summaries for `Fi`, `Pi`, and `Ri` from a trained checkpoint by using the model's existing `inference_encode()` path.

## Comparison status

### Setup comparison
| Paper target | Current reproduction status |
|---|---|
| POP909-based training path | Reproduced |
| 8-beat segments | Reproduced |
| 12-key augmentation | Approximately reproduced via shift range `-6..5` |
| latent dimensions 256/256 | Reproduced |
| batch size 128 | Supported by `train.py` default |
| 6 epochs | Supported by `train.py` default |
| 90/10 split | Approximate current code path uses `portion=8`, i.e. about 8/9 train and 1/9 validation |

### Objective metric comparison (smoke checkpoint proof-of-execution)
Checkpoint evaluated:
- `base_model/result_2026-03-30_171044/models/phase2-local-proof_valid.pt`

Evaluation command used:
```bash
cd /workspace/vae-textures-dev/base_model
python3 evaluate_paper_objective_metrics.py   --checkpoint result_2026-03-30_171044/models/phase2-local-proof_valid.pt   --device cpu   --batch-size 4   --max-batches 2   --output objective_eval_smoke.json
```

Observed qualitative alignment with the paper:
- `Fi`: for small pitch transpositions, `zchd` deltas are much larger than `ztxt` deltas
- `Fi`: `zchd` shows strong sensitivity around `6` semitones and zero change at `12` semitones, while `ztxt` still changes
- `Pi`: `zchd` remains consistently larger than `ztxt` across perturbation probabilities
- `Ri`: `zchd` stays at `0.0` while `ztxt` increases monotonically with perturbation probability

Concrete examples from `objective_eval_smoke.json`:
- `Fi(1)`: mean delta `zchd = 16.81`, `ztxt = 1.73`
- `Fi(6)`: mean delta `zchd = 16.27`, `ztxt = 4.46`
- `Fi(12)`: mean delta `zchd = 0.00`, `ztxt = 6.06`
- `Pi(1.0)`: mean delta `zchd = 15.77`, `ztxt = 1.52`
- `Ri(1.0)`: mean delta `zchd = 0.00`, `ztxt = 13.35`

## Interpretation

This is not yet a full paper-level reproduction claim, because the metric was only run on a short proof checkpoint rather than a more representative training run. However, it is already strong evidence that the restored model path is expressing the same objective disentanglement pattern that the paper describes.

## Verdict

Current paper-driven objective comparison verdict: `partial`

Reason:
- the objective metric protocol from section 5.2 is now implemented and executable
- the observed latent behavior matches the paper qualitatively on a smoke checkpoint
- but a stronger claim still requires a more representative run and then rerunning the same metric suite on that checkpoint
