# Public GitHub repository organization and sync note

Date: 2026-06-03
Repository: `git@github.com:MicaAnt/vae-textures-dev.git`
Branch reviewed/pushed: `experiment-pipeline-and-evaluation`

## Public Repository Shape

The public repository is an academic workspace rather than a minimal package. Its top-level tree currently mixes source code, notebooks, datasets or dataset-derived files, experiment outputs, and documentation artifacts.

Main public areas:

- `base_model/`: POP909/Poly-Dis reproduction path, including canonical training code such as `train.py`, model code, cluster proof helpers, W&B helpers, and evaluation scripts.
- `dl_modules/`: model components used by the VAE stack, including chord, texture, piano-tree, and decoder modules.
- `scripts/`: operational helpers for cluster sync, cluster container entry, COMMU batch workflows, and sbatch timing probes.
- `_artefatos/`: project notes, evidence documents, POP909 loss/W&B guides, paper-vs-code comparison, and operational checklists.
- `NotebooksVAESymTex/`: study notebooks and notebook-side recomputation or visualization workflows.
- `COMMUDataset/`: COMMU metadata and batch files already present in the public repository.
- `result_experiments/`: experiment outputs exist in the repository, but checkpoints and metrics are ignored for future local changes by `.gitignore`.
- `classifier/`, `features/`, `losses/`, `latent_features/`, `model_param/`, and related scripts: analysis/evaluation support code and stored resources from earlier work.

The README currently presents the repository as two connected work lines: POP909 reproduction/validation and COMMU preprocessing/latent analysis/experiment preparation.

## Published Update

On 2026-06-03, the local branch was 3 commits ahead of `origin/experiment-pipeline-and-evaluation`. These commits were checked and pushed:

- `a17b086f7 docs: explain POP909 losses and W&B metrics`
- `b74dd9d8a scripts: add POP909 representative timing probe`
- `fc161a6ec docs: compare POP909 training parameters to paper`

Files published by those commits:

- `_artefatos/POP909 losses and W&B metrics guide.md`
- `_artefatos/POP909 paper-faithful training parameters.md`
- `_artefatos/POP909 representative timing probe checklist.md`
- `base_model/run_pop909_timing_probe.sh`
- `scripts/submit_pop909_timing_probe.sh`

A secret scan over the pushed diff found only the literal variable name `WANDB_API_KEY`, not a key value. `git diff --check` passed before push.

## Deliberately Not Published Yet

The working tree still contains many local modifications and untracked files. They were not pushed because they include caches, generated outputs, notebooks, checkpoint/result files, local notes, and partially triaged operational artifacts.

Examples left local for later review:

- `__pycache__/` and `dl_modules/__pycache__/` changes.
- `result_experiments/.../checkpoints/*.pt` and `metrics.jsonl` changes.
- untracked notebooks/images under `NotebooksVAESymTex/`.
- untracked COMMU enriched loss batches.
- local evidence notes in `_artefatos/` that need a relevance/privacy pass.
- local edits to cluster helper scripts that are not part of the POP909 timing-probe commits.

## Practical Interpretation

The public GitHub repository is now updated with the POP909 loss/W&B guide, the paper-faithful training-parameter comparison, and the representative timing-probe wrapper/checklist. The broader planning state remains local in `.planning/`, while the code/doc artifacts that matter for cluster POP909 reproduction are now available publicly on the active branch.
