# Phase 4 - W&B run flow

## Default behavior

The canonical POP909 training path remains `base_model/train.py`.
W&B is opt-in and disabled by default.

## Required environment variables when W&B is enabled

- `WANDB_ENABLED=1`
- `WANDB_PROJECT=<your-project>`
- `WANDB_API_KEY=<your-api-key>`

Optional variables:
- `WANDB_ENTITY=<entity-or-team>`
- `WANDB_MODE=offline` for a local smoke test without network sync
- `WANDB_GROUP=<group-name>`
- `WANDB_NOTES=<notes>`
- `WANDB_TAGS=tag1,tag2`
- `WANDB_CHECKPOINT_POLICY=valid,final` (default)

## What gets logged

- run config from `train.py`
- train and validation metrics at step boundaries
- epoch-level train/validation loss summaries
- selected checkpoints (`valid` and/or `final` depending on policy)

## Local smoke example

```bash
cd /workspace/vae-textures-dev/base_model
WANDB_ENABLED=1 \
WANDB_MODE=offline \
WANDB_PROJECT=pop909-reproduction \
VAE_BATCH_SIZE=2 \
VAE_N_EPOCH=1 \
VAE_LIMIT_TRAIN_SAMPLES=4 \
VAE_LIMIT_VAL_SAMPLES=2 \
VAE_RUN_NAME=wandb-local-smoke \
python -u train.py
```

## Cluster example

1. Launch the GPU container shell:

```bash
cd ~/vae-textures-dev
scripts/cluster_gpu_shell.sh
```

2. Run the short proof with W&B enabled:

```bash
cd /workspace/base_model
WANDB_ENABLED=1 \
WANDB_PROJECT=pop909-reproduction \
WANDB_ENTITY=<optional-entity> \
WANDB_API_KEY=<your-api-key> \
VAE_RUN_NAME=wandb-cluster-smoke \
./run_cluster_proof.sh
```

## Operational todo

After the flow is working end-to-end:
- run one quick W&B validation with the real credentials
- share the resulting W&B project or run visibility with the supervisor
