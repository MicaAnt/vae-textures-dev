#!/usr/bin/env bash
set -euo pipefail

# Defaults for the Slurm container shell. These are environment-overridable so
# the same script can be reused for a different image, partition, or GPU count.
IMAGE_PATH="${CLUSTER_IMAGE:-/home/${USER}/devcontainer_images/dcli_fidle_tuto.squashfs}"
WORKSPACE_HOST_PATH="${WORKSPACE_HOST_PATH:-/home/${USER}/vae-textures-dev}"
PARTITION="${SLURM_PARTITION:-gpu}"
GPUS="${SLURM_GPUS:-1}"
CPUS="${SLURM_CPUS_PER_TASK:-1}"
NTASKS="${SLURM_NTASKS:-1}"

cat <<MSG
[cluster] Launching GPU container shell
[cluster] image: ${IMAGE_PATH}
[cluster] workspace mount: ${WORKSPACE_HOST_PATH} -> /workspace
[cluster] partition: ${PARTITION}
[cluster] gpus: ${GPUS}
[cluster] cpus-per-task: ${CPUS}
MSG

# Request an interactive Slurm allocation and start a shell inside the image.
# The host repository is mounted at /workspace, which is why later commands use
# /workspace/base_model. The NVIDIA driver libraries come from the cluster host;
# mounting them into /nvidia and updating LD_LIBRARY_PATH is the bridge that lets
# CUDA-enabled PyTorch talk to the real GPU driver.
srun \
  --partition="${PARTITION}" \
  --container-image="${IMAGE_PATH}" \
  --container-mounts="${WORKSPACE_HOST_PATH}:/workspace,/lib64/libcuda.so.1:/nvidia/libcuda.so.1,/lib64/libnvidia-ml.so.1:/nvidia/libnvidia-ml.so.1,/lib64/libnvidia-ptxjitcompiler.so.1:/nvidia/libnvidia-ptxjitcompiler.so.1" \
  --container-workdir=/workspace \
  --gres="gpu:${GPUS}" \
  --ntasks="${NTASKS}" \
  --cpus-per-task="${CPUS}" \
  --pty bash -lc 'export LD_LIBRARY_PATH=/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; export WANDB_CACHE_DIR=/tmp/wandb-cache; export WANDB_CONFIG_DIR=/tmp/wandb-config; export WANDB_DIR=/workspace/base_model; mkdir -p /tmp/wandb-cache /tmp/wandb-config; echo "[cluster] NVIDIA driver libs mounted; LD_LIBRARY_PATH updated."; echo "[cluster] W&B cache/config redirected to /tmp and WANDB_DIR set to /workspace/base_model."; exec bash'
