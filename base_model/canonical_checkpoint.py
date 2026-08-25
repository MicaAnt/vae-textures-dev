"""Canonical full-state checkpoints for faithful training resume.

This file defines the checkpoint format used when we want to continue
training, not merely reload model weights. For an epoch-2 resume to be real,
the checkpoint must carry every piece of state that can affect the next
optimizer step: model weights, optimizer, schedulers, counters, best loss,
configuration, and random-number generator state.
"""

from pathlib import Path
import os
import random
import tempfile
import numpy as np
import torch

# These keys are the public contract for a resume-capable checkpoint.
# If one is missing, we fail immediately instead of silently doing a fake resume.
REQUIRED_TRAINING_STATE_KEYS = {
    # Learned parameters of DisentangleVAE.
    'model_state_dict',
    # Adam state, including moment estimates accumulated during training.
    'optimizer_state_dict',
    # Learning-rate scheduler state; without this, LR restarts incorrectly.
    'lr_scheduler_state_dict',
    # Extra step counter used by the project optimizer wrapper.
    'optimizer_scheduler_step',
    # Teacher forcing, beta/KL annealing, and constant scheduler counters.
    'param_scheduler_steps',
    # Completed epoch count. If this is 1, the next loop iteration is epoch 2.
    'epoch',
    # TensorBoard/W&B train and validation step counters.
    'train_step',
    'val_step',
    # Best validation loss so far, needed for correct best-checkpoint behavior.
    'best_valid_loss',
    # Human-readable/run config copied into checkpoint evidence.
    'config',
    # Randomness state required for exact deterministic continuation.
    'rng_state',
}

# RNG state has its own contract so the resume proof can inspect it directly.
REQUIRED_RNG_STATE_KEYS = {
    'python_random_state',
    'numpy_random_state',
    'torch_rng_state',
    'torch_cuda_rng_state_all',
}


def _atomic_torch_save(payload, target_path, use_new_zip=True):
    """Write a checkpoint via temp file, then replace the target atomically.

    This avoids leaving a half-written `.pt` file if the process crashes while
    saving. The fallback caller can switch serialization mode if needed.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix='.tmp_training_state_', suffix='.pt', dir=str(target.parent)
    )
    os.close(fd)
    try:
        torch.save(payload, tmp_name,
                   _use_new_zipfile_serialization=use_new_zip)
        os.replace(tmp_name, target_path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def save_training_state(path, payload):
    """Validate and save one full-state training checkpoint.

    This function is intentionally strict. If a future code path forgets to
    include optimizer, scheduler, counters, config, or RNG state, saving fails
    here instead of producing a checkpoint that looks resumable but is not.
    """
    missing = REQUIRED_TRAINING_STATE_KEYS.difference(payload.keys())
    if missing:
        raise ValueError(
            'Training state checkpoint is missing required keys: '
            + ', '.join(sorted(missing))
        )
    try:
        _atomic_torch_save(payload, path, use_new_zip=True)
    except Exception:
        _atomic_torch_save(payload, path, use_new_zip=False)


def load_training_state(path, map_location='cpu'):
    """Load and validate a full-state checkpoint before resume.

    Returning only after validation means the training loop can trust that the
    payload has all state required for real continuation.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f'Training state checkpoint is not a dict: {path}')
    missing = REQUIRED_TRAINING_STATE_KEYS.difference(payload.keys())
    if missing:
        raise ValueError(
            'Training state checkpoint is missing required keys: '
            + ', '.join(sorted(missing))
        )
    return payload


def capture_rng_state():
    """Capture random-number generators that can affect the next epoch.

    Python `random` is used by teacher forcing decisions, NumPy is used by
    dataset preparation/splitting, and PyTorch RNG is used by latent sampling.
    CUDA RNG is captured when available for GPU runs.
    """
    return {
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'torch_cuda_rng_state_all': torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else [],
    }


def restore_rng_state(rng_state):
    """Restore RNG state so the resumed job continues the same sequence.

    Without this, epoch 2 may start with correct weights and optimizer state
    but draw different random samples or teacher-forcing decisions.
    """
    if not isinstance(rng_state, dict):
        raise ValueError('Training state checkpoint rng_state is not a dict')
    missing = REQUIRED_RNG_STATE_KEYS.difference(rng_state.keys())
    if missing:
        raise ValueError(
            'Training state checkpoint rng_state is missing required keys: '
            + ', '.join(sorted(missing))
        )

    random.setstate(rng_state['python_random_state'])
    np.random.set_state(rng_state['numpy_random_state'])
    torch.set_rng_state(rng_state['torch_rng_state'].cpu())
    if torch.cuda.is_available() and rng_state['torch_cuda_rng_state_all']:
        # torch.load(..., map_location='cuda') may place saved CUDA RNG states
        # on the GPU. torch.cuda.set_rng_state_all expects CPU ByteTensors, so
        # normalize them before restoring a checkpoint on the cluster.
        cuda_rng_states = [
            state.detach().cpu() if isinstance(state, torch.Tensor) else state
            for state in rng_state['torch_cuda_rng_state_all']
        ]
        torch.cuda.set_rng_state_all(cuda_rng_states)
