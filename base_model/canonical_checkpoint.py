from pathlib import Path
import os
import tempfile
import torch

REQUIRED_TRAINING_STATE_KEYS = {
    'model_state_dict',
    'optimizer_state_dict',
    'lr_scheduler_state_dict',
    'optimizer_scheduler_step',
    'param_scheduler_steps',
    'epoch',
    'train_step',
    'val_step',
    'best_valid_loss',
    'config',
}


def _atomic_torch_save(payload, target_path, use_new_zip=True):
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
