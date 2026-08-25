#!/usr/bin/env python3
"""Shared helpers for the human checkpoint/resume continuity test.

The scripts in this directory are intentionally split by experiment step. This
file keeps the repeated mechanics in one place while the numbered scripts stay
short enough to read before running.
"""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


TEST_DIR = Path(__file__).resolve().parent
BASE_MODEL_DIR = TEST_DIR.parent
OUTPUT_DIR = TEST_DIR / 'outputs'
REPORT_DIR = OUTPUT_DIR / 'reports'
MANIFEST_PATH = OUTPUT_DIR / 'manifest.json'

DEFAULT_SEED = 3345
DEFAULT_LIMIT_TRAIN = 4
DEFAULT_LIMIT_VAL = 2
DEFAULT_BATCH_SIZE = 2
DEFAULT_LIMIT_TRAIN_SHUFFLE = True


def ensure_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def env_int(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


def load_manifest():
    ensure_outputs()
    if MANIFEST_PATH.exists() and not env_flag('RESUME_CONTINUITY_NEW_RUN'):
        manifest = json.loads(MANIFEST_PATH.read_text())
        manifest.setdefault(
            'device_mode',
            'gpu' if env_flag('RESUME_CONTINUITY_USE_GPU') else 'cpu',
        )
        return manifest
    run_id = f'{int(time.time())}-{os.getpid()}'
    manifest = {
        'run_id': run_id,
        'seed': env_int('RESUME_CONTINUITY_SEED', DEFAULT_SEED),
        'limit_train_samples': env_int(
            'RESUME_CONTINUITY_LIMIT_TRAIN_SAMPLES', DEFAULT_LIMIT_TRAIN
        ),
        'limit_val_samples': env_int(
            'RESUME_CONTINUITY_LIMIT_VAL_SAMPLES', DEFAULT_LIMIT_VAL
        ),
        'batch_size': env_int('RESUME_CONTINUITY_BATCH_SIZE', DEFAULT_BATCH_SIZE),
        'limit_train_shuffle': env_flag(
            'RESUME_CONTINUITY_LIMIT_TRAIN_SHUFFLE',
            DEFAULT_LIMIT_TRAIN_SHUFFLE,
        ),
        'device_mode': 'gpu' if env_flag('RESUME_CONTINUITY_USE_GPU') else 'cpu',
        'wandb_enabled': env_flag('RESUME_CONTINUITY_WANDB'),
        'wandb_group': os.getenv(
            'WANDB_GROUP', f'resume-continuity-{run_id}'
        ),
        'paths': {},
    }
    save_manifest(manifest)
    return manifest


def save_manifest(manifest):
    ensure_outputs()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


def run_name(manifest, leg):
    return f'resume-test-{leg}-{manifest["run_id"]}'


def run_train(manifest, leg, n_epoch, run_epochs_this_job=0, resume_from=None):
    """Run the canonical train.py entrypoint for one visible test leg."""
    env = os.environ.copy()
    env.update({
        'PYTHONHASHSEED': str(manifest['seed']),
        # W&B stays off for the default CPU exact proof. The cluster validation
        # can enable it with RESUME_CONTINUITY_WANDB=1 so each leg appears as a
        # separate run under one shared W&B group.
        'WANDB_ENABLED': '1' if manifest.get('wandb_enabled') else '0',
        'WANDB_PROJECT': os.getenv('WANDB_PROJECT', 'pop909-reproduction'),
        'WANDB_GROUP': manifest.get('wandb_group', f'resume-continuity-{manifest["run_id"]}'),
        'WANDB_TAGS': os.getenv(
            'WANDB_TAGS', 'pop909,resume-continuity,phase7'
        ),
        'VAE_RUN_NAME': run_name(manifest, leg),
        'VAE_SEED': str(manifest['seed']),
        'VAE_N_EPOCH': str(n_epoch),
        'VAE_BATCH_SIZE': str(manifest['batch_size']),
        'VAE_LIMIT_TRAIN_SAMPLES': str(manifest['limit_train_samples']),
        'VAE_LIMIT_VAL_SAMPLES': str(manifest['limit_val_samples']),
        'VAE_LIMIT_TRAIN_SHUFFLE': (
            '1' if manifest.get('limit_train_shuffle', True) else '0'
        ),
        'VAE_FULL_CHECKPOINT_POLICY': 'epoch-state,last-state,final-state',
    })
    if manifest.get('device_mode') != 'gpu':
        # CPU remains the default because this is the strict equality proof.
        # GPU runs answer the operational cluster question and may need
        # deterministic-kernel qualification if bitwise equality differs.
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        env.pop('CUDA_VISIBLE_DEVICES', None)
    if run_epochs_this_job:
        env['VAE_RUN_EPOCHS_THIS_JOB'] = str(run_epochs_this_job)
    else:
        env.pop('VAE_RUN_EPOCHS_THIS_JOB', None)
    if resume_from is not None:
        env['VAE_RESUME_FROM'] = str(resume_from)
    else:
        env.pop('VAE_RESUME_FROM', None)

    print(f'[test] leg={leg}')
    print(f'[test] run_name={run_name(manifest, leg)}')
    print(f'[test] n_epoch={n_epoch}')
    print(f'[test] run_epochs_this_job={run_epochs_this_job or "until target"}')
    print(f'[test] resume_from={resume_from or ""}')
    print(f'[test] device_mode={manifest.get("device_mode", "cpu")}')
    print(f'[test] limit_train_shuffle={manifest.get("limit_train_shuffle", True)}')
    print(f'[test] wandb_enabled={manifest.get("wandb_enabled", False)}')
    print(f'[test] wandb_group={env.get("WANDB_GROUP", "")}')
    subprocess.run([sys.executable, '-u', 'train.py'], cwd=BASE_MODEL_DIR,
                   env=env, check=True)


def latest_checkpoint(manifest, leg, kind):
    pattern = f'result_*/models/{run_name(manifest, leg)}_{kind}_state.pt'
    matches = sorted(BASE_MODEL_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No checkpoint found for leg={leg} kind={kind}')
    if len(matches) > 1:
        rendered = '\n'.join(f'  - {path}' for path in matches)
        raise RuntimeError(
            f'Ambiguous checkpoints for leg={leg} kind={kind}. '
            'Clean resume_continuity_test outputs/result dirs or start a fresh '
            f'manifest instead of relying on filesystem mtime:\n{rendered}'
        )
    return matches[0]


def record_path(manifest, key, path):
    manifest['paths'][key] = str(Path(path).resolve())
    save_manifest(manifest)
    print(f'[test] {key}={manifest["paths"][key]}')


def load_state(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def compare_values(label, left, right, diffs):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            diffs.append(f'{label}: tensor/type mismatch')
            return
        if left.shape != right.shape:
            diffs.append(f'{label}: shape mismatch {tuple(left.shape)} != {tuple(right.shape)}')
            return
        if not torch.equal(left.cpu(), right.cpu()):
            max_abs = torch.max(torch.abs(left.cpu() - right.cpu())).item() if left.numel() else 0
            diffs.append(f'{label}: tensor values differ max_abs={max_abs}')
        return

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            diffs.append(f'{label}: ndarray/type mismatch')
            return
        if left.shape != right.shape or not np.array_equal(left, right):
            diffs.append(f'{label}: ndarray values differ')
        return

    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            diffs.append(f'{label}: dict/type mismatch')
            return
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        if left_keys != right_keys:
            diffs.append(f'{label}: keys differ {sorted(left_keys ^ right_keys)}')
            return
        for key in sorted(left_keys, key=str):
            compare_values(f'{label}.{key}', left[key], right[key], diffs)
        return

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            diffs.append(f'{label}: sequence/type mismatch')
            return
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            compare_values(f'{label}[{idx}]', left_item, right_item, diffs)
        return

    if left != right:
        diffs.append(f'{label}: {left!r} != {right!r}')


CONTINUITY_CATEGORIES = {
    'model': ('model_state_dict',),
    'optimizer': ('optimizer_state_dict',),
    'scheduler': (
        'lr_scheduler_state_dict',
        'optimizer_scheduler_step',
        'param_scheduler_steps',
    ),
    'counters': ('epoch', 'train_step', 'val_step', 'best_valid_loss'),
    'rng': ('rng_state',),
}


def compare_training_states(direct, resumed):
    category_diffs = {category: [] for category in CONTINUITY_CATEGORIES}
    for category, keys in CONTINUITY_CATEGORIES.items():
        for label, state in [('direct', direct), ('resumed', resumed)]:
            missing = set(keys).difference(state.keys())
            if missing:
                category_diffs[category].append(
                    f'{label}: missing keys {sorted(missing)}'
                )
        if category_diffs[category]:
            continue
        for key in keys:
            compare_values(
                key, direct[key], resumed[key], category_diffs[category]
            )
    return category_diffs


def flatten_category_diffs(category_diffs):
    diffs = []
    for category, category_items in category_diffs.items():
        diffs.extend(f'{category}: {diff}' for diff in category_items)
    return diffs


def write_weight_diff_csv(direct, resumed):
    rows = []
    direct_model = direct['model_state_dict']
    resumed_model = resumed['model_state_dict']
    for name in sorted(direct_model):
        left = direct_model[name].detach().cpu()
        right = resumed_model[name].detach().cpu()
        diff = torch.abs(left - right)
        rows.append({
            'tensor': name,
            'shape': 'x'.join(str(dim) for dim in left.shape),
            'numel': int(left.numel()),
            'exact_equal': bool(torch.equal(left, right)),
            'max_abs_diff': float(diff.max().item()) if diff.numel() else 0.0,
            'mean_abs_diff': float(diff.float().mean().item()) if diff.numel() else 0.0,
            'direct_mean': float(left.float().mean().item()) if left.numel() else 0.0,
            'resumed_mean': float(right.float().mean().item()) if right.numel() else 0.0,
        })

    csv_path = REPORT_DIR / 'weight_diff_report.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, rows
