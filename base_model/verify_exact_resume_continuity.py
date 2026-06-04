#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent


def run_train(run_name, n_epoch, seed, limit_train, limit_val, batch_size,
              run_epochs_this_job=0, resume_from=None):
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': '',
        'PYTHONHASHSEED': str(seed),
        'WANDB_ENABLED': '0',
        'VAE_RUN_NAME': run_name,
        'VAE_SEED': str(seed),
        'VAE_N_EPOCH': str(n_epoch),
        'VAE_BATCH_SIZE': str(batch_size),
        'VAE_LIMIT_TRAIN_SAMPLES': str(limit_train),
        'VAE_LIMIT_VAL_SAMPLES': str(limit_val),
        'VAE_FULL_CHECKPOINT_POLICY': 'epoch-state,last-state,final-state',
    })
    if run_epochs_this_job:
        env['VAE_RUN_EPOCHS_THIS_JOB'] = str(run_epochs_this_job)
    else:
        env.pop('VAE_RUN_EPOCHS_THIS_JOB', None)
    if resume_from is not None:
        env['VAE_RESUME_FROM'] = str(resume_from)
    else:
        env.pop('VAE_RESUME_FROM', None)

    print(f'[verify] run={run_name} n_epoch={n_epoch} resume_from={resume_from or ""}', flush=True)
    subprocess.run([sys.executable, '-u', 'train.py'], cwd=BASE_DIR, env=env, check=True)


def latest_checkpoint(run_name, kind):
    pattern = f'result_*/models/{run_name}_{kind}_state.pt'
    matches = list(BASE_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No checkpoint found for {run_name} kind={kind}')
    return max(matches, key=lambda path: path.stat().st_mtime)


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
        for idx, (l_item, r_item) in enumerate(zip(left, right)):
            compare_values(f'{label}[{idx}]', l_item, r_item, diffs)
        return

    if left != right:
        diffs.append(f'{label}: {left!r} != {right!r}')


def compare_training_states(direct, resumed):
    diffs = []
    required = {
        'model_state_dict',
        'optimizer_state_dict',
        'lr_scheduler_state_dict',
        'optimizer_scheduler_step',
        'param_scheduler_steps',
        'epoch',
        'train_step',
        'val_step',
        'best_valid_loss',
        'rng_state',
    }
    for name, state in [('direct', direct), ('resumed', resumed)]:
        missing = required.difference(state.keys())
        if missing:
            diffs.append(f'{name}: missing keys {sorted(missing)}')

    if diffs:
        return diffs

    for key in sorted(required):
        compare_values(key, direct[key], resumed[key], diffs)
    return diffs


def main():
    parser = argparse.ArgumentParser(
        description='Verify exact epoch-boundary checkpoint/resume continuity.'
    )
    parser.add_argument('--seed', type=int, default=3345)
    parser.add_argument('--limit-train-samples', type=int, default=4)
    parser.add_argument('--limit-val-samples', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()

    suffix = f'{int(time.time())}-{os.getpid()}'
    direct_name = f'exact-resume-direct-{suffix}'
    initial_name = f'exact-resume-initial-{suffix}'
    resumed_name = f'exact-resume-resumed-{suffix}'

    run_train(
        direct_name, n_epoch=2, seed=args.seed,
        limit_train=args.limit_train_samples, limit_val=args.limit_val_samples,
        batch_size=args.batch_size,
    )
    direct_final = latest_checkpoint(direct_name, 'final-state')

    run_train(
        initial_name, n_epoch=1, seed=args.seed,
        limit_train=args.limit_train_samples, limit_val=args.limit_val_samples,
        batch_size=args.batch_size, run_epochs_this_job=1,
    )
    initial_last = latest_checkpoint(initial_name, 'last-state')

    run_train(
        resumed_name, n_epoch=2, seed=args.seed,
        limit_train=args.limit_train_samples, limit_val=args.limit_val_samples,
        batch_size=args.batch_size, run_epochs_this_job=1,
        resume_from=initial_last,
    )
    resumed_final = latest_checkpoint(resumed_name, 'final-state')

    direct_state = load_state(direct_final)
    resumed_state = load_state(resumed_final)
    diffs = compare_training_states(direct_state, resumed_state)

    print('[verify] direct_final=', direct_final, sep='')
    print('[verify] initial_last=', initial_last, sep='')
    print('[verify] resumed_final=', resumed_final, sep='')

    if diffs:
        print('[verify] exact resume continuity FAILED')
        for diff in diffs[:50]:
            print(f'  - {diff}')
        if len(diffs) > 50:
            print(f'  ... {len(diffs) - 50} more differences')
        return 1

    print('[verify] exact resume continuity PASSED')
    print('[verify] resumed epoch 2 matches uninterrupted two-epoch training state')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
