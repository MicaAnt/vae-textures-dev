import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.checkpoint import checkpoint_paths, save_checkpoint
from experiments.config import build_arg_parser, load_config, save_config_snapshot
from experiments.data_factory import batch_to_inputs, make_commu_loaders
from experiments.freeze import apply_freeze_policy, build_optimizer_for_trainable
from experiments.logging_utils import JsonlLogger, print_train_log
from experiments.model_factory import build_model
from experiments.schedulers import resolve_train_params


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, val_loader, device, train_cfg, epoch, step):
    model.eval()
    losses = []
    max_batches = train_cfg.get('val_max_batches')
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            x, c, pr_mat = batch_to_inputs(batch, device)
            params = resolve_train_params(train_cfg, global_step=step, epoch=epoch)
            params['tfr1'] = 0.0
            params['tfr2'] = 0.0
            params['tfr3'] = 0.0
            out = model('train', x, c, pr_mat, **params)
            losses.append(float(out[0].item()))
            if max_batches is not None and (idx + 1) >= max_batches:
                break
    if not losses:
        return float('inf')
    return sum(losses) / len(losses)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    exp_cfg = cfg.get('experiment', {})
    train_cfg = cfg.get('train', {})
    freeze_cfg = cfg.get('freeze', {})

    set_seed(exp_cfg.get('seed', 3345))
    device = torch.device(exp_cfg.get('device', 'cpu'))
    output_dir = exp_cfg.get('output_dir', 'result_configurable_runner')
    os.makedirs(output_dir, exist_ok=True)
    save_config_snapshot(cfg, output_dir)

    print(f'Iniciando experimento: {exp_cfg.get("name", "unnamed")} | device={device}')
    model = build_model(device, cfg.get('model', {}), exp_cfg.get('name', 'configurable-runner')).to(device)

    train_loader, val_loader = make_commu_loaders(
        cfg.get('data', {}),
        train_batch_size=train_cfg.get('batch_size', 2),
        val_batch_size=train_cfg.get('val_batch_size', train_cfg.get('batch_size', 2)),
    )

    _, trainable = apply_freeze_policy(model, freeze_cfg, epoch=0, step=0)
    optimizer = build_optimizer_for_trainable(model, train_cfg.get('optimizer', {'lr': 1e-3}))

    logger = JsonlLogger(output_dir)
    clip = train_cfg.get('clip_grad_norm', 1.0)
    max_steps = train_cfg.get('max_steps', 100)
    max_epochs = train_cfg.get('epochs', 1)
    log_every = train_cfg.get('log_every', 10)
    ckpt_cfg = cfg.get('checkpoint', {})
    save_every = ckpt_cfg.get('save_every_steps', 50)
    include_optimizer_state = ckpt_cfg.get('include_optimizer_state', True)

    global_step = 0
    best_val = float('inf')
    last_freeze_signature = tuple(trainable)
    start = time.time()

    for epoch in range(max_epochs):
        model.train()

        _, trainable = apply_freeze_policy(model, freeze_cfg, epoch=epoch, step=global_step)
        if tuple(trainable) != last_freeze_signature:
            optimizer = build_optimizer_for_trainable(model, train_cfg.get('optimizer', {'lr': 1e-3}))
            last_freeze_signature = tuple(trainable)
            print(f'[freeze] optimizer recriado no epoch={epoch}; parâmetros treináveis={len(trainable)}')

        for batch in train_loader:
            if global_step >= max_steps:
                break
            x, c, pr_mat = batch_to_inputs(batch, device)
            params = resolve_train_params(train_cfg, global_step=global_step, epoch=epoch)

            optimizer.zero_grad()
            out = model('train', x, c, pr_mat, **params)
            loss = out[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                clip,
            )
            optimizer.step()

            row = {
                'epoch': epoch,
                'step': global_step + 1,
                'loss': float(out[0].item()),
                'recon_loss': float(out[1].item()),
                'kl_loss': float(out[4].item()),
                'chord_loss': float(out[7].item()),
                'tfr1': params['tfr1'],
                'tfr2': params['tfr2'],
                'tfr3': params['tfr3'],
                'beta': params['beta'],
                'weights': list(params['weights']),
            }
            logger.log(row)

            if global_step == 0 or (global_step + 1) % log_every == 0:
                print_train_log('train', row)

            if (global_step + 1) % save_every == 0:
                paths = checkpoint_paths(output_dir, epoch, global_step + 1)
                try:
                    save_checkpoint(
                        paths['periodic'],
                        model,
                        optimizer,
                        epoch,
                        global_step + 1,
                        best_val,
                        include_optimizer_state=include_optimizer_state,
                    )
                except Exception as exc:
                    print(f'[checkpoint][warn] falha ao salvar periódico: {exc}')

            global_step += 1

        val_loss = evaluate(model, val_loader, device, train_cfg, epoch, global_step)
        logger.log({'epoch': epoch, 'step': global_step, 'val_loss': val_loss})
        print(f'[val] epoch={epoch} step={global_step} val_loss={val_loss:.4f}')

        paths = checkpoint_paths(output_dir, epoch, global_step)
        try:
            save_checkpoint(
                paths['last'],
                model,
                optimizer,
                epoch,
                global_step,
                best_val,
                include_optimizer_state=include_optimizer_state,
            )
        except Exception as exc:
            print(f'[checkpoint][warn] falha ao salvar last.pt: {exc}')
        if val_loss < best_val:
            best_val = val_loss
            try:
                save_checkpoint(
                    paths['best'],
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    best_val,
                    include_optimizer_state=include_optimizer_state,
                )
            except Exception as exc:
                print(f'[checkpoint][warn] falha ao salvar best.pt: {exc}')

        if global_step >= max_steps:
            break

    elapsed = time.time() - start
    print(f'Treino concluído em {elapsed:.2f}s | best_val={best_val:.4f} | steps={global_step}')


if __name__ == '__main__':
    main()
