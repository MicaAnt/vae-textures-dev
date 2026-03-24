import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.checkpoint import checkpoint_paths, load_checkpoint, save_checkpoint
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


def resolve_device(exp_cfg: dict) -> torch.device:
    requested = exp_cfg.get('device', 'cpu')
    allow_fallback = exp_cfg.get('allow_cpu_fallback', False)
    device = torch.device(requested)

    if device.type != 'cuda':
        return device

    cuda_compiled = torch.version.cuda is not None
    cuda_available = torch.cuda.is_available()
    if cuda_compiled and cuda_available:
        return device

    message = (
        f"Config pediu device={requested}, mas este ambiente não consegue usar CUDA. "
        f"torch.version.cuda={torch.version.cuda!r}, torch.cuda.is_available()={cuda_available}. "
        "Isso normalmente significa que o PyTorch instalado é CPU-only ou que o job não recebeu GPU/CUDA visível."
    )
    if allow_fallback:
        print(f'[device][warn] {message} Fazendo fallback para cpu porque experiment.allow_cpu_fallback=true.')
        return torch.device('cpu')
    raise RuntimeError(
        message + ' Ajuste o ambiente para uma build CUDA do PyTorch ou use experiment.device=cpu.'
    )


def configure_runtime(device: torch.device, exp_cfg: dict):
    if device.type != 'cpu':
        return

    disable_mkldnn = exp_cfg.get('disable_mkldnn_on_cpu', False)
    if disable_mkldnn and hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.enabled:
        torch.backends.mkldnn.enabled = False
        print('[device][cpu] MKLDNN desabilitado por configuração para evitar falhas/illegal instruction em CPU.')

    cpu_threads = exp_cfg.get('cpu_threads')
    if cpu_threads is not None:
        cpu_threads = max(1, int(cpu_threads))
        torch.set_num_threads(cpu_threads)
        if hasattr(torch, 'set_num_interop_threads'):
            try:
                torch.set_num_interop_threads(cpu_threads)
            except RuntimeError:
                pass
        print(f'[device][cpu] torch.set_num_threads({cpu_threads}) aplicado.')


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    exp_cfg = cfg.get('experiment', {})
    train_cfg = cfg.get('train', {})
    freeze_cfg = cfg.get('freeze', {})

    set_seed(exp_cfg.get('seed', 3345))
    device = resolve_device(exp_cfg)
    configure_runtime(device, exp_cfg)
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

    resume_cfg = train_cfg.get('resume', {})
    pretrained_path = train_cfg.get('pretrained_path')
    pretrained_strict = train_cfg.get('pretrained_strict', True)
    if pretrained_path:
        meta = load_checkpoint(
            pretrained_path,
            model,
            optimizer=None,
            map_location=device,
            strict=pretrained_strict,
        )
        print(f'[load] Pesos carregados de pretrained_path={pretrained_path} (meta={meta})')

    start_epoch = 0
    global_step = 0
    best_val = float('inf')
    resume_from = resume_cfg.get('from_checkpoint')
    if resume_from:
        resume_strict = resume_cfg.get('strict', True)
        load_optimizer_state = resume_cfg.get('load_optimizer_state', True)
        meta = load_checkpoint(
            resume_from,
            model,
            optimizer=optimizer if load_optimizer_state else None,
            map_location=device,
            strict=resume_strict,
        )
        start_epoch = int(meta.get('epoch', 0))
        global_step = int(meta.get('step', 0))
        best_val = meta.get('best_val_loss', float('inf'))
        if best_val is None:
            best_val = float('inf')
        print(
            f'[resume] Carregado checkpoint={resume_from} | '
            f'epoch={start_epoch} step={global_step} best_val={best_val}'
        )

    logger = JsonlLogger(output_dir)
    clip = train_cfg.get('clip_grad_norm', 1.0)
    log_every = train_cfg.get('log_every', 10)
    ckpt_cfg = cfg.get('checkpoint', {})
    save_every = ckpt_cfg.get('save_every_steps', 50)
    include_optimizer_state = ckpt_cfg.get('include_optimizer_state', True)

    last_freeze_signature = tuple(trainable)
    start = time.time()
    steps_per_epoch = max(len(train_loader), 1)
    first_epoch_start_batch = resume_cfg.get('start_batch_in_epoch')
    if first_epoch_start_batch is None:
        first_epoch_start_batch = global_step % steps_per_epoch

    if resume_from and resume_cfg.get('advance_epoch_on_boundary', True):
        if first_epoch_start_batch == 0 and global_step > 0:
            start_epoch += 1
            print(f'[resume] avanço automático para epoch={start_epoch} (checkpoint em fronteira de época)')

    run_steps_budget = train_cfg.get('max_steps', 100)
    stop_at_step = global_step + run_steps_budget
    run_epochs_budget = train_cfg.get('epochs', 1)
    stop_at_epoch = start_epoch + run_epochs_budget

    for epoch in range(start_epoch, stop_at_epoch):
        model.train()

        _, trainable = apply_freeze_policy(model, freeze_cfg, epoch=epoch, step=global_step)
        if tuple(trainable) != last_freeze_signature:
            optimizer = build_optimizer_for_trainable(model, train_cfg.get('optimizer', {'lr': 1e-3}))
            last_freeze_signature = tuple(trainable)
            print(f'[freeze] optimizer recriado no epoch={epoch}; parâmetros treináveis={len(trainable)}')

        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < first_epoch_start_batch:
                continue

            if global_step >= stop_at_step:
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

        if global_step >= stop_at_step:
            break

    elapsed = time.time() - start
    print(f'Treino concluído em {elapsed:.2f}s | best_val={best_val:.4f} | steps={global_step}')


if __name__ == '__main__':
    main()
