from pathlib import Path
import os
import tempfile
import torch


def _atomic_torch_save(payload, target_path: str, use_new_zip: bool = True):
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix='.tmp_ckpt_', suffix='.pt', dir=str(target.parent))
    os.close(fd)
    try:
        torch.save(payload, tmp_name, _use_new_zipfile_serialization=use_new_zip)
        os.replace(tmp_name, target_path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def save_checkpoint(
    path: str,
    model,
    optimizer,
    epoch: int,
    step: int,
    best_val_loss=None,
    include_optimizer_state: bool = True,
):
    payload = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'step': step,
        'best_val_loss': best_val_loss,
    }
    if include_optimizer_state and optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()

    # tentativa 1: formato zip padrão (mais recente)
    try:
        _atomic_torch_save(payload, path, use_new_zip=True)
        return
    except Exception as first_exc:
        # tentativa 2: serialização legada (mais tolerante em alguns FS)
        try:
            _atomic_torch_save(payload, path, use_new_zip=False)
            print(f'[checkpoint] fallback legacy save aplicado em: {path}')
            return
        except Exception as second_exc:
            raise RuntimeError(
                f'Falha ao salvar checkpoint em {path}. '
                f'Erro zip={first_exc}; erro legacy={second_exc}'
            )


def checkpoint_paths(output_dir: str, epoch: int, step: int):
    base = Path(output_dir) / 'checkpoints'
    return {
        'last': str(base / 'last.pt'),
        'best': str(base / 'best.pt'),
        'periodic': str(base / f'epoch{epoch:03d}_step{step:06d}.pt'),
    }


def load_checkpoint(path: str, model, optimizer=None, map_location='cpu', strict: bool = True):
    payload = torch.load(path, map_location=map_location, weights_only=False)

    # checkpoint completo do runner
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        model.load_state_dict(payload['model_state_dict'], strict=strict)
        if optimizer is not None and 'optimizer_state_dict' in payload:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        return {
            'epoch': int(payload.get('epoch', 0)),
            'step': int(payload.get('step', 0)),
            'best_val_loss': payload.get('best_val_loss'),
        }

    # state_dict puro
    if isinstance(payload, dict):
        model.load_state_dict(payload, strict=strict)
        return {'epoch': 0, 'step': 0, 'best_val_loss': None}

    raise ValueError(f'Formato de checkpoint não suportado: {path}')
