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
