from pathlib import Path
import torch


def save_checkpoint(path: str, model, optimizer, epoch: int, step: int, best_val_loss=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'best_val_loss': best_val_loss,
    }
    torch.save(payload, path)


def checkpoint_paths(output_dir: str, epoch: int, step: int):
    base = Path(output_dir) / 'checkpoints'
    return {
        'last': str(base / 'last.pt'),
        'best': str(base / 'best.pt'),
        'periodic': str(base / f'epoch{epoch:03d}_step{step:06d}.pt'),
    }

