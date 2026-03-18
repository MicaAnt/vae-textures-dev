from pathlib import Path
import os
import tempfile
import torch


def _atomic_torch_save(payload, target_path: str, use_new_zip: bool = True):
    target = Path(target_path) # Convert the destination path string into a Path object so we can manipulate it more easily.
    target.parent.mkdir(parents=True, exist_ok=True) # Ensure that the parent folder of the target file exists. If it does not exist, create it (including intermediate folders).

    # Create a temporary file in the same directory as the final target file.
    # mkstemp returns:
    #   - fd: a low-level file descriptor
    #   - tmp_name: the temporary file path
    # The file is created with a prefix and suffix to make it identifiable.

    fd, tmp_name = tempfile.mkstemp(prefix='.tmp_ckpt_', suffix='.pt', dir=str(target.parent))
    
    # Close the file descriptor immediately because torch.save will handle writing using the file path, not this descriptor.
    
    os.close(fd)
    try:
        torch.save(payload, tmp_name, _use_new_zipfile_serialization=use_new_zip) # Save the payload to the temporary file first.
        os.replace(tmp_name, target_path) # Atomically replace the final target file with the temporary file.
    finally:
        if os.path.exists(tmp_name): # Cleanup step
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
    # Build the checkpoint dictionary with the essential training state.
    payload = {
        'model_state_dict': model.state_dict(), # Save the model parameters
        'epoch': epoch, # Save the current epoch number
        'step': step, # Save the current training step
        'best_val_loss': best_val_loss, # Save the best validation loss seen so far
    }
    # Optionally include the optimizer state so training can be resumed exactly.
    if include_optimizer_state and optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()

    # Attempt 1: save using the default modern zip-based PyTorch format.
    try:
        _atomic_torch_save(payload, path, use_new_zip=True)
        return
    except Exception as first_exc:
        # Attempt 2: if the modern format fails, try the legacy serialization format.
        try:
            _atomic_torch_save(payload, path, use_new_zip=False)
            print(f'[checkpoint] fallback legacy save aplicado em: {path}')
            return
        except Exception as second_exc: # If both save attempts fail, raise a clear error showing both causes.
            raise RuntimeError(
                f'Falha ao salvar checkpoint em {path}. '
                f'Erro zip={first_exc}; erro legacy={second_exc}'
            )


def checkpoint_paths(output_dir: str, epoch: int, step: int):
    base = Path(output_dir) / 'checkpoints' # Define the base directory where all checkpoint files will be stored.
    return {
        'last': str(base / 'last.pt'), # Path for the most recent checkpoint, usually overwritten every save.
        'best': str(base / 'best.pt'), # Path for the best checkpoint, usually overwritten only when validation improves.
        # Path for a periodic checkpoint with epoch and step encoded in the filename.
        # Example: epoch005_step000120.pt
        'periodic': str(base / f'epoch{epoch:03d}_step{step:06d}.pt'),
    }
