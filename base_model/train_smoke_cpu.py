import argparse
import os
import random
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset

from model import DisentangleVAE
from ptvae import RnnEncoder, TextureEncoder, PtvaeDecoder, RnnDecoder
from dataset_loaders import MusicDataLoaders
from dataset import SEED


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(device: torch.device) -> DisentangleVAE:
    chd_encoder = RnnEncoder(36, 1024, 256)
    # TextureEncoder expects pr_mat with shape (B, 32, 128), which is exactly
    # what this smoke script feeds as the 3rd model input.
    rhy_encoder = TextureEncoder(emb_size=256, hidden_dim=1024, z_dim=256)
    chd_decoder = RnnDecoder(z_dim=256)
    pt_decoder = PtvaeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)
    model = DisentangleVAE('disvae-smoke-cpu', device, chd_encoder, rhy_encoder, pt_decoder, chd_decoder)
    return model


def make_small_loaders(batch_size: int, limit_train_samples: int, limit_val_samples: int):
    data_loaders = MusicDataLoaders.get_loaders(
        SEED,
        bs_train=batch_size,
        bs_val=batch_size,
        portion=8,
        shift_low=-6,
        shift_high=5,
        num_bar=2,
        contain_chord=True,
        random_train=False,
        random_val=False,
    )

    train_dataset = data_loaders.train_loader.dataset
    val_dataset = data_loaders.val_loader.dataset

    train_count = min(limit_train_samples, len(train_dataset))
    val_count = min(limit_val_samples, len(val_dataset))

    train_subset = Subset(train_dataset, list(range(train_count)))
    val_subset = Subset(val_dataset, list(range(val_count)))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_count, val_count


def batch_to_inputs(batch, device: torch.device):
    _, _, pr_mat, x, c, _ = batch
    pr_mat = pr_mat.to(device).float()
    x = x.to(device).long()
    c = c.to(device).float()
    return x, c, pr_mat


def main():
    parser = argparse.ArgumentParser(description='Smoke test de treino VAE em CPU')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--limit-train-samples', type=int, default=100)
    parser.add_argument('--limit-val-samples', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default='result_smoke_cpu')
    parser.add_argument('--seed', type=int, default=3345)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cpu')
    print(f'Usando dispositivo: {device}')

    start = time.time()
    model = build_model(device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader, train_count, val_count = make_small_loaders(
        args.batch_size, args.limit_train_samples, args.limit_val_samples
    )

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f'Train samples usados: {train_count} | Val samples usados: {val_count}')
    print(f'Passos máximos de treino: {args.max_steps}')

    first_batch = next(iter(train_loader))
    x, c, pr_mat = batch_to_inputs(first_batch, device)
    print('Exemplo do batch de treino (shapes):')
    print(f'  x: {tuple(x.shape)} | c: {tuple(c.shape)} | pr_mat: {tuple(pr_mat.shape)}')
    print(f'  x[0,0,0,:6] = {x[0, 0, 0, :6].tolist()}')
    print(f'  c[0,0,:6] = {[round(v, 3) for v in c[0, 0, :6].tolist()]}')

    global_step = 0
    train_logs = []

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            inputs = batch_to_inputs(batch, device)
            optimizer.zero_grad()
            outputs = model('train', *inputs, tfr1=0.6, tfr2=0.5, tfr3=0.5, beta=0.1, weights=(1.0, 0.5))
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            log_item = {
                'step': global_step + 1,
                'loss': float(outputs[0].item()),
                'recon_loss': float(outputs[1].item()),
                'kl_loss': float(outputs[4].item()),
                'chord_loss': float(outputs[7].item()),
            }
            train_logs.append(log_item)

            if (global_step + 1) % args.log_every == 0 or global_step == 0:
                print(
                    f"[train] step={log_item['step']:03d} "
                    f"loss={log_item['loss']:.4f} "
                    f"recon={log_item['recon_loss']:.4f} "
                    f"kl={log_item['kl_loss']:.4f} "
                    f"chord={log_item['chord_loss']:.4f}"
                )

            if (global_step + 1) % 25 == 0:
                ckpt_path = os.path.join(ckpt_dir, f'epoch{epoch + 1:02d}_step{global_step + 1:03d}.pt')
                torch.save(model.state_dict(), ckpt_path)
                print(f'Checkpoint salvo em: {ckpt_path}')

            global_step += 1

        if global_step >= args.max_steps:
            break

    model.eval()
    with torch.no_grad():
        val_batch = next(iter(val_loader))
        val_inputs = batch_to_inputs(val_batch, device)
        val_out = model('train', *val_inputs, tfr1=0.0, tfr2=0.0, tfr3=0.0, beta=0.1, weights=(1.0, 0.5))
        print(f"[val] loss={val_out[0].item():.4f} recon={val_out[1].item():.4f} kl={val_out[4].item():.4f}")

    final_ckpt = os.path.join(ckpt_dir, 'final_smoke.pt')
    torch.save(model.state_dict(), final_ckpt)
    elapsed = time.time() - start
    print(f'Checkpoint final salvo em: {final_ckpt}')
    print(f'Tempo total (s): {elapsed:.2f}')


if __name__ == '__main__':
    main()