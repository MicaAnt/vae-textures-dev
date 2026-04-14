
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'base_model'))
sys.path.insert(0, str(REPO_ROOT))

from base_model.dataset import wrap_dataset
from base_model.model import DisentangleVAE
from interface import PolyDisVAE
from amc_dl.torch_plus.train_utils import get_zs_from_dists


def load_model(device: torch.device) -> DisentangleVAE:
    interface = PolyDisVAE.init_model(device=device)
    interface.load_model(str(REPO_ROOT / 'model_param' / 'polydis-v1.pt'))
    model = DisentangleVAE(
        'disvae',
        interface.device,
        interface.chd_encoder,
        interface.txt_encoder,
        interface.pnotree_decoder,
        interface.chd_decoder,
    )
    model.eval()
    return model


def prepare_tensors(x, c, pr_mat, device: torch.device):
    x = torch.tensor(x).long().unsqueeze(0).to(device)
    c = torch.tensor(c).float().unsqueeze(0).to(device)
    pr_mat = torch.tensor(pr_mat).float().unsqueeze(0).to(device)
    return x, c, pr_mat


def compute_segment(model: DisentangleVAE, x, c, pr_mat):
    embedded_x, lengths = model.decoder.emb_x(x)
    dist_chd = model.chd_encoder(c)
    dist_rhy = model.rhy_encoder(pr_mat)
    z_chd, z_txt = get_zs_from_dists([dist_chd, dist_rhy], True)
    dec_z = torch.cat([z_chd, z_txt], dim=-1)
    recon_pitch, recon_dur = model.decoder(dec_z, False, embedded_x, lengths, 0.0, 0.0)
    recon_root, recon_chroma, recon_bass = model.chd_decoder(z_chd, False, 0.0, c)
    loss_values = model.loss_function(
        x,
        c,
        recon_pitch,
        recon_dur,
        dist_chd,
        dist_rhy,
        recon_root,
        recon_chroma,
        recon_bass,
        beta=0.1,
        weights=(1, 0.5),
    )
    total_loss, recon_loss, pl, dl, kl_loss, kl_chd, kl_rhy, chord_loss, root_loss, chroma_loss, bass_loss = loss_values
    return {
        'z_chd': z_chd.squeeze(0).detach().cpu().numpy(),
        'z_txt': z_txt.squeeze(0).detach().cpu().numpy(),
        'final_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'pitch_loss': pl.item(),
        'duration_loss': dl.item(),
        'kl_loss': kl_loss.item(),
        'kl_chd': kl_chd.item(),
        'kl_rhy': kl_rhy.item(),
        'chord_loss': chord_loss.item(),
        'root_loss': root_loss.item(),
        'chroma_loss': chroma_loss.item(),
        'bass_loss': bass_loss.item(),
    }


def scalar_meta(payload, key: str):
    if key not in payload:
        return None
    value = payload[key]
    arr = np.asarray(value)
    return arr.item() if arr.shape == () else arr


def recompute_file(npz_path: Path, model: DisentangleVAE, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = np.load(npz_path, allow_pickle=True)
    dataset = wrap_dataset([str(npz_path)], [0], shift_low=0, shift_high=0, num_bar=2, contain_chord=True)
    meta_keys = [
        'audio_key', 'chord_progressions', 'pitch_range', 'num_measures',
        'bpm', 'genre', 'track_role', 'inst', 'sample_rhythm', 'time_signature'
    ]
    metadata = {key: scalar_meta(payload, key) for key in meta_keys}

    for idx in range(len(dataset)):
        _, _, pr_mat, x, c, _ = dataset[idx]
        x_t, c_t, pr_t = prepare_tensors(x, c, pr_mat, model.device)
        result = compute_segment(model, x_t, c_t, pr_t)
        segment_name = f'{npz_path.stem}-{idx + 1:03d}.npz'
        np.savez_compressed(output_dir / segment_name, **result, **metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description='Recompute richer COMMU per-segment loss components for study notebooks.')
    parser.add_argument('--input-dir', type=Path, default=REPO_ROOT / 'COMMUDataset' / 'npzFiles')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'NotebooksVAESymTex' / '_cache' / 'commu_enriched_losses')
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    files = sorted(args.input_dir.glob('*.npz'))
    if args.max_files is not None:
        files = files[:args.max_files]
    if not files:
        raise FileNotFoundError(f'No npz files found in {args.input_dir}')

    device = torch.device(args.device)
    model = load_model(device)

    for path in tqdm(files, desc='Recomputing COMMU losses'):
        recompute_file(path, model, args.output_dir)


if __name__ == '__main__':
    main()
