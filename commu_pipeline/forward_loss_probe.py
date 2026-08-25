#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "base_model"))

from base_model.dataset import wrap_dataset  # noqa: E402
from base_model.model import DisentangleVAE  # noqa: E402
from interface import PolyDisVAE  # noqa: E402
from amc_dl.torch_plus.train_utils import get_zs_from_dists  # noqa: E402

LOSS_KEYS = ["final_loss", "recon_loss", "pitch_loss", "duration_loss", "kl_loss", "kl_chd", "kl_rhy", "chord_loss", "root_loss", "chroma_loss", "bass_loss"]


def load_model(device: torch.device, checkpoint: Path) -> DisentangleVAE:
    interface = PolyDisVAE.init_model(device=device)
    interface.load_model(str(checkpoint))
    model = DisentangleVAE("disvae", interface.device, interface.chd_encoder, interface.txt_encoder, interface.pnotree_decoder, interface.chd_decoder)
    model.eval()
    return model


def prepare_tensors(x, c, pr_mat, device: torch.device):
    return torch.tensor(x).long().unsqueeze(0).to(device), torch.tensor(c).float().unsqueeze(0).to(device), torch.tensor(pr_mat).float().unsqueeze(0).to(device)


def compute_segment(model, x, c, pr_mat):
    embedded_x, lengths = model.decoder.emb_x(x)
    dist_chd = model.chd_encoder(c)
    dist_rhy = model.rhy_encoder(pr_mat)
    z_chd, z_txt = get_zs_from_dists([dist_chd, dist_rhy], True)
    dec_z = torch.cat([z_chd, z_txt], dim=-1)
    recon_pitch, recon_dur = model.decoder(dec_z, False, embedded_x, lengths, 0.0, 0.0)
    recon_root, recon_chroma, recon_bass = model.chd_decoder(z_chd, False, 0.0, c)
    values = model.loss_function(x, c, recon_pitch, recon_dur, dist_chd, dist_rhy, recon_root, recon_chroma, recon_bass, beta=0.1, weights=(1, 0.5))
    total_loss, recon_loss, pl, dl, kl_loss, kl_chd, kl_rhy, chord_loss, root_loss, chroma_loss, bass_loss = values
    tensors = [total_loss, recon_loss, pl, dl, kl_loss, kl_chd, kl_rhy, chord_loss, root_loss, chroma_loss, bass_loss]
    return {k: float(v.item()) for k, v in zip(LOSS_KEYS, tensors)}


def resolve_files(input_dir: Path, track_ids: str | None, max_files: int | None) -> list[Path]:
    if track_ids:
        files = [input_dir / f"{tid.strip()}.npz" for tid in track_ids.split(",") if tid.strip()]
    else:
        files = sorted(input_dir.glob("*.npz"))
    files = [p for p in files if p.exists()]
    if max_files is not None:
        files = files[:max_files]
    return files


def run_probe(input_dir: Path, track_ids: str | None, max_files: int | None, device: torch.device, checkpoint: Path) -> dict:
    files = resolve_files(input_dir, track_ids, max_files)
    model = load_model(device, checkpoint)
    samples = []
    errors = []
    segments = 0
    with torch.no_grad():
        for path in files:
            try:
                dataset = wrap_dataset([str(path)], [0], shift_low=0, shift_high=0, num_bar=2, contain_chord=True)
                for idx in range(min(len(dataset), 1)):
                    _, _, pr_mat, x, c, _ = dataset[idx]
                    x_t, c_t, pr_t = prepare_tensors(x, c, pr_mat, model.device)
                    losses = compute_segment(model, x_t, c_t, pr_t)
                    samples.append({"npz": str(path), "segment_index": idx, "losses": losses})
                    segments += 1
            except Exception as exc:  # noqa: BLE001
                errors.append({"npz": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {"checkpoint": str(checkpoint), "device": str(device), "files_checked": len(files), "segments_checked": segments, "loss_keys": LOSS_KEYS, "errors": errors, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded COMMU forward/loss compatibility probe; does not train or compare checkpoints.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--track-ids", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = run_probe(args.input_dir, args.track_ids, args.max_files, torch.device(args.device), args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
    if data["errors"] or data["segments_checked"] == 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
