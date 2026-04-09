import argparse
import json
from pathlib import Path

import torch

from dataset import SEED
from dataset_loaders import MusicDataLoaders
from model import DisentangleVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate objective disentanglement metrics inspired by ISMIR 2020 paper section 5.2")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pt checkpoint")
    parser.add_argument("--device", default=None, help="torch device, default: cuda if available else cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--portion", type=int, default=8, help="Dataset split portion used by current train.py (8 -> 8/9 train, 1/9 val)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-batches", type=int, default=0, help="Limit number of val batches for quick smoke evaluation")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def get_device(name):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, device):
    model = DisentangleVAE.init_model(device=device)
    model.load_model(checkpoint_path, map_location=device)
    model.eval()
    return model


def get_val_loader(batch_size, portion, seed):
    loaders = MusicDataLoaders.get_loaders(
        seed,
        bs_train=batch_size,
        bs_val=batch_size,
        portion=portion,
        shift_low=-6,
        shift_high=5,
        num_bar=2,
        contain_chord=True,
        random_train=False,
        random_val=False,
    )
    return loaders.val_loader


def batch_to_eval_tensors(batch, device):
    _, _, pr_mat, _, c, _ = batch
    return pr_mat.to(device).float(), c.to(device).float()


def rotate_chord(chord, semitone):
    semitone = semitone % 12
    if semitone == 0:
        return chord.clone()
    out = chord.clone()
    out[..., 0:12] = torch.roll(chord[..., 0:12], shifts=semitone, dims=-1)
    out[..., 12:24] = torch.roll(chord[..., 12:24], shifts=semitone, dims=-1)
    out[..., 24:36] = torch.roll(chord[..., 24:36], shifts=semitone, dims=-1)
    return out


def transpose_pr_mat(pr_mat, semitone):
    semitone = int(semitone)
    if semitone == 0:
        return pr_mat.clone()
    out = torch.zeros_like(pr_mat)
    if semitone > 0:
        out[..., semitone:] = pr_mat[..., :-semitone]
    else:
        out[..., :semitone] = pr_mat[..., -semitone:]
    return out


def apply_fi(pr_mat, chord, semitone):
    return transpose_pr_mat(pr_mat, semitone), rotate_chord(chord, semitone)


def apply_pi(pr_mat, chord, probability, generator):
    pr_aug = pr_mat.clone()
    ch_aug = chord.clone()
    bs = pr_mat.size(0)
    beat_mask = torch.rand((bs, 8), generator=generator, device=pr_mat.device) < probability
    sign = torch.randint(0, 2, (bs, 8), generator=generator, device=pr_mat.device) * 2 - 1
    shifts = beat_mask.long() * sign

    for b in range(bs):
        for beat in range(8):
            shift = int(shifts[b, beat].item())
            if shift == 0:
                continue
            start = beat * 4
            end = start + 4
            pr_aug[b, start:end, :] = transpose_pr_mat(pr_aug[b, start:end, :], shift)
            ch_aug[b, beat:beat + 1, :] = rotate_chord(ch_aug[b, beat:beat + 1, :], shift)
    return pr_aug, ch_aug


def apply_ri(pr_mat, probability, generator):
    mask = (torch.rand(pr_mat.shape, generator=generator, device=pr_mat.device) < probability) & (pr_mat > 0)
    halved = torch.ceil(pr_mat / 2.0)
    return torch.where(mask, halved, pr_mat)


def encode_means(model, pr_mat, chord):
    dist_chd, dist_rhy = model.inference_encode(pr_mat, chord)
    return dist_chd.mean, dist_rhy.mean


def l1_delta(base, aug):
    return torch.abs(aug - base).sum(dim=-1)


def summarize(values):
    if not values:
        return {"sum": 0.0, "mean": 0.0, "count": 0}
    stacked = torch.cat(values, dim=0)
    return {
        "sum": float(stacked.sum().item()),
        "mean": float(stacked.mean().item()),
        "count": int(stacked.numel()),
    }


def evaluate(model, val_loader, device, max_batches=0):
    generator = torch.Generator(device=device.type if device.type != 'cpu' else 'cpu')
    generator.manual_seed(1234)

    fi_results = {str(i): {"zchd": [], "ztxt": []} for i in range(1, 13)}
    probs = [round(i / 10.0, 1) for i in range(1, 11)]
    pi_results = {f"{p:.1f}": {"zchd": [], "ztxt": []} for p in probs}
    ri_results = {f"{p:.1f}": {"zchd": [], "ztxt": []} for p in probs}

    for batch_idx, batch in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break
        pr_mat, chord = batch_to_eval_tensors(batch, device)
        base_chd, base_txt = encode_means(model, pr_mat, chord)

        for i in range(1, 13):
            pr_aug, ch_aug = apply_fi(pr_mat, chord, i)
            aug_chd, aug_txt = encode_means(model, pr_aug, ch_aug)
            fi_results[str(i)]["zchd"].append(l1_delta(base_chd, aug_chd).detach().cpu())
            fi_results[str(i)]["ztxt"].append(l1_delta(base_txt, aug_txt).detach().cpu())

        for p in probs:
            pr_pi, ch_pi = apply_pi(pr_mat, chord, p, generator)
            pi_chd, pi_txt = encode_means(model, pr_pi, ch_pi)
            pi_results[f"{p:.1f}"]["zchd"].append(l1_delta(base_chd, pi_chd).detach().cpu())
            pi_results[f"{p:.1f}"]["ztxt"].append(l1_delta(base_txt, pi_txt).detach().cpu())

            pr_ri = apply_ri(pr_mat, p, generator)
            ri_chd, ri_txt = encode_means(model, pr_ri, chord)
            ri_results[f"{p:.1f}"]["zchd"].append(l1_delta(base_chd, ri_chd).detach().cpu())
            ri_results[f"{p:.1f}"]["ztxt"].append(l1_delta(base_txt, ri_txt).detach().cpu())

    fi_summary = {k: {"zchd": summarize(v["zchd"]), "ztxt": summarize(v["ztxt"])} for k, v in fi_results.items()}
    pi_summary = {k: {"zchd": summarize(v["zchd"]), "ztxt": summarize(v["ztxt"])} for k, v in pi_results.items()}
    ri_summary = {k: {"zchd": summarize(v["zchd"]), "ztxt": summarize(v["ztxt"])} for k, v in ri_results.items()}

    return {
        "paper_protocol_notes": {
            "paper_train_split": "90/10 at song level",
            "current_eval_split": f"{portion_from_loader(val_loader)} validation proxy from current code path",
            "note": "Current code path defaults to portion=8 (about 8/9 train, 1/9 val). This evaluation reports on the held-out validation split of the restored code path.",
            "latent_statistic": "Posterior mean used as latent representation for delta computation.",
        },
        "Fi_all_note_transposition": fi_summary,
        "Pi_beatwise_random_pitch_perturbation": pi_summary,
        "Ri_random_duration_halving": ri_summary,
    }


def portion_from_loader(val_loader):
    try:
        val_size = len(val_loader.dataset)
        batch = next(iter(val_loader))
        _ = batch
        return f"val_size={val_size}"
    except Exception:
        return "validation split"


def main():
    args = parse_args()
    device = get_device(args.device)
    checkpoint = Path(args.checkpoint)
    model = load_model(str(checkpoint), device)
    val_loader = get_val_loader(args.batch_size, args.portion, args.seed)
    results = evaluate(model, val_loader, device, max_batches=args.max_batches)
    results["checkpoint"] = str(checkpoint)
    results["device"] = str(device)
    results["max_batches"] = args.max_batches

    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
