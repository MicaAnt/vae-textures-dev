#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def render(npz_path: Path, output_dir: Path, track_id: str | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = np.load(npz_path, allow_pickle=True)
    piano = payload["piano"]
    chord = payload["chord"]
    tid = track_id or npz_path.stem
    matrix = np.zeros((128, max(int(np.ceil(piano[:, 3].max())) + 1 if piano.size else 1, 1)), dtype=float)
    for row in piano:
        start = int(row[0])
        end = max(start + 1, int(np.ceil(row[3] + row[4] / max(row[5], 1))))
        pitch = int(row[6])
        if 0 <= pitch < 128:
            matrix[pitch, start:end] = 1
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{tid}_pianoroll.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(matrix, origin="lower", aspect="auto", cmap="gray_r")
    ax.set_title(f"{tid} piano matrix")
    ax.set_xlabel("beat")
    ax.set_ylabel("MIDI pitch")
    fig.tight_layout()
    fig.savefig(image_path, dpi=140)
    plt.close(fig)
    meta = {
        "track_id": tid,
        "npz_path": str(npz_path),
        "track_role": str(payload["track_role"].item()) if "track_role" in payload.files else None,
        "chord_shape": list(chord.shape),
        "piano_shape": list(piano.shape),
        "image_path": str(image_path),
    }
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Render bounded COMMU NPZ visual examples.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", default=None)
    args = parser.parse_args()
    meta = render(args.npz, args.output_dir, args.track_id)
    manifest = args.output_dir / f"{meta['track_id']}_manifest.json"
    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(meta["image_path"])

if __name__ == "__main__":
    main()
