#!/usr/bin/env python3
"""Build a pickle index of .npz files for COMMUDataset."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def build_index(npz_dir: Path, output_pkl: Path, absolute_paths: bool = False) -> int:
    if not npz_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {npz_dir}")
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {npz_dir}")

    files = sorted(p for p in npz_dir.iterdir() if p.is_file() and p.suffix.lower() == ".npz")
    if absolute_paths:
        index = [str(p.resolve()) for p in files]
    else:
        index = [str(p) for p in files]

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(index, f)

    return len(index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Varre uma pasta com .npz e gera um arquivo .pkl com a lista de caminhos."
    )
    parser.add_argument(
        "--npz-dir",
        default="./COMMUDataset/npzFiles",
        help="Pasta de entrada contendo arquivos .npz (default: ./COMMUDataset/npzFiles)",
    )
    parser.add_argument(
        "--output",
        default="./COMMUDataset/ind.pkl",
        help="Caminho do arquivo .pkl de saída (default: ./COMMUDataset/ind.pkl)",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Salva caminhos absolutos no .pkl em vez de relativos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_dir = Path(args.npz_dir)
    output_pkl = Path(args.output)

    count = build_index(npz_dir=npz_dir, output_pkl=output_pkl, absolute_paths=args.absolute_paths)
    print(f"Arquivo criado: {output_pkl}")
    print(f"Total de arquivos .npz indexados: {count}")


if __name__ == "__main__":
    main()

