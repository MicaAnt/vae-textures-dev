
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

SECONDS_PER_FILE_DEFAULT = 8.264


def discover_track_ids(input_dir: Path) -> list[str]:
    return sorted(path.stem for path in input_dir.glob('*.npz'))


def chunked(values: list[str], chunk_size: int) -> list[list[str]]:
    return [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description='Create overnight-friendly batch files for the COMMU enriched loss dataset generation.')
    parser.add_argument('--input-dir', type=Path, default=Path('/workspace/vae-textures-dev/COMMUDataset/npzFiles'))
    parser.add_argument('--output-dir', type=Path, default=Path('/workspace/vae-textures-dev/COMMUDataset/enriched_loss_batches'))
    parser.add_argument('--batch-size', type=int, default=None, help='Number of source COMMU files per batch.')
    parser.add_argument('--target-hours', type=float, default=8.0, help='If batch-size is omitted, derive it from this target runtime.')
    parser.add_argument('--seconds-per-file', type=float, default=SECONDS_PER_FILE_DEFAULT, help='Measured or assumed runtime per source COMMU file.')
    args = parser.parse_args()

    track_ids = discover_track_ids(args.input_dir)
    if not track_ids:
        raise FileNotFoundError(f'No npz files found in {args.input_dir}')

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = max(1, int((args.target_hours * 3600.0) / args.seconds_per_file))

    batches = chunked(track_ids, batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, batch in enumerate(batches):
        batch_path = args.output_dir / f'batch_{index:03d}.txt'
        batch_path.write_text('\n'.join(batch) + '\n')

    est_hours = (batch_size * args.seconds_per_file) / 3600.0
    print(f'track_count={len(track_ids)}')
    print(f'batch_size={batch_size}')
    print(f'batch_count={len(batches)}')
    print(f'estimated_hours_per_batch={est_hours:.2f}')
    print(f'output_dir={args.output_dir}')


if __name__ == '__main__':
    main()
