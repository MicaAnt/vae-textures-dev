#!/usr/bin/env python3
"""Print a human inspection view of model weights and differences."""

import csv

from test_common import REPORT_DIR, load_manifest, load_state


def main():
    manifest = load_manifest()
    direct_path = manifest['paths'].get('direct_final')
    resumed_path = manifest['paths'].get('resumed_final')
    if not direct_path or not resumed_path:
        raise SystemExit(
            'Missing direct_final or resumed_final in outputs/manifest.json. '
            'Run steps 01, 02, 03, and 04 first.'
        )

    direct = load_state(direct_path)
    resumed = load_state(resumed_path)
    direct_model = direct['model_state_dict']
    resumed_model = resumed['model_state_dict']

    print('[test] STEP 05: inspect model weights')
    print(f'[test] direct_final={direct_path}')
    print(f'[test] resumed_final={resumed_path}')
    print()
    print('tensor, shape, exact_equal, direct_mean, resumed_mean, max_abs_diff')

    rows = []
    for name in sorted(direct_model):
        left = direct_model[name].detach().cpu()
        right = resumed_model[name].detach().cpu()
        diff = (left - right).abs()
        row = {
            'tensor': name,
            'shape': 'x'.join(str(dim) for dim in left.shape),
            'exact_equal': bool((left == right).all().item()),
            'direct_mean': float(left.float().mean().item()) if left.numel() else 0.0,
            'resumed_mean': float(right.float().mean().item()) if right.numel() else 0.0,
            'max_abs_diff': float(diff.max().item()) if diff.numel() else 0.0,
        }
        rows.append(row)
        print(
            f'{row["tensor"]}, {row["shape"]}, {row["exact_equal"]}, '
            f'{row["direct_mean"]:.9g}, {row["resumed_mean"]:.9g}, '
            f'{row["max_abs_diff"]:.9g}'
        )

    csv_path = REPORT_DIR / 'weight_inspection.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f'[test] csv={csv_path}')


if __name__ == '__main__':
    main()
