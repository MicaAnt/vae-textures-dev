#!/usr/bin/env python3
"""Compare direct final state against resumed final state."""

from test_common import (REPORT_DIR, compare_training_states,
                         flatten_category_diffs, load_manifest, load_state,
                         write_weight_diff_csv)


def main():
    manifest = load_manifest()
    direct_path = manifest['paths'].get('direct_final')
    resumed_path = manifest['paths'].get('resumed_final')
    if not direct_path or not resumed_path:
        raise SystemExit(
            'Missing direct_final or resumed_final in outputs/manifest.json. '
            'Run steps 01, 02, and 03 first.'
        )

    print('[test] STEP 04: compare final training states')
    direct = load_state(direct_path)
    resumed = load_state(resumed_path)
    category_diffs = compare_training_states(direct, resumed)
    diffs = flatten_category_diffs(category_diffs)
    csv_path, rows = write_weight_diff_csv(direct, resumed)

    report_path = REPORT_DIR / 'state_comparison.txt'
    lines = [
        'Checkpoint/resume continuity comparison',
        f'direct_final={direct_path}',
        f'resumed_final={resumed_path}',
        f'weight_diff_report={csv_path}',
        '',
        'Category summary:',
    ]
    for category, category_items in category_diffs.items():
        status = 'PASS' if not category_items else 'FAIL'
        lines.append(f'- {category}: {status}')
    lines.append('')
    if diffs:
        lines.append('RESULT=FAILED')
        lines.extend(f'- {diff}' for diff in diffs)
    else:
        lines.append('RESULT=PASSED')
        lines.append('All continuity-defining state matched exactly.')
    report_path.write_text('\n'.join(lines) + '\n')

    non_equal_weights = [row for row in rows if not row['exact_equal']]
    print(f'[test] report={report_path}')
    print(f'[test] weight_diff_report={csv_path}')
    print(f'[test] non_equal_weight_tensors={len(non_equal_weights)}')
    if diffs:
        print('[test] RESULT=FAILED')
        for category, category_items in category_diffs.items():
            status = 'PASS' if not category_items else 'FAIL'
            print(f'[test] category={category} status={status}')
        for diff in diffs[:20]:
            print(f'[test] diff={diff}')
        raise SystemExit(1)
    for category in category_diffs:
        print(f'[test] category={category} status=PASS')
    print('[test] RESULT=PASSED')


if __name__ == '__main__':
    main()
