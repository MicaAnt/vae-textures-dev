#!/usr/bin/env python3
"""Path B, leg 2: load epoch-1 checkpoint and run epoch 2."""

from test_common import latest_checkpoint, load_manifest, record_path, run_train


def main():
    manifest = load_manifest()
    initial_last = manifest['paths'].get('initial_last')
    if not initial_last:
        raise SystemExit(
            'Missing initial_last in outputs/manifest.json. '
            'Run 02_train_one_epoch_checkpoint.py first.'
        )

    print('[test] STEP 03: resume from epoch-1 checkpoint and run epoch 2')
    run_train(manifest, leg='resumed', n_epoch=2, run_epochs_this_job=1,
              resume_from=initial_last)
    record_path(manifest, 'resumed_final',
                latest_checkpoint(manifest, 'resumed', 'final-state'))


if __name__ == '__main__':
    main()
