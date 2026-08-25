#!/usr/bin/env python3
"""Path B, leg 1: train only epoch 1 and save the resume checkpoint."""

from test_common import latest_checkpoint, load_manifest, record_path, run_train


def main():
    manifest = load_manifest()
    print('[test] STEP 02: interrupted path, first job runs exactly one epoch')
    run_train(manifest, leg='initial', n_epoch=1, run_epochs_this_job=1)
    record_path(manifest, 'initial_last',
                latest_checkpoint(manifest, 'initial', 'last-state'))


if __name__ == '__main__':
    main()
