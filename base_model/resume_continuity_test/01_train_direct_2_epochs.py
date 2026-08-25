#!/usr/bin/env python3
"""Path A: uninterrupted training for epoch 1 and epoch 2."""

import os

from test_common import latest_checkpoint, load_manifest, record_path, run_train


def main():
    os.environ['RESUME_CONTINUITY_NEW_RUN'] = '1'
    manifest = load_manifest()
    print('[test] STEP 01: direct uninterrupted two-epoch training')
    run_train(manifest, leg='direct', n_epoch=2)
    record_path(manifest, 'direct_final',
                latest_checkpoint(manifest, 'direct', 'final-state'))


if __name__ == '__main__':
    main()
