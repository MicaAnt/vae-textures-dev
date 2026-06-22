import glob
import io
import os
import re
import sys
from contextlib import redirect_stdout

from dataset import DATA_PATH, SEED
from dataset_loaders import MusicDataLoaders


def env_int(name, default):
    value = os.getenv(name)
    return default if value is None or value.strip() == '' else int(value)


def positive_limit(name):
    return env_int(name, 0) > 0


def line(message):
    print(f'[phase8-preflight] {message}')


def main():
    train_limit = env_int('VAE_LIMIT_TRAIN_SAMPLES', 0)
    val_limit = env_int('VAE_LIMIT_VAL_SAMPLES', 0)
    if positive_limit('VAE_LIMIT_TRAIN_SAMPLES') or positive_limit('VAE_LIMIT_VAL_SAMPLES'):
        line(
            'ERROR positive sample limits are not valid for representative '
            f'training: VAE_LIMIT_TRAIN_SAMPLES={train_limit}, '
            f'VAE_LIMIT_VAL_SAMPLES={val_limit}'
        )
        return 2

    seed = env_int('VAE_SEED', SEED)
    batch_size = env_int('VAE_BATCH_SIZE', 128)
    npz_count = len(glob.glob(os.path.join(DATA_PATH, '*.npz')))

    line(f'VAE_SEED={seed}')
    line(f'VAE_BATCH_SIZE={batch_size}')
    line(f'VAE_LIMIT_TRAIN_SAMPLES={train_limit}')
    line(f'VAE_LIMIT_VAL_SAMPLES={val_limit}')
    line(f'npz_count={npz_count}')
    line(
        'loader_config=portion=8 shift_low=-6 shift_high=5 num_bar=2 '
        'contain_chord=True random_train=True random_val=False'
    )

    loader_output = io.StringIO()
    with redirect_stdout(loader_output):
        data_loaders = MusicDataLoaders.get_loaders(
            seed,
            bs_train=batch_size,
            bs_val=batch_size,
            portion=8,
            shift_low=-6,
            shift_high=5,
            num_bar=2,
            contain_chord=True,
            random_train=True,
            random_val=False,
        )

    selected_count = 'unavailable'
    for raw_line in loader_output.getvalue().splitlines():
        print(raw_line)
        match = re.search(r'Selected (\d+) files, all are in duple meter\.', raw_line)
        if match:
            selected_count = match.group(1)

    train_len = len(data_loaders.train_loader.dataset)
    val_len = len(data_loaders.val_loader.dataset)
    line(f'selected_duple_meter_count={selected_count}')
    line(f'train_dataset_length={train_len}')
    line(f'val_dataset_length={val_len}')
    line(f'num_train_batch={data_loaders.num_train_batch}')
    line(f'num_val_batch={data_loaders.num_val_batch}')
    line('status=ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
