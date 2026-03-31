from torch.utils.data import DataLoader, Subset

from dataset import SEED
from dataset_loaders_commu import MusicDataLoaders


def make_commu_loaders(data_cfg: dict, train_batch_size: int, val_batch_size: int):
    loaders = MusicDataLoaders.get_loaders(
        SEED,
        bs_train=train_batch_size,
        bs_val=val_batch_size,
        portion=data_cfg.get('portion', 8),
        shift_low=data_cfg.get('shift_low', -6),
        shift_high=data_cfg.get('shift_high', 5),
        num_bar=data_cfg.get('num_bar', 2),
        contain_chord=data_cfg.get('contain_chord', True),
        random_train=data_cfg.get('random_train', False),
        random_val=data_cfg.get('random_val', False),
    )

    train_dataset = loaders.train_loader.dataset
    val_dataset = loaders.val_loader.dataset

    train_limit = data_cfg.get('limit_train_samples')
    val_limit = data_cfg.get('limit_val_samples')

    if train_limit is not None:
        train_subset = Subset(train_dataset, list(range(min(train_limit, len(train_dataset)))))
        train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=False)
    else:
        train_loader = loaders.train_loader

    if val_limit is not None:
        val_subset = Subset(val_dataset, list(range(min(val_limit, len(val_dataset)))))
        val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False)
    else:
        val_loader = loaders.val_loader

    return train_loader, val_loader


def batch_to_inputs(batch, device):
    _, _, pr_mat, x, c, _ = batch
    pr_mat = pr_mat.to(device).float()
    x = x.to(device).long()
    c = c.to(device).float()
    return x, c, pr_mat

