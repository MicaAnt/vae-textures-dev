import os
import random
import warnings

import numpy as np

warnings.simplefilter('ignore', UserWarning)
from model import DisentangleVAE
from ptvae import RnnEncoder, TextureEncoder, PtvaeEncoder, PtvaeDecoder, \
    RnnDecoder
from dataset_loaders import MusicDataLoaders, TrainingVAE
from dataset import SEED
from amc_dl.torch_plus import LogPathManager, SummaryWriters, \
    ParameterScheduler, OptimizerScheduler, MinExponentialLR, \
    TeacherForcingScheduler, ConstantScheduler
from amc_dl.torch_plus.train_utils import kl_anealing
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from wandb_helper import WandbRunLogger, env_flag


def env_int(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


def env_float(name, default):
    value = os.getenv(name)
    return default if value is None else float(value)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed = env_int('VAE_SEED', SEED)
set_global_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
readme_fn = './train.py'
batch_size = env_int('VAE_BATCH_SIZE', 128)
n_epoch = env_int('VAE_N_EPOCH', 6)
clip = 1
parallel = False
weights = [1, 0.5]
beta = 0.1
tf_rates = [(0.6, 0), (0.5, 0), (0.5, 0)]
lr = env_float('VAE_LR', 1e-3)
name = os.getenv('VAE_RUN_NAME', 'disvae-nozoth')
limit_train_samples = env_int('VAE_LIMIT_TRAIN_SAMPLES', 0)
limit_val_samples = env_int('VAE_LIMIT_VAL_SAMPLES', 0)
resume_from = os.getenv('VAE_RESUME_FROM', '')
run_epochs_this_job = env_int('VAE_RUN_EPOCHS_THIS_JOB', 0)
full_checkpoint_policy = os.getenv(
    'VAE_FULL_CHECKPOINT_POLICY', 'epoch-state,last-state,final-state'
)
wandb_enabled = env_flag('WANDB_ENABLED', False)

parallel = parallel if torch.cuda.is_available() and \
                       torch.cuda.device_count() > 1 else False
# train_model
chd_encoder = RnnEncoder(36, 1024, 256)
rhy_encoder = TextureEncoder(256, 1024, 256)
# rhy_encoder = PtvaeEncoder(device=device, z_size=256, max_pitch=39 - 8, min_pitch=0)
# pt_encoder = PtvaeEncoder(device=device, z_size=152)
chd_decoder = RnnDecoder(z_dim=256)
pt_decoder = PtvaeDecoder(note_embedding=None,
                          dec_dur_hid_size=64, z_size=512)
model = DisentangleVAE(name, device, chd_encoder,
                       rhy_encoder, pt_decoder, chd_decoder)

# data loaders
data_loaders = \
    MusicDataLoaders.get_loaders(seed, bs_train=batch_size, bs_val=batch_size,
                                 portion=8, shift_low=-6, shift_high=5,
                                 num_bar=2,
                                 contain_chord=True)

# Optional sample limits keep the canonical entrypoint intact while allowing
# short local proof runs before committing to a full reproduction run.
if limit_train_samples > 0:
    train_dataset = data_loaders.train_loader.dataset
    train_subset = Subset(train_dataset,
                          list(range(min(limit_train_samples,
                                         len(train_dataset)))))
    data_loaders.train_loader = DataLoader(train_subset,
                                           batch_size=batch_size,
                                           shuffle=False)

if limit_val_samples > 0:
    val_dataset = data_loaders.val_loader.dataset
    val_subset = Subset(val_dataset,
                        list(range(min(limit_val_samples,
                                       len(val_dataset)))))
    data_loaders.val_loader = DataLoader(val_subset,
                                         batch_size=batch_size,
                                         shuffle=False)

log_path_mng = LogPathManager(readme_fn)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = MinExponentialLR(optimizer, gamma=0.9999, minimum=1e-5)
optimizer_scheduler = OptimizerScheduler(optimizer, scheduler, clip)

writer_names = ['loss', 'recon_loss', 'pl', 'dl', 'kl_loss', 'kl_chd',
                'kl_rhy', 'chord_loss', 'root_loss', 'chroma_loss', 'bass_loss']
#, 'chord', 'root', 'chroma', 'bass']
tags = {'loss': None}
summary_writers = SummaryWriters(writer_names, tags, log_path_mng.writer_path)
tfr1_scheduler = TeacherForcingScheduler(*tf_rates[0])
tfr2_scheduler = TeacherForcingScheduler(*tf_rates[1])
tfr3_scheduler = TeacherForcingScheduler(*tf_rates[2])
weights_scheduler = ConstantScheduler(weights)
beta_scheduler = TeacherForcingScheduler(beta, 0., f=kl_anealing)
params_dic = dict(tfr1=tfr1_scheduler, tfr2=tfr2_scheduler,
                  tfr3=tfr3_scheduler,
                  beta=beta_scheduler, weights=weights_scheduler)
param_scheduler = ParameterScheduler(**params_dic)

wandb_config = {
    'run_name': name,
    'device': str(device),
    'batch_size': batch_size,
    'n_epoch': n_epoch,
    'learning_rate': lr,
    'clip': clip,
    'beta': beta,
    'weights': weights,
    'tf_rates': tf_rates,
    'parallel': parallel,
    'limit_train_samples': limit_train_samples,
    'limit_val_samples': limit_val_samples,
    'seed': seed,
    'shift_low': -6,
    'shift_high': 5,
    'num_bar': 2,
    'contain_chord': True,
    'train_portion': 8,
    'writer_names': writer_names,
    'resume_from': resume_from,
    'run_epochs_this_job': run_epochs_this_job,
    'full_checkpoint_policy': full_checkpoint_policy,
    'wandb_run_id': os.getenv('WANDB_RUN_ID', ''),
    'wandb_resume': os.getenv('WANDB_RESUME', ''),
}

run_logger = WandbRunLogger.from_env(name, wandb_config, log_path_mng)

training = TrainingVAE(device, model, parallel, log_path_mng,
                       data_loaders, summary_writers, optimizer_scheduler,
                       param_scheduler, n_epoch, run_logger=run_logger)
resume_meta = None
if resume_from:
    resume_meta = training.load_training_state_checkpoint(resume_from)
max_epochs_this_job = run_epochs_this_job if run_epochs_this_job > 0 else None
try:
    if resume_meta is None:
        training.run(
            max_epochs_this_job=max_epochs_this_job,
            checkpoint_config=wandb_config,
        )
    else:
        training.run(
            start_epoch=resume_meta['epoch'],
            start_train_step=resume_meta['train_step'],
            start_val_step=resume_meta['val_step'],
            best_valid_loss=resume_meta['best_valid_loss'],
            max_epochs_this_job=max_epochs_this_job,
            checkpoint_config=wandb_config,
        )
finally:
    if run_logger is not None:
        run_logger.finish()

if __name__ == '__main__':
    pass
