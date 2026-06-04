import time
import os
import torch
from canonical_checkpoint import save_training_state, load_training_state, capture_rng_state, restore_rng_state
from torch import nn
from .train_utils import epoch_time


class PytorchModel(nn.Module):

    def __init__(self, name, device):
        self.name = name
        super(PytorchModel, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu')
        self.device = device

    def run(self, *input):
        """A general way to run the model.
        Usually tensor input -> tensor output"""
        raise NotImplementedError

    def loss(self, *input, **kwargs):
        """Call it during training. The output is loss and possibly others to
        display on tensorboard."""
        raise NotImplementedError

    def inference(self, *input):
        """Call it during inference.
        The output is usually numpy after argmax."""
        raise NotImplementedError

    def loss_function(self, *input):
        raise NotImplementedError

    def forward(self, mode, *input, **kwargs):
        if mode in ["run", 0]:
            return self.run(*input, **kwargs)
        elif mode in ['loss', 'train', 1]:
            return self.loss(*input, **kwargs)
        elif mode in ['inference', 'eval', 'val', 2]:
            return self.inference(*input, **kwargs)
        else:
            raise NotImplementedError

    def load_model(self, model_path, map_location=None):
        if map_location is None:
            map_location = self.device
        dic = torch.load(model_path, map_location=map_location)
        for name in list(dic.keys()):
            dic[name.replace('module.', '')] = dic.pop(name)
        self.load_state_dict(dic)
        self.to(self.device)

    @staticmethod
    def init_model(*inputs):
        raise NotImplementedError


class TrainingInterface:

    def __init__(self, device, model, parallel, log_path_mng, data_loaders,
                 summary_writers,
                 opt_scheduler, param_scheduler, n_epoch, **kwargs):
        self.model = model
        self.model.device = device
        if parallel:
            self.model = nn.DataParallel(self.model)
        self.model.to(device)
        self.path_mng = log_path_mng
        self.summary_writers = summary_writers
        self.data_loaders = data_loaders
        self.opt_scheduler = opt_scheduler
        self.param_scheduler = param_scheduler
        self.device = device
        self.n_epoch = n_epoch
        self.epoch = 0
        self.train_step = 0
        self.val_step = 0
        self.parallel = parallel
        self.run_logger = kwargs.pop('run_logger', None)
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def name(self):
        if self.parallel:
            return self.model.module.name
        else:
            return self.model.name

    @property
    def log_path(self):
        return self.path_mng.log_path

    @property
    def model_path(self):
        return self.path_mng.model_path

    @property
    def writer_path(self):
        return self.path_mng.writer_path

    @property
    def writer_names(self):
        return self.summary_writers.writer_names

    def _init_loss_dic(self):
        loss_dic = {}
        for key in self.writer_names:
            loss_dic[key] = 0.
        return loss_dic

    def _accumulate_loss_dic(self, loss_dic, loss_items):
        assert len(self.writer_names) == len(loss_items)
        for key, val in zip(self.writer_names, loss_items):
            loss_dic[key] += val.item()
        return loss_dic

    def _write_loss_to_dic(self, loss_items):
        loss_dic = {}
        assert len(self.writer_names) == len(loss_items)
        for key, val in zip(self.writer_names, loss_items):
            loss_dic[key] = val.item()
        return loss_dic

    def _batch_to_inputs(self, batch):
        raise NotImplementedError

    def train(self, **kwargs):
        self.model.train()
        self.param_scheduler.train()
        epoch_loss_dic = self._init_loss_dic()

        for i, batch in enumerate(self.data_loaders.train_loader):
            inputs = self._batch_to_inputs(batch)
            self.opt_scheduler.optimizer_zero_grad()
            input_params = self.param_scheduler.step()
            outputs = self.model('train', *inputs, **input_params)
            outputs = self._sum_parallel_loss(outputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.opt_scheduler.clip)
            self.opt_scheduler.step()
            self._accumulate_loss_dic(epoch_loss_dic, outputs)
            batch_loss_dic = self._write_loss_to_dic(outputs)
            self.summary_writers.write_task('train', batch_loss_dic,
                                            self.train_step)
            if self.run_logger is not None:
                self.run_logger.log_task_metrics('train', batch_loss_dic, self.train_step)
            self.train_step += 1
        return epoch_loss_dic

    def _sum_parallel_loss(self, loss):
        if self.parallel:
            if isinstance(loss, tuple):
                return tuple([x.mean() for x in loss])
            else:
                return loss.mean()
        else:
            return loss

    def eval(self):
        self.model.eval()
        self.param_scheduler.eval()
        epoch_loss_dic = self._init_loss_dic()

        for i, batch in enumerate(self.data_loaders.val_loader):
            inputs = self._batch_to_inputs(batch)
            input_params = self.param_scheduler.step()
            with torch.no_grad():
                outputs = self.model('train', *inputs, **input_params)
                outputs = self._sum_parallel_loss(outputs)
            self._accumulate_loss_dic(epoch_loss_dic, outputs)
            batch_loss_dic = self._write_loss_to_dic(outputs)
            self.summary_writers.write_task('val', batch_loss_dic,
                                            self.val_step)
            if self.run_logger is not None:
                self.run_logger.log_task_metrics('val', batch_loss_dic, self.val_step)
            self.val_step += 1
        return epoch_loss_dic

    def _model_state_dict(self):
        if self.parallel:
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _load_model_state_dict(self, state_dict):
        cleaned = {}
        for key, value in state_dict.items():
            cleaned[key.replace('module.', '')] = value
        if self.parallel:
            self.model.module.load_state_dict(cleaned)
        else:
            self.model.load_state_dict(cleaned)

    def save_model(self, fn):
        torch.save(self._model_state_dict(), fn)

    def _param_scheduler_steps(self):
        return {
            key: getattr(scheduler, '_step', 0)
            for key, scheduler in self.param_scheduler.schedulers.items()
        }

    def _restore_param_scheduler_steps(self, steps):
        for key, step in (steps or {}).items():
            if key in self.param_scheduler.schedulers:
                self.param_scheduler.schedulers[key]._step = int(step)

    def _state_checkpoint_path(self, kind):
        return os.path.join(self.model_path, f'{self.name}_{kind}_state.pt')

    def _training_state_payload(self, best_valid_loss, config=None):
        return {
            'model_state_dict': self._model_state_dict(),
            'optimizer_state_dict': self.opt_scheduler.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.opt_scheduler.scheduler.state_dict(),
            'optimizer_scheduler_step': getattr(self.opt_scheduler, '_step', 0),
            'param_scheduler_steps': self._param_scheduler_steps(),
            'epoch': self.epoch,
            'train_step': self.train_step,
            'val_step': self.val_step,
            'best_valid_loss': best_valid_loss,
            'config': config or {},
            'rng_state': capture_rng_state(),
        }

    def save_training_state_checkpoint(self, kind, best_valid_loss,
                                       config=None):
        state_path = self._state_checkpoint_path(kind)
        save_training_state(
            state_path,
            self._training_state_payload(best_valid_loss, config=config),
        )
        print(f'[checkpoint] Saved training state ({kind}): {state_path}',
              flush=True)
        if self.run_logger is not None:
            self.run_logger.log_checkpoint(kind, state_path)
        return state_path

    def load_training_state_checkpoint(self, path):
        payload = load_training_state(path, map_location=self.device)
        self._load_model_state_dict(payload['model_state_dict'])
        self.opt_scheduler.optimizer.load_state_dict(
            payload['optimizer_state_dict']
        )
        self.opt_scheduler.scheduler.load_state_dict(
            payload['lr_scheduler_state_dict']
        )
        self.opt_scheduler._step = int(payload['optimizer_scheduler_step'])
        self._restore_param_scheduler_steps(payload['param_scheduler_steps'])
        self.epoch = int(payload['epoch'])
        self.train_step = int(payload['train_step'])
        self.val_step = int(payload['val_step'])
        restore_rng_state(payload['rng_state'])
        best_valid_loss = payload.get('best_valid_loss')
        if best_valid_loss is None:
            best_valid_loss = float('inf')
        print(
            f'[resume] Loaded training state from {path} | '
            f'epoch={self.epoch} train_step={self.train_step} '
            f'val_step={self.val_step} best_valid_loss={best_valid_loss}',
            flush=True,
        )
        return {
            'epoch': self.epoch,
            'train_step': self.train_step,
            'val_step': self.val_step,
            'best_valid_loss': best_valid_loss,
            'config': payload.get('config', {}),
        }

    def epoch_report(self, start_time, end_time, train_loss, valid_loss):
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {self.epoch + 1:02} | '
              f'Time: {epoch_mins}m {epoch_secs}s',
              flush=True)
        print(
            f'\tTrain Loss: {train_loss:.3f}', flush=True)
        print(
            f'\t Valid. Loss: {valid_loss:.3f}', flush=True)
        if self.run_logger is not None:
            self.run_logger.log_epoch_metrics(self.epoch + 1, train_loss, valid_loss, epoch_mins, epoch_secs)

    def run(self, start_epoch=0, start_train_step=0, start_val_step=0,
            best_valid_loss=float('inf'), max_epochs_this_job=None,
            checkpoint_config=None):
        self.epoch = start_epoch
        self.train_step = start_train_step
        self.val_step = start_val_step
        if best_valid_loss is None:
            best_valid_loss = float('inf')

        epochs_run_this_job = 0
        while self.epoch < self.n_epoch:
            if max_epochs_this_job is not None and                     epochs_run_this_job >= max_epochs_this_job:
                break
            start_time = time.time()
            train_loss = self.train()['loss']
            val_loss = self.eval()['loss']
            end_time = time.time()
            epoch_model_path = self.path_mng.epoch_model_path(self.name)
            self.save_model(epoch_model_path)
            next_epoch = self.epoch + 1
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                valid_model_path = self.path_mng.valid_model_path(self.name)
                self.save_model(valid_model_path)
                if self.run_logger is not None:
                    self.run_logger.log_checkpoint('valid', valid_model_path)
            self.epoch_report(start_time, end_time, train_loss, val_loss)
            self.epoch = next_epoch
            self.save_training_state_checkpoint(
                'epoch-state', best_valid_loss, config=checkpoint_config
            )
            self.save_training_state_checkpoint(
                'last-state', best_valid_loss, config=checkpoint_config
            )
            epochs_run_this_job += 1

        final_model_path = self.path_mng.final_model_path(self.name)
        self.save_model(final_model_path)
        if self.run_logger is not None:
            self.run_logger.log_checkpoint('final', final_model_path)
        self.save_training_state_checkpoint(
            'final-state', best_valid_loss, config=checkpoint_config
        )
        print('Model saved.')




