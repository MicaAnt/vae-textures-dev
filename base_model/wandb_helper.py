import os
from pathlib import Path


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


class WandbRunLogger:

    def __init__(self, wandb_module, run, checkpoint_kinds):
        self.wandb = wandb_module
        self.run = run
        self.checkpoint_kinds = set(checkpoint_kinds)

    @classmethod
    def from_env(cls, run_name, config, log_path_mng):
        if not env_flag('WANDB_ENABLED', False):
            return None

        project = os.getenv('WANDB_PROJECT')
        if not project:
            raise RuntimeError(
                'WANDB_ENABLED is set, but WANDB_PROJECT is missing. '
                'Set WANDB_PROJECT to the target Weights & Biases project.'
            )

        mode = os.getenv('WANDB_MODE', '').strip().lower() or None
        api_key = os.getenv('WANDB_API_KEY')
        if mode != 'offline' and not api_key:
            raise RuntimeError(
                'WANDB_ENABLED is set, but WANDB_API_KEY is missing. '
                'Export WANDB_API_KEY or set WANDB_MODE=offline for a local smoke test.'
            )

        import wandb

        init_kwargs = {
            'project': project,
            'name': run_name,
            'config': config,
            'dir': log_path_mng.log_path,
        }
        entity = os.getenv('WANDB_ENTITY')
        if entity:
            init_kwargs['entity'] = entity
        if mode:
            init_kwargs['mode'] = mode
        run_id = os.getenv('WANDB_RUN_ID')
        if run_id:
            init_kwargs['id'] = run_id
        resume = os.getenv('WANDB_RESUME')
        if resume:
            init_kwargs['resume'] = resume
        group = os.getenv('WANDB_GROUP')
        if group:
            init_kwargs['group'] = group
        notes = os.getenv('WANDB_NOTES')
        if notes:
            init_kwargs['notes'] = notes
        tags = [tag.strip() for tag in os.getenv('WANDB_TAGS', '').split(',') if tag.strip()]
        if tags:
            init_kwargs['tags'] = tags

        run = wandb.init(**init_kwargs)
        run.define_metric('train/step')
        run.define_metric('train/*', step_metric='train/step')
        run.define_metric('val/step')
        run.define_metric('val/*', step_metric='val/step')
        run.define_metric('epoch')
        run.define_metric('epoch/*', step_metric='epoch')
        run.config.update(
            {
                'log_path': log_path_mng.log_path,
                'writer_path': log_path_mng.writer_path,
                'model_path': log_path_mng.model_path,
            },
            allow_val_change=True,
        )

        checkpoint_policy = os.getenv(
            'WANDB_CHECKPOINT_POLICY',
            'valid,final,epoch-state,last-state,final-state',
        )
        checkpoint_kinds = [kind.strip() for kind in checkpoint_policy.split(',') if kind.strip()]
        return cls(wandb, run, checkpoint_kinds)

    def log_task_metrics(self, task, vals_dic, step):
        payload = {f'{task}/{key}': value for key, value in vals_dic.items()}
        payload[f'{task}/step'] = step
        self.run.log(payload)

    def log_epoch_metrics(self, epoch, train_loss, valid_loss, epoch_mins, epoch_secs):
        duration_seconds = (epoch_mins * 60) + epoch_secs
        self.run.log(
            {
                'epoch': epoch,
                'epoch/train_loss': train_loss,
                'epoch/valid_loss': valid_loss,
                'epoch/duration_seconds': duration_seconds,
            }
        )

    def log_checkpoint(self, kind, path):
        if kind not in self.checkpoint_kinds:
            return
        checkpoint = Path(path)
        if not checkpoint.exists():
            return

        artifact = self.wandb.Artifact(
            name=f'{self.run.name}-{kind}',
            type='model',
            metadata={
                'kind': kind,
                'path': str(checkpoint),
            },
        )
        artifact.add_file(str(checkpoint), name=checkpoint.name)
        aliases = [kind]
        if kind == 'valid':
            aliases.append('best')
        if kind == 'last-state':
            aliases.append('resume')
        self.run.log_artifact(artifact, aliases=aliases)

    def finish(self):
        self.run.finish()
