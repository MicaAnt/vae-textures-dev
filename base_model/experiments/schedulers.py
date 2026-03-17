
def linear_schedule(start: float, end: float, total_steps: int, step: int) -> float:
    if total_steps <= 0:
        return end
    ratio = min(max(step / float(total_steps), 0.0), 1.0)
    return start + ratio * (end - start)


def resolve_param(base_value, schedule_cfg: dict | None, global_step: int, epoch: int):
    if not schedule_cfg:
        return base_value

    schedule_type = schedule_cfg.get('type', 'constant')
    if schedule_type == 'constant':
        return schedule_cfg.get('value', base_value)

    if schedule_type == 'linear':
        start = schedule_cfg.get('start', base_value)
        end = schedule_cfg.get('end', base_value)
        mode = schedule_cfg.get('mode', 'step')
        index = global_step if mode == 'step' else epoch
        total = schedule_cfg.get('total_steps' if mode == 'step' else 'total_epochs', 1)
        return linear_schedule(start, end, total, index)

    raise ValueError(f'Schedule não suportado: {schedule_type}')


def resolve_train_params(train_cfg: dict, global_step: int, epoch: int) -> dict:
    tfr_cfg = train_cfg.get('teacher_forcing', {})
    loss_cfg = train_cfg.get('loss', {})

    tfr1 = resolve_param(tfr_cfg.get('tfr1', 0.6), tfr_cfg.get('tfr1_schedule'), global_step, epoch)
    tfr2 = resolve_param(tfr_cfg.get('tfr2', 0.5), tfr_cfg.get('tfr2_schedule'), global_step, epoch)
    tfr3 = resolve_param(tfr_cfg.get('tfr3', 0.5), tfr_cfg.get('tfr3_schedule'), global_step, epoch)

    beta = resolve_param(loss_cfg.get('beta', 0.1), loss_cfg.get('beta_schedule'), global_step, epoch)
    w0 = resolve_param(loss_cfg.get('weights', [1.0, 0.5])[0], loss_cfg.get('weight0_schedule'), global_step, epoch)
    w1 = resolve_param(loss_cfg.get('weights', [1.0, 0.5])[1], loss_cfg.get('weight1_schedule'), global_step, epoch)

    return {'tfr1': tfr1, 'tfr2': tfr2, 'tfr3': tfr3, 'beta': beta, 'weights': (w0, w1)}

