import re

MODULE_MAP = {
    'chd_encoder': 'chd_encoder',
    'rhy_encoder': 'rhy_encoder',
    'decoder': 'decoder',
    'chd_decoder': 'chd_decoder',
}


def _set_module_requires_grad(module, enabled: bool):
    for param in module.parameters():
        param.requires_grad = enabled


def _freeze_all(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


def _unfreeze_all(model):
    for _, param in model.named_parameters():
        param.requires_grad = True


def _apply_module_freeze(model, freeze_modules: list[str]):
    _unfreeze_all(model)
    for module_name in freeze_modules:
        attr = MODULE_MAP.get(module_name)
        if not attr:
            raise ValueError(f'Módulo desconhecido para freeze: {module_name}')
        _set_module_requires_grad(getattr(model, attr), False)


def _apply_module_train_only(model, train_only_modules: list[str]):
    _freeze_all(model)
    for module_name in train_only_modules:
        attr = MODULE_MAP.get(module_name)
        if not attr:
            raise ValueError(f'Módulo desconhecido para train_only: {module_name}')
        _set_module_requires_grad(getattr(model, attr), True)


def _apply_name_regex_freeze(model, patterns: list[str]):
    for name, param in model.named_parameters():
        for pattern in patterns:
            if re.match(pattern, name):
                param.requires_grad = False
                break


def apply_freeze_policy(model, freeze_cfg: dict, epoch: int, step: int):
    policy = freeze_cfg.get('policy', 'no_freeze')
    changed_tag = f'policy:{policy}'

    if policy == 'no_freeze':
        _unfreeze_all(model)
    elif policy == 'freeze_modules':
        _apply_module_freeze(model, freeze_cfg.get('freeze_modules', []))
    elif policy == 'train_only_modules':
        _apply_module_train_only(model, freeze_cfg.get('train_only_modules', []))
    elif policy == 'gradual_unfreeze':
        _apply_module_train_only(model, freeze_cfg.get('initial_train_only_modules', []))
        milestones = freeze_cfg.get('unfreeze_at_epochs', [])
        for milestone in milestones:
            if epoch >= milestone['epoch']:
                for module_name in milestone.get('modules', []):
                    _set_module_requires_grad(getattr(model, MODULE_MAP[module_name]), True)
        changed_tag = f'policy:{policy}@epoch{epoch}'
    else:
        raise ValueError(f'Freeze policy desconhecida: {policy}')

    regex_patterns = freeze_cfg.get('freeze_name_regex', [])
    if regex_patterns:
        _apply_name_regex_freeze(model, regex_patterns)

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    return changed_tag, trainable


def build_optimizer_for_trainable(model, optimizer_cfg: dict):
    import torch

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError('Nenhum parâmetro treinável após aplicar freeze policy.')

    return torch.optim.Adam(
        trainable,
        lr=optimizer_cfg.get('lr', 1e-3),
        weight_decay=optimizer_cfg.get('weight_decay', 0.0),
    )

