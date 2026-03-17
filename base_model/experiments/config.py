import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if lowered == 'null':
        return None
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _apply_overrides(config: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    output = copy.deepcopy(config)
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'Override inválido: {item}. Use chave.subchave=valor')
        dotted_key, raw_value = item.split('=', 1)
        keys = dotted_key.split('.')
        cursor = output
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = _parse_scalar(raw_value)
    return output


def load_config(config_path: str, overrides: list[str] | None = None) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'Arquivo de config não encontrado: {config_path}')

    raw = path.read_text(encoding='utf-8')
    if path.suffix.lower() in {'.yaml', '.yml'}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError('PyYAML não disponível para ler .yaml/.yml') from exc
        config = yaml.safe_load(raw)
    elif path.suffix.lower() == '.json':
        config = json.loads(raw)
    else:
        raise ValueError('Formato de config suportado: .yaml/.yml/.json')

    if not isinstance(config, dict):
        raise ValueError('Config precisa ser um objeto de topo (dict/map).')

    if overrides:
        config = _apply_overrides(config, overrides)
    return config


def save_config_snapshot(config: Dict[str, Any], output_dir: str) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshot_path = out / 'resolved_config.json'
    snapshot_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    return str(snapshot_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Configurable Experiment Runner para COMMU')
    parser.add_argument('--config', type=str, required=True, help='Path para config .yaml/.json')
    parser.add_argument(
        '--set',
        nargs='*',
        default=[],
        help='Overrides pontuais no formato chave.subchave=valor',
    )
    return parser

