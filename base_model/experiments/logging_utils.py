import json
from pathlib import Path


class JsonlLogger:
    def __init__(self, output_dir: str):
        self.path = Path(output_dir) / 'metrics.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict):
        with self.path.open('a', encoding='utf-8') as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + '\n')


def print_train_log(prefix: str, row: dict):
    print(
        f"[{prefix}] epoch={row.get('epoch')} step={row.get('step')} "
        f"loss={row.get('loss'):.4f} recon={row.get('recon_loss'):.4f} "
        f"kl={row.get('kl_loss'):.4f} chord={row.get('chord_loss'):.4f}"
    )

