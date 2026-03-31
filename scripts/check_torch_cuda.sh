#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
else
  echo "[phase3][erro] Nenhum interpretador Python encontrado (python/python3)." >&2
  exit 127
fi

"${PYTHON_CMD[@]}" - <<'PY'
import json
import socket
import torch

info = {
    "hostname": socket.gethostname(),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
}
print(json.dumps(info, indent=2))
PY
