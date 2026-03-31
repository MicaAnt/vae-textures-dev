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

WHEEL_DIR="${1:-${WHEEL_DIR:-}}"
if [[ -z "$WHEEL_DIR" ]]; then
  echo "Uso: scripts/install_torch_from_wheels.sh /caminho/para/torch_wheels_cp311" >&2
  exit 2
fi

if [[ ! -d "$WHEEL_DIR" ]]; then
  echo "[phase3][erro] Diretorio de wheels nao encontrado: $WHEEL_DIR" >&2
  exit 2
fi

if ! find "$WHEEL_DIR" -maxdepth 1 -name 'torch-2.5.1+cu12*-cp311-cp311-linux_x86_64.whl' | grep -q .; then
  echo "[phase3][erro] Nao encontrei wheel do torch cp311 CUDA em $WHEEL_DIR" >&2
  exit 3
fi

echo "[phase3] Python: ${PYTHON_CMD[*]}"
echo "[phase3] Diretorio de wheels: $WHEEL_DIR"
echo "[phase3] Vou reinstalar torch/torchvision/torchaudio a partir dos wheels locais."
read -r -p "Continuar? [y/N] " answer
if [[ ! "$answer" =~ ^[Yy]$ ]]; then
  echo "Cancelado."
  exit 0
fi

echo "[phase3] Removendo instalacoes atuais"
"${PYTHON_CMD[@]}" -m pip uninstall -y torch torchvision torchaudio || true

echo "[phase3] Instalando offline a partir dos wheels locais"
"${PYTHON_CMD[@]}" -m pip install \
  --no-index \
  --find-links "$WHEEL_DIR" \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124

echo
echo "[phase3] Verificacao apos instalacao:"
/workspace/vae-textures-dev/scripts/check_torch_cuda.sh

echo
echo "[phase3] Se torch_cuda_version != null, o proximo passo e fazer seu fluxo dcli e rerrodar o probe no cluster."
