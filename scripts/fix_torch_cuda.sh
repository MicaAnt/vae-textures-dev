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

PYTHON_LABEL="${PYTHON_CMD[*]}"
PIP_CMD=("${PYTHON_CMD[@]}" -m pip)
CUDA_VARIANT="${CUDA_VARIANT:-cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
PIP_VERBOSE="${PIP_VERBOSE:-1}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

if [[ "$CUDA_VARIANT" != "cu124" && "$CUDA_VARIANT" != "cu121" ]]; then
  echo "CUDA_VARIANT precisa ser cu124 ou cu121" >&2
  exit 2
fi

INDEX_URL="https://download.pytorch.org/whl/${CUDA_VARIANT}"
TORCH_URL="${INDEX_URL}/torch/"

preflight_check() {
  echo "[phase3] Testando conectividade com $TORCH_URL"
  "${PYTHON_CMD[@]}" - <<PY
import sys
import urllib.request
url = ${TORCH_URL@Q}
try:
    with urllib.request.urlopen(url, timeout=20) as resp:
        print(f"[phase3] Preflight OK: HTTP {resp.status} em {url}")
except Exception as exc:
    print(f"[phase3][erro] Falha ao acessar {url}: {exc}", file=sys.stderr)
    sys.exit(3)
PY
}

echo "[phase3] Python: $PYTHON_LABEL"
echo "[phase3] CUDA_VARIANT: $CUDA_VARIANT"
echo "[phase3] Index URL: $INDEX_URL"
echo "[phase3] Vou substituir torch/torchvision/torchaudio pela variante CUDA correspondente."
echo "[phase3] Isso afeta o ambiente local que depois sera empacotado pelo seu fluxo dcli."
read -r -p "Continuar? [y/N] " answer
if [[ ! "$answer" =~ ^[Yy]$ ]]; then
  echo "Cancelado."
  exit 0
fi

preflight_check

PIP_INSTALL_CMD=(
  "${PYTHON_CMD[@]}" -m pip install
  "torch==${TORCH_VERSION}"
  "torchvision==${TORCHVISION_VERSION}"
  "torchaudio==${TORCHAUDIO_VERSION}"
  --index-url "$INDEX_URL"
  --progress-bar on
)

if [[ "$PIP_VERBOSE" == "1" ]]; then
  PIP_INSTALL_CMD+=( -v )
fi

if [[ -n "$PIP_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( $PIP_EXTRA_ARGS )
  PIP_INSTALL_CMD+=( "${EXTRA[@]}" )
fi

echo "[phase3] Removendo pacotes CPU-only atuais"
"${PIP_CMD[@]}" uninstall -y torch torchvision torchaudio || true

echo "[phase3] Comando de instalacao:"
printf '  %q' "${PIP_INSTALL_CMD[@]}"
printf '\n\n'

"${PIP_INSTALL_CMD[@]}"

echo
echo "[phase3] Verificacao apos instalacao:"
/workspace/vae-textures-dev/scripts/check_torch_cuda.sh

echo
echo "[phase3] Proximo passo sugerido:"
echo "  1. validar que o output agora mostra torch_cuda_version != null"
echo "  2. fazer seu fluxo de dcli config / dcli push"
echo "  3. rodar no cluster: python base_model/cluster_cuda_probe.py --require-cuda"
