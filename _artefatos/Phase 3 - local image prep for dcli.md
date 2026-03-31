# Phase 3 - local image prep for dcli

Estes comandos sao para a sua maquina/local image antes do `dcli push`.

## 1. Checar o runtime local que sera empacotado

```bash
cd /workspace/vae-textures-dev
scripts/check_torch_cuda.sh
```

Se aparecer algo como:
- `torch_version = 2.5.1+cpu`
- `torch_cuda_version = null`

o ambiente local ainda esta CPU-only e isso tende a ir junto para o cluster no seu fluxo atual.

## 2. Caminho recomendado: instalar offline a partir de wheels baixados fora da imagem

Quando a imagem/workspace nao consegue baixar direto de `download.pytorch.org`, use os wheels `cp311` ja baixados em outro ambiente e instale localmente:

```bash
cd /workspace/vae-textures-dev
scripts/install_torch_from_wheels.sh /caminho/para/torch_wheels_cp311
```

O diretorio de wheels precisa conter, no minimo, estes arquivos:
- `torch-2.5.1+cu124-cp311-cp311-linux_x86_64.whl`
- `torchvision-0.20.1+cu124-cp311-cp311-linux_x86_64.whl`
- `torchaudio-2.5.1+cu124-cp311-cp311-linux_x86_64.whl`

## 3. Alternativa: tentativa de instalacao online

Se o seu ambiente conseguir acessar o indice da PyTorch diretamente, ainda e possivel tentar:

```bash
cd /workspace/vae-textures-dev
CUDA_VARIANT=cu124 scripts/fix_torch_cuda.sh
```

Alternativa se precisar testar `cu121`:

```bash
cd /workspace/vae-textures-dev
CUDA_VARIANT=cu121 scripts/fix_torch_cuda.sh
```

## 4. Depois disso

Quando o check local mostrar `torch_cuda_version != null`, siga o seu fluxo normal:
- `dcli config`
- `dcli push`

## 5. Validacao no cluster

Depois do push, no cluster:

```bash
cd /workspace/base_model
python cluster_cuda_probe.py --require-cuda
```

Se isso passar, rode a prova curta com o mesmo `train.py`.
