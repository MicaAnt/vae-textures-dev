# Phase 3 - cluster GPU diagnosis and proof run

Este documento registra o caminho operacional da Fase 3 para validar e corrigir o ambiente do cluster sem abrir um segundo caminho de treino.

## 1. Diagnostico minimo no job real do cluster

Rode no ambiente real do job:

```bash
cd /path/to/vae-textures-dev
python base_model/cluster_cuda_probe.py
```

Se quiser falhar explicitamente quando CUDA nao estiver disponivel:

```bash
cd /path/to/vae-textures-dev
python base_model/cluster_cuda_probe.py --require-cuda
```

### Como interpretar
- `torch_cuda_version = null` e `cuda_available = false`
  Isso indica fortemente build CPU-only do PyTorch.
- `torch_cuda_version != null` e `cuda_available = false`
  Isso sugere build com CUDA, mas sem GPU visivel/alocada no job.
- `cuda_available = true`
  O runtime esta apto para prova de treino em GPU.

## 2. Evidencia coletada ate agora

### Workspace local
Validacao local feita em 2026-03-30:

```json
{
  "torch_version": "2.5.1+cpu",
  "torch_cuda_version": null,
  "cuda_available": false,
  "cuda_device_count": 0
}
```

### Cluster real
Saida reportada pelo usuario em `diflives1`:

```json
{
  "hostname": "diflives1",
  "torch_version": "2.5.1+cpu",
  "torch_cuda_version": null,
  "cuda_available": false,
  "cuda_device_count": 0,
  "devices": []
}
```

Conclusao atual:
- o problema esta confirmado como build CPU-only do PyTorch no ambiente do cluster
- a proxima etapa nao e mais diagnostico, e sim reparo da imagem/runtime

## 3. Reparar a imagem/runtime

Siga a estrategia ja descrita em:
- `_artefatos/How to - corrigir o PyTorch da imagem para funcionar no cluster sem quebrar o uso local.md`

Resumo:
- reconstruir uma variante CUDA-enabled da mesma familia de versao
- usar nova tag de imagem para manter rollback
- manter uso local em CPU e usar a mesma imagem no cluster com GPU

## 4. Validacao depois do reparo

Depois de rebuildar a imagem ou atualizar o runtime, rode novamente:

```bash
cd /path/to/vae-textures-dev
python base_model/cluster_cuda_probe.py --require-cuda
```

Esperado no cluster apos o reparo:
- `torch_cuda_version != null`
- `cuda_available = true`
- `cuda_device_count > 0`

## 5. Prova curta em GPU usando o mesmo train.py

Depois que o probe reportar `cuda_available = true`, rode uma prova curta usando o mesmo entrypoint local:

```bash
cd /path/to/vae-textures-dev/base_model
VAE_BATCH_SIZE=2 \
VAE_N_EPOCH=1 \
VAE_LIMIT_TRAIN_SAMPLES=4 \
VAE_LIMIT_VAL_SAMPLES=2 \
VAE_RUN_NAME=phase3-cluster-proof \
python -u train.py
```

## 6. O que precisa ficar registrado para fechar a fase

- comando do probe no cluster apos o reparo
- saida do probe no cluster apos o reparo
- tag/versao da imagem usada no job
- comando do proof run
- evidencia de que `torch.cuda.is_available()` estava verdadeiro no job
- diretorio de resultado gerado pelo `train.py`

## 7. Fronteira da fase

Esta fase termina quando o mesmo `train.py` ja validado localmente for demonstrado no cluster com CUDA disponivel. Integracao com W&B continua sendo Fase 4.
