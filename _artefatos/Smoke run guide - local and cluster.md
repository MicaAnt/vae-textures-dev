# Smoke run guide - local and cluster

Este documento resume os comandos minimos para rodar smoke tests do treino POP909 original, localmente e no cluster.

O objetivo aqui e evitar depender do historico do chat.

## 1. Smoke local sem W&B

Use quando quiser checar rapidamente se o `train.py` continua funcionando no seu ambiente local.

```bash
cd /workspace/vae-textures-dev/base_model

VAE_BATCH_SIZE=2 \
VAE_N_EPOCH=1 \
VAE_LIMIT_TRAIN_SAMPLES=4 \
VAE_LIMIT_VAL_SAMPLES=2 \
VAE_RUN_NAME=local-smoke \
python -u train.py
```

Esperado:
- o dataset POP909 e carregado
- aparece `Epoch: 01`
- aparecem `Train Loss` e `Valid. Loss`
- termina com `Model saved.`

## 2. Smoke local com W&B offline

Use quando quiser validar a integracao do W&B sem depender da internet nem de credenciais reais.

```bash
cd /workspace/vae-textures-dev/base_model

WANDB_ENABLED=1 \
WANDB_MODE=offline \
WANDB_PROJECT=pop909-reproduction \
VAE_BATCH_SIZE=2 \
VAE_N_EPOCH=1 \
VAE_LIMIT_TRAIN_SAMPLES=4 \
VAE_LIMIT_VAL_SAMPLES=2 \
VAE_RUN_NAME=wandb-local-smoke \
python -u train.py
```

Esperado:
- W&B inicializa em modo offline
- metricas de treino/validacao aparecem no resumo do run
- checkpoints `valid` e `final` sao salvos no diretorio de resultado
- o run fica registrado localmente em `result_.../wandb/`

## 3. Preparacao do cluster

### 3.1. Arquivo privado de ambiente do W&B no cluster

No cluster, fora do container, mantenha este arquivo privado no seu home:

```bash
~/.config/wandb/env.sh
```

Exemplo de conteudo:

```bash
export WANDB_API_KEY='SUA_CHAVE_AQUI'
export WANDB_PROJECT='pop909-reproduction'
export WANDB_ENTITY='micael-antunes-lis-cnrs'
export WANDB_CACHE_DIR='/tmp/wandb-cache'
export WANDB_CONFIG_DIR='/tmp/wandb-config'
export WANDB_DATA_DIR='/tmp/wandb-data'
export WANDB_DIR='/workspace/base_model'
```

Proteja o arquivo:

```bash
chmod 600 ~/.config/wandb/env.sh
```

### 3.2. Entrar no container GPU do cluster

No cluster, fora do container:

```bash
cd ~/vae-textures-dev
source ~/.config/wandb/env.sh
scripts/cluster_gpu_shell.sh
```

Esse script:
- sobe o container com GPU
- monta `libcuda.so.1`, `libnvidia-ml.so.1` e `libnvidia-ptxjitcompiler.so.1`
- configura `LD_LIBRARY_PATH`
- redireciona cache/config/data do W&B para `/tmp`

Quando o prompt mudar para algo como:

```bash
micael.antunes@lisnodeX:/workspace$
```

voce esta dentro do container no cluster.

## 4. Smoke no cluster sem W&B

Ja dentro do container no cluster:

```bash
cd /workspace/base_model
./run_cluster_proof.sh
```

O script usa por padrao:
- `VAE_BATCH_SIZE=2`
- `VAE_N_EPOCH=1`
- `VAE_LIMIT_TRAIN_SAMPLES=4`
- `VAE_LIMIT_VAL_SAMPLES=2`
- `VAE_RUN_NAME=phase3-cluster-proof`

Se quiser sobrescrever algum valor:

```bash
cd /workspace/base_model
VAE_RUN_NAME=cluster-smoke-alt ./run_cluster_proof.sh
```

## 5. Smoke no cluster com W&B online

Ja dentro do container no cluster:

```bash
cd /workspace/base_model

WANDB_ENABLED=1 \
VAE_RUN_NAME=wandb-cluster-smoke \
./run_cluster_proof.sh
```

Importante:
- `WANDB_PROJECT`, `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_DATA_DIR`, etc. ja devem ter vindo do:

```bash
source ~/.config/wandb/env.sh
```

Esperado:
- W&B faz login com `WANDB_API_KEY`
- aparece URL do projeto e do run
- o treino completa com `Epoch: 01`
- aparecem `Train Loss` e `Valid. Loss`
- termina com `Model saved.`
- o resumo final do W&B mostra metricas
- artifacts/checkpoints sao sincronizados

## 6. Onde olhar o resultado

### No filesystem local ou no cluster
Cada run cria um diretorio do tipo:

```bash
result_YYYY-MM-DD_HHMMSS/
```

Dentro dele, normalmente voce encontra:
- `models/*_epoch.pt`
- `models/*_valid.pt`
- `models/*_final.pt`
- `wandb/` quando W&B esta habilitado
- `writers/` com os logs do writer atual

### No W&B
Procure o projeto:

```text
pop909-reproduction
```

E confira no run:
- config
- metricas `train/*`
- metricas `val/*`
- metricas `epoch/*`
- artifacts do checkpoint quando habilitados

## 7. Quando precisa de novo push da imagem

### Precisa de `dcli push`
Quando voce muda a imagem/base do ambiente, por exemplo:
- instala ou troca PyTorch
- instala bibliotecas Python do ambiente
- muda a imagem que sera empacotada

### Nao precisa de `dcli push`
Quando voce muda apenas:
- o comando `srun`
- `--container-mounts`
- `LD_LIBRARY_PATH`
- `WANDB_*` via `source ~/.config/wandb/env.sh`
- scripts do repo que ja estao presentes no cluster

## 8. Arquivos principais deste fluxo

- `base_model/train.py`
- `base_model/run_cluster_proof.sh`
- `base_model/cluster_cuda_probe.py`
- `base_model/wandb_helper.py`
- `scripts/cluster_gpu_shell.sh`
- `_artefatos/Phase 3 - cluster GPU diagnosis and proof run.md`
- `_artefatos/Phase 4 - wandb run flow.md`
