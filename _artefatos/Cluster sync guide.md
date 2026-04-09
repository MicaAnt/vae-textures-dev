# Cluster sync guide

## Objetivo

Sincronizar o repositorio local com o cluster sem subir resultados de treino, checkpoints pesados ou caches locais.

Scripts usados:
- `scripts/sync_to_cluster.sh`
- `.rsyncignore-cluster`

## O que o sync exclui

O arquivo `.rsyncignore-cluster` filtra por padrao:
- resultados de treino (`result_*`, `result_experiments/`, `wandb/`)
- checkpoints e pesos (`*.pt`, `*.pth`, `*.ckpt`)
- caches (`__pycache__/`, `.ipynb_checkpoints/`)
- alguns diretorios grandes de dados ja geridos separadamente

## Comandos

Ir para o repo:

```bash
cd ~/Documents/VAE-Textures/vae-textures-dev
```

### 1. Preview seguro

```bash
scripts/sync_to_cluster.sh
```

Esse e o modo padrao: `dry-run`.
Ele mostra o que seria sincronizado e o que seria removido, sem aplicar mudancas.

### 2. Sincronizacao real

```bash
scripts/sync_to_cluster.sh sync
```

Esse modo aplica a sincronizacao e usa `--delete` para limpar do cluster apenas os arquivos que estao sendo gerenciados por esse sync.

### 3. Sincronizacao sem apagar nada no destino

```bash
scripts/sync_to_cluster.sh no-delete
```

Esse modo envia atualizacoes, mas nao remove nada do cluster.

## Seguranca sobre o `--delete`

O script usa `--delete`, mas tambem usa `--exclude-from .rsyncignore-cluster`.

Importante:
- o script **nao** usa `--delete-excluded`
- por isso, caminhos excluidos nao sao apagados no cluster

Na pratica, isso protege no cluster os diretorios/arquivos excluidos pelo `.rsyncignore-cluster`, como resultados de treino e arquivos `.pt`.

## Recomendacoes

Fluxo recomendado:

```bash
scripts/sync_to_cluster.sh
scripts/sync_to_cluster.sh sync
```

Use `no-delete` quando estiver em duvida.

## Ajustes futuros

Se surgir algum diretorio novo que voce nao quer enviar para o cluster, adicione esse caminho em `.rsyncignore-cluster`.
