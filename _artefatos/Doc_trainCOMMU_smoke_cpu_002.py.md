# Documentação técnica — `trainCOMMU_smoke_cpu_002.py`

Este documento descreve o script `base_model/trainCOMMU_smoke_cpu_002.py` para facilitar sua evolução para um **configurable experiment runner**.

## 1) Objetivo do script

O script implementa um **smoke test de treino em CPU** para o modelo VAE de disentanglement musical. Ele:

1. Parseia argumentos de linha de comando.
2. Inicializa seed e dispositivo (`cpu`).
3. Constrói o modelo (`DisentangleVAE`) com encoders/decoders específicos.
4. Monta loaders reduzidos (subsets pequenos) de treino/validação.
5. Executa um loop curto de treino com limite de passos.
6. Faz uma validação rápida em 1 batch.
7. Salva checkpoints periódicos e final.

---

## 2) Dependências

## 2.1 Dependências da biblioteca padrão (Python)

- `os`: manipulação de paths e diretórios de saída/checkpoints.
- `sys`: ajuste de `sys.path` para importar módulos do diretório pai.
- `argparse`: parsing dos argumentos CLI.
- `random`: seed para aleatoriedade do Python.
- `time`: medição do tempo total da execução.

## 2.2 Dependências externas (pip/conda)

- `numpy` (`np`): seed do NumPy.
- `torch`: tensores, dispositivo, autograd, otimização, checkpoint.
  - `torch.optim` (`Adam`): otimizador.
  - `torch.utils.data.DataLoader`: dataloaders.
  - `torch.utils.data.Subset`: subset de datasets para smoke test.

## 2.3 Dependências internas do projeto

- `model.DisentangleVAE`: classe principal do VAE.
- `dl_modules`:
  - `ChordEncoder`
  - `ChordDecoder`
  - `TextureEncoder`
  - `PianoTreeDecoder`
- `dataset_loaders.MusicDataLoaders`: fábrica de loaders do dataset.
- `dataset.SEED`: seed global usada na construção dos loaders.

## 2.4 Dependências de dados/estrutura esperadas

- O script presume que os datasets e metadados necessários já estão acessíveis conforme a implementação de `MusicDataLoaders.get_loaders(...)`.
- Presume formato de batch compatível com o unpack:
  - `_, _, pr_mat, x, c, _ = batch`
- Presume que `DisentangleVAE.__call__('train', ...)` retorne uma tupla/indexável com perdas em posições específicas (0, 1, 4, 7).

---

## 3) Funções: responsabilidade, entradas e saídas

## 3.1 `set_seed(seed: int) -> None`

**O que faz**
- Define sementes de aleatoriedade para:
  - Python (`random.seed`)
  - NumPy (`np.random.seed`)
  - PyTorch (`torch.manual_seed`)

**Entradas**
- `seed` (`int`): valor da semente.

**Saída**
- `None`.

**Observações**
- Não configura flags adicionais de determinismo do backend, pois o script roda em CPU smoke test e mantém simplicidade.

---

## 3.2 `build_model(device: torch.device) -> DisentangleVAE`

**O que faz**
- Instancia os módulos de encoder/decoder usados no pipeline.
- Monta a instância de `DisentangleVAE` com nome fixo (`'disvae-smoke-cpu'`) e com os módulos mapeados para os parâmetros esperados pela classe.

**Entradas**
- `device` (`torch.device`): dispositivo alvo (no script atual, sempre CPU).

**Saída**
- Objeto `DisentangleVAE` inicializado (sem carregar pesos pré-treinados).

**Detalhes de arquitetura definidos internamente**
- `ChordEncoder(36, 1024, 256)`
- `TextureEncoder(256, 1024, 256, 10)`
- `ChordDecoder(z_dim=256)`
- `PianoTreeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=512)`

**Acoplamento relevante**
- O comentário indica compatibilidade com a família de módulos usada por `interface.py`.
- `txt_encoder` é passado como `rhy_encoder` e `pnotree_decoder` como `decoder` na assinatura de `DisentangleVAE`.

---

## 3.3 `make_small_loaders(batch_size: int, limit_train_samples: int, limit_val_samples: int)`

**O que faz**
- Cria loaders de treino e validação a partir de `MusicDataLoaders.get_loaders(...)`.
- Recorta datasets para subsets pequenos (smoke test) com limites máximos fornecidos.
- Retorna novos `DataLoader`s sem shuffle.

**Entradas**
- `batch_size` (`int`): tamanho do batch em treino e validação.
- `limit_train_samples` (`int`): máximo de amostras de treino.
- `limit_val_samples` (`int`): máximo de amostras de validação.

**Saídas**
- `train_loader` (`DataLoader`): loader com subset de treino.
- `val_loader` (`DataLoader`): loader com subset de validação.
- `train_count` (`int`): número real de amostras de treino usadas.
- `val_count` (`int`): número real de amostras de validação usadas.

**Parâmetros fixos internos do dataset**
- `portion=8`
- `shift_low=-6`, `shift_high=5`
- `num_bar=2`
- `contain_chord=True`
- `random_train=False`, `random_val=False`

Isso reforça o caráter determinístico/controlado do smoke test.

---

## 3.4 `batch_to_inputs(batch, device: torch.device)`

**O que faz**
- Converte um batch bruto do dataset para os tensores de entrada do modelo.
- Seleciona apenas os componentes necessários ao forward.
- Move para o dispositivo e ajusta dtype.

**Entradas**
- `batch`: tupla/estrutura retornada pelo DataLoader.
- `device` (`torch.device`): dispositivo destino.

**Saídas**
- `x` (`torch.Tensor`, `long`): entrada simbólica/eventos para decoder.
- `c` (`torch.Tensor`, `float`): representação de acordes.
- `pr_mat` (`torch.Tensor`, `float`): matriz de piano-roll/textura.

**Contrato implícito de batch**
- Espera layout com pelo menos 6 itens, onde:
  - índice 2: `pr_mat`
  - índice 3: `x`
  - índice 4: `c`

---

## 3.5 `main() -> None`

**O que faz**
Orquestra a execução completa de treino/validação smoke em CPU.

**Entradas**
- Não recebe argumentos diretamente; usa `argparse`.

**Saída**
- `None` (efeitos colaterais: prints em stdout e arquivos de checkpoint em disco).

**Argumentos CLI disponíveis**
- `--batch-size` (default: `2`)
- `--epochs` (default: `1`)
- `--max-steps` (default: `50`)
- `--log-every` (default: `5`)
- `--limit-train-samples` (default: `100`)
- `--limit-val-samples` (default: `20`)
- `--lr` (default: `1e-3`)
- `--output-dir` (default: `result_smoke_cpu`)
- `--seed` (default: `3345`)

**Fluxo interno detalhado**
1. Parse de argumentos.
2. Seed e seleção de dispositivo CPU.
3. Construção do modelo e otimizador Adam.
4. Criação dos loaders pequenos (treino/val).
5. Preparação de diretórios de saída/checkpoints.
6. Inspeção do primeiro batch (shapes + sample values).
7. Loop de treino:
   - converte batch para inputs,
   - `zero_grad`,
   - forward em modo `'train'` com hiperparâmetros fixos (`tfr1/tfr2/tfr3/beta/weights`),
   - backward,
   - clip de gradiente,
   - `optimizer.step`,
   - logging periódico,
   - checkpoint a cada 25 passos.
8. Saída antecipada quando `global_step >= max_steps`.
9. Validação rápida em 1 batch (sem grad).
10. Salvamento de checkpoint final + tempo total.

---

## 4) Relação entre funções (visão sistêmica)

A relação é **pipeline-orquestrada por `main`**:

1. `main` chama `set_seed` para estabilizar execução.
2. `main` chama `build_model` para obter a arquitetura pronta.
3. `main` chama `make_small_loaders` para obter dados de treino/val em escala reduzida.
4. Dentro dos loops (treino e validação), `main` chama `batch_to_inputs` para adaptar o batch ao contrato do modelo.
5. `main` executa forward/backward/otimização/checkpoint com os objetos produzidos pelas funções anteriores.

Em termos de dependência:
- `set_seed`, `build_model`, `make_small_loaders`, `batch_to_inputs` são **funções utilitárias puras de preparação/transformação**.
- `main` é a **função coordenadora** com efeitos colaterais (I/O, treino, logs, persistência).

---

## 5) Pontos de acoplamento importantes (para refatorar em experiment runner configurável)

Os principais acoplamentos atuais que você provavelmente vai transformar em configuração externa são:

1. **Arquitetura hardcoded em `build_model`**
   - dimensões dos módulos e tipo de decoder/encoder.

2. **Parâmetros de dataset hardcoded em `make_small_loaders`**
   - `portion`, `shift_low/high`, `num_bar`, `contain_chord`, randomização.

3. **Hiperparâmetros de treinamento hardcoded no forward**
   - `tfr1=0.6`, `tfr2=0.5`, `tfr3=0.5`, `beta=0.1`, `weights=(1.0, 0.5)`.

4. **Política de checkpoint/logging fixa**
   - checkpoint a cada 25 passos, seleção de métricas por índice de saída.

5. **Contrato frágil do output do modelo por índices mágicos**
   - perdas em `outputs[0]`, `outputs[1]`, `outputs[4]`, `outputs[7]`.

---

## 6) Mapa rápido de execução (call graph simplificado)

```text
if __name__ == '__main__':
    main()

main()
 ├─ parse_args()
 ├─ set_seed(args.seed)
 ├─ build_model(device)
 ├─ make_small_loaders(args.batch_size, ...)
 ├─ batch_to_inputs(first_batch, device)
 ├─ [loop treino]
 │   ├─ batch_to_inputs(batch, device)
 │   ├─ model('train', ...)
 │   └─ optimizer step + logs + checkpoints
 ├─ [validação]
 │   ├─ batch_to_inputs(val_batch, device)
 │   └─ model('train', ... sem teacher forcing)
 └─ salva checkpoint final
```

---

## 7) Resumo executivo

O script atual já separa utilidades essenciais (seed, modelo, loaders, batch adapter), o que ajuda bastante a migração para um runner configurável. A maior parte da refatoração vai concentrar-se em:

- externalizar parâmetros hardcoded para arquivo de configuração/CLI;
- reduzir acoplamento por índices mágicos no retorno do modelo;
- padronizar os objetos de log/checkpoint/avaliação.

