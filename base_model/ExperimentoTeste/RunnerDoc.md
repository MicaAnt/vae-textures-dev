# Configurable Experiment Runner (COMMU) — Especificação e guia prático

Este documento materializa o design de `runnerDesingA.md` em uma proposta implementável com foco em:

- preservar o fluxo de dados do `trainCOMMU_smoke_cpu_002.py`;
- incorporar os ganhos estruturais do `train.py` (parâmetros dinâmicos/scheduler);
- suportar todas as variações de **freezing/unfreezing** pedidas.

## 1) O que foi consolidado dos scripts existentes

### Herdado do `trainCOMMU_smoke_cpu_002.py`
- Data pipeline COMMU com `MusicDataLoaders.get_loaders(...)`.
- Subsets limitados para smoke/debug rápido (`limit_train_samples`, `limit_val_samples`).
- Loop explícito, bom para injetar freeze policy e validações pontuais.
- Checkpoint simples e direto em `.pt`.

### Herdado do `train.py`
- Organização por blocos lógicos de parâmetros de treino.
- Conceito de parâmetros dinâmicos para `tfr1/tfr2/tfr3`, `beta` e `weights`.
- Prática de separar dados/modelo/treino para facilitar evolução.

## 2) Arquitetura implementada

```text
base_model/
  experiments/
    runner.py
    config.py
    data_factory.py
    model_factory.py
    freeze.py
    schedulers.py
    checkpoint.py
    logging_utils.py
  configs/
    commu_baseline.yaml
    commu_freeze_decoders.yaml
    commu_gradual_unfreeze.yaml
```

### Responsabilidades
- `runner.py`: orquestra treino, validação, checkpoint e reconfiguração de optimizer.
- `config.py`: leitura YAML/JSON + overrides CLI (`--set chave.subchave=valor`) + snapshot final.
- `data_factory.py`: criação de loaders COMMU + limitação de amostras para smoke.
- `model_factory.py`: montagem do `DisentangleVAE` com módulos compatíveis com o baseline.
- `freeze.py`: políticas de freeze e rebuild de optimizer para parâmetros treináveis.
- `schedulers.py`: resolução de parâmetros fixos/lineares por step ou epoch.
- `checkpoint.py`: `last.pt`, `best.pt` e checkpoints periódicos.
- `logging_utils.py`: `metrics.jsonl` + logging amigável no stdout.

## 3) Políticas de freezing cobertas

## 3.1 Prontas via `policy`
1. `no_freeze`
2. `freeze_modules`
3. `train_only_modules`
4. `gradual_unfreeze`

Com `freeze_modules` e `train_only_modules`, os módulos válidos são:
- `chd_encoder`
- `rhy_encoder`
- `decoder`
- `chd_decoder`

## 3.2 Mapeamento para os cenários solicitados

- **No freeze (full fine-tune)**
  - `policy: no_freeze`

- **Freeze both encoders; train only decoders**
  - `policy: train_only_modules`
  - `train_only_modules: [decoder, chd_decoder]`

- **Freeze decoders; train only encoders**
  - `policy: train_only_modules`
  - `train_only_modules: [chd_encoder, rhy_encoder]`

- **Freeze only chd_encoder**
  - `policy: freeze_modules`
  - `freeze_modules: [chd_encoder]`

- **Freeze only rhy_encoder**
  - `policy: freeze_modules`
  - `freeze_modules: [rhy_encoder]`

- **Freeze only chd_decoder**
  - `policy: freeze_modules`
  - `freeze_modules: [chd_decoder]`

- **Freeze only decoder**
  - `policy: freeze_modules`
  - `freeze_modules: [decoder]`

- **Gradual unfreezing**
  - `policy: gradual_unfreeze`
  - `initial_train_only_modules: [...]`
  - `unfreeze_at_epochs: [{epoch: N, modules: [...]}, ...]`

- **Partial freeze by layer name (advanced)**
  - usar `freeze_name_regex` com regex de `named_parameters()`.
  - exemplo:

```yaml
freeze:
  policy: no_freeze
  freeze_name_regex:
    - '^decoder\\.dec_time_gru\\..*'
```

> Observação: quando a assinatura dos parâmetros treináveis muda em `gradual_unfreeze`, o runner recria o optimizer automaticamente.

## 4) Configuração mínima esperada

```yaml
experiment:
  name: commu-baseline
  seed: 3345
  device: cpu
  output_dir: result_experiments/commu_baseline

data:
  portion: 8
  shift_low: -6
  shift_high: 5
  num_bar: 2
  contain_chord: true
  random_train: false
  random_val: false
  limit_train_samples: 100
  limit_val_samples: 20

model:
  chd_latent_dim: 256
  rhy_latent_dim: 256
  num_channel: 10

train:
  epochs: 1
  max_steps: 50
  batch_size: 2
  val_batch_size: 2
  clip_grad_norm: 1.0
  log_every: 5
  val_max_batches: 1
  optimizer:
    lr: 0.001
  teacher_forcing:
    tfr1: 0.6
    tfr2: 0.5
    tfr3: 0.5
  loss:
    beta: 0.1
    weights: [1.0, 0.5]

freeze:
  policy: no_freeze

checkpoint:
  save_every_steps: 25
```

## 5) Schedulers suportados

Parâmetros com schedule opcional:
- `teacher_forcing.tfr1_schedule`
- `teacher_forcing.tfr2_schedule`
- `teacher_forcing.tfr3_schedule`
- `loss.beta_schedule`
- `loss.weight0_schedule`
- `loss.weight1_schedule`

Formato atual:
- `constant`
- `linear` por `step` ou por `epoch`

Exemplo:

```yaml
train:
  teacher_forcing:
    tfr1: 0.6
    tfr1_schedule:
      type: linear
      mode: step
      start: 0.6
      end: 0.2
      total_steps: 120
```

## 6) Como executar

Da raiz do repositório:

```bash
python base_model/experiments/runner.py --config base_model/configs/commu_baseline.yaml
```

Com override rápido via CLI:

```bash
python base_model/experiments/runner.py \
  --config base_model/configs/commu_gradual_unfreeze.yaml \
  --set train.max_steps=20 train.batch_size=1 experiment.name=debug-run
```

## 7) Artefatos de saída por experimento

No `output_dir`:
- `resolved_config.json` (snapshot final da config)
- `metrics.jsonl` (linhas com métricas de treino/validação)
- `checkpoints/last.pt`
- `checkpoints/best.pt`
- `checkpoints/epochXXX_stepYYYYYY.pt` (periódicos)

## 8) Checklist de evolução futura

- adicionar scheduler de LR opcional;
- adicionar validação completa por padrão em runs não-smoke;
- carregar checkpoint para resume de treino;
- adicionar relatório final agregando curvas de loss/kl/chord.

