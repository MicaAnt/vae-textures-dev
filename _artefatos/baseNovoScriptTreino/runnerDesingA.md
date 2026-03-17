# Diferenças entre `trainCOMMU_smoke_cpu_002.py` e `train.py` + proposta de Configurable Experiment Runner

> Objetivo deste documento: explicar, de forma prática, o que cada script já resolve, o que falta para um runner configurável, e como incorporar estratégias de **freeze/unfreeze** mantendo o foco em treino com o **COMMU dataset** (base de dados do smoke).

---

## 1) Visão geral dos dois scripts

## `trainCOMMU_smoke_cpu_002.py` (foco COMMU + smoke/controlado)

**Pontos fortes**
- Usa pipeline de dados COMMU (`datasetCOMMU` via `MusicDataLoaders`/`dataset_loaders`).
- Define subsets pequenos de treino/val (`limit_train_samples`, `limit_val_samples`) para ciclos rápidos.
- Treina com CPU por padrão (bom para debug/reprodutibilidade inicial).
- Faz checkpoint periódico e final (`.pt`).
- É simples de entender e modificar.

**Limitações**
- Vários hiperparâmetros estão hardcoded na chamada do forward (`tfr1/tfr2/tfr3/beta/weights`).
- Sem scheduler de parâmetros de treino.
- Sem arquitetura de experimentos (nome/versionamento/metadata de run) mais formal.
- Sem suporte nativo a freeze/unfreeze de módulos.

---

## `train.py` (pipeline original mais completo)

**Pontos fortes**
- Estrutura de treino completa com `TrainingVAE` e `TrainingInterface`.
- Usa `ParameterScheduler` para `tfr1/tfr2/tfr3/beta/weights`.
- Inclui scheduler de LR, logs e organização de treino por época.
- Mais próximo de um runner “de produção” do código original.

**Limitações para o seu caso**
- Não é voltado ao fluxo COMMU smoke que você quer preservar como base de dados/referência.
- Menos transparente para experimentos rápidos de engenharia (mais indireção em classes e schedulers).
- Configuração ainda está em variáveis fixas no próprio script (não em arquivo de config).

---

## 2) Diferenças-chave por dimensão

## 2.1 Dados (o mais importante para você)
- **Base desejada por você:** manter o comportamento de dados do `trainCOMMU_smoke_cpu_002.py`.
- Isso implica preservar:
  - `MusicDataLoaders.get_loaders(..., portion=8, shift_low=-6, shift_high=5, num_bar=2, contain_chord=True, random_train=False, random_val=False)` (ou tornar isso configurável, mas com defaults equivalentes).
  - possibilidade de limitar número de amostras para smoke/debug.

**Conclusão prática:**
- Para seu runner, **a camada de dados deve nascer do script COMMU smoke**.

## 2.2 Loop de treino
- `trainCOMMU_smoke_cpu_002.py`: loop explícito (fácil de customizar freeze/unfreeze).
- `train.py`: loop encapsulado em `TrainingVAE` (melhor para consistência, pior para customizações rápidas sem mexer na infra).

**Conclusão prática:**
- Para estratégia de freezing rica, o loop explícito do smoke tende a acelerar evolução.

## 2.3 Hiperparâmetros dinâmicos
- Smoke: valores fixos (hardcoded).
- Original: schedulers para TFR e beta.

**Conclusão prática:**
- Reaproveitar a **ideia** de scheduler do `train.py`, mas aplicar em runner próprio sobre base COMMU.

## 2.4 Logging/checkpoint
- Smoke: simples e direto (stdout + `.pt`).
- Original: logging mais integrado via interfaces.

**Conclusão prática:**
- Começar simples (stdout + JSON + checkpoints `.pt`) e evoluir depois.

---

## 3) O que aproveitar de cada script para o Configurable Runner

## Aproveitar de `trainCOMMU_smoke_cpu_002.py`
1. **Construção de dataloaders COMMU** (base do seu requisito).
2. **Funções utilitárias claras** (`set_seed`, `build_model`, `batch_to_inputs`).
3. **Loop explícito** de treino/val para injetar lógica de freeze/unfreeze.
4. **Estratégia de smoke/debug** (limites de amostras e passos).

## Aproveitar de `train.py`
1. **Conceito de schedulers** para `tfr1/tfr2/tfr3/beta/weights`.
2. **Parâmetros de treino organizados** por bloco lógico.
3. **Separação de responsabilidades** (dados, otimização, parâmetros dinâmicos).

---

## 4) Estratégias de freezing que seu runner deve suportar

Você listou as possibilidades abaixo; aqui está como encaixar no runner:

1. **No freeze (full fine-tune)**
   - `requires_grad=True` para todos os parâmetros.

2. **Freeze both encoders; train only decoders**
   - congelar: `chd_encoder`, `rhy_encoder`
   - treinar: `decoder`, `chd_decoder`

3. **Freeze decoders; train only encoders**
   - congelar: `decoder`, `chd_decoder`
   - treinar: `chd_encoder`, `rhy_encoder`

4. **Freeze only chd_encoder**
5. **Freeze only rhy_encoder**
6. **Freeze only chd_decoder**
7. **Freeze only decoder**
   - casos unitários por módulo.

8. **Gradual unfreezing**
   - iniciar com conjunto congelado e, após N épocas/steps, destravar módulos por estágio.
   - exemplo: epoch 0–1: só decoders; epoch 2: destrava `rhy_encoder`; epoch 3: destrava `chd_encoder`.

9. **Partial freeze by layer name (advanced)**
   - aplicar regex/lista de nomes de parâmetros (ex.: `decoder.dec_time_gru.*`) para congelar parcialmente.

## Recomendação de implementação
- Implementar função central:
  - `apply_freeze_policy(model, policy, epoch=None, step=None)`
- E outra para rebuild do optimizer quando conjunto treinável muda:
  - `build_optimizer_for_trainable(model, lr, weight_decay, ...)`

> Observação importante: quando mudar freeze no meio do treino (gradual unfreezing), é recomendável recriar optimizer para refletir os parâmetros treináveis atuais.

---

## 5) O que seu script configurable experiment runner PRECISA ter

## 5.1 Configuração (obrigatório)
- Fonte única de configuração (YAML/JSON + CLI override).
- Blocos mínimos:
  - `experiment`: nome, seed, output_dir
  - `data`: parâmetros COMMU (portion, shifts, num_bar, limits)
  - `model`: dims/variante de encoders e decoders
  - `train`: epochs, max_steps, batch_size, lr, clip
  - `loss`: beta, weights
  - `teacher_forcing`: tfr1, tfr2, tfr3 (+ opcional schedules)
  - `freeze`: policy + cronograma de unfreeze
  - `checkpoint`: save_every_steps, save_best, save_last

## 5.2 Reprodutibilidade
- Salvar snapshot da config usada em cada run.
- Salvar estado RNG (opcional avançado).
- Definir seed no início.

## 5.3 Checkpointing `.pt`
Você pediu saída `.pt` para experimentos. Mínimo recomendado:
- `last.pt`: últimos pesos.
- `best.pt`: melhor validação.
- `epochXX_stepYYY.pt`: opcionais periódicos.
- Ideal: salvar também `optimizer_state_dict`, epoch e step.

## 5.4 Logs úteis para experimento
- `metrics.jsonl` com `step/epoch`, `loss`, `recon_loss`, `kl_loss`, `chord_loss`.
- Print resumido em stdout.
- (Opcional) TensorBoard/W&B depois.

## 5.5 Avaliação mínima por validação
- Ao menos 1 batch (como smoke) no começo.
- Ideal: validação completa do val_loader (quando custo permitir).

---

## 6) Arquitetura: 1 script só vs múltiplos arquivos

## Opção A — tudo em um script (rápido para começar)
**Prós**
- Implementação inicial muito rápida.
- Menos overhead de organização.

**Contras**
- Cresce mal quando você adiciona freeze policies/schedulers/loggers.
- Mais difícil de testar e reutilizar.

## Opção B — múltiplos arquivos (recomendado para seu objetivo)
**Prós**
- Melhor manutenção para muitos experimentos.
- Facilita adicionar freeze gradual e policies avançadas.
- Permite testes unitários por componente.

**Contras**
- Exige um pouco mais de estrutura inicial.

## Recomendação objetiva
Para o seu caso (runner configurável + várias políticas de freeze), use **múltiplos arquivos** desde o início, com estrutura enxuta:

```text
base_model/experiments/
  runner.py              # loop principal de treino/val
  config.py              # schema + leitura YAML/JSON + overrides
  model_factory.py       # build_model por config
  data_factory.py        # loaders COMMU + subsets smoke
  freeze.py              # apply_freeze_policy + gradual unfreeze
  schedulers.py          # tfr/beta schedules (ou wrappers)
  checkpoint.py          # save/load checkpoints .pt
  logging_utils.py       # jsonl + stdout
configs/
  commu_baseline.yaml
  commu_freeze_decoders.yaml
  commu_gradual_unfreeze.yaml
```

Se você quiser começar incrementalmente:
1. criar `runner.py` único funcional,
2. extrair `freeze.py` e `config.py`,
3. extrair `data_factory.py` e `model_factory.py`.

---

## 7) Plano de evolução sugerido (curto e prático)

1. **Fase 1 (baseline configurável):**
   - replicar exatamente o comportamento do `trainCOMMU_smoke_cpu_002.py`, mas lendo config.

2. **Fase 2 (freezing policies):**
   - adicionar todas as políticas fixas (no freeze, freeze encoders, etc.).

3. **Fase 3 (gradual unfreezing):**
   - cronograma por época/step com rebuild de optimizer.

4. **Fase 4 (advanced):**
   - freeze por nome de camada + schedules de TFR/Beta.

---

## 8) Decisão final para o seu contexto

Como você quer **treinar com base COMMU** e comparar muitos experimentos de fine-tuning, a estratégia mais sólida é:

- **Dados/loop baseados no `trainCOMMU_smoke_cpu_002.py`** (por requisito de dataset e controle).
- **Incorporar conceitos do `train.py`** (scheduler e organização de parâmetros).
- **Estruturar em múltiplos arquivos** (mínimo modular), para suportar freeze/unfreeze sem virar script monolítico difícil de manter.

Isso te dá velocidade agora e escalabilidade para próximos experimentos.


