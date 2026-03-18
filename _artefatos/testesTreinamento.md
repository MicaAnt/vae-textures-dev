
# 18 de Março de 2026 

# 12:45 - Update Git

## Teste 2 

### Código 

**No terminal**

```
python base_model/experiments/runner.py --config base_model/configs/commu_smoke_minimal.yaml

```

**Opção no `commu_smoke_minimal.yaml`**

```
name: commu-smoke-minimal_18032026-1158
save_every_steps: 2
output_dir: result_experiments/commu_smoke_minimal_18032026-1158
include_optimizer_state: true
```

## Saída 

No bugs...


## Teste 1

### Código 

**No terminal**

```
python base_model/experiments/runner.py --config base_model/configs/commu_smoke_minimal.yaml

```

**Opção no `commu_smoke_minimal.yaml`**

```
name: commu-smoke-minimal_18032026-1158
output_dir: result_experiments/commu_smoke_minimal_18032026-1158
include_optimizer_state: true
```

## Saída 

No bugs...


## Ação

Vou começar vendo o script commu_smoke_minimal e checkpoint pra ver como eles salvam meus arquivos...


# 17 de Março de 2026

## Teste 1 - 13:09

### Código no terminal

```
python base_model/experiments/runner.py --config base_model/configs/commu_smoke_minimal.yaml
```

### Saída

```
Iniciando experimento: commu-smoke-minimal | device=cpu
The folder contains 8924 .npz files.
Selected 8924 files, all are in quadruple meter.
526248 5603
[train] epoch=0 step=1 loss=10.6615 recon=5.1742 kl=0.0014 chord=5.4872
[train] epoch=0 step=2 loss=10.6524 recon=4.8635 kl=0.0097 chord=5.7879
[val] epoch=0 step=2 val_loss=10.4220
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/torch/serialization.py", line 850, in save
    _save(
  File "/usr/local/lib/python3.11/site-packages/torch/serialization.py", line 1114, in _save
    zip_file.write_record(name, storage, num_bytes)
RuntimeError: [enforce fail at inline_container.cc:778] . PytorchStreamWriter failed writing file data/223: file write failed

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/workspace/vae-tuning-texutres/base_model/experiments/runner.py", line 152, in <module>
    main()
  File "/workspace/vae-tuning-texutres/base_model/experiments/runner.py", line 139, in main
    save_checkpoint(paths['last'], model, optimizer, epoch, global_step, best_val)
  File "/workspace/vae-tuning-texutres/base_model/experiments/checkpoint.py", line 14, in save_checkpoint
    torch.save(payload, path)
  File "/usr/local/lib/python3.11/site-packages/torch/serialization.py", line 849, in save
    with _open_zipfile_writer(f) as opened_zipfile:
  File "/usr/local/lib/python3.11/site-packages/torch/serialization.py", line 690, in __exit__
    self.file_like.write_end_of_file()
RuntimeError: [enforce fail at inline_container.cc:603] . unexpected pos 140059712 vs 140059604


```
## Teste 2 - 13:46

### Modificação nos seguintes arquivos

- checkpoint.py
- runner.py
- commu_smoke_minimal.yaml

### Código 

**No terminal**

```
python base_model/experiments/runner.py --config base_model/configs/commu_smoke_minimal.yaml

```

**Opção no `commu_smoke_minimal.yaml`**

```
include_optimizer_state: false
```

### Saída

Rodou tudo certo, com saida em `result_experiments`

## 14:04 - Update no Github

## Teste 3 - 14:50 

### Código 

**No terminal**

```
python base_model/experiments/runner.py --config base_model/configs/commu_smoke_minimal.yaml

```

**Opção no `commu_smoke_minimal.yaml`**

```
name: commu-smoke-minimal_17032026_1449
include_optimizer_state: true
```

## Saída 

-> Aparentemente ele não salvou a saída! :/
