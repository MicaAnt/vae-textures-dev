# GitLab Repo       : https://gitlab.lis-lab.fr/sicomp/dcli
# Tutos             : https://gitlab.lis-lab.fr/sicomp/devcontainer_formation
# Contact us        : https://mattermost.lis-lab.fr/cluster

# Rodando train_smoke_cpu

# Questões

-[] Como que ele está realizando o batch?

# 04/03/2026

Rodei o código

```
python train_smoke_cpu_002.py --epochs 1 --max-steps 5 --batch-size 1 --log-every 1 --limit-train-samples 8 --limit-val-samples 4 --output-dir base_model/result_smoke_cpu
```

e obtive a saída

```
Usando dispositivo: cpu
The folder contains 886 .npz files.
Selected 858 files, all are in duple meter.
702756 7718
Train samples usados: 8 | Val samples usados: 4
Passos máximos de treino: 5
Exemplo do batch de treino (shapes):
  x: (1, 32, 16, 6) | c: (1, 8, 36) | pr_mat: (1, 32, 128)
  x[0,0,0,:6] = [128, 2, 2, 2, 2, 2]
  c[0,0,:6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[train] step=001 loss=10.8182 recon=5.1788 kl=0.0020 chord=5.6391
[train] step=002 loss=10.4731 recon=4.5815 kl=0.0130 chord=5.8902
[train] step=003 loss=8.9283 recon=3.1819 kl=0.0709 chord=5.7393
[train] step=004 loss=10.1398 recon=4.3328 kl=0.5389 chord=5.7531
[train] step=005 loss=9.3804 recon=3.5908 kl=0.2575 chord=5.7638
[val] loss=8.7095 recon=3.1168 kl=0.0704
Checkpoint final salvo em: base_model/result_smoke_cpu/checkpoints/final_smoke.pt
Tempo total (s): 197.10
```

- O código aparentemente rodou do começo ao fim, isso é bom.
- Eu não achei nenhum arquivo salvo, isso é bizarro.
- Eu ainda estou perdido com todas as informações de treino. Vamos aos poucos. 

# 03/03/2026

## Erro obtido na ultima versão do código

python train_smoke_cpu.py --epochs 1 --max-steps 50 --batch-size 2 --log-every 10 --limit-train-samples 100 --limit-val-samples 20 --output-dir base_model/result_smoke_cpu
Usando dispositivo: cpu
The folder contains 886 .npz files.
Selected 858 files, all are in duple meter.
702756 7718
Train samples usados: 100 | Val samples usados: 20
Passos máximos de treino: 50
Exemplo do batch de treino (shapes):
  x: (2, 32, 16, 6) | c: (2, 8, 36) | pr_mat: (2, 32, 128)
  x[0,0,0,:6] = [128, 2, 2, 2, 2, 2]
  c[0,0,:6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[train] step=001 loss=10.8451 recon=5.1778 kl=0.0020 chord=5.6671
[train] step=010 loss=8.6676 recon=2.7292 kl=0.0461 chord=5.9337
[train] step=020 loss=6.6539 recon=2.8249 kl=0.5975 chord=3.7693
Checkpoint salvo em: base_model/result_smoke_cpu/checkpoints/epoch01_step025.pt
[train] step=030 loss=7.8941 recon=2.4915 kl=0.3929 chord=5.3633
[train] step=040 loss=6.6296 recon=2.5561 kl=0.1719 chord=4.0563
[train] step=050 loss=3.5508 recon=2.1706 kl=1.1768 chord=1.2624
Checkpoint salvo em: base_model/result_smoke_cpu/checkpoints/epoch01_step050.pt
[val] loss=12.6086 recon=2.5170 kl=0.5654
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/torch/serialization.py", line 850, in save
    _save(
  File "/usr/local/lib/python3.11/site-packages/torch/serialization.py", line 1114, in _save
    zip_file.write_record(name, storage, num_bytes)
RuntimeError: [enforce fail at inline_container.cc:778] . PytorchStreamWriter failed writing file data/34: file write failed

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/workspace/vae-tuning-texutres/base_model/train_smoke_cpu.py", line 170, in <module>
    main()
  File "/workspace/vae-tuning-texutres/base_model/train_smoke_cpu.py", line 163, in main
    torch.save(model.state_dict(), final_ckpt)
  File "/usr/local/lib/python3.11/site-packages/torch/serialization.py", line 849, in save
    with _open_zipfile_writer(f) as opened_zipfile:
  File "/usr/local/lib/python3.11/site-packages/torch/serialization.py", line 690, in __exit__
    self.file_like.write_end_of_file()
RuntimeError: [enforce fail at inline_container.cc:603] . unexpected pos 53126464 vs 53126352