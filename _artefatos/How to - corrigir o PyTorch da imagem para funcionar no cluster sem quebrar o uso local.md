# Tutorial objetivo: corrigir o PyTorch da imagem para funcionar no cluster sem quebrar o uso local

## Diagnóstico
Se este comando retorna algo como:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

```text
2.5.1+cpu
None
False
```

então a imagem está com **PyTorch CPU-only**.

Como a imagem local é a mesma que vai para o cluster, a correção deve ser feita **na imagem local** e depois essa mesma imagem deve ser enviada para o cluster.

---

## O que fazer
Troque o PyTorch atual por uma build **da mesma família de versão** com CUDA:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

Se a infraestrutura do cluster não aceitar `cu124`, tente a mesma versão com `cu121`:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Fonte oficial: <https://pytorch.org/get-started/previous-versions/>

---

## Como fazer isso sem ferrar o que já está funcionando
Não sobrescreva a imagem antiga.

Faça uma **nova tag** de imagem, por exemplo:

- imagem antiga: `fidle:cpu`
- imagem nova: `fidle:cu124`

Assim você consegue testar sem perder rollback.

---

## O que muda no uso
### Local
Continue usando CPU no YAML:

```yaml
experiment:
  device: cpu
```

Mesmo com PyTorch CUDA-enabled instalado, isso continua funcionando normalmente na sua máquina sem GPU.

### Cluster
Use a mesma imagem, mas com:

```yaml
experiment:
  device: cuda
```

---

## Validação rápida
Depois de rebuildar a imagem, rode:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### Esperado no local
Algo como:

```text
2.5.1+cu124
12.4
False
```

Isso está **correto** no local se você sempre roda em CPU.

### Esperado no cluster
Algo como:

```text
2.5.1+cu124
12.4
True
```

Se aparecer `True`, então `device: cuda` deve funcionar.

---

## Resumo em uma linha
Você **não precisa mexer no código** para resolver isso; precisa apenas reconstruir a imagem com **PyTorch 2.5.1 com CUDA**, manter `device: cpu` localmente e usar `device: cuda` no cluster.
