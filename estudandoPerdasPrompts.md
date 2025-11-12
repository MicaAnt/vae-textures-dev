# 24 de Junho

Entendendo que

`z_chd` - é a representação latente da harmonia e
`z_txt` - é a representação latente da textura (também as vezes chamada de ritmo no modelo)

Eu quero que você faça uma nova versão de `./compute_single_loss.py` aproveitando também a função loss_function, com as seguintes caracerísticas.

**Entrada:** um arquivo .npz, como os da pasta `./dataSet`

**Saída:** para cada segmento extraido do arquivo `.npz`, quero retornar: `z_chd`, `z_txt`, `kl_loss`, `kl_chd`, `kl_rhy` e `Final loss`



# 23 de Junho 

Existe uma relação de dependência entre as funções `loss_function`, `chord_loss`, `kl_loss` e `loss`? Isto é, uma depende de outra pra funcionar? Existe outras partes da função de perda que eu não citei?

O fluxo é: loss → loss_function → (recon_loss, kl_loss, chord_loss). Cada uma dessas funções depende das anteriores para produzir os valores finais de perda.

Quanto a partes adicionais da função de perda, o código não mostra outros termos além de reconstrução, KL e acorde. Esses três componentes são combinados com o fator beta para ponderar a divergência KL. Portanto, não há outras parcelas de perda implementadas além das citadas.

Sim. Os métodos formam uma cadeia de chamadas:

loss

Este método é o ponto de entrada durante o treinamento. Ele executa self.run() para obter as saídas do modelo e, em seguida, chama loss_function para calcular as perdas:

`def loss(self, x, c, pr_mat, tfr1=0., tfr2=0., tfr3=0.,
         beta=0.1, weights=(1, 0.5)):
    outputs = self.run(x, c, pr_mat, tfr1, tfr2, tfr3)
    loss = self.loss_function(x, c, *outputs, beta, weights)
    return loss`

loss_function

Combina três componentes de perda: recon_loss (chamada de decoder.recon_loss), kl_loss e chord_loss:

`def loss_function(self, x, c, recon_pitch, recon_dur, dist_chd,
                  dist_rhy, recon_root, recon_chroma, recon_bass,
                  beta, weights, weighted_dur=False):
    recon_loss, pl, dl = self.decoder.recon_loss(x, recon_pitch, recon_dur,
                                                 weights, weighted_dur)
    kl_loss, kl_chd, kl_rhy = self.kl_loss(dist_chd, dist_rhy)
    chord_loss, root, chroma, bass = self.chord_loss(c, recon_root,
                                                     recon_chroma,
                                                     recon_bass)
    loss = recon_loss + beta * kl_loss + chord_loss
    return loss, recon_loss, pl, dl, kl_loss, kl_chd, kl_rhy, chord_loss, \
           root, chroma, bass
`
chord_loss e kl_loss

São chamadas diretamente por loss_function, mas funcionam de forma independente:

`def chord_loss(self, c, recon_root, recon_chroma, recon_bass):
    loss_fun = nn.CrossEntropyLoss()
    ...
    chord_loss = root_loss + chroma_loss + bass_loss
    return chord_loss, root_loss, chroma_loss, bass_loss

def kl_loss(self, *dists):
    kl_chd = kl_with_normal(dists[0])
    kl_rhy = kl_with_normal(dists[1])
    kl_loss = kl_chd + kl_rhy
    return kl_loss, kl_chd, kl_rhy`

decoder.recon_loss

Calcula a parte de reconstrução (pitch e duração) usada por loss_function:

def recon_loss(self, x, recon_pitch, recon_dur, weights=(1, 0.5),
               weighted_dur=False):
    pitch_loss_func = nn.CrossEntropyLoss(ignore_index=self.pitch_pad)
    ...
    loss = weights[0] * pitch_loss + weights[1] * dur_loss
    return loss, pitch_loss, dur_loss

# Questão 2

*quais dados eu preciso para rodar loss? o que é self, x, c e pr_mat? você é capaz de me mostrar no modelo como são geradas esses dados?*

Para calcular a perda no DisentangleVAE é necessário fornecer três tensores:

x – representação “piano tree” da sequência (32×N×6).

Gerado em dataset.__getitem__: os piano‑rolls são convertidos para matrizes de duração (pr_mats) e, em seguida, para grids 3D (p_grids), que se tornam o x utilizado pelo modelo:

prs = np.array([pr_to_onehot_pr(pr) for pr in acc_segments])
pr_mats = np.array([piano_roll_to_target(pr) for pr in prs])
p_grids = np.array([target_to_3dtarget(pr_mat, max_note_count=16,
                                       max_pitch=128, min_pitch=0,
                                       pitch_pad_ind=130,
                                       pitch_sos_ind=128,
                                       pitch_eos_ind=129)
                    for pr_mat in pr_mats])
# ...
pr_mats = pr_mats[0]
p_grids = p_grids[0]
...
dt_x = detrend_pianotree(p_grids, chord)  # (32, 16, 39)
return mel_segments, prs, pr_mats, p_grids, chord, dt_x

c – vetores de acordes (8×36) contendo raiz, cromas e baixo.

Também retornados por dataset.__getitem__ após expand_chord:

chord = [x[-1] for x in data]
chord = np.concatenate(chord, axis=0)
chord = np.array([expand_chord(c, shift) for c in chord])

pr_mat – matriz piano‑roll (32×128) com duração codificada por inteiro.

Produzida pela função piano_roll_to_target:

def piano_roll_to_target(pr):
    pr[:, :, 1] = np.logical_not(np.logical_or(pr[:, :, 0], pr[:, :, 2]))
    pr = pr.astype(int)
    pr_matrix = np.zeros((32, 128))
    for i in range(31, -1, -1):
        onset_idx = np.where(pr[i, :, 0] == 1)[0]
        pr_matrix[i, onset_idx] = pr[i, onset_idx, 1] + 1
        if i == 0:
            break
        pr[i, onset_idx, 1] = 0
        pr[i - 1, :, 1] += pr[i, :, 1]
    return pr_matrix

Durante o treinamento, MusicDataLoaders.batch_to_inputs transforma o batch do DataLoader nesses três tensores:

_, _, pr_mat, x, c, dt_x = batch
pr_mat = pr_mat.to(self.device).float()
x = x.to(self.device).long()
c = c.to(self.device).float()
dt_x = dt_x.to(self.device).float()
return x, c, pr_mat, dt_x

O método loss do modelo recebe exatamente x, c e pr_mat (além de parâmetros opcionais) e combina as perdas de reconstrução, KL e acordes:

def loss(self, x, c, pr_mat, tfr1=0., tfr2=0., tfr3=0.,
         beta=0.1, weights=(1, 0.5)):
    outputs = self.run(x, c, pr_mat, tfr1, tfr2, tfr3)
    loss = self.loss_function(x, c, *outputs, beta, weights)
    return loss

Nesse contexto:

self é a instância do DisentangleVAE.

x contém o piano tree (grids de pitch/duração) que serve como alvo de reconstrução.

c guarda a informação de acordes por batida.

pr_mat é o piano‑roll original transformado em matriz de duração, usado pelo codificador de textura.

Esses tensores são obtidos automaticamente a partir do dataset e fornecidos à função loss durante o treinamento.

# Questão 3

Você consegue fazer um código python que 

1 - Carrega apenas um único arquivo: ./vae-tuning-texutres/dataSet/POP09-PIANOROLL-4-bin-quantization/001.npz
2 - Extrai os dados

`x` contém o piano tree (grids de pitch/duração) que serve como alvo de reconstrução.
`c` guarda a informação de acordes por batida.
`pr_mat` é o piano‑roll original transformado em matriz de duração, usado pelo codificador de textura.

3 - Envia os dados para o modelo, se necessário, usando inteface.py
4 - Calcula a loss para esse arquivo.

É possível?

```
import torch
import numpy as np

from base_model.dataset import wrap_dataset
from base_model.model import DisentangleVAE


def load_sample(path):
    dataset = wrap_dataset([path], [0], shift_low=0, shift_high=0,
                           num_bar=2, contain_chord=True)
    mel, prs, pr_mat, x, c, dt_x = dataset[0]
    return x, c, pr_mat


def prepare_tensors(x, c, pr_mat, device):
    x = torch.tensor(x).long().unsqueeze(0).to(device)
    c = torch.tensor(c).float().unsqueeze(0).to(device)
    pr_mat = torch.tensor(pr_mat).float().unsqueeze(0).to(device)
    return x, c, pr_mat


def main():
    data_path = './dataSet/POP09-PIANOROLL-4-bin-quantization/001.npz'
    model = DisentangleVAE.init_model()
    x, c, pr_mat = load_sample(data_path)
    x, c, pr_mat = prepare_tensors(x, c, pr_mat, model.device)

    loss, *_ = model.loss(x, c, pr_mat)
    print('Loss:', loss.item())


if __name__ == '__main__':
    main()
```

# Código revisado pra levar em conta o modelo

import torch
import numpy as np

from base_model.dataset import wrap_dataset
from base_model.model import DisentangleVAE
from interface import PolyDisVAE


def load_sample(path):
    dataset = wrap_dataset([path], [0], shift_low=0, shift_high=0,
                           num_bar=2, contain_chord=True)
    mel, prs, pr_mat, x, c, dt_x = dataset[0]
    return x, c, pr_mat


def prepare_tensors(x, c, pr_mat, device):
    x = torch.tensor(x).long().unsqueeze(0).to(device)
    c = torch.tensor(c).float().unsqueeze(0).to(device)
    pr_mat = torch.tensor(pr_mat).float().unsqueeze(0).to(device)
    return x, c, pr_mat


def load_model(device):
    interface = PolyDisVAE.init_model(device=device)
    interface.load_model('./model_param/polydis-v1.pt')
    model = DisentangleVAE('disvae', device,
                           interface.chd_encoder,
                           interface.txt_encoder,
                           interface.pnotree_decoder,
                           interface.chd_decoder)
    model.eval()
    return model


def main():
    data_path = './dataSet/POP09-PIANOROLL-4-bin-quantization/001.npz'
    model = load_model(device=None)
    x, c, pr_mat = load_sample(data_path)
    x, c, pr_mat = prepare_tensors(x, c, pr_mat, model.device)

    loss, *_ = model.loss(x, c, pr_mat)
    print('Loss:', loss.item())


if __name__ == '__main__':
    main()
===

```
import torch
import numpy as np

from base_model.dataset import wrap_dataset
from base_model.model import DisentangleVAE
from interface import PolyDisVAE


def load_sample(path):
    dataset = wrap_dataset([path], [0], shift_low=0, shift_high=0,
                           num_bar=2, contain_chord=True)
    mel, prs, pr_mat, x, c, dt_x = dataset[0]
    return x, c, pr_mat


def prepare_tensors(x, c, pr_mat, device):
    x = torch.tensor(x).long().unsqueeze(0).to(device)
    c = torch.tensor(c).float().unsqueeze(0).to(device)
    pr_mat = torch.tensor(pr_mat).float().unsqueeze(0).to(device)
    return x, c, pr_mat


def load_model(device):
    interface = PolyDisVAE.init_model(device=device)
    interface.load_model('./model_param/polydis-v1.pt')
    model = DisentangleVAE('disvae', device,
                           interface.chd_encoder,
                           interface.txt_encoder,
                           interface.pnotree_decoder,
                           interface.chd_decoder)
    model.eval()
    return model


def debug_run(model, x, c, pr_mat):
    names = [
        'recon_pitch', 'recon_dur', 'dist_chd', 'dist_rhy',
        'recon_root', 'recon_chroma', 'recon_bass'
    ]
    outputs = model.run(x, c, pr_mat, 0., 0., 0.)
    print('--- run() outputs ---')
    for n, o in zip(names, outputs):
        if isinstance(o, torch.distributions.Distribution):
            print(f'{n}: mean {tuple(o.mean.shape)}, std {tuple(o.stddev.shape)}')
        else:
            print(f'{n}: {tuple(o.shape)}')
    return outputs


def debug_loss_function(model, x, c, run_outputs):
    labels = [
        'total_loss', 'recon_loss', 'pitch_loss', 'dur_loss',
        'kl_loss', 'kl_chd', 'kl_rhy',
        'chord_loss', 'root_loss', 'chroma_loss', 'bass_loss'
    ]
    loss_values = model.loss_function(x, c, *run_outputs, beta=0.1, weights=(1, 0.5))
    print('--- loss_function() breakdown ---')
    for label, val in zip(labels, loss_values):
        if torch.is_tensor(val):
            if val.dim() == 0:
                print(f'{label}: {val.item()}')
            else:
                print(f'{label}: {tuple(val.shape)}')
        else:
            print(f'{label}: {val}')
    return loss_values


def main():
    data_path = './dataSet/POP09-PIANOROLL-4-bin-quantization/001.npz'
    model = load_model(device=None)
    x, c, pr_mat = load_sample(data_path)
    x, c, pr_mat = prepare_tensors(x, c, pr_mat, model.device)

    run_outs = debug_run(model, x, c, pr_mat)
    debug_loss_function(model, x, c, run_outs)

    loss, *_ = model.loss(x, c, pr_mat)
    print('Final loss from model.loss():', loss.item())


if __name__ == '__main__':
    main()
```