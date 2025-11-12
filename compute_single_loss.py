import torch
import numpy as np

import sys
sys.path.append('./base_model')

from base_model.dataset import wrap_dataset
from base_model.model import DisentangleVAE
from interface import PolyDisVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    data_path = '../dataSet/POP09-PIANOROLL-4-bin-quantization/001.npz'
    model = load_model(device=None)
    x, c, pr_mat = load_sample(data_path)
    x, c, pr_mat = prepare_tensors(x, c, pr_mat, model.device)

    loss, *_ = model.loss(x, c, pr_mat)
    print('Loss:', loss.item())


if __name__ == '__main__':
    main()
