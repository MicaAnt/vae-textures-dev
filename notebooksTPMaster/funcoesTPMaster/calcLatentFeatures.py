import argparse
import sys
import torch
import numpy as np
import os
from tqdm import tqdm

sys.path.append('./base_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from base_model.dataset import wrap_dataset
from base_model.model import DisentangleVAE
from interface import PolyDisVAE
from amc_dl.torch_plus.train_utils import get_zs_from_dists

def load_model(device=device):
    interface = PolyDisVAE.init_model(device=device)
    interface.load_model('drive/MyDrive/tp20252/model_param/polydis-v1.pt')
    model = DisentangleVAE('disvae', interface.device,
                           interface.chd_encoder,
                           interface.txt_encoder,
                           interface.pnotree_decoder,
                           interface.chd_decoder)
    model.eval()
    return model

def prepare_tensors(x, c, pr_mat, device):
    x = torch.tensor(x).long().unsqueeze(0).to(device)
    c = torch.tensor(c).float().unsqueeze(0).to(device)
    pr_mat = torch.tensor(pr_mat).float().unsqueeze(0).to(device)
    return x, c, pr_mat

def run_with_latents(model, x, c, pr_mat):
    embedded_x, lengths = model.decoder.emb_x(x)
    dist_chd = model.chd_encoder(c)
    dist_rhy = model.rhy_encoder(pr_mat)
    z_chd, z_txt = get_zs_from_dists([dist_chd, dist_rhy], True)
    dec_z = torch.cat([z_chd, z_txt], dim=-1)
    recon_pitch, recon_dur = model.decoder(dec_z, False, embedded_x, lengths, 0., 0.)
    recon_root, recon_chroma, recon_bass = model.chd_decoder(z_chd, False, 0., c)
    return recon_pitch, recon_dur, dist_chd, dist_rhy, recon_root, recon_chroma, recon_bass, z_chd, z_txt

def compute_segment(model, x, c, pr_mat):
    recon_pitch, recon_dur, dist_chd, dist_rhy, recon_root, recon_chroma, recon_bass, z_chd, z_txt = run_with_latents(model, x, c, pr_mat)
    loss_values = model.loss_function(x, c, recon_pitch, recon_dur,
                                      dist_chd, dist_rhy,
                                      recon_root, recon_chroma, recon_bass,
                                      beta=0.1, weights=(1, 0.5))
    total_loss, _, _, _, kl_loss, kl_chd, kl_rhy, _, _, _, _ = loss_values
    return z_chd.squeeze(0), z_txt.squeeze(0), kl_loss, kl_chd, kl_rhy, total_loss

def compute_losses(npz_path, model, output_dir="drive/MyDrive/tp20252/monDataset/features"):

    # Garante que o diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrai nome base do arquivo (sem caminho, sem extensão)
    base_name = os.path.splitext(os.path.basename(npz_path))[0]

    meta = np.load(npz_path, allow_pickle=True)
    track_role = meta["track_role"].item()  # Se for um objeto (tipo string)
    
    #model = load_model(device=device)
    dataset = wrap_dataset([npz_path], [0], shift_low=0, shift_high=0,
                           num_bar=2, contain_chord=True)

    results = []
    
    for idx in tqdm(range(len(dataset)), desc="Calculando perdas"):
    #for idx in tqdm((range(3)), desc="Calculando perdas"):
        
        _, _, pr_mat, x, c, _ = dataset[idx]
        x_t, c_t, pr_mat_t = prepare_tensors(x, c, pr_mat, model.device)
        z_chd, z_txt, kl_loss, kl_chd, kl_rhy, total_loss = compute_segment(
            model, x_t, c_t, pr_mat_t)

        # Monta o nome do arquivo de saída
        segment_name = f"{base_name}-{idx+1:03d}"  # Ex: 001-001
        save_path = os.path.join(output_dir, f"{segment_name}.npz")

        #Salva os dados em arquivo .npz
        np.savez_compressed(save_path,
                            z_chd=z_chd.detach().cpu().numpy(),
                            z_txt=z_txt.detach().cpu().numpy(),
                            kl_loss=kl_loss.item(),
                            kl_chd=kl_chd.item(),
                            kl_rhy=kl_rhy.item(),
                            final_loss=total_loss.item(),
                            track_role = track_role)
        
def main(path):
    """Compute per-segment losses for the given NPZ file.

    Parameters
    ----------
    path : str
        Path to the ``.npz`` file.

    Returns
    -------
    list[dict]
        A list with one entry per segment containing ``z_chd``, ``z_txt`` and
        all loss values.
    """
    results = compute_losses(path)
    return results

def process_folder(folder):

    npz_files = [f for f in os.listdir(folder) if f.endswith(".npz")]
    os.makedirs("losses", exist_ok=True)

    model = load_model(device=device)
    #for fname in os.listdir(folder):
    for fname in tqdm(npz_files, desc="Processando arquivos .npz"):
        if fname.endswith(".npz"):
            fpath = os.path.join(folder, fname)
            print(f"Processando {fpath}")
            results = compute_losses(fpath, model)
            out_name = fname.replace(".npz", ".npy")
            np.save(os.path.join("losses", out_name), results)

#if __name__ == '__main__':
 #   path = sys.argv[1]
  #  main(path)
if __name__ == "__main__":
    folder = sys.argv[1]
    process_folder(folder)