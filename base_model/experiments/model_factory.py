from model import DisentangleVAE
from dl_modules import ChordEncoder, ChordDecoder, TextureEncoder, PianoTreeDecoder


def build_model(device, model_cfg: dict, experiment_name: str) -> DisentangleVAE:
    chd_dim = model_cfg.get('chd_latent_dim', 256)
    rhy_dim = model_cfg.get('rhy_latent_dim', 256)
    num_channel = model_cfg.get('num_channel', 10)

    chd_encoder = ChordEncoder(36, 1024, chd_dim)
    rhy_encoder = TextureEncoder(256, 1024, rhy_dim, num_channel)
    chd_decoder = ChordDecoder(z_dim=chd_dim)
    decoder = PianoTreeDecoder(note_embedding=None, dec_dur_hid_size=64, z_size=chd_dim + rhy_dim)

    return DisentangleVAE(experiment_name, device, chd_encoder, rhy_encoder, decoder, chd_decoder)
