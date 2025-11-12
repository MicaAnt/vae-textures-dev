# VAE Representation of Symbolic Musical Textures

This repository contains tools and scripts for processing and analysing symbolic musical textures using a Variational Autoencoder (VAE) architecture. The work builds upon the model proposed in [polyphonic-chord-texture-disentanglement](https://github.com/ZZWaang/polyphonic-chord-texture-disentanglement), focusing on the representation and analysis of musical textures derived from annotated MIDI data.


## Dataset

All examples and experiments in this repository are based on the **Commu Dataset**. This dataset includes:

- A collection of `.mid` files with symbolic musical data.
- A corresponding metadata file (`commu_meta.csv`) containing annotations such as chord progressions.

The metadata CSV must contain a column named `chord_progressions`, where each row is a string-encoded chord sequence. Each entry is indexed by a unique track ID that matches the name of the corresponding MIDI file.

---

## Preprocessing MIDI Files

All MIDI preprocessing functions are implemented in `utilProcessing.py`.

To convert a folder of `.mid` files and associated metadata into `.npz` files (NumPy archive format), use the script `processMidiPath.py`.

The resulting `.npz` files contain the following arrays:

- `beat` — shape `(n, 6)`, dtype `int32`
- `chord` — shape `(n, 14)`, dtype `float64`
- `melody` — shape `(n, 8)`, dtype `int32`
- `bridge` — shape `(n, 8)`, dtype `int32`
- `piano` — shape `(n, 8)`, dtype `int32`

---

## Analysing VAE Representations

To analyse the quality of the latent representation learned by the model, we compute loss components and latent vectors for each `.npz` file using the script `calc_latent_loss.py`.

This script outputs:

- `z_chd` – latent vector associated with the **chord** encoding.
- `z_txt` – latent vector associated with the **texture** encoding.
- `kl_loss` – total Kullback–Leibler divergence between posterior and prior.
- `kl_chd` – KL divergence for the **chord** latent variable.
- `kl_rhy` – KL divergence for the **texture** latent variable.
- `final_loss` – total reconstruction loss of the VAE.

These outputs can be used to:
- Study how well the model captures harmonic and textural features.
- Visualise latent spaces using dimensionality reduction techniques (e.g. UMAP).
- Compare reconstruction quality across different musical textures.










```

              +------------------+       +----------------+
              |     midifile     |       |  metadata.csv  |
              +--------+---------+       +--------+-------+
                       |                          |
                       v                          v
            +------------------+         +----------------+
            |  midiFileTo4bin  |         |    get_fund    |
            +--------+---------+         +--------+-------+
                     |                          |
          +----------+-----------+              |
          |                      |              |
          v                      v              |
     +--------+            +-----------+<-------+
     | piano  |            |   chord   |
     +--------+            +-----------+
     
```

## To update 

### 31/07/2025

- The dataset for the VAE model was generated using the notebook `FilterCommuDataset`
- The folder COMMUDataset was included in the project. The `midiFiles` folder are the original `raw` data and the `npzFiles` folder the preocessed data using the function `processMidiBatch`. `createBatches.py` allows the creation of batches data stores in the `batches` folder. 
- A new function called `processMidiBatch.py` was created to process the raw `.mid` data into `.npz` data.





















