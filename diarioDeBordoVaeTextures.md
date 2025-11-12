# Init

cd vae-tuning-texutres/COMMUDataset/
cd COMMUDataset/

# TO DO

- [] Inclur mais metadados no calculo da perda!!!

# Comandos recorrentes

`docker exec -it dcli_fidle_tuto /bin/bash`

# 11/08/2025

## Meu data set de representações latentes tem a seguinte estrutura

```
Chaves encontradas no arquivo '../COMMUDataset/losses/commu00001-001.npz':

– z_chd: shape = (256,), dtype = float32
  First 20 rows:
[ 0.10153764  1.3947941   0.38176268 -0.493975   -1.0387502   1.3639889   0.4164051   0.8212657   0.39879447  0.02583198 -1.0974351  -1.8505356  -1.5507616  -1.4575522   1.8703055  -1.2324191  -0.55191296 -0.44315654  1.6455977   0.97078454]

– z_txt: shape = (256,), dtype = float32
  First 20 rows:
[-0.8357064   1.1716447   0.23959124  0.53362155  0.48109812  0.32163465 -0.22896728  0.15017265  0.16173217  0.08637981  0.53888404 -0.15645075  1.4217248   1.3644127   0.48116958  1.9895382   0.5313427   0.9260218  -0.6943377  -0.7437458 ]

– kl_loss: shape = (), dtype = float64
  Value: 1.432845115661621

– kl_chd: shape = (), dtype = float64
  Value: 0.8210890889167786

– kl_rhy: shape = (), dtype = float64
  Value: 0.6117559671401978

– final_loss: shape = (), dtype = float64
  Value: 1.6189824342727661

– audio_key: shape = (), dtype = <U6
  Value: aminor

– chord_progressions: shape = (), dtype = <U346
  Value: [['Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'Dm', 'Dm', 'Dm', 'Dm', 'Dm', 'Dm', 'Dm', 'Dm', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'Am', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D']]

– pitch_range: shape = (), dtype = <U3
  Value: mid

– num_measures: shape = (), dtype = <U1
  Value: 8

– bpm: shape = (), dtype = <U3
  Value: 120

– genre: shape = (), dtype = <U9
  Value: cinematic

– track_role: shape = (), dtype = <U11
  Value: main_melody

– inst: shape = (), dtype = <U15
  Value: string_ensemble

– sample_rhythm: shape = (), dtype = <U8
  Value: standard

– time_signature: shape = (), dtype = <U3
  Value: 4/4

```

## A localização é

```
folder_path="../COMMUDataset/losses/"
```

## Os arquivos que eu quero carregar são:

```
{'accompaniment': {'min': ['commu01310-005.npz', 'commu01287-001.npz', 'commu02622-001.npz', 'commu10104-001.npz'], 'max': ['commu04212-004.npz', 'commu04212-007.npz', 'commu03128-007.npz', 'commu03950-003.npz']}, 'bass': {'min': ['commu10308-004.npz', 'commu10308-002.npz', 'commu10825-003.npz', 'commu10643-002.npz'], 'max': ['commu09796-001.npz', 'commu10103-001.npz', 'commu09796-004.npz', 'commu11096-001.npz']}, 'main_melody': {'min': ['commu00788-001.npz', 'commu06083-005.npz', 'commu02617-004.npz', 'commu05918-005.npz'], 'max': ['commu03994-002.npz', 'commu03994-006.npz', 'commu03151-005.npz', 'commu02846-002.npz']}, 'pad': {'min': ['commu02966-002.npz', 'commu02876-004.npz', 'commu03113-004.npz', 'commu10077-003.npz'], 'max': ['commu10236-003.npz', 'commu10236-002.npz', 'commu00951-002.npz', 'commu04582-006.npz']}, 'riff': {'min': ['commu03851-001.npz', 'commu11141-002.npz', 'commu09946-002.npz', 'commu09946-004.npz'], 'max': ['commu00374-007.npz', 'commu00370-003.npz', 'commu00370-002.npz', 'commu00370-004.npz']}, 'sub_melody': {'min': ['commu05514-002.npz', 'commu05321-002.npz', 'commu02565-002.npz', 'commu04823-001.npz'], 'max': ['commu01630-004.npz', 'commu01630-005.npz', 'commu01630-007.npz', 'commu01630-003.npz']}}

```

## E quero reconstruir em arquivos midi inspirado no código

```

from pretty_midi import PrettyMIDI, Instrument, TimeSignature
import pretty_midi

from poly_dis.model import PolyDisVAE

# initialize the model
polydis_model = PolyDisVAE.init_model()

# load model parameters
polydis_param_path = "./poly_dis/model_param/polydis-v1.pt"
polydis_model.load_model(polydis_param_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

out_dir = "./HigherLowerLosses"
os.makedirs(out_dir, exist_ok=True)

for filename, (z_h, z_t) in combos.items():
    # Decode pianotree and remove batch dimension
    ptree = polydis_model.pnotree_decode(z_h, z_t).squeeze(0)
    if isinstance(ptree, torch.Tensor):
        ptree = ptree.detach().cpu().numpy()

    # Convert pianotree to PrettyMIDI Note objects with fixed BPM = 80
    notes = polydis_model.pnotree_to_notes(ptree, bpm=80, start=0.0)

    # Create PrettyMIDI object and add one instrument
    midi_obj = PrettyMIDI()
    instrument = Instrument(program=0, name=filename.replace(".mid", ""))
    instrument.notes = notes
    midi_obj.instruments.append(instrument)

    # Write individual MIDI file
    path = os.path.join(out_dir, filename)
    midi_obj.write(path)
    print(f"Saved {path} at 80 BPM")

```

# 13/08/2025

para salvar o modelo

`torch.save(model.state_dict(), "meu_modelo.pth")`

# 12/08/2025

parei no batch 672

`python run_batches.py --start 672`

# 08/08/2025

`docker exec -it dcli_fidle_tuto /bin/bash`


# 07/08/2025

- Eu estou rodando o script `NEWcalcLatentBatchLoos.py`. 
	- Por acaso, eu rodei com a lista errada de batches, com 100 batches. Mas ele está fazendo em um numero razoavel. É bom pra ter uma referência de como eu vou fazer com o cluster do LIS.

- Seria bom eu desenhar bem a minha pipeline. 

- Mas antes, eu vou trabalhar no notebook `umapPlots`
	- Aqui, eu tenho que ver se está tudo ok com os arquvos do `COMMUDataset/losses/` 

### 01/08/2025

- I want to calculate the loss of my dataset. To do it, I need to use the script `calc_latent_loss.py`.
- Eu vou criar o arquivo `calclatentlossBatch.py`.

# Tentando rodar um batche

python calcLatentBatchLoss.py --folder ./COMMUDataset/npzFiles --batch ./COMMUDataset/batchesNPZ/batch_004.txt
