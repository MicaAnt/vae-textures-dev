import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# beat - shape(n,6), dtype = int32
# chord - shape(n,14), dtype = float64
# melody - shape(n, 8), dtype = int32
# bridge - shape(n,8), dtype = int32
# piano - shape(n, 8), dtype = int32

from utilProcessing import midiToPitchHarmony, genBeatTable

def ViewDataSet(trackId = "commu00002", dataSetPath = "../midiDataTest/", csv_path= "../midiDataTest/commu_meta.csv"):

    piano, chord = midiToPitchHarmony(trackId, dataSetPath, csv_path)

    beat = genBeatTable(len(chord))

    print("Piano", piano, "Shape", piano.shape)
    print("Chord", chord, "Shape", chord.shape)
    print("Beat", beat, "Shape", beat.shape)

def GenDataSet(trackId="commu00002", dataSetPath="../midiDataTest/", csv_path="../midiDataTest/commu_meta.csv"):
    # Gera piano e chord
    piano, chord = midiToPitchHarmony(trackId, dataSetPath, csv_path)

    # Gera beat com shape (n, 6) e dtype int32
    beat = genBeatTable(len(chord)).astype(np.int32)

    # Converte chord para float64 se não estiver
    chord = chord.astype(np.float64)

    # melody e bridge vazios com shape (0, 8) e dtype int32
    melody = np.empty((0, 8), dtype=np.int32)
    bridge = np.empty((0, 8), dtype=np.int32)

    # Converte piano para int32 se necessário
    piano = piano.astype(np.int32)

    # Caminho de saída
    save_path = f"../{trackId}.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Salva o arquivo
    np.savez(
        save_path,
        beat=beat,
        chord=chord,
        melody=melody,
        bridge=bridge,
        piano=piano
    )

    print(f"Arquivo salvo em {save_path}")

GenDataSet()