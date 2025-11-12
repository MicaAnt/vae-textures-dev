import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilProcessing import get_fund, prToChroma, midiFileTo4bin

# --------------------------------------------

def combineFundChroma(fund, chroma):
    n = len(fund)
    m = chroma.shape[0]

    if m > n:
        raise ValueError("chroma has more rows than fund")

    output = np.zeros((n, 14))
    output[:, 0] = fund

    # Preenche as colunas 1 a 13 com os dados disponíveis de chroma
    output[:m, 1:] = chroma

    # Se chroma tiver menos linhas que fund, completa o restante com -1 na última coluna
    if m < n:
        output[m:, -1] = -1

    return output

def buildMidiPath(path, trackId):
    return os.path.join(path, f"{trackId}.mid")

def midiToHarmony(trackId = "commu00002", midiPath = "../midiDataTest/", csv_path= "../midiDataTest/commu_meta.csv" ):

    #csv_path="../midiDataTest/commu_meta.csv"
    midi_path = buildMidiPath(midiPath, trackId)
    


    funds = get_fund(csv_path, trackId)

    pitches = midiFileTo4bin(midi_path)
    chroma = prToChroma(pitches)

    return combineFundChroma(funds, chroma)

print(midiToHarmony("commu00003"))
