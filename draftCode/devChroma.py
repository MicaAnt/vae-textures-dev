import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import pretty_midi as pm

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from utilProcessing import midiFileTo4bin, get_fund

import numpy as np
from pychord import utils

def prToChroma(notes: np.ndarray) -> np.ndarray:
    """Generate per-beat pitch class activity vectors from quantized notes.

    Parameters
    ----------
    notes : np.ndarray
        Array with shape (n, 8) containing quantized note information in the
        format [beat_on, subdiv_on, quant, beat_off, subdiv_off, quant, pitch,
        velocity].

    Returns
    -------
    np.ndarray
        Matrix with shape (n_beats, 13). Columns 0-11 indicate active pitch
        classes for each beat. Column 12 stores the pitch class of the lowest
        active note or ``-1`` if no notes are active.
    """
    if notes.size == 0:
        return np.zeros((0, 13), dtype=int)

    #beat_on = notes[:, 0]
    #beat_off = notes[:, 3]
    #beat_on_full = notes[:, 0] + notes[:, 1] / notes[:, 2]
    #beat_off_full = notes[:, 3] + notes[:, 4] / notes[:, 5]

    #beat_on = beat_on_full
    #beat_off = beat_off_full
    beat_on = notes[:, 0] + notes[:, 1] / notes[:, 2]
    beat_off = notes[:, 3] + notes[:, 4] / notes[:, 5]
    pitches = notes[:, 6].astype(int)

    start_beat = int(np.floor(beat_on.min()))
    end_beat = int(np.ceil(beat_off.max())) 

    n_beats = max(0, end_beat - start_beat)
    result = np.zeros((n_beats, 13), dtype=int)
    result[:, 12] = -1

    for b in range(start_beat, end_beat):
        mask = (beat_on <= b) & (b < beat_off)
        # include notes that start or end within this beat
        mask = (beat_on < b + 1) & (beat_off > b) # incluiu essa linha, mas tem um problema na nota mais grave
        active_pitches = pitches[mask]
        if active_pitches.size == 0:
            continue

        # Compute pitch classes; use pychord if available
        if utils is not None and hasattr(utils, "pitch_to_note"):
            pcs = [utils.note_to_val(utils.pitch_to_note(int(p))) % 12 for p in active_pitches]
        else:
            pcs = [int(p) % 12 for p in active_pitches]

        pcs = np.array(pcs, dtype=int)
        beat_idx = b - start_beat
        result[beat_idx, pcs] = 1
        lowest_pitch = active_pitches[active_pitches.argmin()]
        result[beat_idx, 12] = int(lowest_pitch % 12)

    return result

dados = midiFileTo4bin(midi_path = "../midiDataTest/commu00002.mid")
funds = get_fund(csv_path="../midiDataTest/commu_meta.csv",track_id="commu00001")

chroma = prToChroma(dados)
print("Chroma", chroma)
print("Dados", dados)

print("Fundamentais", funds)

#print("Shape dos dados", dados.shape)
print("Shape das fundamentais", funds.shape)
print("Shape de Chroma", chroma.shape)

