import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import pretty_midi as pm

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from utilProcessing import midiFileTo4bin, get_fund
from devChroma import prToChroma

import numpy as np
from pychord import utils

def midiToPitchClass(midi_values):
    return np.array(midi_values) % 12

dados = midiFileTo4bin(midi_path = "../midiDataTest/commu00002.mid")
dadosExemplo = dados[:4]

#print(type(dadosExemplo))
#print(dadosExemplo.shape)

pitches = dados[:, [6]].flatten()
pitchClasses = midiToPitchClass(pitches)

print("Pitches", pitches)
print("Pitch Classes", pitchClasses)