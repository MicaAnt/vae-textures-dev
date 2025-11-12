# This is to test if get_fund and prToChroma has the same shape

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilProcessing import get_fund, prToChroma, midiFileTo4bin

csv_path="../midiDataTest/commu_meta.csv"
midi_path = "../midiDataTest/commu00003.mid"

funds = get_fund(csv_path, track_id="commu00003")

pitches = midiFileTo4bin(midi_path)
chroma = prToChroma(pitches)

print("Shape de fundamentais é", funds.shape)
print("Shape de Chroma é", chroma.shape)

# Para o commu00001
#Shape de fundamentais é (32,)
#Shape de Chroma é (31, 13)

# Para o commu00002
#Shape de fundamentais é (32,)
#Shape de Chroma é (32, 13)

# Para o commu0003
#Shape de fundamentais é (32,)
#Shape de Chroma é (32, 13)