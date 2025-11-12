import unittest
import numpy as np
import pretty_midi as pm
import matplotlib.pyplot as plt
from utilProcessing import parseCOMU, pitchDataProcessing


def quantized_to_midi(q_notes, tempo=120):
    """Convert quantized note matrix back to a PrettyMIDI object."""
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    for row in q_notes:
        start = (row[0] + row[1] / row[2]) * (60.0 / tempo)
        end = (row[3] + row[4] / row[5]) * (60.0 / tempo)
        inst.notes.append(pm.Note(velocity=int(row[7]), pitch=int(row[6]), start=start, end=end))
    midi.instruments.append(inst)
    return midi


def plot_piano_rolls(midi_notes, npz_notes, filename="npz_compare.png"):
    """Save piano roll comparison between MIDI notes and NPZ data."""
    midi_pm = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    inst.notes = midi_notes
    midi_pm.instruments.append(inst)
    midi_roll = midi_pm.get_piano_roll()

    npz_pm = quantized_to_midi(npz_notes)
    npz_roll = npz_pm.get_piano_roll()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].set_title("MIDI Piano Roll")
    axs[0].imshow(midi_roll, origin="lower", aspect="auto", cmap="gray_r")
    axs[1].set_title("NPZ Piano Roll")
    axs[1].imshow(npz_roll, origin="lower", aspect="auto", cmap="gray_r")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


class TestNPZIntegrity(unittest.TestCase):
    def test_commu00001_npz(self):
        midi_path = "./midiDataTest/commu00001.mid"
        npz_path = "./commu00001_4bin.npz"

        notes = parseCOMU(midi_path)
        quantized_midi = pitchDataProcessing(midi_path, notes, 4)
        npz_data = np.load(npz_path)["quantized_notes"]

        plot_piano_rolls(notes, npz_data, filename="commu00001_compare.png")

        self.assertEqual(
            quantized_midi.shape,
            npz_data.shape,
            "Shape mismatch between MIDI quantized data and NPZ data.",
        )

        midi_set = {tuple(row) for row in quantized_midi}
        npz_set = {tuple(row) for row in npz_data}

        missing_from_npz = midi_set - npz_set
        extra_in_npz = npz_set - midi_set

        if missing_from_npz:
            missing_notes = sorted({row[6] for row in missing_from_npz})
            missing_vels = sorted({row[7] for row in missing_from_npz})
            self.fail(
                f"Missing notes in npz: {missing_notes}, velocities: {missing_vels}"
            )

        if extra_in_npz:
            extra_notes = sorted({row[6] for row in extra_in_npz})
            extra_vels = sorted({row[7] for row in extra_in_npz})
            self.fail(
                f"Extra notes in npz: {extra_notes}, velocities: {extra_vels}"
            )


if __name__ == "__main__":
    unittest.main()