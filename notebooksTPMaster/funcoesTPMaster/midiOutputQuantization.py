from typing import Sequence
import pretty_midi

def writeQuantizedMidi_22(
    notes: Sequence,                 # objetos com .start, .end, .pitch, .velocity
    bpm: float = 80,
    output_path: str = "./output.mid",   # use um caminho de ARQUIVO (.mid)
    length_beats_22: int = 8,        # nº de "batidas" em 2/2 (mínimas). Ex.: 8 => 8 mínimas
    grid_divs_per_quarter: int = 4,  # 4 => quantização em 1/16
    program: int = 0                 # Acoustic Grand
) -> pretty_midi.PrettyMIDI:
    """
    Quantiza uma lista de notas para uma grade fixa e escreve um MIDI em compasso 2/2.

    - Quantização: 1/(4*grid_divs_per_quarter) da semibreve (padrão: 1/16 da semínima).
    - Duração total: length_beats_22 mínimas (cada mínima = 2 semínimas).
    """
    qsec = 60.0 / bpm
    step = qsec / grid_divs_per_quarter
    end_limit = (2 * length_beats_22) * qsec

    def q(x: float) -> float:
        return round(x / step) * step

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    pm.time_signature_changes = [pretty_midi.TimeSignature(2, 2, 0.0)]
    inst = pretty_midi.Instrument(program=program)

    for n in notes:
        s_q = max(0.0, q(float(n.start)))
        e_q = max(q(float(n.end)), s_q + step)
        if s_q >= end_limit:
            continue
        if e_q > end_limit:
            e_q = end_limit
        inst.notes.append(pretty_midi.Note(
            velocity=int(max(1, min(127, int(n.velocity)))),
            pitch=int(n.pitch),
            start=s_q,
            end=e_q
        ))

    inst.notes.sort(key=lambda x: (x.start, x.pitch))
    pm.instruments.append(inst)
    pm.write(output_path)
    return pm

def writeQuantizedMidi_24(
    notes: Sequence,                 # objetos com .start, .end, .pitch, .velocity
    bpm: float = 80,
    output_path: str = "./output.mid",   # use um caminho de ARQUIVO (.mid)
    length_beats_24: int = 8,        # nº de "batidas" em 2/2 (mínimas). Ex.: 8 => 8 mínimas
    grid_divs_per_quarter: int = 4,  # 4 => quantização em 1/16
    program: int = 0                 # Acoustic Grand
) -> pretty_midi.PrettyMIDI:
    """
    Quantiza uma lista de notas para uma grade fixa e escreve um MIDI em compasso 2/2.

    - Quantização: 1/(4*grid_divs_per_quarter) da semibreve (padrão: 1/16 da semínima).
    - Duração total: length_beats_22 mínimas (cada mínima = 2 semínimas).
    """
    qsec = 60.0 / bpm
    step = qsec / grid_divs_per_quarter
    end_limit = (2 * length_beats_24) * qsec

    def q(x: float) -> float:
        return round(x / step) * step

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    pm.time_signature_changes = [pretty_midi.TimeSignature(2, 4, 0.0)]
    inst = pretty_midi.Instrument(program=program)

    for n in notes:
        s_q = max(0.0, q(float(n.start)))
        e_q = max(q(float(n.end)), s_q + step)
        if s_q >= end_limit:
            continue
        if e_q > end_limit:
            e_q = end_limit
        inst.notes.append(pretty_midi.Note(
            velocity=int(max(1, min(127, int(n.velocity)))),
            pitch=int(n.pitch),
            start=s_q,
            end=e_q
        ))

    inst.notes.sort(key=lambda x: (x.start, x.pitch))
    pm.instruments.append(inst)
    pm.write(output_path)
    return pm

# -------------------- 

def writeQuantizedMidi_44( # mudar o nome
    notes: Sequence,                 # objetos com .start, .end, .pitch, .velocity
    bpm: float = 80,
    output_path: str = "./output.mid",   # use um caminho de ARQUIVO (.mid)
    length_beats_44: int = 8,        # MUDAR, nº de "batidas" em 2/2 (mínimas). Ex.: 8 => 8 mínimas
    grid_divs_per_quarter: int = 4,  # 4 => quantização em 1/16
    program: int = 0                 # Acoustic Grand
) -> pretty_midi.PrettyMIDI:
    """
    Quantiza uma lista de notas para uma grade fixa e escreve um MIDI em compasso 2/2.

    - Quantização: 1/(4*grid_divs_per_quarter) da semibreve (padrão: 1/16 da semínima).
    - Duração total: length_beats_22 mínimas (cada mínima = 2 semínimas).
    """
    qsec = 60.0 / bpm
    step = qsec / grid_divs_per_quarter
    end_limit = (2 * length_beats_44) * qsec # MUDAR

    def q(x: float) -> float:
        return round(x / step) * step

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    pm.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]
    inst = pretty_midi.Instrument(program=program)

    for n in notes:
        s_q = max(0.0, q(float(n.start)))
        e_q = max(q(float(n.end)), s_q + step)
        if s_q >= end_limit:
            continue
        if e_q > end_limit:
            e_q = end_limit
        inst.notes.append(pretty_midi.Note(
            velocity=int(max(1, min(127, int(n.velocity)))),
            pitch=int(n.pitch),
            start=s_q,
            end=e_q
        ))

    inst.notes.sort(key=lambda x: (x.start, x.pitch))
    pm.instruments.append(inst)
    pm.write(output_path)
    return pm