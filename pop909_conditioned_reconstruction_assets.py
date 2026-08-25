#!/usr/bin/env python3
"""Generate perceptual review assets for POP909 conditioned reconstruction runs."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
import math
import wave
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pop909_conditioned_reconstruction import (  # noqa: E402
    iter_real_segments,
    load_model_for_checkpoint,
    parse_config,
)
from pop909_conditioned_reconstruction_review import ReviewRun  # noqa: E402


def safe_case_dir_name(compound_id: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in compound_id)[:180]


def find_config_for_run(run: ReviewRun, explicit_config: Optional[str] = None):
    if explicit_config:
        return parse_config(Path(explicit_config))
    summary = run.summary()
    run_id = summary.get('run_id')
    for cfg_path in (ROOT / 'configs').glob('pop909_conditioned_reconstruction_*.json'):
        try:
            cfg = parse_config(cfg_path)
        except Exception:
            continue
        if cfg.run_id == run_id:
            return cfg
    raise RuntimeError(f'Could not find original config for run_id={run_id}; pass --config explicitly')


def load_segment_by_dataset_index(cfg, dataset_index: int):
    for identity, x, c, pr_mat in iter_real_segments(cfg):
        if int(identity['dataset_index']) == int(dataset_index):
            return identity, x, c, pr_mat
    raise RuntimeError(f'dataset_index not found in configured stream: {dataset_index}')



def load_segments_by_dataset_indices(cfg, dataset_indices: Iterable[int]):
    wanted = {int(idx) for idx in dataset_indices}
    found = {}
    print(f'[phase9-assets] loading {len(wanted)} selected dataset segments; max dataset_index={max(wanted) if wanted else None}', flush=True)
    scanned = 0
    for identity, x, c, pr_mat in iter_real_segments(cfg):
        scanned += 1
        if scanned % 500 == 0:
            print(f'[phase9-assets] scanned {scanned} segments, found {len(found)}/{len(wanted)}', flush=True)
        dataset_index = int(identity['dataset_index'])
        if dataset_index in wanted:
            found[dataset_index] = (identity, x, c, pr_mat)
            print(f'[phase9-assets] cached dataset_index={dataset_index} ({len(found)}/{len(wanted)})', flush=True)
            if len(found) == len(wanted):
                break
    missing = sorted(wanted.difference(found))
    if missing:
        raise RuntimeError(f'dataset_index values not found in configured stream: {missing}')
    return found

def grid_to_notes(model, grid, bpm=80):
    arr = grid.detach().cpu().numpy() if hasattr(grid, 'detach') else np.asarray(grid)
    if arr.ndim == 4:
        arr = arr[0]
    return model.decoder.grid_to_pr_and_notes(arr, bpm=bpm, start=0.0)[1]


def symbolic_step_seconds(bpm: float = 80) -> float:
    return 0.25 * 60.0 / float(bpm)


def notes_to_midi(notes, path: Path, bpm=80) -> None:
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    midi.time_signature_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]
    inst = pretty_midi.Instrument(program=0, name='piano')
    inst.notes.extend(notes)
    midi.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def notes_extent(notes, bpm: float = 80) -> Tuple[int, int, int]:
    step_sec = symbolic_step_seconds(bpm)
    max_step = 32
    pitches = [int(note.pitch) for note in notes if 0 <= int(note.pitch) < 128]
    for note in notes:
        max_step = max(max_step, int(math.ceil(float(note.end) / step_sec)))
    if not pitches:
        return max_step, 48, 84
    return max_step, max(0, min(pitches) - 2), min(127, max(pitches) + 2)


def midi_pitch_label(pitch: int) -> str:
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f'{names[pitch % 12]}{pitch // 12 - 1}'


def write_piano_roll(notes, path: Path, title: str, bpm: float = 80) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    step_sec = symbolic_step_seconds(bpm)
    max_step, min_pitch, max_pitch = notes_extent(notes, bpm=bpm)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for note in sorted(notes, key=lambda item: (item.pitch, item.start, item.end)):
        start_step = float(note.start) / step_sec
        end_step = float(note.end) / step_sec
        duration = max(0.15, end_step - start_step)
        intensity = max(0.25, min(1.0, float(note.velocity) / 127.0))
        ax.broken_barh(
            [(start_step, duration)],
            (int(note.pitch) - 0.42, 0.84),
            facecolors=(0.1, 0.38, 0.78, intensity),
            edgecolors=(0.02, 0.08, 0.16, 0.85),
            linewidth=0.45,
        )
    ax.set_xlim(0, max(32, max_step))
    ax.set_ylim(min_pitch, max_pitch)
    ax.set_title(title)
    ax.set_xlabel(f'symbolic position: 1 step = 1/16 note = {step_sec:.4f}s at {bpm:g} BPM')
    ax.set_ylabel('pitch (MIDI / note)')
    beat_ticks = list(range(0, max(32, max_step) + 1, 4))
    ax.set_xticks(beat_ticks)
    ax.set_xticklabels([str(t) for t in beat_ticks])
    for beat in beat_ticks:
        ax.axvline(beat, color='0.82', linewidth=0.8, zorder=0)
    for bar in range(0, max(32, max_step) + 1, 16):
        ax.axvline(bar, color='0.25', linewidth=1.15, zorder=0)
        ax.text(bar + 0.15, max_pitch - 0.4, f'bar {bar // 16 + 1}', fontsize=8, va='top', color='0.2')
    pitch_ticks = list(range((min_pitch // 12) * 12, max_pitch + 1, 12))
    pitch_ticks = [p for p in pitch_ticks if min_pitch <= p <= max_pitch]
    ax.set_yticks(pitch_ticks)
    ax.set_yticklabels([f'{p} / {midi_pitch_label(p)}' for p in pitch_ticks])
    ax.grid(axis='y', color='0.9', linewidth=0.6)
    secax = ax.secondary_xaxis('top', functions=(lambda step: step * step_sec, lambda sec: sec / step_sec))
    secax.set_xlabel('seconds')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def synthesize_wav_from_notes(notes, wav_path: Path, bpm: float = 80, sample_rate: int = 22050) -> Dict[str, Any]:
    if not notes:
        return {'status': 'fallback', 'reason': 'no notes available for sine synthesis', 'wav_path': None, 'method': 'sine'}
    duration = max(float(note.end) for note in notes) + 0.35
    audio = np.zeros(int(duration * sample_rate) + 1, dtype=np.float32)
    for note in notes:
        start = max(0, int(float(note.start) * sample_rate))
        end = min(len(audio), max(start + 1, int(float(note.end) * sample_rate)))
        n = end - start
        if n <= 0:
            continue
        t = np.arange(n, dtype=np.float32) / float(sample_rate)
        freq = 440.0 * (2.0 ** ((int(note.pitch) - 69) / 12.0))
        tone = np.sin(2.0 * np.pi * freq * t)
        tone += 0.35 * np.sin(2.0 * np.pi * freq * 2.0 * t)
        envelope = np.ones(n, dtype=np.float32)
        attack = min(n, int(0.01 * sample_rate))
        release = min(n, int(0.04 * sample_rate))
        if attack > 1:
            envelope[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
        if release > 1:
            envelope[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
        audio[start:end] += tone.astype(np.float32) * envelope * (float(note.velocity) / 127.0) * 0.18
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.88
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (audio * 32767.0).astype('<i2')
    with wave.open(str(wav_path), 'wb') as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
    return {
        'status': 'ok',
        'reason': 'generated with built-in sine fallback; install fluidsynth + soundfont for piano timbre',
        'wav_path': str(wav_path),
        'method': 'sine',
        'bpm': bpm,
        'sample_rate': sample_rate,
    }


def find_soundfont(explicit: Optional[str] = None) -> Optional[str]:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.getenv('SF2_PATH') or os.getenv('SOUNDFONT')
    if env:
        candidates.append(Path(env))
    for base in [Path('/usr/share/sounds/sf2'), Path('/usr/share/soundfonts'), ROOT]:
        if base.exists():
            candidates.extend(list(base.rglob('*.sf2')) + list(base.rglob('*.sf3')))
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def synthesize_wav(midi_path: Path, wav_path: Path, soundfont: Optional[str] = None, notes=None, bpm: float = 80) -> Dict[str, Any]:
    fluidsynth = shutil.which('fluidsynth')
    sf = find_soundfont(soundfont)
    if fluidsynth and sf:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [fluidsynth, '-ni', sf, str(midi_path), '-F', str(wav_path), '-r', '44100']
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode == 0:
            return {'status': 'ok', 'reason': None, 'wav_path': str(wav_path), 'method': 'fluidsynth', 'soundfont': sf}
        if notes is None:
            return {'status': 'fallback', 'reason': result.stderr.strip() or result.stdout.strip(), 'wav_path': None}
    if notes is not None:
        fallback = synthesize_wav_from_notes(notes, wav_path, bpm=bpm)
        missing = []
        if not fluidsynth:
            missing.append('fluidsynth executable not found')
        if not sf:
            missing.append('no .sf2/.sf3 soundfont found')
        if missing and fallback['status'] == 'ok':
            fallback['source_fallback_reason'] = '; '.join(missing)
        return fallback
    if not fluidsynth:
        return {'status': 'fallback', 'reason': 'fluidsynth executable not found', 'wav_path': None}
    return {'status': 'fallback', 'reason': 'no .sf2/.sf3 soundfont found; set SF2_PATH or pass --soundfont', 'wav_path': None}



def resolve_local_checkpoint_refs(cfg) -> Tuple[Any, Dict[str, Dict[str, str]]]:
    remapped = {}
    checkpoints = {}
    for label, ref in cfg.checkpoints.items():
        raw_path = Path(ref.path)
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(ROOT / raw_path)
        candidates.extend((ROOT / '_artefatos').glob(f'**/{raw_path.name}'))
        chosen = next((path for path in candidates if path.exists()), None)
        if chosen is not None and str(chosen) != ref.path:
            checkpoints[label] = replace(ref, path=str(chosen))
            remapped[label] = {'configured_path': ref.path, 'local_path': str(chosen)}
        else:
            checkpoints[label] = ref
    if remapped:
        cfg = replace(cfg, checkpoints=checkpoints)
    return cfg, remapped

def _case_ids_from_selection(run: ReviewRun, selection_manifest: Optional[str] = None) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    manifest_path = Path(selection_manifest) if selection_manifest else run.selection_manifest_path()
    if not manifest_path.is_absolute():
        manifest_path = run.run_dir / manifest_path
    if not manifest_path.exists():
        return [], None
    selection = json.loads(manifest_path.read_text())
    case_ids = [item["compound_id"] for item in selection.get("selected_cases", [])]
    return case_ids, selection


def _nonzero_file(path: Optional[str]) -> bool:
    return bool(path) and Path(path).exists() and Path(path).stat().st_size > 0


def generate_assets(
    run_dir: str,
    strata: Optional[List[str]] = None,
    max_cases: Optional[int] = None,
    config: Optional[str] = None,
    soundfont: Optional[str] = None,
    selection_manifest: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    run = ReviewRun.from_run_dir(run_dir)
    cfg = find_config_for_run(run, config)
    if device:
        cfg = replace(cfg, device=device)
    cfg, checkpoint_remaps = resolve_local_checkpoint_refs(cfg)
    selection_case_ids, selection = _case_ids_from_selection(run, selection_manifest)
    if selection_case_ids:
        case_ids = selection_case_ids
        selection_mode = 'selection_manifest'
    else:
        limit = 2 if max_cases is None else max_cases
        case_ids = run.selected_case_ids(strata=strata, max_cases=limit)
        selection_mode = 'ranking_strata'
    models = {label: load_model_for_checkpoint(ref, cfg.device) for label, ref in cfg.checkpoints.items()}
    authors_model = models['authors']
    rows_by_case = {compound_id: run.row_by_id(compound_id) for compound_id in case_ids}
    segment_cache = load_segments_by_dataset_indices(cfg, [int(row['dataset_index']) for row in rows_by_case.values()])
    import torch
    manifest = {
        'run_dir': str(run.run_dir),
        'selection_mode': selection_mode,
        'selection_manifest_path': selection.get('selection_manifest_path') if selection else None,
        'references': ['original'] + list(models.keys()),
        'checkpoint_remaps': checkpoint_remaps,
        'cases': [],
    }
    with torch.no_grad():
        for case_number, compound_id in enumerate(case_ids, start=1):
            print(f'[phase9-assets] rendering case {case_number}/{len(case_ids)}: {compound_id}', flush=True)
            row = rows_by_case[compound_id]
            identity, x, c, pr_mat = segment_cache[int(row['dataset_index'])]
            device = next(authors_model.parameters()).device
            x = x.long().to(device)
            c = c.float().to(device)
            pr_mat = pr_mat.float().to(device)
            refs = {'original': grid_to_notes(authors_model, x, bpm=80)}
            for label, model in models.items():
                refs[label] = grid_to_notes(model, model.inference(pr_mat, c, sample=False), bpm=80)
            case_dir = run.assets_dir / 'review_cases' / safe_case_dir_name(compound_id)
            case_record = {'compound_id': compound_id, 'case_dir': str(case_dir), 'references': {}, 'integrity': {}}
            for ref_name, notes in refs.items():
                midi_path = case_dir / ref_name / f'{ref_name}.mid'
                png_path = case_dir / ref_name / f'{ref_name}_piano_roll.png'
                wav_path = case_dir / ref_name / f'{ref_name}.wav'
                notes_to_midi(notes, midi_path)
                write_piano_roll(notes, png_path, f'{ref_name}: {compound_id}', bpm=80)
                audio = synthesize_wav(midi_path, wav_path, soundfont=soundfont, notes=notes, bpm=80)
                ref_record = {
                    'midi_path': str(midi_path),
                    'piano_roll_path': str(png_path),
                    'audio': audio,
                    'note_count': len(notes),
                    'bpm': 80,
                    'symbolic_step': '1/16 note',
                    'symbolic_step_seconds': symbolic_step_seconds(80),
                }
                ref_record['integrity'] = {
                    'midi_nonzero': _nonzero_file(ref_record['midi_path']),
                    'piano_roll_nonzero': _nonzero_file(ref_record['piano_roll_path']),
                    'wav_nonzero': _nonzero_file(audio.get('wav_path')),
                }
                case_record['references'][ref_name] = ref_record
            expected_refs = ['original'] + list(models.keys())
            case_record['integrity'] = {
                'expected_references': expected_refs,
                'missing_references': [ref for ref in expected_refs if ref not in case_record['references']],
                'all_midi_nonzero': all(case_record['references'].get(ref, {}).get('integrity', {}).get('midi_nonzero') for ref in expected_refs),
                'all_piano_roll_nonzero': all(case_record['references'].get(ref, {}).get('integrity', {}).get('piano_roll_nonzero') for ref in expected_refs),
                'all_wav_nonzero': all(case_record['references'].get(ref, {}).get('integrity', {}).get('wav_nonzero') for ref in expected_refs),
            }
            manifest['cases'].append(case_record)
            print(f'[phase9-assets] completed case {case_number}/{len(case_ids)}', flush=True)
    out = run.asset_manifest_path()
    manifest['asset_manifest_path'] = str(out)
    manifest['integrity'] = {
        'case_count': len(manifest['cases']),
        'all_cases_have_expected_references': all(not case['integrity']['missing_references'] for case in manifest['cases']),
        'all_midi_nonzero': all(case['integrity']['all_midi_nonzero'] for case in manifest['cases']),
        'all_piano_roll_nonzero': all(case['integrity']['all_piano_roll_nonzero'] for case in manifest['cases']),
        'all_wav_nonzero': all(case['integrity']['all_wav_nonzero'] for case in manifest['cases']),
    }
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'[phase9-assets] wrote {out}', flush=True)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--config')
    ap.add_argument('--strata', nargs='*', default=None)
    ap.add_argument('--max-cases', type=int, default=None)
    ap.add_argument('--selection-manifest', help='Path to selected_cases_24.json; defaults to RUN_DIR/review_selection/selected_cases_24.json when present')
    ap.add_argument('--device', help='Override config device for local asset rendering, for example cpu')
    ap.add_argument('--soundfont')
    args = ap.parse_args(argv)
    result = generate_assets(args.run_dir, strata=args.strata, max_cases=args.max_cases, config=args.config, soundfont=args.soundfont, selection_manifest=args.selection_manifest, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
