#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utilProcessing import get_fund  # noqa: E402


def metadata_row(csv_path: Path, track_id: str) -> dict[str, str]:
    with csv_path.open(newline="") as f:
        row = next((r for r in csv.DictReader(f) if r.get("id") == track_id), None)
    if row is None:
        raise ValueError(f"Track id {track_id} not found in {csv_path}")
    return row


def reduced_chords_from_row(row: dict[str, str]) -> list[str]:
    chord_list = ast.literal_eval(row["chord_progressions"])[0]
    return [chord_list[i] for i in range(0, len(chord_list), 2)]


def trace(metadata_csv: Path, npz_path: Path, track_id: str, max_beats: int | None = None) -> dict[str, Any]:
    row = metadata_row(metadata_csv, track_id)
    reduced = reduced_chords_from_row(row)
    fundamentals = get_fund(str(metadata_csv), track_id)
    payload = np.load(npz_path, allow_pickle=True)
    chord = payload["chord"]
    compare_n = min(len(fundamentals), chord.shape[0])
    if max_beats is not None:
        compare_n = min(compare_n, max_beats)
    expected = fundamentals[:compare_n].astype(float)
    actual = chord[:compare_n, 0].astype(float)
    matches = bool(np.array_equal(expected, actual))
    return {
        "track_id": track_id,
        "metadata_csv": str(metadata_csv),
        "npz": str(npz_path),
        "metadata_chord_progressions": row["chord_progressions"],
        "reduced_chords": reduced[:compare_n],
        "fundamentals": fundamentals[:compare_n].astype(int).tolist(),
        "npz_chord_shape": list(chord.shape),
        "npz_chord_first_column": actual.astype(int).tolist(),
        "fundamental_column_matches": matches,
        "checked_beats": compare_n,
        "chroma_columns_nonzero": int(np.count_nonzero(chord[:compare_n, 1:13])),
        "bass_like_column_values": sorted({int(v) for v in chord[:compare_n, 13].tolist()}),
        "notes": "chord_progressions metadata is parsed by get_fund into fundamentals stored in chord[:, 0]; chroma/bass-like columns come from MIDI pitch activity via prToChroma/combineFundChroma.",
    }


def write_md(data: dict[str, Any], path: Path) -> None:
    status = "confirmed" if data["fundamental_column_matches"] else "mismatch"
    path.write_text("\n".join([
        "# COMMU Harmony Trace",
        "",
        f"Track: `{data['track_id']}`",
        f"Status: **{status}**",
        "",
        "## Interpretation",
        "",
        "The Phase 10 expectation is confirmed in a qualified way: `chord_progressions` from COMMU metadata is used to derive the fundamental/root column of the model-facing `chord` matrix. The remaining chroma and bass-like columns are derived from MIDI pitch activity, not directly from the text metadata.",
        "",
        "## Evidence",
        "",
        f"- NPZ chord shape: `{data['npz_chord_shape']}`",
        f"- Checked beats: `{data['checked_beats']}`",
        f"- Fundamental column matches parsed metadata: `{data['fundamental_column_matches']}`",
        f"- Reduced chord examples: `{data['reduced_chords'][:8]}`",
        f"- Fundamental examples: `{data['fundamentals'][:8]}`",
        f"- Bass-like column values: `{data['bass_like_column_values']}`",
        "",
    ]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace COMMU chord_progressions metadata into NPZ chord arrays.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--track-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-beats", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = trace(args.metadata_csv, args.npz, args.track_id, args.max_beats)
    json_path = args.output_dir / "commu_harmony_trace.json"
    md_path = args.output_dir / "commu_harmony_trace.md"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    write_md(data, md_path)
    print(json_path)
    print(md_path)
    if not data["fundamental_column_matches"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
