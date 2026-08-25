#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utilProcessing import GenDataSet  # noqa: E402


def read_track_ids(track_id: str | None, track_list: Path | None) -> list[str]:
    ids: list[str] = []
    if track_id:
        ids.append(track_id)
    if track_list:
        ids.extend([line.strip() for line in track_list.read_text().splitlines() if line.strip()])
    seen = set()
    result = []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            result.append(tid)
    if not result:
        raise ValueError("Provide --track-id or --track-list")
    return result


def process_tracks(track_ids: Iterable[str], midi_dir: Path, metadata_csv: Path, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for track_id in track_ids:
        midi = midi_dir / f"{track_id}.mid"
        out = output_dir / f"{track_id}.npz"
        row = {"track_id": track_id, "source_midi": str(midi), "metadata_csv": str(metadata_csv), "output_npz": str(out), "status": "pending", "error": None}
        try:
            GenDataSet(trackId=track_id, dataSetPath=str(midi_dir), csv_path=str(metadata_csv), output_dir=str(output_dir))
            row["status"] = "ok" if out.exists() else "missing_output"
        except Exception as exc:  # noqa: BLE001 - manifest should preserve per-track failures
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean COMMU-first wrapper for MIDI + metadata -> NPZ generation.")
    parser.add_argument("--midi-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-id", default=None)
    parser.add_argument("--track-list", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    track_ids = read_track_ids(args.track_id, args.track_list)
    rows = process_tracks(track_ids, args.midi_dir, args.metadata_csv, args.output_dir)
    manifest = args.manifest or (args.output_dir / "preprocess_manifest.json")
    manifest.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n")
    print(manifest)

if __name__ == "__main__":
    main()
