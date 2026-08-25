#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_KEYS = ["beat", "chord", "melody", "bridge", "piano"]
META_KEYS = ["audio_key", "chord_progressions", "pitch_range", "num_measures", "bpm", "genre", "track_role", "inst", "sample_rhythm", "time_signature"]


def scalar(payload, key: str):
    if key not in payload.files:
        return None
    arr = np.asarray(payload[key])
    return arr.item() if arr.shape == () else arr.tolist()


def audit_file(path: Path) -> dict[str, Any]:
    anomalies: list[str] = []
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        return {"track_id": path.stem, "path": str(path), "load_error": f"{type(exc).__name__}: {exc}", "anomalies": ["load_error"], "anomaly_count": 1}
    missing = [k for k in EXPECTED_KEYS + META_KEYS if k not in data.files]
    if missing:
        anomalies.append("missing_keys:" + ",".join(missing))
    shapes = {k: list(data[k].shape) for k in data.files if k in EXPECTED_KEYS}
    dtypes = {k: str(data[k].dtype) for k in data.files if k in EXPECTED_KEYS}
    if "chord" in data.files and (len(data["chord"].shape) != 2 or data["chord"].shape[1] != 14):
        anomalies.append("unexpected_chord_shape")
    if "beat" in data.files and (len(data["beat"].shape) != 2 or data["beat"].shape[1] != 6):
        anomalies.append("unexpected_beat_shape")
    if "piano" in data.files and (len(data["piano"].shape) != 2 or data["piano"].shape[1] != 8):
        anomalies.append("unexpected_piano_shape")
    if "chord" in data.files and "beat" in data.files and data["chord"].shape[0] != data["beat"].shape[0]:
        anomalies.append("beat_chord_length_mismatch")
    meta_missing = [k for k in META_KEYS if scalar(data, k) in (None, "")]
    if meta_missing:
        anomalies.append("metadata_missing:" + ",".join(meta_missing))
    return {
        "track_id": path.stem,
        "path": str(path),
        "track_role": scalar(data, "track_role"),
        "time_signature": scalar(data, "time_signature"),
        "metadata_missing": meta_missing,
        "shapes": shapes,
        "dtypes": dtypes,
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
    }


def audit(npz_dir: Path, max_files: int | None, sample_per_role: int) -> dict[str, Any]:
    files = sorted(npz_dir.glob("*.npz"))
    if max_files is not None:
        files = files[:max_files]
    rows = [audit_file(p) for p in files]
    role_counts = Counter(str(r.get("track_role")) for r in rows if r.get("track_role") is not None)
    time_counts = Counter(str(r.get("time_signature")) for r in rows if r.get("time_signature") is not None)
    metadata_coverage = {}
    for key in META_KEYS:
        present = 0
        for p in files:
            try:
                d = np.load(p, allow_pickle=True)
                if scalar(d, key) not in (None, ""):
                    present += 1
            except Exception:
                pass
        metadata_coverage[key] = {"present": present, "checked": len(files)}
    shape_summary: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for k, shape in r.get("shapes", {}).items():
            shape_summary[k][str(shape)] += 1
    samples: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        role = r.get("track_role")
        if role and len(samples[str(role)]) < sample_per_role:
            samples[str(role)].append(r["track_id"])
    anomalies = [r for r in rows if r.get("anomaly_count", 0) > 0]
    return {
        "schema_version": "1.0",
        "total_files_checked": len(rows),
        "role_counts": dict(role_counts),
        "metadata_coverage": metadata_coverage,
        "time_signature_counts": dict(time_counts),
        "shape_summary": {k: dict(v) for k, v in shape_summary.items()},
        "anomalies": anomalies,
        "representative_samples": dict(samples),
        "rows": rows,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["track_id", "track_role", "time_signature", "beat_shape", "chord_shape", "piano_shape", "metadata_missing", "anomaly_count"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            shapes = r.get("shapes", {})
            writer.writerow({
                "track_id": r.get("track_id"),
                "track_role": r.get("track_role"),
                "time_signature": r.get("time_signature"),
                "beat_shape": shapes.get("beat"),
                "chord_shape": shapes.get("chord"),
                "piano_shape": shapes.get("piano"),
                "metadata_missing": ";".join(r.get("metadata_missing", [])),
                "anomaly_count": r.get("anomaly_count", 0),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit treated COMMU NPZ files.")
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sample-per-role", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = audit(args.npz_dir, args.max_files, args.sample_per_role)
    json_path = args.output_dir / "commu_npz_audit_summary.json"
    csv_path = args.output_dir / "commu_npz_audit_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_csv(summary["rows"], csv_path)
    print(json_path)
    print(csv_path)

if __name__ == "__main__":
    main()
