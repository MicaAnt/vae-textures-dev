#!/usr/bin/env python3
"""Review helpers for POP909 conditioned reconstruction outputs."""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

VALID_LABELS = {"authors better", "epoch4 better", "epoch6 better", "ours better", "comparable", "conflict", "unclear"}
FIXED_REVIEW_PLAN = [
    ("near_tie", 4, "Near metric ties for comparable/equivalence inspection"),
    ("authors_much_better", 4, "Authors lower-loss cases"),
    ("epoch4_lower_loss", 4, "Epoch 4 lower-loss or closest cases"),
    ("epoch6_lower_loss", 4, "Epoch 6 lower-loss or closest cases"),
    ("epoch4_epoch6_disagreement", 4, "Largest epoch 4 versus epoch 6 disagreements"),
    ("median_representative", 4, "Median/representative cases"),
]
FINAL_REVIEW_MIN_ROWS = 24


class ReviewError(RuntimeError):
    pass


@dataclass
class ReviewRun:
    run_dir: Path
    csv_path: Path
    manifest_path: Path
    rankings_path: Path
    summary_json_path: Path
    summary_md_path: Path
    assets_dir: Path
    notes_dir: Path

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "ReviewRun":
        base = Path(run_dir).expanduser()
        if not base.is_absolute():
            base = Path.cwd() / base
        paths = {
            "csv_path": base / "tables" / "comparison_wide.csv",
            "manifest_path": base / "manifests" / "comparison_manifest.jsonl",
            "rankings_path": base / "rankings" / "ranking_strata.json",
            "summary_json_path": base / "summaries" / "summary.json",
            "summary_md_path": base / "summaries" / "summary.md",
        }
        missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
        if missing:
            raise ReviewError("Missing required review files:\n" + "\n".join(missing))
        assets = base / "assets"
        notes = base / "review_notes"
        assets.mkdir(parents=True, exist_ok=True)
        notes.mkdir(parents=True, exist_ok=True)
        return cls(base, assets_dir=assets, notes_dir=notes, **paths)

    def selection_dir(self) -> Path:
        path = self.run_dir / "review_selection"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def selection_manifest_path(self) -> Path:
        return self.selection_dir() / "selected_cases_24.json"

    def asset_manifest_path(self) -> Path:
        return self.assets_dir / "review_asset_manifest.json"

    def file_map(self) -> Dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "csv": str(self.csv_path),
            "manifest_jsonl": str(self.manifest_path),
            "rankings_json": str(self.rankings_path),
            "summary_json": str(self.summary_json_path),
            "summary_md": str(self.summary_md_path),
            "assets_dir": str(self.assets_dir),
            "review_notes_dir": str(self.notes_dir),
            "human_notes_jsonl": str(self.notes_path()),
            "selection_manifest": str(self.selection_manifest_path()),
            "asset_manifest": str(self.asset_manifest_path()),
        }

    def rows(self) -> List[Dict[str, Any]]:
        with self.csv_path.open(newline="") as fh:
            return list(csv.DictReader(fh))

    def manifest_records(self) -> List[Dict[str, Any]]:
        with self.manifest_path.open() as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def rankings(self) -> Dict[str, List[str]]:
        return json.loads(self.rankings_path.read_text())

    def summary(self) -> Dict[str, Any]:
        return json.loads(self.summary_json_path.read_text())

    def notes_path(self) -> Path:
        return self.notes_dir / "human_notes.jsonl"

    def global_stats(self) -> Dict[str, Any]:
        rows = self.rows()
        summary = self.summary()
        rankings = self.rankings()
        stratum_counts = {name: len(ids) for name, ids in rankings.items()}
        loss_stats = summary.get("loss_component_stats", {})
        return {
            "run_id": summary.get("run_id"),
            "run_role": summary.get("run_role"),
            "manifest_run_role": summary.get("manifest_run_role"),
            "split": summary.get("split"),
            "row_count": len(rows),
            "non_final_warning": bool(summary.get("non_final_warning")),
            "checkpoints": summary.get("checkpoints", {}),
            "candidate_labels": summary.get("candidate_labels", list(summary.get("checkpoints", {}).keys())),
            "delta_pairs": summary.get("delta_pairs", []),
            "canonical_delta_pair": summary.get("canonical_delta_pair"),
            "loss_recon_conflict_count": summary.get("loss_recon_conflict_count", 0),
            "stratum_counts": stratum_counts,
            "loss_component_stats": loss_stats,
            "loss_stats": loss_stats.get("loss", {}),
            "recon_loss_stats": loss_stats.get("recon_loss", {}),
            "delta_loss": loss_stats.get("loss", {}).get("delta", {}),
            "delta_recon_loss": loss_stats.get("recon_loss", {}).get("delta", {}),
            "delta_chord_loss": loss_stats.get("chord_loss", {}).get("delta", {}),
        }

    def selected_case_ids(self, strata: Optional[Iterable[str]] = None, max_cases: Optional[int] = None) -> List[str]:
        rankings = self.rankings()
        wanted = list(strata) if strata else ["near_tie", "authors_much_better", "ours_much_better", "epoch4_lower_loss", "epoch6_lower_loss", "epoch4_epoch6_disagreement", "median_representative", "loss_recon_conflict"]
        selected: List[str] = []
        seen = set()
        for stratum in wanted:
            for compound_id in rankings.get(stratum, []):
                if compound_id not in seen:
                    selected.append(compound_id)
                    seen.add(compound_id)
                if max_cases and len(selected) >= max_cases:
                    return selected
        return selected

    def assert_final_review_input(self, diagnostic_override: bool = False, min_rows: int = FINAL_REVIEW_MIN_ROWS) -> None:
        stats = self.global_stats()
        row_count = int(stats.get("row_count") or 0)
        summary = self.summary()
        split_policy = summary.get("split_policy", {})
        problems = []
        if row_count < min_rows:
            problems.append(f"row_count {row_count} is smaller than required final-review minimum {min_rows}")
        if stats.get("non_final_warning"):
            problems.append("summary marks this RUN_DIR as non-final evidence")
        if split_policy and not split_policy.get("full_split_target", False):
            problems.append("split_policy.full_split_target is not true")
        if split_policy and split_policy.get("fallback_used", False):
            problems.append("split_policy.fallback_used is true")
        if problems and not diagnostic_override:
            raise ReviewError("Refusing final 24-case selection from diagnostic/smoke RUN_DIR: " + "; ".join(problems))

    def select_fixed_review_cases(self, diagnostic_override: bool = False, write: bool = True) -> Dict[str, Any]:
        self.assert_final_review_input(diagnostic_override=diagnostic_override)
        rankings = self.rankings()
        rows_by_id = {row.get("compound_id"): row for row in self.rows()}
        source_by_id: Dict[str, List[str]] = {}
        for source, ids in rankings.items():
            for cid in ids:
                source_by_id.setdefault(cid, []).append(source)

        selected: List[Dict[str, Any]] = []
        seen = set()
        substitutions: List[Dict[str, Any]] = []

        def add_case(requested: str, actual: str, cid: str, reason: str, substitution: bool) -> bool:
            if cid in seen or cid not in rows_by_id:
                return False
            seen.add(cid)
            row = rows_by_id[cid]
            selected.append({
                "slot": len(selected) + 1,
                "compound_id": cid,
                "requested_stratum": requested,
                "actual_source_stratum": actual,
                "selection_reason": reason,
                "substitution": substitution,
                "dataset_index": int(row.get("dataset_index", 0)),
                "loader_index": int(row.get("loader_index", 0)),
                "sorted_file_index": int(row.get("sorted_file_index", 0)),
                "loss_recon_conflict": str(row.get("loss_recon_conflict", "")).lower() == "true",
                "authors_loss": _maybe_float(row.get("authors_loss")),
                "ours_epoch4_loss": _maybe_float(row.get("ours_epoch4_loss")),
                "ours_epoch6_loss": _maybe_float(row.get("ours_epoch6_loss")),
                "delta_ours_epoch4_minus_authors_loss": _maybe_float(row.get("delta_ours_epoch4_minus_authors_loss")),
                "delta_ours_epoch6_minus_authors_loss": _maybe_float(row.get("delta_ours_epoch6_minus_authors_loss")),
                "delta_ours_epoch6_minus_ours_epoch4_loss": _maybe_float(row.get("delta_ours_epoch6_minus_ours_epoch4_loss")),
            })
            return True

        for requested, target_count, reason in FIXED_REVIEW_PLAN:
            before = sum(1 for item in selected if item["requested_stratum"] == requested)
            for cid in rankings.get(requested, []):
                if sum(1 for item in selected if item["requested_stratum"] == requested) >= target_count:
                    break
                add_case(requested, requested, cid, reason, False)
            after = sum(1 for item in selected if item["requested_stratum"] == requested)
            if after - before < target_count:
                substitutions.append({
                    "requested_stratum": requested,
                    "requested_count": target_count,
                    "selected_count": after - before,
                    "needed_substitutions": target_count - (after - before),
                    "reason": "stratum did not have enough unique cases after deduplication",
                })

        target_total = sum(count for _, count, _ in FIXED_REVIEW_PLAN)
        fallback_order = [
            "loss_recon_conflict",
            "ours_much_better",
            "near_tie",
            "authors_much_better",
            "epoch4_lower_loss",
            "epoch6_lower_loss",
            "epoch4_epoch6_disagreement",
            "median_representative",
        ]
        for source in fallback_order:
            if len(selected) >= target_total:
                break
            for cid in rankings.get(source, []):
                if len(selected) >= target_total:
                    break
                add_case("substitution_fill", source, cid, "Substitution fill to maintain 24 unique review cases", True)

        if len(selected) < target_total and not diagnostic_override:
            raise ReviewError(f"Only selected {len(selected)} unique cases; expected {target_total}")

        manifest = {
            "schema_version": "1.0",
            "run_dir": str(self.run_dir),
            "selection_manifest_path": str(self.selection_manifest_path()),
            "selection_count": len(selected),
            "target_count": target_total,
            "diagnostic_override": diagnostic_override,
            "source_summary": {
                "run_id": self.summary().get("run_id"),
                "run_role": self.summary().get("run_role"),
                "row_count": self.summary().get("row_count"),
                "candidate_labels": self.summary().get("candidate_labels"),
                "split_policy": self.summary().get("split_policy"),
            },
            "requested_distribution": [
                {"stratum": name, "target_count": count, "reason": reason}
                for name, count, reason in FIXED_REVIEW_PLAN
            ],
            "substitutions": substitutions,
            "selected_cases": selected,
        }
        if write:
            self.selection_manifest_path().write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest

    def load_selection_manifest(self, path: Optional[str | Path] = None) -> Dict[str, Any]:
        manifest_path = Path(path) if path else self.selection_manifest_path()
        if not manifest_path.is_absolute():
            manifest_path = self.run_dir / manifest_path
        if not manifest_path.exists():
            raise ReviewError(f"selection manifest not found: {manifest_path}")
        return json.loads(manifest_path.read_text())

    def row_by_id(self, compound_id: str) -> Dict[str, Any]:
        for row in self.rows():
            if row.get("compound_id") == compound_id:
                return row
        raise ReviewError(f"case not found in CSV: {compound_id}")

    def notes(self) -> List[Dict[str, Any]]:
        path = self.notes_path()
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def write_note(self, compound_id: str, label: str, notes: str = "", reviewer: str = "human") -> Dict[str, Any]:
        if label not in VALID_LABELS:
            raise ValueError(f"invalid label {label!r}; expected one of {sorted(VALID_LABELS)}")
        record = {"compound_id": compound_id, "label": label, "notes": notes, "reviewer": reviewer}
        with self.notes_path().open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def label_summary(self) -> Dict[str, Any]:
        notes = self.notes()
        counts = Counter(note["label"] for note in notes)
        total = sum(counts.values())
        return {
            "total_notes": total,
            "counts": dict(counts),
            "percentages": {k: (v / total * 100.0 if total else 0.0) for k, v in counts.items()},
        }


def print_file_map(run: ReviewRun) -> None:
    for key, value in run.file_map().items():
        print(f"{key}: {value}")


def print_global_stats(run: ReviewRun) -> None:
    print(json.dumps(run.global_stats(), indent=2, sort_keys=True))


def _maybe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
