#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

CLASSIFICATION_LABELS = ["canonical", "candidate", "legacy", "exploration", "generated artifact", "unknown"]

SOURCE_RULES = {
    "utilProcessing.py": ("candidate", "Core MIDI + metadata -> NPZ treatment functions: GenDataSet, get_fund, prToChroma, combineFundChroma."),
    "processMidiBatch.py": ("candidate", "Operational COMMU batch wrapper around GenDataSet; path handling needs hardening."),
    "processMidiPath.py": ("legacy", "Folder wrapper useful for small proofs; defaults target midiDataTest/commuTestNPZ."),
    "base_model/datasetCOMMU.py": ("candidate", "COMMU loader candidate with path resolution, ID normalization, and 4/4 filtering."),
    "base_model/dataset_loaders_commu.py": ("candidate", "Training/loader adapter for datasetCOMMU."),
    "base_model/experiments/data_factory.py": ("candidate", "Config-driven COMMU loader integration with bounded sample support."),
    "NotebooksVAESymTex/generate_commu_enriched_loss_dataset.py": ("candidate", "Rich per-segment forward/loss output path preserving metadata."),
    "NotebooksVAESymTex/recompute_commu_loss_components.py": ("legacy", "Older overlapping enriched loss recomputation path."),
    "NEWcalcLatentBatchLoos.py": ("legacy", "Older batch latent/loss script with metadata preservation and naming/path debt."),
    "calc_latent_loss.py": ("legacy", "Older latent/loss path; classify before reuse."),
    "calcLatentBatchLoss.py": ("legacy", "Older latent/loss path; classify before reuse."),
}

ARTIFACT_DIRS = [
    "COMMUDataset/npzFiles",
    "COMMUDataset/losses",
    "COMMUDataset/losses_enriched",
    "COMMUDataset/midiFiles",
    "COMMUDataset/batches",
    "COMMUDataset/batchesNPZ",
    "COMMUDataset/enriched_loss_batches",
    "commuTestNPZ",
    "commuTestNPZ/losses",
    "losses",
    "NotebooksVAESymTex/_cache",
]

EXPLORATION_HINTS = ["draftCode", "testeFunctions", "notebooksTPMaster", ".ipynb_checkpoints"]


def rel(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def summarize_dir(path: Path, repo_root: Path, limit: int) -> dict[str, Any]:
    files = [p for p in path.rglob("*") if p.is_file()]
    ext_counts = Counter(p.suffix or "[no suffix]" for p in files)
    examples = [rel(p, repo_root) for p in sorted(files)[:limit]]
    return {
        "path": rel(path, repo_root),
        "classification": "generated artifact",
        "file_count": len(files),
        "extension_counts": dict(sorted(ext_counts.items())),
        "examples": examples,
        "bounded_examples": len(examples),
    }


def collect_sources(repo_root: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for rel_path, (classification, reason) in SOURCE_RULES.items():
        p = repo_root / rel_path
        sources.append({
            "path": rel_path,
            "exists": p.exists(),
            "classification": classification if p.exists() else "unknown",
            "reason": reason if p.exists() else "Expected COMMU-related path not found during inventory.",
        })
    for p in sorted(repo_root.rglob("*")):
        if not p.is_file():
            continue
        rp = rel(p, repo_root)
        if any(part in rp for part in EXPLORATION_HINTS) and ("commu" in rp.lower() or p.suffix in {".ipynb", ".png"}):
            sources.append({"path": rp, "exists": True, "classification": "exploration", "reason": "Exploratory/notebook/test visual area; do not treat as canonical without review."})
    return sources


def build_inventory(repo_root: Path, limit: int) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    sources = collect_sources(repo_root)
    artifact_dirs = []
    for rel_dir in ARTIFACT_DIRS:
        p = repo_root / rel_dir
        if p.exists() and p.is_dir():
            artifact_dirs.append(summarize_dir(p, repo_root, limit))
    classifications = {label: [] for label in CLASSIFICATION_LABELS}
    for item in sources:
        classifications.setdefault(item["classification"], []).append(item["path"])
    for item in artifact_dirs:
        classifications.setdefault("generated artifact", []).append(item["path"])
    quarantine_candidates = [
        {"path": item["path"], "reason": item["reason"]}
        for item in sources
        if item["classification"] in {"legacy", "exploration", "unknown"}
    ]
    return {
        "schema_version": "1.0",
        "repo_root": str(repo_root),
        "classification_labels": CLASSIFICATION_LABELS,
        "sources": sources,
        "artifact_directories": artifact_dirs,
        "classifications": classifications,
        "quarantine_candidates": quarantine_candidates,
    }


def write_markdown(inv: dict[str, Any], out: Path) -> None:
    lines = [
        "# COMMU Pipeline Inventory",
        "",
        "Phase 10 inventory classifies existing code and generated artifacts before any quarantine/move.",
        "",
        "## Classification Labels",
        "",
    ]
    for label in CLASSIFICATION_LABELS:
        lines.append(f"- `{label}`")
    lines += ["", "## Source Files", "", "| Path | Classification | Exists | Reason |", "|------|----------------|--------|--------|"]
    for item in inv["sources"]:
        lines.append(f"| `{item['path']}` | `{item['classification']}` | {item['exists']} | {item['reason']} |")
    lines += ["", "## Generated Artifact Directories", "", "| Path | File Count | Extension Counts | Bounded Examples |", "|------|------------|------------------|------------------|"]
    for item in inv["artifact_directories"]:
        lines.append(f"| `{item['path']}` | {item['file_count']} | `{item['extension_counts']}` | {item['bounded_examples']} |")
    lines += ["", "## Quarantine Candidates", ""]
    if inv["quarantine_candidates"]:
        for item in inv["quarantine_candidates"]:
            lines.append(f"- `{item['path']}` — {item['reason']}")
    else:
        lines.append("None identified.")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory COMMU pipeline code and artifacts without moving files.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-artifact-list", type=int, default=25)
    args = parser.parse_args()
    inv = build_inventory(args.repo_root, args.limit_artifact_list)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "commu_pipeline_inventory.json"
    md_path = args.output_dir / "commu_pipeline_inventory.md"
    json_path.write_text(json.dumps(inv, indent=2, ensure_ascii=False) + "\n")
    write_markdown(inv, md_path)
    print(json_path)
    print(md_path)

if __name__ == "__main__":
    main()
