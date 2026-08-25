#!/usr/bin/env python3
"""POP909 conditioned reconstruction comparison pipeline.

This script prepares Phase 9 authors-vs-ours evidence without changing the
canonical Phase 8 training path. It is config-first and intentionally strict:
ambiguous checkpoint roles or partial loads fail before evidence is written.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOSS_COMPONENTS = [
    "loss",
    "recon_loss",
    "pl",
    "dl",
    "kl_loss",
    "kl_chd",
    "kl_rhy",
    "chord_loss",
    "root_loss",
    "chroma_loss",
    "bass_loss",
]

VALID_RUN_ROLES = {"smoke", "dry_run", "official_final", "train_diagnostic"}
VALID_MANIFEST_RUN_ROLES = {"dry_run", "official_final", "train_diagnostic"}
VALID_SPLITS = {"validation", "train"}
ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA = ROOT / "_artefatos" / "pop909_conditioned_reconstruction_manifest_schema.json"
DEFAULT_OUTPUT_ROOT = ROOT / "_artefatos" / "pop909-conditioned-reconstruction"


class ConfigError(ValueError):
    """Configuration or protocol contract error."""


class CheckpointError(RuntimeError):
    """Checkpoint loading or compatibility error."""


@dataclass(frozen=True)
class CheckpointRef:
    role: str
    path: str
    provenance_note: str
    run_id: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_group: Optional[str] = None
    accepted: bool = False
    payload_format: str = "auto"
    epoch: Optional[int] = None

    @classmethod
    def from_config(cls, raw: Mapping[str, Any], label: str) -> "CheckpointRef":
        required = ["role", "path", "provenance_note"]
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise ConfigError(f"checkpoint '{label}' missing required fields: {', '.join(missing)}")
        return cls(
            role=str(raw["role"]),
            path=str(raw["path"]),
            provenance_note=str(raw["provenance_note"]),
            run_id=raw.get("run_id"),
            wandb_run_id=raw.get("wandb_run_id"),
            wandb_group=raw.get("wandb_group"),
            accepted=bool(raw.get("accepted", False)),
            payload_format=str(raw.get("payload_format", "auto")),
            epoch=int(raw["epoch"]) if raw.get("epoch") is not None else None,
        )

    def manifest_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "run_id": self.run_id,
            "wandb_run_id": self.wandb_run_id,
            "wandb_group": self.wandb_group,
            "provenance_note": self.provenance_note,
            "accepted": self.accepted,
            "payload_format": self.payload_format,
            "epoch": self.epoch,
        }


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    run_role: str
    split: str
    sample_count: Optional[int]
    selection_seed: int
    ordering_rule: str
    output_root: Path
    device: str
    loader: Dict[str, Any]
    split_policy: Dict[str, Any]
    checkpoints: Dict[str, CheckpointRef]
    asset_generation: Dict[str, Any]
    schema_path: Path
    overwrite: bool = False

    @property
    def manifest_run_role(self) -> str:
        return "dry_run" if self.run_role == "smoke" else self.run_role

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id


def _now_run_id(prefix: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON config {path}: {exc}") from exc


def parse_config(path: Path) -> RunConfig:
    raw = load_json(path)
    required = ["run_role", "split", "selection_seed", "ordering_rule", "checkpoints"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ConfigError(f"config missing required fields: {', '.join(missing)}")

    run_role = str(raw["run_role"])
    if run_role not in VALID_RUN_ROLES:
        raise ConfigError(f"invalid run_role '{run_role}', expected one of {sorted(VALID_RUN_ROLES)}")
    split = str(raw["split"])
    if split not in VALID_SPLITS:
        raise ConfigError(f"invalid split '{split}', expected one of {sorted(VALID_SPLITS)}")
    if run_role == "train_diagnostic" and split != "train":
        raise ConfigError("run_role=train_diagnostic must use split=train")
    if split == "train" and run_role != "train_diagnostic":
        raise ConfigError("train split is diagnostic only; use run_role=train_diagnostic")

    checkpoints_raw = raw["checkpoints"]
    if not isinstance(checkpoints_raw, Mapping) or "authors" not in checkpoints_raw:
        raise ConfigError("config.checkpoints must contain at least authors")
    checkpoints = {
        str(label): CheckpointRef.from_config(ref_raw, str(label))
        for label, ref_raw in checkpoints_raw.items()
    }
    validate_checkpoint_roles(run_role, checkpoints)

    sample_count = raw.get("sample_count")
    if sample_count is not None:
        sample_count = int(sample_count)
        if sample_count < 1:
            raise ConfigError("sample_count must be positive or null")

    loader = dict(raw.get("loader", {}))
    loader.setdefault("batch_size", 1)
    loader.setdefault("portion", 8)
    loader.setdefault("shift_low", -6)
    loader.setdefault("shift_high", 5)
    loader.setdefault("num_bar", 2)
    loader.setdefault("contain_chord", True)
    loader.setdefault("seed", int(raw["selection_seed"]))
    loader.setdefault("data_path", "base_model/data/POP09-PIANOROLL-4-bin-quantization")
    loader.setdefault("index_file_path", "base_model/data/index.xlsx")

    fallback_used = bool(raw.get("fallback_used", sample_count is not None))
    split_policy = {
        "official": run_role == "official_final" and split == "validation",
        "full_split_target": bool(raw.get("full_split_target", sample_count is None and split == "validation")),
        "fallback_used": fallback_used,
        "fallback_reason": raw.get("fallback_reason") if fallback_used else None,
        "fallback_seed": int(raw["selection_seed"]) if fallback_used else None,
        "fallback_sample_count": sample_count if fallback_used else None,
        "ordering_rule": str(raw["ordering_rule"]),
    }
    if fallback_used and not split_policy["fallback_reason"]:
        raise ConfigError("fallback_used requires fallback_reason")

    run_id = str(raw.get("run_id") or _now_run_id(f"pop909-{run_role}"))
    return RunConfig(
        run_id=run_id,
        run_role=run_role,
        split=split,
        sample_count=sample_count,
        selection_seed=int(raw["selection_seed"]),
        ordering_rule=str(raw["ordering_rule"]),
        output_root=Path(raw.get("output_root", str(DEFAULT_OUTPUT_ROOT))),
        device=str(raw.get("device", "cpu")),
        loader=loader,
        split_policy=split_policy,
        checkpoints=checkpoints,
        asset_generation=dict(raw.get("asset_generation", {})),
        schema_path=Path(raw.get("manifest_schema", str(DEFAULT_SCHEMA))),
        overwrite=bool(raw.get("overwrite", False)),
    )


def validate_checkpoint_roles(run_role: str, checkpoints: Mapping[str, CheckpointRef]) -> None:
    if "authors" not in checkpoints:
        raise ConfigError("config.checkpoints must contain authors")
    authors = checkpoints["authors"]
    if authors.role != "authors_reference":
        raise ConfigError("authors checkpoint role must be authors_reference")
    candidate_labels = [label for label in checkpoints if label != "authors"]
    if not candidate_labels:
        raise ConfigError("config.checkpoints must contain at least one reconstruction candidate besides authors")
    if run_role in {"smoke", "dry_run"}:
        for label in candidate_labels:
            if checkpoints[label].role not in {"ours_dry_run", "ours_official_final", "ours_best_validation_epoch4", "ours_protocol_final_epoch6"}:
                raise ConfigError(f"{run_role} has unsupported candidate role for {label}: {checkpoints[label].role}")
    if run_role == "official_final":
        required = {"authors", "ours_epoch4", "ours_epoch6"}
        missing = sorted(required - set(checkpoints))
        if missing:
            raise ConfigError("official_final requires checkpoint(s): " + ", ".join(missing))
        expected_roles = {
            "ours_epoch4": "ours_best_validation_epoch4",
            "ours_epoch6": "ours_protocol_final_epoch6",
        }
        for label, role in expected_roles.items():
            ref = checkpoints[label]
            if ref.role != role:
                raise ConfigError(f"official_final requires {label} role {role}")
            if not ref.accepted:
                raise ConfigError(f"official_final requires accepted=true for {label}")
            if not ref.path or "TODO" in ref.path:
                raise ConfigError(f"official_final requires a real checkpoint path for {label}")
    if run_role == "train_diagnostic":
        for label in candidate_labels:
            if checkpoints[label].role not in {"ours_dry_run", "ours_official_final", "ours_best_validation_epoch4", "ours_protocol_final_epoch6"}:
                raise ConfigError(f"train_diagnostic has unsupported candidate role for {label}: {checkpoints[label].role}")


def candidate_labels(cfg: RunConfig) -> List[str]:
    return list(cfg.checkpoints.keys())


def reconstruction_candidate_labels(cfg: RunConfig) -> List[str]:
    return [label for label in candidate_labels(cfg) if label != "original"]


def canonical_delta_pair(cfg: RunConfig) -> Tuple[str, str]:
    if "ours" in cfg.checkpoints:
        return "ours", "authors"
    if "ours_epoch6" in cfg.checkpoints:
        return "ours_epoch6", "authors"
    labels = [label for label in cfg.checkpoints if label != "authors"]
    return labels[0], "authors"


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def validate_config_files(cfg: RunConfig, check_files: bool = False) -> Dict[str, Any]:
    info = resolved_metadata(cfg)
    if check_files:
        missing = [str(resolve_path(ref.path)) for ref in cfg.checkpoints.values() if not resolve_path(ref.path).exists()]
        if missing:
            raise CheckpointError("missing checkpoint file(s): " + ", ".join(missing))
    return info


def resolved_metadata(cfg: RunConfig) -> Dict[str, Any]:
    return {
        "run_id": cfg.run_id,
        "run_role": cfg.run_role,
        "manifest_run_role": cfg.manifest_run_role,
        "split": cfg.split,
        "sample_count": cfg.sample_count,
        "selection_seed": cfg.selection_seed,
        "ordering_rule": cfg.ordering_rule,
        "output_root": str(cfg.output_root),
        "run_dir": str(cfg.run_dir),
        "split_policy": cfg.split_policy,
        "loader": cfg.loader,
        "checkpoints": {k: v.manifest_dict() for k, v in cfg.checkpoints.items()},
    }


def _import_torch_stack():
    base = ROOT / "base_model"
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch  # type: ignore
    from base_model.model import DisentangleVAE  # type: ignore
    return torch, DisentangleVAE


def _rename_author_keys(state: Mapping[str, Any]) -> Dict[str, Any]:
    renamed = dict(state)
    for name in list(renamed.keys()):
        renamed[name.replace("rhy_encoder", "txt_encoder")] = renamed.pop(name)
    for name in list(renamed.keys()):
        renamed[name.replace("chd_decoder", "xxyy")] = renamed.pop(name)
    for name in list(renamed.keys()):
        renamed[name.replace("decoder", "pnotree_decoder")] = renamed.pop(name)
    for name in list(renamed.keys()):
        renamed[name.replace("xxyy", "chd_decoder")] = renamed.pop(name)
    return renamed


def extract_model_state(payload: Any, ref: CheckpointRef) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"checkpoint payload is not a dict: {ref.path}")
    if ref.role == "authors_reference":
        if "model_state_dict" in payload:
            raise CheckpointError("authors_reference should be a plain released checkpoint, not full training state")
        return _rename_author_keys(payload)
    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
        if not isinstance(state, Mapping):
            raise CheckpointError("model_state_dict is not a mapping")
        return state
    if ref.payload_format in {"plain_state_dict", "auto"}:
        return payload
    raise CheckpointError(f"unsupported checkpoint payload for role {ref.role}")


def load_model_for_checkpoint(ref: CheckpointRef, device_name: str = "cpu"):
    torch, DisentangleVAE = _import_torch_stack()
    path = resolve_path(ref.path)
    if not path.exists():
        raise CheckpointError(f"checkpoint not found: {path}")
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))

    if ref.role == "authors_reference":
        # The released authors checkpoint uses the interface.py naming scheme
        # (`txt_encoder`, `pnotree_decoder`). Reuse the established adapter from
        # compute_single_loss.py, then wrap the loaded submodules in the
        # canonical base_model.DisentangleVAE used by the training path.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from interface import PolyDisVAE  # type: ignore

        interface = PolyDisVAE.init_model(device=device)
        try:
            interface.load_model(str(path))
        except Exception as exc:
            raise CheckpointError(f"failed to load authors checkpoint {path}: {exc}") from exc
        model = DisentangleVAE(
            "disvae",
            device,
            interface.chd_encoder,
            interface.txt_encoder,
            interface.pnotree_decoder,
            interface.chd_decoder,
        )
        model.to(device)
        model.eval()
        return model

    model = DisentangleVAE.init_model(device=device)
    payload = torch.load(str(path), map_location=device, weights_only=False)
    state = extract_model_state(payload, ref)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise CheckpointError(f"failed to load checkpoint {path}: {exc}") from exc
    model.to(device)
    model.eval()
    return model


def prepare_output_dirs(cfg: RunConfig) -> Dict[str, Path]:
    run_dir = cfg.run_dir
    if run_dir.exists():
        if cfg.overwrite:
            shutil.rmtree(run_dir)
        else:
            raise ConfigError(f"run directory already exists: {run_dir}; set overwrite=true to replace")
    dirs = {
        "run": run_dir,
        "config": run_dir / "config",
        "tables": run_dir / "tables",
        "manifests": run_dir / "manifests",
        "summaries": run_dir / "summaries",
        "rankings": run_dir / "rankings",
        "assets": run_dir / "assets",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_git_metadata() -> Dict[str, Any]:
    def run(args: Sequence[str]) -> Optional[str]:
        try:
            return subprocess.check_output(args, cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_segment_identity(split: str, dataset_index: int, loader_index: int, cfg: RunConfig, npz_path: Optional[str] = None, sorted_file_index: Optional[int] = None, shift: int = 0) -> Dict[str, Any]:
    npz_path = npz_path or f"unknown-pop909-source-{dataset_index:06d}.npz"
    sorted_file_index = dataset_index if sorted_file_index is None else sorted_file_index
    num_bar = int(cfg.loader.get("num_bar", 2))
    seed = int(cfg.loader.get("seed", cfg.selection_seed))
    compound = f"{split}|seed={seed}|file={sorted_file_index}|dataset={dataset_index}|loader={loader_index}|shift={shift}|bars={num_bar}"
    return {
        "compound_id": compound,
        "npz_path": npz_path,
        "sorted_file_index": int(sorted_file_index),
        "dataset_index": int(dataset_index),
        "loader_index": int(loader_index),
        "shift": int(shift),
        "num_bar": num_bar,
        "loader_seed": seed,
        "metadata": {},
    }


def iter_real_segments(cfg: RunConfig):
    torch, _ = _import_torch_stack()
    from base_model.dataset import prepare_dataset  # type: ignore

    cwd = Path.cwd()
    os.chdir(ROOT / "base_model")
    try:
        train_loader, val_loader = prepare_dataset(
            int(cfg.loader.get("seed", cfg.selection_seed)),
            int(cfg.loader.get("batch_size", 1)),
            int(cfg.loader.get("batch_size", 1)),
            portion=int(cfg.loader.get("portion", 8)),
            shift_low=int(cfg.loader.get("shift_low", -6)),
            shift_high=int(cfg.loader.get("shift_high", 5)),
            num_bar=int(cfg.loader.get("num_bar", 2)),
            contain_chord=bool(cfg.loader.get("contain_chord", True)),
            random_train=False,
            random_val=False,
        )
        loader = val_loader if cfg.split == "validation" else train_loader
        count = 0
        for batch in loader:
            _, _, pr_mat, x, c, _ = batch
            batch_size = x.shape[0]
            for i in range(batch_size):
                if cfg.sample_count is not None and count >= cfg.sample_count:
                    return
                dataset_index = count
                identity = build_segment_identity(cfg.split, dataset_index, count, cfg, shift=0)
                yield identity, x[i:i+1], c[i:i+1], pr_mat[i:i+1]
                count += 1
    finally:
        os.chdir(cwd)


def tensor_loss_dict(loss_tuple: Sequence[Any]) -> Dict[str, float]:
    values = list(loss_tuple)
    if len(values) != len(LOSS_COMPONENTS):
        raise RuntimeError(f"expected {len(LOSS_COMPONENTS)} loss components, got {len(values)}")
    result: Dict[str, float] = {}
    for name, value in zip(LOSS_COMPONENTS, values):
        if hasattr(value, "detach"):
            scalar = value.detach().mean().cpu().item()
        else:
            scalar = float(value)
        result[name] = float(scalar)
    return result


def compute_deltas(baseline: Mapping[str, float], candidate: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in LOSS_COMPONENTS:
        signed = float(candidate[name]) - float(baseline[name])
        out[f"{name}_signed"] = signed
        out[f"{name}_abs"] = abs(signed)
    return out


def pair_delta_prefix(candidate: str, baseline: str) -> str:
    return f"delta_{candidate}_minus_{baseline}"


def pair_abs_delta_prefix(candidate: str, baseline: str) -> str:
    return f"abs_delta_{candidate}_minus_{baseline}"


def loss_recon_conflict(delta_loss: float, delta_recon_loss: float) -> bool:
    if delta_loss == 0 or delta_recon_loss == 0:
        return False
    return (delta_loss < 0 < delta_recon_loss) or (delta_recon_loss < 0 < delta_loss)


def configured_delta_pairs(cfg: RunConfig) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if "authors" in cfg.checkpoints:
        for label in cfg.checkpoints:
            if label != "authors":
                pairs.append((label, "authors"))
    if "ours_epoch4" in cfg.checkpoints and "ours_epoch6" in cfg.checkpoints:
        pairs.append(("ours_epoch6", "ours_epoch4"))
    return pairs


def make_row(cfg: RunConfig, segment_id: Mapping[str, Any], losses: Mapping[str, Mapping[str, float]], legacy_ours: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    # Backward-compatible test/API path: make_row(cfg, ident, authors, ours).
    if legacy_ours is not None:
        losses = {"authors": losses, "ours": legacy_ours}  # type: ignore[dict-item]
    row: Dict[str, Any] = {
        "run_id": cfg.run_id,
        "run_role": cfg.manifest_run_role,
        "split": cfg.split,
        "compound_id": segment_id["compound_id"],
        "npz_path": segment_id["npz_path"],
        "sorted_file_index": segment_id["sorted_file_index"],
        "dataset_index": segment_id["dataset_index"],
        "loader_index": segment_id["loader_index"],
        "shift": segment_id["shift"],
        "num_bar": segment_id["num_bar"],
        "loader_seed": segment_id["loader_seed"],
        "stratum": None,
    }
    for label, ref in cfg.checkpoints.items():
        row[f"{label}_role"] = ref.role
        row[f"{label}_path"] = ref.path
    for label, label_losses in losses.items():
        for name in LOSS_COMPONENTS:
            row[f"{label}_{name}"] = float(label_losses[name])

    for candidate, baseline in configured_delta_pairs(cfg):
        if candidate not in losses or baseline not in losses:
            continue
        deltas = compute_deltas(losses[baseline], losses[candidate])
        for name in LOSS_COMPONENTS:
            row[f"{pair_delta_prefix(candidate, baseline)}_{name}"] = float(deltas[f"{name}_signed"])
            row[f"{pair_abs_delta_prefix(candidate, baseline)}_{name}"] = float(deltas[f"{name}_abs"])

    canonical_candidate, canonical_baseline = canonical_delta_pair(cfg)
    canonical_prefix = pair_delta_prefix(canonical_candidate, canonical_baseline)
    canonical_abs_prefix = pair_abs_delta_prefix(canonical_candidate, canonical_baseline)
    for name in LOSS_COMPONENTS:
        row[f"delta_{name}"] = float(row[f"{canonical_prefix}_{name}"])
        row[f"abs_delta_{name}"] = float(row[f"{canonical_abs_prefix}_{name}"])
    row["loss_recon_conflict"] = loss_recon_conflict(float(row["delta_loss"]), float(row["delta_recon_loss"]))
    return row


def _loss_value(row: Mapping[str, Any], label: str) -> float:
    return float(row[f"{label}_loss"])


def rank_rows(rows: List[Dict[str, Any]], per_stratum: int = 6) -> Dict[str, List[Dict[str, Any]]]:
    for row in rows:
        row["rank_by_loss_abs"] = None
        row["rank_by_authors_advantage"] = None
        row["rank_by_ours_advantage"] = None
        row["rank_by_epoch4_advantage"] = None
        row["rank_by_epoch6_advantage"] = None
        row["stratum"] = None

    near = sorted(rows, key=lambda r: (float(r["abs_delta_loss"]), r["compound_id"]))
    authors = sorted(rows, key=lambda r: (-float(r["delta_loss"]), r["compound_id"]))
    ours = sorted(rows, key=lambda r: (float(r["delta_loss"]), r["compound_id"]))
    median_sorted = sorted(rows, key=lambda r: (float(r["delta_loss"]), r["compound_id"]))
    mid = len(median_sorted) // 2
    half = per_stratum // 2
    median = median_sorted[max(0, mid - half): max(0, mid - half) + per_stratum]

    for idx, row in enumerate(near, start=1):
        row["rank_by_loss_abs"] = idx
    for idx, row in enumerate(authors, start=1):
        row["rank_by_authors_advantage"] = idx
    for idx, row in enumerate(ours, start=1):
        row["rank_by_ours_advantage"] = idx

    strata = {
        "near_tie": near[:per_stratum],
        "authors_much_better": [r for r in authors if float(r["delta_loss"]) > 0][:per_stratum],
        "ours_much_better": [r for r in ours if float(r["delta_loss"]) < 0][:per_stratum],
        "median_representative": median,
        "loss_recon_conflict": [r for r in rows if bool(r["loss_recon_conflict"])][:per_stratum],
    }
    if rows and "ours_epoch4_loss" in rows[0]:
        epoch4 = sorted(rows, key=lambda r: (_loss_value(r, "ours_epoch4"), r["compound_id"]))
        for idx, row in enumerate(epoch4, start=1):
            row["rank_by_epoch4_advantage"] = idx
        strata["epoch4_lower_loss"] = [r for r in epoch4 if _loss_value(r, "ours_epoch4") <= min(_loss_value(r, label) for label in ["authors", "ours_epoch6"] if f"{label}_loss" in r)][:per_stratum]
    if rows and "ours_epoch6_loss" in rows[0]:
        epoch6 = sorted(rows, key=lambda r: (_loss_value(r, "ours_epoch6"), r["compound_id"]))
        for idx, row in enumerate(epoch6, start=1):
            row["rank_by_epoch6_advantage"] = idx
        strata["epoch6_lower_loss"] = [r for r in epoch6 if _loss_value(r, "ours_epoch6") <= min(_loss_value(r, label) for label in ["authors", "ours_epoch4"] if f"{label}_loss" in r)][:per_stratum]
    if rows and "delta_ours_epoch6_minus_ours_epoch4_loss" in rows[0]:
        strata["epoch4_epoch6_disagreement"] = sorted(rows, key=lambda r: (-abs(float(r["delta_ours_epoch6_minus_ours_epoch4_loss"])), r["compound_id"]))[:per_stratum]

    for name, values in strata.items():
        if name == "loss_recon_conflict":
            continue
        for row in values:
            if row["stratum"] is None:
                row["stratum"] = name
    return strata


def csv_fieldnames(cfg: Optional[RunConfig] = None) -> List[str]:
    labels = list(cfg.checkpoints.keys()) if cfg else ["authors", "ours", "ours_epoch4", "ours_epoch6"]
    pairs = configured_delta_pairs(cfg) if cfg else [("ours", "authors"), ("ours_epoch4", "authors"), ("ours_epoch6", "authors"), ("ours_epoch6", "ours_epoch4")]
    base = [
        "run_id", "run_role", "split", "compound_id", "npz_path",
        "sorted_file_index", "dataset_index", "loader_index", "shift",
        "num_bar", "loader_seed", "loss_recon_conflict", "stratum",
        "rank_by_loss_abs", "rank_by_authors_advantage", "rank_by_ours_advantage",
        "rank_by_epoch4_advantage", "rank_by_epoch6_advantage",
    ]
    for label in labels:
        base.extend([f"{label}_role", f"{label}_path"])
    for name in LOSS_COMPONENTS:
        for label in labels:
            base.append(f"{label}_{name}")
        base.extend([f"delta_{name}", f"abs_delta_{name}"])
        for candidate, baseline in pairs:
            base.extend([f"{pair_delta_prefix(candidate, baseline)}_{name}", f"{pair_abs_delta_prefix(candidate, baseline)}_{name}"])
    return list(dict.fromkeys(base))


def write_csv(rows: List[Dict[str, Any]], path: Path, cfg: Optional[RunConfig] = None) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fieldnames(cfg), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def manifest_record(cfg: RunConfig, row: Mapping[str, Any]) -> Dict[str, Any]:
    losses = {
        label: {name: float(row[f"{label}_{name}"]) for name in LOSS_COMPONENTS}
        for label in cfg.checkpoints
    }
    deltas: Dict[str, Any] = {}
    for candidate, baseline in configured_delta_pairs(cfg):
        pair_name = f"{candidate}_minus_{baseline}"
        deltas[pair_name] = {
            name: {
                "signed": float(row[f"{pair_delta_prefix(candidate, baseline)}_{name}"]),
                "abs": float(row[f"{pair_abs_delta_prefix(candidate, baseline)}_{name}"]),
            }
            for name in LOSS_COMPONENTS
        }
    # Legacy flat fields consumed by the existing smoke notebook/tests.
    deltas.update({
        "loss_signed": float(row["delta_loss"]),
        "loss_abs": float(row["abs_delta_loss"]),
        "recon_loss_signed": float(row["delta_recon_loss"]),
        "recon_loss_abs": float(row["abs_delta_recon_loss"]),
        "chord_loss_signed": float(row["delta_chord_loss"]),
        "kl_rhy_signed": float(row["delta_kl_rhy"]),
    })
    assets = {
        "original_midi_path": None,
        "original_audio_path": None,
        "piano_roll_figure_path": None,
        "notebook_notes_path": None,
    }
    for label in cfg.checkpoints:
        assets[f"{label}_midi_path"] = None
        assets[f"{label}_audio_path"] = None
    return {
        "manifest_version": "1.0",
        "run_role": cfg.manifest_run_role,
        "split": cfg.split,
        "split_policy": cfg.split_policy,
        "segment_id": {
            "compound_id": row["compound_id"],
            "npz_path": row["npz_path"],
            "sorted_file_index": int(row["sorted_file_index"]),
            "dataset_index": int(row["dataset_index"]),
            "loader_index": int(row["loader_index"]),
            "shift": int(row["shift"]),
            "num_bar": int(row["num_bar"]),
            "loader_seed": int(row["loader_seed"]),
            "metadata": {},
        },
        "loader_config": {
            "batch_size": cfg.loader.get("batch_size"),
            "portion": cfg.loader.get("portion"),
            "shift_low": cfg.loader.get("shift_low"),
            "shift_high": cfg.loader.get("shift_high"),
            "contain_chord": cfg.loader.get("contain_chord"),
            "data_path": cfg.loader.get("data_path"),
            "index_file_path": cfg.loader.get("index_file_path"),
        },
        "checkpoints": {label: ref.manifest_dict() for label, ref in cfg.checkpoints.items()},
        "losses": losses,
        "deltas": deltas,
        "ranking": {
            "stratum": row.get("stratum"),
            "rank_by_loss_abs": row.get("rank_by_loss_abs"),
            "rank_by_authors_advantage": row.get("rank_by_authors_advantage"),
            "rank_by_ours_advantage": row.get("rank_by_ours_advantage"),
            "rank_by_epoch4_advantage": row.get("rank_by_epoch4_advantage"),
            "rank_by_epoch6_advantage": row.get("rank_by_epoch6_advantage"),
        },
        "assets": assets,
    }


def validate_manifest_record(record: Mapping[str, Any], schema: Optional[Mapping[str, Any]] = None) -> None:
    required = ["manifest_version", "run_role", "split", "segment_id", "loader_config", "checkpoints", "losses", "deltas", "ranking"]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"manifest missing required keys: {missing}")
    if record["manifest_version"] != "1.0":
        raise ValueError("manifest_version must be 1.0")
    if record["run_role"] not in VALID_MANIFEST_RUN_ROLES:
        raise ValueError(f"invalid manifest run_role: {record['run_role']}")
    for side, losses in record["losses"].items():
        for name in LOSS_COMPONENTS:
            if name not in losses:
                raise ValueError(f"missing loss component {side}.{name}")


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path, schema_path: Optional[Path] = None) -> int:
    schema = None
    if schema_path and schema_path.exists():
        schema = json.loads(schema_path.read_text())
    count = 0
    with path.open("w") as fh:
        for rec in records:
            validate_manifest_record(rec, schema)
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            count += 1
    return count


def distribution(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {"count": n, "min": ordered[0], "max": ordered[-1], "mean": sum(ordered) / n, "median": median}


def build_summary(cfg: RunConfig, rows: List[Dict[str, Any]], paths: Mapping[str, str], strata: Mapping[str, List[Mapping[str, Any]]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for name in LOSS_COMPONENTS:
        component: Dict[str, Any] = {}
        for label in cfg.checkpoints:
            component[label] = distribution([float(r[f"{label}_{name}"]) for r in rows])
        component["delta"] = distribution([float(r[f"delta_{name}"]) for r in rows])
        for candidate, baseline in configured_delta_pairs(cfg):
            component[f"delta_{candidate}_minus_{baseline}"] = distribution([
                float(r[f"{pair_delta_prefix(candidate, baseline)}_{name}"]) for r in rows
            ])
        stats[name] = component
    return {
        "run_id": cfg.run_id,
        "run_role": cfg.run_role,
        "manifest_run_role": cfg.manifest_run_role,
        "split": cfg.split,
        "row_count": len(rows),
        "split_policy": cfg.split_policy,
        "checkpoints": {k: v.manifest_dict() for k, v in cfg.checkpoints.items()},
        "candidate_labels": list(cfg.checkpoints.keys()),
        "delta_pairs": [f"{candidate}_minus_{baseline}" for candidate, baseline in configured_delta_pairs(cfg)],
        "canonical_delta_pair": f"{canonical_delta_pair(cfg)[0]}_minus_{canonical_delta_pair(cfg)[1]}",
        "non_final_warning": cfg.run_role in {"smoke", "dry_run"},
        "claim_boundary": "Phase 4 packages quantitative/perceptual evidence only; Phase 9 and human review own final reproduction acceptance.",
        "loss_component_stats": stats,
        "loss_recon_conflict_count": sum(1 for r in rows if r["loss_recon_conflict"]),
        "strata": {name: [r["compound_id"] for r in values] for name, values in strata.items()},
        "paths": dict(paths),
        "environment": {"python": sys.version, "cwd": str(ROOT), "git": get_git_metadata()},
    }


def write_markdown_summary(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# POP909 Conditioned Reconstruction Comparison Summary",
        "",
        f"Run id: `{summary['run_id']}`",
        f"Run role: `{summary['run_role']}`",
        f"Split: `{summary['split']}`",
        f"Rows: `{summary['row_count']}`",
        f"Canonical delta pair: `{summary.get('canonical_delta_pair')}`",
        "",
    ]
    if summary.get("non_final_warning"):
        lines += ["**Non-final evidence warning:** this run is smoke/dry-run evidence only, not official Phase 9 evidence.", ""]
    lines += ["## Checkpoints", ""]
    for label, ref in summary["checkpoints"].items():
        lines.append(f"- {label}: `{ref['role']}` — `{ref['path']}`")
    lines += ["", "## Delta Sign Convention", "", "Pairwise deltas are `candidate - baseline`; positive total-loss delta means the baseline has lower total loss.", "", "## Ranking Strata", ""]
    for name, ids in summary["strata"].items():
        lines.append(f"- `{name}`: {len(ids)} candidate(s)")
    lines += ["", "## Claim Boundary", "", str(summary["claim_boundary"]), ""]
    path.write_text("\n".join(lines))


def run_comparison(cfg: RunConfig) -> Dict[str, Any]:
    validate_config_files(cfg, check_files=True)
    dirs = prepare_output_dirs(cfg)
    (dirs["config"] / "resolved_config.json").write_text(json.dumps(resolved_metadata(cfg), indent=2, sort_keys=True) + "\n")
    set_global_seeds(cfg.selection_seed)
    torch, _ = _import_torch_stack()
    models = {label: load_model_for_checkpoint(ref, cfg.device) for label, ref in cfg.checkpoints.items()}

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for segment_id, x, c, pr_mat in iter_real_segments(cfg):
            device = next(models["authors"].parameters()).device
            x = x.long().to(device)
            c = c.float().to(device)
            pr_mat = pr_mat.float().to(device)
            losses = {
                label: tensor_loss_dict(model.loss(x, c, pr_mat))
                for label, model in models.items()
            }
            rows.append(make_row(cfg, segment_id, losses))
    if not rows:
        raise RuntimeError("comparison produced zero rows")

    strata = rank_rows(rows)
    csv_path = dirs["tables"] / "comparison_wide.csv"
    manifest_path = dirs["manifests"] / "comparison_manifest.jsonl"
    summary_json_path = dirs["summaries"] / "summary.json"
    summary_md_path = dirs["summaries"] / "summary.md"
    ranking_path = dirs["rankings"] / "ranking_strata.json"
    write_csv(rows, csv_path, cfg)
    write_jsonl((manifest_record(cfg, row) for row in rows), manifest_path, cfg.schema_path)
    ranking_path.write_text(json.dumps({k: [r["compound_id"] for r in v] for k, v in strata.items()}, indent=2) + "\n")
    paths = {
        "csv": str(csv_path),
        "manifest_jsonl": str(manifest_path),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
        "rankings_json": str(ranking_path),
    }
    summary = build_summary(cfg, rows, paths, strata)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown_summary(summary, summary_md_path)
    return summary
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="POP909 conditioned reconstruction comparison")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--list-segments", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = parse_config(Path(args.config))
    if args.validate_config:
        print(json.dumps(validate_config_files(cfg, check_files=args.check_files), indent=2, sort_keys=True))
        return 0
    if args.list_segments:
        for idx, (identity, _x, _c, _pr) in enumerate(iter_real_segments(cfg)):
            print(json.dumps(identity, sort_keys=True))
            if idx + 1 >= args.list_segments:
                break
        return 0
    summary = run_comparison(cfg)
    print(json.dumps({"run_id": summary["run_id"], "row_count": summary["row_count"], "paths": summary["paths"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
