
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import umap

BASE_LOSS_COLUMNS = ["final_loss", "kl_loss", "kl_chd", "kl_rhy"]
ENRICHED_LOSS_COLUMNS = [
    "recon_loss",
    "pitch_loss",
    "duration_loss",
    "chord_loss",
    "root_loss",
    "chroma_loss",
    "bass_loss",
]
METADATA_COLUMNS = [
    "audio_key",
    "genre",
    "track_role",
    "inst",
    "sample_rhythm",
    "time_signature",
    "pitch_range",
    "num_measures",
    "bpm",
    "chord_progressions",
]


@dataclass(frozen=True)
class CommuPaths:
    repo_root: Path
    loss_dir: Path
    enriched_loss_dir: Path
    cache_dir: Path
    cache_table: Path
    cache_umap_prefix: Path


def default_paths(repo_root: str | Path | None = None) -> CommuPaths:
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    cache_dir = repo_root / 'NotebooksVAESymTex' / '_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return CommuPaths(
        repo_root=repo_root,
        loss_dir=repo_root / 'COMMUDataset' / 'losses',
        enriched_loss_dir=repo_root / 'COMMUDataset' / 'losses_enriched',
        cache_dir=cache_dir,
        cache_table=cache_dir / 'commu_loss_table.pkl',
        cache_umap_prefix=cache_dir / 'commu_umap',
    )


def normalize_track_role(role: str, merge_melodies: bool) -> str:
    if role is None:
        return 'unknown'
    if merge_melodies and role in {'main_melody', 'sub_melody'}:
        return 'melody'
    return role


def discover_loss_files(loss_dir: str | Path) -> list[Path]:
    loss_dir = Path(loss_dir)
    return sorted(path for path in loss_dir.glob('*.npz') if path.is_file())


def _scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr


def _load_loss_rows(files: list[Path], columns: list[str]) -> list[dict]:
    rows = []
    for path in files:
        with np.load(path, allow_pickle=True) as payload:
            row = {
                'segment_id': path.stem,
                'source_file': path.name,
                'track_id': path.stem.split('-')[0],
                'segment_index': int(path.stem.split('-')[-1]),
            }
            if 'z_chd' in payload:
                row['z_chd'] = np.asarray(payload['z_chd'], dtype=np.float32)
            if 'z_txt' in payload:
                row['z_txt'] = np.asarray(payload['z_txt'], dtype=np.float32)
            for column in columns + METADATA_COLUMNS:
                row[column] = _scalar(payload[column]) if column in payload else np.nan
            rows.append(row)
    return rows


def load_commu_loss_table(
    loss_dir: str | Path,
    *,
    enriched_loss_dir: str | Path | None = None,
    max_files: int | None = None,
    use_cache: bool = True,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    loss_dir = Path(loss_dir)
    cache_path = Path(cache_path) if cache_path else None
    enriched_loss_dir = Path(enriched_loss_dir) if enriched_loss_dir else None

    if use_cache and cache_path and cache_path.exists() and max_files is None:
        return pd.read_pickle(cache_path)

    files = discover_loss_files(loss_dir)
    if max_files is not None:
        files = files[:max_files]

    rows = _load_loss_rows(files, BASE_LOSS_COLUMNS)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f'No COMMU loss files found in {loss_dir}')

    if enriched_loss_dir and enriched_loss_dir.exists():
        enriched_files = discover_loss_files(enriched_loss_dir)
        enriched_lookup = {path.stem: path for path in enriched_files}
        selected_enriched = [enriched_lookup[row['segment_id']] for row in rows if row['segment_id'] in enriched_lookup]
        if selected_enriched:
            enriched = pd.DataFrame(_load_loss_rows(selected_enriched, ENRICHED_LOSS_COLUMNS))
            keep_columns = ['segment_id'] + ENRICHED_LOSS_COLUMNS
            table = table.merge(enriched[keep_columns], on='segment_id', how='left')
        else:
            for column in ENRICHED_LOSS_COLUMNS:
                table[column] = np.nan
    else:
        for column in ENRICHED_LOSS_COLUMNS:
            table[column] = np.nan

    table['track_role_grouped'] = table['track_role'].map(lambda value: normalize_track_role(value, True))
    table['track_role_detailed'] = table['track_role'].map(lambda value: normalize_track_role(value, False))
    table['is_melody'] = table['track_role'].isin(['main_melody', 'sub_melody'])

    numeric_columns = BASE_LOSS_COLUMNS + ENRICHED_LOSS_COLUMNS + ['bpm', 'num_measures', 'segment_index']
    for column in numeric_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors='coerce')

    if use_cache and cache_path and max_files is None:
        table.to_pickle(cache_path)
    return table


def build_latent_matrix(table: pd.DataFrame, latent_space: str) -> np.ndarray:
    if latent_space == 'z_chd':
        return np.stack(table['z_chd'].to_numpy())
    if latent_space == 'z_txt':
        return np.stack(table['z_txt'].to_numpy())
    if latent_space == 'z_both':
        z_chd = np.stack(table['z_chd'].to_numpy())
        z_txt = np.stack(table['z_txt'].to_numpy())
        return np.concatenate([z_chd, z_txt], axis=1)
    raise ValueError(f'Unknown latent_space={latent_space!r}')


def compute_umap_embedding(
    table: pd.DataFrame,
    latent_space: str,
    *,
    n_neighbors: int = 20,
    min_dist: float = 0.08,
    metric: str = 'cosine',
    random_state: int = 42,
) -> pd.DataFrame:
    latent = build_latent_matrix(table, latent_space)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        transform_seed=random_state,
    )
    emb = reducer.fit_transform(latent)
    output = table.copy()
    output['umap_x'] = emb[:, 0]
    output['umap_y'] = emb[:, 1]
    output['latent_space'] = latent_space
    return output


def metric_availability_table(table: pd.DataFrame | None = None) -> pd.DataFrame:
    def status_for(column: str) -> str:
        if table is None or column not in table.columns:
            return 'unknown'
        return 'available now' if table[column].notna().any() else 'needs recomputation'

    return pd.DataFrame([
        {
            'display_name': 'Total loss',
            'column': 'final_loss',
            'status': status_for('final_loss') if table is not None else 'available now',
            'meaning': 'The total scalar training loss saved with each segment.',
        },
        {
            'display_name': 'Reconstruction loss',
            'column': 'recon_loss',
            'status': status_for('recon_loss'),
            'meaning': 'Most direct global reconstruction-quality view when enriched losses are present.',
        },
        {
            'display_name': 'Harmony proxy: chord loss',
            'column': 'chord_loss',
            'status': status_for('chord_loss'),
            'meaning': 'Aggregate harmonic reconstruction loss from root + chroma + bass.',
        },
        {
            'display_name': 'Harmony detail: chroma loss',
            'column': 'chroma_loss',
            'status': status_for('chroma_loss'),
            'meaning': 'Didactic harmonic view focused on chroma reconstruction.',
        },
        {
            'display_name': 'Texture proxy: duration loss',
            'column': 'duration_loss',
            'status': status_for('duration_loss'),
            'meaning': 'Didactic rhythm/texture view based on note-duration reconstruction.',
        },
        {
            'display_name': 'Pitch loss',
            'column': 'pitch_loss',
            'status': status_for('pitch_loss'),
            'meaning': 'Pitch reconstruction component, useful as a counterpoint to duration/chord losses.',
        },
        {
            'display_name': 'Chord KL',
            'column': 'kl_chd',
            'status': status_for('kl_chd') if table is not None else 'available now',
            'meaning': 'Latent regularization signal for the chord branch.',
        },
        {
            'display_name': 'Texture KL',
            'column': 'kl_rhy',
            'status': status_for('kl_rhy') if table is not None else 'available now',
            'meaning': 'Latent regularization signal for the texture branch.',
        },
    ])
