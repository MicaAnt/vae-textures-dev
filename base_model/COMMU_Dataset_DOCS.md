# `dataset.py` vs `datasetCOMMU.py`

This document summarizes the practical differences between the original POP909 loader (`dataset.py`) and the COMMU loader (`datasetCOMMU.py`), and explains the current data-loading flow used by `trainCOMMU_smoke_cpu_002.py`.

---

## 1) High-level purpose

- `dataset.py`: dataset pipeline for the POP909 assets (`POP09-PIANOROLL-4-bin-quantization`) with POP909 metadata format.
- `datasetCOMMU.py`: adapted pipeline for COMMU assets (`COMMUnpzFiles`) and COMMU metadata (`CommuVAEDataset.xlsx`).

The core tensor preparation path is intentionally kept similar (segment extraction, augmentation, piano-roll conversion, and optional chord-conditioned features), so model-side expectations stay compatible.

---

## 2) Configuration and path resolution differences

### `dataset.py`
- Uses relative paths:
  - `DATA_PATH = data/POP09-PIANOROLL-4-bin-quantization`
  - `INDEX_FILE_PATH = data/index.xlsx`
- Behavior depends on process working directory (CWD).

### `datasetCOMMU.py`
- Uses a module-relative base directory:
  - `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`
  - `DATA_PATH = BASE_DIR/data/COMMUnpzFiles`
  - `INDEX_FILE_PATH = BASE_DIR/data/CommuVAEDataset.xlsx`
  - `INDICES_CACHE_PATH = BASE_DIR/data/indCOMMU.pkl` (kept as constant, not used by active loading flow)
- This avoids path breakage when the training script is launched from a different CWD.

---

## 3) Metadata matching (`song_id`) differences

### `dataset.py`
- Expects numeric IDs directly from the beginning of filenames (`stem[0:3]`) and casts with `int(...)`.
- Works for POP909 naming pattern where first characters are numeric.

### `datasetCOMMU.py`
- Adds `_extract_song_id(value)` using regex `r'(\d+)$'`.
- Extracts trailing numeric ID from:
  - filename stems (e.g., `commu06755` -> `6755`)
  - Excel `song_id` column entries (supports strings such as `commu06755` and numeric-like values)
- Builds `song_id_numeric` in the dataframe and matches by this normalized key.
- Skips rows/files whose IDs cannot be parsed.

Why: COMMU filenames and metadata may include string prefixes, which break direct `int(stem[:3])` parsing.

---

## 4) Meter filtering differences

POP909 keeps duple meter while COMMUDataset keeps quadruple meter

- POP909 loader: metadata row selected via direct numeric filename ID extraction.
- COMMU loader: metadata row selected via normalized `song_id_numeric`.

After metadata match, both keep only `num_beats_per_measure == 4`.

---

## 5) Cache behavior differences (`ind*.pkl`)

### `dataset.py`
- Reads `data/ind.pkl` inside `prepare_dataset()` and replaces discovered file list with cache content.

### `datasetCOMMU.py`
- Does **not** override discovered files with `indCOMMU.pkl` in `prepare_dataset()`.
- Active flow uses `collect_data_fns()` as the source of truth.

Why this matters:
- Legacy pickles can contain stale relative paths from another machine/folder layout (for example `COMMUDataset/npzFiles/...`) and cause `FileNotFoundError`.

---

## 6) What stays equivalent between both files

The following parts are intentionally similar/compatible:

- `ArrangementDataset` class behavior (`__getitem__`, shifts, segment combination)
- Conversion pipeline:
  - `ext_nmat_to_mel_pr`
  - `ext_nmat_to_pr`
  - `augment_*`
  - `pr_to_onehot_pr`
  - `piano_roll_to_target`
  - `target_to_3dtarget`
- Chord-conditioned branch and `detrend_pianotree`
- Split/wrap interface:
  - `split_dataset(...)`
  - `wrap_dataset(...)`
  - `prepare_dataset(...)` return shape `(train_loader, val_loader)`

This is why existing model code can keep using the same loader interface.

---

## 7) Current COMMU data-loading flow (when running `trainCOMMU_smoke_cpu_002.py`)

1. `trainCOMMU_smoke_cpu_002.py` calls `MusicDataLoaders.get_loaders(...)`.
2. `dataset_loaders.py` imports `prepare_dataset` from `datasetCOMMU.py` and delegates to it.
3. `prepare_dataset(...)` calls `collect_data_fns()`.
4. `collect_data_fns()`:
   - scans `DATA_PATH` for `*.npz`
   - loads `CommuVAEDataset.xlsx`
   - computes `song_id_numeric` by `_extract_song_id(...)`
   - parses each filename ID with `_extract_song_id(...)`
   - keeps files whose metadata match exists and `num_beats_per_measure == 4`
5. `prepare_dataset(...)` splits indices with `split_dataset(...)` using the provided seed/portion.
6. `wrap_dataset(...)` builds train and val datasets:
   - opens each NPZ via `init_music(fn)`
   - converts raw music to segment-level training examples with `music.prepare_data(...)`
7. `DataLoader` objects are created and returned to the smoke trainer.
8. The smoke script may further subselect indices (start/limit) and define subset shuffling behavior before training.

---

## 8) Practical implications for future training runs

With the current COMMU pipeline, you control:

- Which files are considered valid (via metadata + 4/4 filter)
- Train/validation split randomness (`seed`) and proportion (`portion`)
- Data augmentation transposition range (`shift_low`, `shift_high`)
- Number of bars per training sample (`num_bar`)
- Subset windows and batch/shuffle behavior at smoke-script level

The main operational recommendation is to keep COMMU metadata `song_id` values parseable by trailing digits (e.g., `commu06755`).

