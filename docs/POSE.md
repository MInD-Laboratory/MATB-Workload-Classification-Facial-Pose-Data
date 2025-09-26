# Facial Pose Analysis: Complete Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Data Input and Format](#data-input-and-format)
3. [Processing Pipeline Architecture](#processing-pipeline-architecture)
4. [Modules & Key Functions](#modules--key-functions)
5. [Feature Extraction Details](#feature-extraction-details)
6. [Coordinate Normalization Methods](#coordinate-normalization-methods)
7. [Temporal Filtering](#temporal-filtering)
8. [Statistical Analysis](#statistical-analysis)
9. [Notebook Utilities & Visualizations](#notebook-utilities--visualizations)
10. [Outputs & Directory Layout](#outputs--directory-layout)
11. [Configuration & Flags](#configuration--flags)

---

## Overview

Modular pipeline for processing OpenPose facial landmarks into research-ready metrics. Steps:

1) Load CSVs → keep relevant points  
2) Mask low-confidence detections  
3) Interpolate short gaps + Butterworth filter  
4) Normalize to screen size  
5) Build templates (global + participant)  
6) Extract features: Procrustes (global), Procrustes (participant), Original  
7) Window metrics (drop windows containing any NaNs per metric)  
8) Linear metrics (mean |vel|, mean |acc|, RMS)  
9) Write a JSON summary (config, masking stats, window drops)

---

## Data Input and Format

### Expected Input
CSV files with 1-based landmark triplets:

```
x1,y1,prob1,x2,y2,prob2,...,x70,y70,prob70
```

- Landmarks 1–68: standard 68-point face
- 69–70: pupils (if present)

### Filenames
`<participantID>_<condition>.csv` (e.g., `472_H.csv`) — conditions typically `L/M/H`.

---

## Processing Pipeline Architecture

```
Raw CSVs
   ↓
Filter to relevant keypoints
   ↓
Mask (conf < threshold)
   ↓
Interpolate short gaps + Butterworth LPF
   ↓
Normalize to screen (width,height)
   ↓
Templates (global + per-participant)
   ↓
Per-frame features (3 routes)
   ↓
Windowing (drop NaN windows) → features/*.csv
   ↓
Linear metrics (vel/acc/RMS) → linear_metrics/*.csv
   ↓
pipeline_summary.json
```

---

## Modules & Key Functions

### `pose_pipeline.py` (orchestrator)
- `run_pipeline()`: Executes the full workflow based on flags & `CFG`.

### `utils/__init__.py`
Holds:
- `Config` / `CFG`
- All run flags (e.g., `RUN_FILTER`, `RUN_TEMPLATES`, …)
- Toggles like `SCALE_BY_INTEROCULAR`
- `SCIPY_AVAILABLE`

### `utils/io.py`
- **I/O & housekeeping**
  - `ensure_dirs()`
  - `load_raw_files()`
  - `save_json_summary(path, obj)`
- **Naming / parsing**
  - `parse_participant_condition(filename)`
  - `detect_conf_prefix_case_insensitive(columns)`  → finds `prob*` / `c*` / `confidence*`
  - `relevant_indices()`  → union of sets used across features (eyes, nose, mouth, pupils)
- **Data selection & masking**
  - `filter_df_to_relevant(df, conf_prefix, indices)`
  - `confidence_mask(df_reduced, conf_prefix, indices, thr)` → masked df + stats
- **Per-frame output**
  - `write_per_frame_metrics(out_root, source, participant, condition, perframe, interocular, n_frames)`

### `utils/signal_utils.py`
- `interpolate_run_limited(series, max_run)` → only fill gaps ≤ `max_run`
- `butterworth_segment_filter(series, order, cutoff_hz, fs)` → zero-phase (per contiguous segment)

### `utils/normalize_utils.py`
- `normalize_to_screen(df, width, height)` → divide x/y by width/height
- `interocular_series(df, conf_prefix=None)` → per-frame distance (37↔46)

### `utils/window_utils.py`
- `windows_indices(n, win, hop)` → (start, end, idx)
- `window_features(metric_map, interocular, fps, win, hop)` → per-window means + drop counts
- `is_distance_like_metric(name)` → for interocular scaling logic
- `linear_metrics(x, fps)` → mean |vel|, mean |acc|, RMS

### `utils/features_utils.py`
- **Per-frame features**
  - `procrustes_features_for_file(df_norm, template_df, rel_indices)`  
    Returns dict arrays:  
    `head_rotation_rad, head_tx, head_ty, head_scale, head_motion_mag, blink_aperture, mouth_aperture, pupil_metric`
  - `original_features_for_file(df_norm)`  
    Returns dict arrays:  
    `head_rotation_rad, blink_aperture, mouth_aperture, pupil_metric, center_face_magnitude`
- **Linear metrics from saved per-frame**
  - `compute_linear_from_perframe_dir(per_frame_dir, out_csv_path, fps, window_seconds, overlap, scale_by_interocular)`

### `utils/nb_utils.py` (notebook helpers)
- Disk status & selection:
  - `outputs_exist(out_base)`
  - `pick_norm_file(out_base)`
  - `slice_first_seconds(df, fps, seconds)`
- Series helpers:
  - `series_num(df, axis, i, n)`
- Plot utilities (bars):
  - `candidate_metric_cols(df)`
  - `default_metric(cols)`
  - `ensure_condition_order(df)`
  - `bar_by_condition(df, metric, title_suffix="")`

### `utils/stats_utils.py`
- `compare_groups_statistical(df, metric, test_type="auto")`  
  Returns:
  - `omnibus` (ANOVA/t-test or Kruskal/Mann-Whitney)
  - `descriptives` (mean/std/sem/median/n)
  - `pairwise` (Holm–Bonferroni corrected)

### `utils/viz_utils.py`
- `create_interactive_pose_timeseries_viewer(df_raw, df_features, ...)`
- `procrustes_transform_series(df_norm_window, template_df)`  
  → `(df_pose_aligned, features_dict)` for quick interactive previews

---

## Feature Extraction Details

All routes are computed on **normalized** coordinates (screen-normalized first; Procrustes routes additionally align to a template). Distance-like metrics are optionally **scaled by interocular** in the linear stage.

**Core metrics (per-frame):**
- `head_rotation_rad` — angle of (37→46)
- `blink_aperture` — average lid separation (L/R)
- `mouth_aperture` — ||(63) − (67)||
- `pupil_metric` — pupil offset magnitude from eye-ring centers (avg L/R)
- `center_face_magnitude` *(Original route only)* — RMS of nose points (28–36) around per-file mean
- `head_tx, head_ty, head_scale, head_motion_mag` *(Procrustes routes)* — transform params & composite motion

**Windowing:** mean across frames within each window (NaN in any frame → window NaN for that metric).  
**Linear metrics per window:** mean |vel|, mean |acc|, RMS of each per-frame metric (optionally interocular-scaled).

---

## Coordinate Normalization Methods

### Original (no Procrustes)
- Use raw normalized coordinates (divided by screen width/height).
- Head rotation derived directly from eye corners (37,46).

### Procrustes (Global / Participant)
- Solve similarity transform per frame (SVD; translation + rotation + uniform scale) against:
  - **Global template** (pooled mean across files)
  - **Participant template** (mean across that participant’s files)
- More robust to pose and individual geometry.

---

## Temporal Filtering

- Interpolation: linear, **limited to runs ≤ `CFG.MAX_INTERP_RUN`** (longer runs remain NaN).
- Filtering: **zero-phase Butterworth** (order `CFG.FILTER_ORDER`, cutoff `CFG.CUTOFF_HZ` Hz, fs=`CFG.FPS`), applied **per contiguous non-NaN segment** to preserve NaN structure and avoid `filtfilt` edge issues.

---

## Statistical Analysis

Use `utils/stats_utils.compare_groups_statistical` in the notebook UI:

- **Omnibus**: one-way ANOVA (parametric) or Kruskal–Wallis (nonparametric); auto-select via Shapiro tests when possible.
- **Pairwise**: t-tests or Mann–Whitney with **Holm–Bonferroni** correction.
- Widgets let you pick:
  - Normalization (Original / Procrustes Global / Procrustes Participant)
  - Metric (auto-listed from file)
  - Test strategy (auto/parametric/nonparametric)

---

## Notebook Utilities & Visualizations

- **Interactive viewer** (`viz_utils.create_interactive_pose_timeseries_viewer`):  
  Live pose scatter + feature time series with slider + play controls.  
  Use with `procrustes_transform_series` for quick per-file previews (global or participant template).
- **Bars/Stats UI** (final section of the notebook):  
  Only bar charts + stats (no pose/time-series redraw), wired with `nb_utils` and `stats_utils`.

---

## Outputs & Directory Layout

All under `CFG.OUT_BASE`:

```
processed/
├─ reduced/                 # filtered landmark triplets
├─ masked/                  # after confidence masking
├─ interp_filtered/         # after interpolation + Butterworth
├─ norm_screen/             # screen-normalized coordinates
├─ templates/
│   ├─ global_template.csv
│   └─ participant_<PID>_template.csv
├─ features/
│   ├─ procrustes_global_features.csv
│   ├─ procrustes_participant_features.csv
│   ├─ original_features.csv
│   └─ per_frame/
│       ├─ procrustes_global/<PID>_<COND>_perframe.csv
│       ├─ procrustes_participant/<PID>_<COND>_perframe.csv
│       └─ original/<PID>_<COND>_perframe.csv
├─ linear_metrics/
│   ├─ procrustes_global_linear.csv
│   ├─ procrustes_participant_linear.csv
│   └─ original_linear.csv
└─ pipeline_summary.json
```

`pipeline_summary.json` includes:
- `config` (exact `CFG`)
- `flags` (all toggles used)
- `masking_overall` (per-file)
- `window_drops` (per route + linear metrics)

---

## Configuration & Flags

### `Config` (`utils.__init__.py`)
| Field | Meaning | Typical |
|---|---|---|
| `RAW_DIR` | Input CSV folder | `data/raw` |
| `OUT_BASE` | Output base folder | `data/processed` |
| `FPS` | Sampling rate (Hz) | `60` |
| `IMG_WIDTH, IMG_HEIGHT` | Screen size for normalization | `2560, 1440` |
| `CONF_THRESH` | Mask if conf < thr | `0.30` |
| `MAX_INTERP_RUN` | Max gap to interpolate (frames) | `60` |
| `FILTER_ORDER` | Butterworth order | `4` |
| `CUTOFF_HZ` | Butterworth cutoff (Hz) | `10.0` |

### Run Flags (booleans in `utils.__init__.py`)
- `RUN_FILTER`, `RUN_MASK`, `RUN_INTERP_FILTER`, `RUN_NORM`, `RUN_TEMPLATES`  
- `RUN_FEATURES_PROCRUSTES_GLOBAL`, `RUN_FEATURES_PROCRUSTES_PARTICIPANT`, `RUN_FEATURES_ORIGINAL`  
- `RUN_LINEAR`  
- Outputs: `SAVE_REDUCED`, `SAVE_MASKED`, `SAVE_INTERP_FILTERED`, `SAVE_NORM`, per-frame saves per route  
- Overwrite: `OVERWRITE`, `OVERWRITE_TEMPLATES`  
- Scaling: `SCALE_BY_INTEROCULAR`
