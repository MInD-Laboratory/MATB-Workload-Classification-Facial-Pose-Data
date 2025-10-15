# Facial Pose Analysis Pipeline

Modular pipeline for processing facial pose data from OpenPose landmark detection, extracting features, and computing linear metrics including velocity and acceleration statistics.

## Overview

This pipeline processes raw OpenPose CSV files through 8 sequential steps:

1. **Load raw data** - Import OpenPose CSVs with 70 landmarks (x, y, confidence per landmark)
2. **Filter landmarks** - Extract relevant facial keypoints based on feature requirements
3. **Mask low confidence** - Set landmarks below confidence threshold to NaN
4. **Interpolate and filter** - Fill short gaps (up to 1 second), apply Butterworth low-pass filter
5. **Normalize coordinates** - Scale to screen dimensions (0-1 range)
6. **Build templates** - Create global template and per-participant reference poses
7. **Extract features** - Compute windowed features using three normalization approaches:
   - **Original**: Raw normalized coordinates (no alignment)
   - **Procrustes Global**: Aligned to global template across all participants
   - **Procrustes Participant**: Aligned to participant-specific template
8. **Compute linear metrics** - Calculate velocity, acceleration, and statistical features (min, max, mean, RMS, std, median, quartiles, autocorrelation)

The pipeline supports smart resumption (automatically skips completed steps) and flexible execution (start from any step).

**Output**: Per-frame time series, windowed features, linear metrics with temporal derivatives, and processing summaries.

## Directory Structure

```
Pose/
├── process_pose_data.py            # Main pipeline (steps 1-8)
│
├── utils/
│   ├── config.py                   # Configuration parameters and processing flags
│   ├── io_utils.py                 # File I/O operations, directory management
│   ├── preprocessing_utils.py      # Landmark filtering, confidence masking, condition mapping
│   ├── signal_utils.py             # Interpolation and Butterworth filtering
│   ├── normalize_utils.py          # Screen normalization, inter-ocular distance calculation
│   ├── geometry_utils.py           # Procrustes alignment, geometric transformations
│   ├── features_utils.py           # Feature extraction, derivative computation, linear metrics
│   └── window_utils.py             # Sliding window operations
│
├── data/
│   ├── pose_data/                  # Raw OpenPose CSV files (input)
│   │   ├── 3101_01_pose.csv        # Participant 3101, trial 1
│   │   ├── 3101_02_pose.csv        # Participant 3101, trial 2
│   │   └── ...
│   │
│   ├── processed/                  # Pipeline outputs (see structure below)
│   │   ├── reduced/                # Step 2: Filtered landmarks
│   │   ├── masked/                 # Step 3: Low-confidence masked
│   │   ├── interp_filtered/        # Step 4: Interpolated and filtered
│   │   ├── norm_screen/            # Step 5: Normalized coordinates
│   │   ├── templates/              # Step 6: Procrustes templates
│   │   ├── features/               # Step 7: Windowed and per-frame features
│   │   ├── linear_metrics/         # Step 8: Linear metrics with derivatives
│   │   └── processing_summary.json # Processing metadata and statistics
│   │
│   └── participant_info.csv        # Participant metadata with condition mapping
│
└── pose_stats_figures.ipynb        # Statistical analysis and visualization notebook
```

## Data Format

### Input Files

**Location**: `data/pose_data/`

**Naming convention**: `<participantID>_<trial>_pose.csv`

**Example**: `3101_01_pose.csv` (participant 3101, trial 1)

**Columns**: x1, y1, prob1, ..., x70, y70, prob70 (210 columns total)
- 70 facial landmarks from OpenPose
- Each landmark has x-coordinate, y-coordinate, and confidence value
- Confidence prefix auto-detected: `prob*`, `c*`, or `confidence*`

**Sampling rate**: 60 Hz (configurable in `config.py`)

### Participant Info File

**Location**: `data/participant_info.csv`

**Required columns**:
- `participant`: Participant ID (e.g., 3101)
- `trial_1`, `trial_2`, `trial_3`: Condition codes for each trial

**Condition codes**: L (Low), M (Moderate), H (High)

**Example**:
```
participant,trial_1,trial_2,trial_3
3101,L,M,H
3102,M,H,L
```

This file maps trial numbers to experimental conditions. The pipeline uses this to generate condition-based output filenames (e.g., `3101_L_norm.csv` instead of `3101_01_norm.csv`).

## Output Structure

```
data/processed/
├── reduced/                        # Step 2: Filtered to relevant landmarks
│   └── <pid>_<cond>_reduced.csv
│
├── masked/                         # Step 3: Low-confidence points masked to NaN
│   └── <pid>_<cond>_masked.csv
│
├── interp_filtered/                # Step 4: Interpolated and Butterworth filtered
│   └── <pid>_<cond>_interp_filt.csv
│
├── norm_screen/                    # Step 5: Normalized to screen coordinates (0-1)
│   └── <pid>_<cond>_norm.csv
│
├── templates/                      # Step 6: Procrustes reference templates
│   ├── global_template.csv         # Single template from all participants
│   └── participant_<pid>_template.csv  # One per participant
│
├── features/                       # Step 7: Extracted features
│   ├── original.csv                # Window-level features (no alignment)
│   ├── procrustes_global.csv       # Window-level features (global alignment)
│   ├── procrustes_participant.csv  # Window-level features (participant alignment)
│   │
│   └── per_frame/                  # Frame-level time series
│       ├── original/
│       │   └── <pid>_<cond>_perframe.csv
│       ├── procrustes_global/
│       │   └── <pid>_<cond>_perframe.csv
│       └── procrustes_participant/
│           └── <pid>_<cond>_perframe.csv
│
├── linear_metrics/                 # Step 8: Statistical metrics with derivatives
│   ├── original_linear.csv
│   ├── procrustes_global_linear.csv
│   └── procrustes_participant_linear.csv
│
└── processing_summary.json         # Configuration, flags, and processing statistics
```

**Filename convention**: `<pid>_<cond>_<suffix>.csv`
- `<pid>`: Participant ID (e.g., 3101)
- `<cond>`: Condition letter (L, M, or H)
- `<suffix>`: Processing stage identifier

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scipy tqdm python-dotenv
```

**Required packages**:
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `scipy`: Signal processing (Butterworth filter)
- `tqdm`: Progress bars
- `python-dotenv`: Environment variable management (optional)

### 2. Configure Data Paths

**Option A: Use environment variables (recommended for development)**

Create a `.env` file in the project root:

```bash
# .env file
POSE_RAW_DIR=/path/to/your/data/pose_data
POSE_OUT_BASE=/path/to/your/output/processed
PARTICIPANT_INFO_FILE=participant_info.csv
```

The `.env` file is not committed to version control and allows each developer to use custom paths.

**Option B: Use default paths (recommended for published data)**

Ensure your data is in the standard location:
```
Pose/
└── data/
    ├── pose_data/              # Raw OpenPose CSVs here
    └── participant_info.csv    # Participant metadata here
```

No configuration needed - the pipeline will use these default paths.

### 3. Run the Pipeline

**Full pipeline** (all 8 steps):
```bash
cd Pose
python process_pose_data.py
```

**Smart resumption**: The pipeline automatically detects if steps 1-5 are complete by checking for condition-based normalized files. If found, it loads existing data and runs only steps 6-8.

**Force reprocessing** (ignore existing files):
```bash
python process_pose_data.py --overwrite
```

**Start from specific step** (requires existing data from prior steps):
```bash
python process_pose_data.py --start-step 6  # Start from template building
python process_pose_data.py --start-step 7  # Start from feature extraction
python process_pose_data.py --start-step 8  # Only compute linear metrics
```

### 4. Monitor Progress

The pipeline displays progress bars and status messages for each step:
- File loading progress
- Processing status for each step
- Warnings if data is missing
- Summary of outputs created

Check `data/processed/processing_summary.json` for detailed processing statistics.

## Configuration

### Key Parameters

Edit `utils/config.py` to modify processing parameters:

```python
# Detection and filtering
CONF_THRESH = 0.30          # Minimum confidence for valid landmarks (0-1)
MAX_INTERP_RUN = 60         # Max consecutive frames to interpolate (1 sec @ 60 Hz)
FILTER_ORDER = 4            # Butterworth filter order
CUTOFF_HZ = 10.0            # Low-pass cutoff frequency (Hz)

# Windowing
WINDOW_SECONDS = 60         # Window size in seconds
WINDOW_OVERLAP = 0.5        # Window overlap fraction (0.5 = 50%)

# Feature scaling
SCALE_BY_INTEROCULAR = True # Scale features by inter-ocular distance
```

### Processing Flags

Control which steps and outputs are generated:

```python
# Core processing steps
RUN_FILTER = True           # Step 2: Filter to relevant landmarks
RUN_MASK = True             # Step 3: Mask low confidence
RUN_INTERP_FILTER = True    # Step 4: Interpolate and filter
RUN_NORM = True             # Step 5: Normalize coordinates
RUN_TEMPLATES = True        # Step 6: Build templates

# Feature extraction (Step 7)
RUN_FEATURES_PROCRUSTES_GLOBAL = True      # Global template alignment
RUN_FEATURES_PROCRUSTES_PARTICIPANT = True # Participant template alignment
RUN_FEATURES_ORIGINAL = True               # No alignment

# Linear metrics (Step 8)
RUN_LINEAR = True           # Compute linear metrics

# Output control
SAVE_REDUCED = True         # Save intermediate outputs
SAVE_MASKED = True
SAVE_INTERP_FILTERED = True
SAVE_NORM = True
SAVE_PER_FRAME_PROCRUSTES_GLOBAL = True
SAVE_PER_FRAME_PROCRUSTES_PARTICIPANT = True
SAVE_PER_FRAME_ORIGINAL = True

OVERWRITE = False           # Skip processing if outputs exist
OVERWRITE_TEMPLATES = False # Preserve existing templates
```

### Facial Landmarks

The pipeline uses these landmark subsets (based on MediaPipe topology):

```python
PROCRUSTES_REF = (28, 31, 37, 46)  # Stable points for alignment
BLINK_L_TOP = (38, 39)              # Left eye upper eyelid
BLINK_L_BOT = (41, 42)              # Left eye lower eyelid
BLINK_R_TOP = (44, 45)              # Right eye upper eyelid
BLINK_R_BOT = (47, 48)              # Right eye lower eyelid
HEAD_ROT = (37, 46)                 # Eye corners for head rotation
MOUTH = (63, 67)                    # Mouth corners
CENTER_FACE = (28-36)               # Nose bridge region
```

## Processing Details

### Step 1: Load Raw Data

Reads OpenPose CSV files from `RAW_DIR`. Each file contains 70 landmarks with x, y, and confidence values.

**Input**: `<pid>_<trial>_pose.csv`
**Output**: Raw DataFrames in memory

### Step 2: Filter to Relevant Landmarks

Reduces data to landmarks needed for feature extraction (eyes, mouth, nose bridge, face center). Auto-detects confidence column prefix (`prob`, `c`, or `confidence`).

**Input**: Raw DataFrames
**Output**: `data/processed/reduced/<pid>_<cond>_reduced.csv`

### Step 3: Mask Low Confidence

Sets landmarks with confidence below `CONF_THRESH` (default 0.30) to NaN. Tracks masking statistics per file.

**Input**: Reduced DataFrames
**Output**: `data/processed/masked/<pid>_<cond>_masked.csv`

### Step 4: Interpolate and Filter

**Interpolation**: Fills NaN runs up to `MAX_INTERP_RUN` frames (default 60 = 1 second) using linear interpolation.

**Filtering**: Applies Butterworth low-pass filter to contiguous segments:
- Order: `FILTER_ORDER` (default 4)
- Cutoff: `CUTOFF_HZ` (default 10 Hz)
- Sampling rate: `FPS` (default 60 Hz)

Segments separated by NaN are filtered independently to avoid edge artifacts.

**Input**: Masked DataFrames
**Output**: `data/processed/interp_filtered/<pid>_<cond>_interp_filt.csv`

### Step 5: Normalize Coordinates

Converts pixel coordinates to screen-relative coordinates by dividing:
- x-coordinates by screen width (`IMG_WIDTH`, default 2560)
- y-coordinates by screen height (`IMG_HEIGHT`, default 1440)

Result: Coordinates in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right.

**Input**: Filtered DataFrames
**Output**: `data/processed/norm_screen/<pid>_<cond>_norm.csv`

### Step 6: Build Templates

**Global template**: Mean pose computed from all frames of all participants.

**Participant templates**: Mean pose computed from all frames of each participant individually.

Templates are used as reference poses for Procrustes alignment. If templates already exist and `OVERWRITE_TEMPLATES=False`, they are loaded from disk rather than recomputed.

**Input**: Normalized DataFrames
**Output**:
- `data/processed/templates/global_template.csv`
- `data/processed/templates/participant_<pid>_template.csv`

### Step 7: Extract Features

Computes features using three normalization approaches:

#### 7A: Procrustes Global Alignment

Each frame is aligned to the global template using Procrustes analysis:
1. Center both frame and template at origin
2. Find optimal rotation matrix (SVD decomposition)
3. Apply non-uniform scaling (separate x and y scales)
4. Transform frame to aligned coordinates

Features extracted: head translation (tx, ty), rotation (rad), pupil distances (dx, dy), blink aperture, mouth aperture.

**Output**:
- `data/processed/features/procrustes_global.csv` (window-level)
- `data/processed/features/per_frame/procrustes_global/<pid>_<cond>_perframe.csv` (frame-level)

#### 7B: Procrustes Participant Alignment

Same as global alignment but uses participant-specific template instead of global template. Removes participant-specific anatomical differences.

**Output**:
- `data/processed/features/procrustes_participant.csv` (window-level)
- `data/processed/features/per_frame/procrustes_participant/<pid>_<cond>_perframe.csv` (frame-level)

#### 7C: Original Features (No Alignment)

Features computed directly from normalized coordinates without Procrustes alignment. Preserves head position and orientation variance.

**Output**:
- `data/processed/features/original.csv` (window-level)
- `data/processed/features/per_frame/original/<pid>_<cond>_perframe.csv` (frame-level)

#### Windowing

Features are aggregated into overlapping windows:
- Window size: `WINDOW_SECONDS` seconds (default 60s = 3600 frames)
- Overlap: `WINDOW_OVERLAP` (default 0.5 = 50% = 1800 frames)
- Hop size: (1 - overlap) × window_size (default 30s = 1800 frames)

Windows containing any NaN values for a given feature are dropped independently per feature.

### Step 8: Compute Linear Metrics

Calculates comprehensive statistical metrics including temporal derivatives.

#### Derivative Computation

For each feature in per-frame data:
1. Compute velocity using `np.gradient()` (centered finite differences)
2. Compute acceleration as gradient of velocity
3. Scale by sampling rate for correct time units (per second)

Result: Three signals per feature (value, velocity, acceleration)

#### Statistical Metrics

For each signal (value, vel, acc), compute 9 statistics:
1. **min**: Minimum value
2. **max**: Maximum value
3. **mean**: Arithmetic mean
4. **rms**: Root mean square
5. **std**: Standard deviation
6. **median**: Median value
7. **p25**: 25th percentile (lower quartile)
8. **p75**: 75th percentile (upper quartile)
9. **autocorr1**: Lag-1 autocorrelation (temporal smoothness)

Total: 9 statistics × 3 signals = **27 metrics per feature**

#### Inter-ocular Scaling

If `SCALE_BY_INTEROCULAR=True`, features are scaled by inter-ocular distance (Euclidean distance between eye corners, landmarks 37 and 46) to normalize for face size and distance from camera.

**Input**: Per-frame feature CSVs
**Output**:
- `data/processed/linear_metrics/original_linear.csv`
- `data/processed/linear_metrics/procrustes_global_linear.csv`
- `data/processed/linear_metrics/procrustes_participant_linear.csv`

#### Output Format

Each linear metrics CSV contains columns:
```
participant, source, window_start, window_end,
<feature>_min, <feature>_max, <feature>_mean, <feature>_rms, <feature>_std,
<feature>_median, <feature>_p25, <feature>_p75, <feature>_autocorr1,
<feature>_vel_min, <feature>_vel_max, ...,
<feature>_acc_min, <feature>_acc_max, ...
```

Example features: `head_rotation_rad`, `head_tx`, `head_ty`, `blink_aperture`, `mouth_aperture`, `pupil_dx`, `pupil_dy`

## Feature Descriptions

### Head Pose Features

- **head_tx, head_ty**: Translation in x and y directions (Procrustes alignments only)
- **head_rotation_rad**: Head rotation angle in radians, computed from eye corner positions
- **head_scale_x, head_scale_y**: Scaling factors from Procrustes alignment (Procrustes only)

### Eye Features

- **blink_aperture**: Vertical distance between upper and lower eyelid (averaged across both eyes)
- **pupil_dx, pupil_dy**: Pupil displacement in x and y directions (from face center)

### Mouth Features

- **mouth_aperture**: Horizontal distance between mouth corners

## Normalization Approaches

### Original (No Alignment)

Coordinates are normalized to screen dimensions but no geometric alignment is performed.

**Preserves**:
- Head position variance (translation)
- Head orientation variance (rotation)
- Natural movement patterns

**Use cases**:
- Analyzing global movement patterns
- Studying head position dynamics
- Comparing large-scale movements across conditions

### Procrustes Global

Each frame is aligned to a global template computed from all participants using Procrustes superimposition.

**Removes**:
- Translation (centers each frame)
- Rotation (aligns to template orientation)
- Non-uniform scaling (standardizes size independently in x and y)

**Preserves**:
- Shape changes (deformation)
- Relative landmark positions

**Use cases**:
- Isolating facial deformations from rigid transformations
- Cross-participant comparison
- Studying facial expressions independent of head motion

### Procrustes Participant

Each frame is aligned to a participant-specific template.

**Removes**:
- Translation, rotation, scaling (as in global)
- Participant-specific anatomical differences

**Preserves**:
- Within-subject shape changes
- Temporal dynamics of facial movements

**Use cases**:
- Within-subject analysis
- Removing inter-individual anatomical variability
- Focusing on dynamic facial behavior

## Command-Line Options

```
python process_pose_data.py [--overwrite] [--start-step N]
```

### --overwrite

Force reprocessing of all steps, ignoring existing files.

**Without flag**: Pipeline automatically skips steps 1-5 if condition-based normalized files already exist. This saves processing time when only later steps need to be rerun.

**With flag**: All steps run regardless of existing files.

### --start-step N

Start pipeline from step N (1-8). Requires existing data from previous steps.

**Examples**:
```bash
# Start from template building (requires normalized data)
python process_pose_data.py --start-step 6

# Start from feature extraction (requires normalized data and templates)
python process_pose_data.py --start-step 7

# Only compute linear metrics (requires per-frame features)
python process_pose_data.py --start-step 8
```

When starting from step 6 or later, the pipeline automatically loads:
- Normalized data from `data/processed/norm_screen/`
- Templates from `data/processed/templates/` (if starting from step 7+)

## Processing Summary

After pipeline completion, `data/processed/processing_summary.json` contains:

```json
{
  "config": {
    "CONF_THRESH": 0.3,
    "MAX_INTERP_RUN": 60,
    "WINDOW_SECONDS": 60,
    ...
  },
  "flags": {
    "RUN_FILTER": true,
    "RUN_TEMPLATES": true,
    ...
  },
  "files_processed": 150,
  "participants": 50
}
```

This file documents the exact configuration used for processing, enabling reproducibility.

## Analysis Workflow

1. **Run pipeline**: `python process_pose_data.py`
2. **Check outputs**: Verify files created in `data/processed/`
3. **Inspect linear metrics**: Load `data/processed/linear_metrics/*.csv` for analysis
4. **Statistical analysis**: Use `pose_stats_figures.ipynb` for visualization and statistical tests
5. **Compare approaches**: Analyze differences between original, global, and participant normalization

## Dependencies

**Required**:
- Python 3.8+
- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7 (for Butterworth filter)
- tqdm >= 4.60 (for progress bars)

**Optional**:
- python-dotenv >= 0.19 (for .env file support)
- jupyter >= 1.0 (for notebook analysis)

Install all dependencies:
```bash
pip install numpy pandas scipy tqdm python-dotenv jupyter
```
