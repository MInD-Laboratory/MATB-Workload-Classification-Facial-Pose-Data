# Eye Tracking Processing Pipeline

Processes eye tracker data to extract gaze patterns, pupil size, and eye movement events that indicate cognitive workload.

## What This Does (Simple Explanation)

This pipeline analyzes eye movement data to measure:
- Where you're looking on the screen (gaze position)
- How big your pupils are (changes with mental effort)
- When you blink
- When your eyes are holding still (fixations)
- When your eyes are jumping between locations (saccades)

These patterns change when mental workload increases or decreases.

## Pipeline Overview Diagram

```
Raw EyeLink CSV Files (gaze coordinates & pupil data at 1000 Hz)
           │
           ▼
    ┌──────────────────┐
    │ 1. Load & Validate│
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ 2. Normalize     │
    │    Gaze to       │
    │    Screen (0-1)  │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ 3. Detect        │
    │    Blinks        │
    │  (pupil size)    │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ 4. Detect        │
    │    Fixations     │
    │  (stable gaze)   │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ 5. Detect        │
    │    Saccades      │
    │  (rapid moves)   │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ 6. Extract       │
    │    Windowed      │
    │    Metrics       │
    │  (60s windows)   │
    └────────┬─────────┘
             │
             ▼
    Eye Tracking Features Ready for Analysis
```

## Technical Overview

This pipeline processes raw EyeLink CSV files through the following steps:

1. **Load raw data** - Import EyeLink CSVs with binocular gaze coordinates and pupil size
2. **Normalize gaze** - Convert to screen-relative coordinates (0-1 range), mask invalid data
3. **Detect blinks** - Identify blinks using pupil size with z-score and raw threshold criteria
4. **Detect fixations** - Identify fixations based on maximum gaze dispersion threshold
5. **Detect saccades** - Identify saccades using velocity and acceleration thresholds
6. **Extract metrics** - Compute windowed statistics including velocity, acceleration, dispersion, and event counts

The pipeline integrates with the pose pipeline's condition mapping system and produces output compatible with random forest modeling.

**Output**: Normalized gaze coordinates, detected events, windowed metrics with participant and condition labels.

## Directory Structure

```
eye_tracking/
├── process_eye_data.py             # Main processing script
│
├── utils/
│   ├── config.py                   # Configuration parameters and processing flags
│   ├── eye_gaze_utils.py           # Core processing functions
│   └── __init__.py
│
├── data/
│   ├── eyelink_data/               # Raw EyeLink CSV files (input)
│   │   ├── 3105_session01.csv      # Participant 3105, session 1
│   │   ├── 3105_session02.csv      # Participant 3105, session 2
│   │   └── ...
│   │
│   └── processed/                  # Pipeline outputs (see structure below)
│       ├── normalized/             # Normalized gaze coordinates and pupil data
│       ├── metrics/                # Windowed metrics per file
│       ├── combined/               # Combined metrics across all files
│       └── events/                 # Detected blinks, fixations, saccades (optional)
│
└── eye_gaze_analysis.ipynb         # Statistical analysis and visualization notebook
```

## Data Format

### Input Files

**Location**: `data/eyelink_data/`

**Naming convention**: `<participantID>_session<number>.csv`

**Example**: `3105_session01.csv` (participant 3105, session 1)

**Required columns**:
- `R Gaze X`, `R Gaze Y`: Right eye gaze coordinates (pixels)
- `L Gaze X`, `L Gaze Y`: Left eye gaze coordinates (pixels)
- `R Pupil Size`, `L Pupil Size`: Pupil diameter (arbitrary units)
- `Time Stamp`: Time in milliseconds

**Sampling rate**: 1000 Hz (configurable in `config.py`)

**Session-to-Trial Mapping**: Session numbers (session01, session02, session03) are mapped to trial numbers (1, 2, 3), which are then mapped to condition codes (L, M, H) using the participant info file.

### Participant Info File

**Location**: `../data/participant_info.csv` (project root data directory)

**Required columns**:
- `Participant ID`: Participant ID (e.g., 3105)
- `session01`, `session02`, `session03`: Condition codes for each session

**Condition codes**: L (Low), M (Moderate), H (High)

**Example**:
```
Participant ID,session01,session02,session03
3105,L,M,H
3106,L,M,H
```

This file maps trial numbers to experimental conditions. The pipeline uses this to generate condition-based output filenames (e.g., `3105_L_eyegaze_metrics.csv` instead of `3105_session01_eyegaze_metrics.csv`).

## Output Structure

```
data/processed/
├── normalized/                     # Normalized gaze and pupil data
│   └── <pid>_<cond>_eyegaze_normalized.csv
│
├── metrics/                        # Windowed eye tracking metrics
│   └── <pid>_<cond>_eyegaze_metrics.csv
│
├── combined/                       # Combined datasets
│   └── eyegaze_metrics_all.csv     # All participants and conditions
│
└── events/                         # Detected events (optional)
    └── <pid>_<cond>_eyegaze_events.csv
```

**Filename convention**: `<pid>_<cond>_eyegaze_<type>.csv`
- `<pid>`: Participant ID (e.g., 3105)
- `<cond>`: Condition letter (L, M, or H)
- `<type>`: Output type (normalized, metrics, events)

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas scipy python-dotenv
```

**Required packages**:
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `scipy`: Signal processing (median filter, interpolation)
- `python-dotenv`: Environment variable management (optional)

### 2. Configure Data Paths

**Option A: Use environment variables (recommended for development)**

Create or update the `.env` file in the project root:

```bash
# .env file
EYELINK_RAW_DIR=/path/to/your/data/eyelink_data
EYELINK_OUT_BASE=/path/to/your/output/processed
PARTICIPANT_INFO_FILE=participant_info.csv
```

The `.env` file is not committed to version control and allows each developer to use custom paths.

**Option B: Use default paths (recommended for published data)**

Ensure your data is in the standard location:
```
eye_tracking/
└── data/
    └── eyelink_data/           # Raw EyeLink CSVs here
```

And participant info in project root data directory:
```
MATB-Workload-Classification-Facial-Pose-Data/
└── data/
    └── participant_info.csv        # Participant metadata here
```

No configuration needed - the pipeline will use these default paths.

### 3. Run the Pipeline

**Process new files only** (skip files with existing output):
```bash
cd eye_tracking
python process_eye_data.py
```

**Force reprocessing** (overwrite existing files):
```bash
python process_eye_data.py --overwrite
```

### 4. Monitor Progress

The pipeline displays progress messages for each file:
- Loading and validation status
- Session-to-condition mapping confirmation
- Event detection results
- Output file creation
- Processing summary (successful/skipped/failed)

## Configuration

### Key Parameters

Edit `utils/config.py` to modify processing parameters:

```python
# Screen parameters
SCREEN_WIDTH = 2560             # Screen width in pixels
SCREEN_HEIGHT = 1440            # Screen height in pixels

# Sampling parameters
SAMPLE_RATE = 1000              # Eye tracker sampling rate (Hz)

# Window parameters
WINDOW_SECONDS = 60             # Window size in seconds
WINDOW_OVERLAP = 0.5            # Window overlap fraction (0.5 = 50%)
MISSING_MAX = 0.25              # Maximum proportion of missing data per window

# Blink detection thresholds
BLINK_Z_THRESH = -2.0           # Z-score threshold for pupil size
BLINK_RAW_FLOOR = 30.0          # Minimum pupil size for valid data
BLINK_MAX_DUR = 0.6             # Maximum blink duration (seconds)

# Fixation detection thresholds
FIXATION_MAX_DIST = 0.02        # Maximum gaze dispersion (normalized units)
FIXATION_MIN_DUR = 0.20         # Minimum fixation duration (seconds)

# Saccade detection thresholds
SACCADE_MIN_LEN = 2             # Minimum saccade length (samples)
SACCADE_VEL_THRESH = 0.5        # Velocity threshold (normalized units/second)
SACCADE_ACC_THRESH = 5.0        # Acceleration threshold (normalized units/second²)

# Filtering parameters
MEDIAN_FILTER_KERNEL = 5        # Kernel size for median filter
OUTLIER_N_SD = 5.0              # Standard deviations for outlier clipping
PUPIL_OUTLIER_THRESH = 3.0      # SD threshold for pupil size outliers
```

### Processing Flags

Control which outputs are generated:

```python
# Output control
SAVE_NORMALIZED = True          # Save normalized gaze coordinates
SAVE_EVENTS = False             # Save detected events (blinks, fixations, saccades)
OVERWRITE = False               # Skip processing if outputs exist
```

## Processing Details

### Step 1: Load and Validate Data

**Function**: `load_file_data(directory, filename)`

Reads EyeLink CSV file and performs validation:
- Checks filename format: `<participantID>_session<number>.csv`
- Validates presence of required columns
- Converts columns to numeric values
- Parses participant ID and session number

**Validation errors**: Returns `None` if validation fails, allowing processing to continue with other files.

### Step 2: Normalize Gaze Coordinates

**Function**: `normalize_gaze_to_screen(df, screen_width, screen_height)`

Converts pixel coordinates to screen-relative coordinates:
- Divides x-coordinates by `SCREEN_WIDTH`
- Divides y-coordinates by `SCREEN_HEIGHT`
- Result: Coordinates in range [0, 1] where (0, 0) is top-left and (1, 1) is bottom-right

**Invalid data handling**:
- Values outside [0, screen_dimension] are set to NaN
- Missing values remain as NaN

**Pupil outlier removal**:
- Detects outliers beyond `PUPIL_OUTLIER_THRESH` standard deviations
- Sets outlier pupil sizes to 0 (excluded from blink detection)

### Step 3: Detect Blinks

**Function**: `pupil_blink_detection(pr, pl, t, z_thresh, raw_floor, max_dur)`

Detects blinks using pupil size with dual criteria:

**Detection criteria**:
1. **Z-score threshold**: Pupil size drops below `BLINK_Z_THRESH` standard deviations (default -2.0)
2. **Raw threshold**: Pupil size falls below `BLINK_RAW_FLOOR` (default 30.0)
3. Both eyes must meet criteria simultaneously

**Duration filtering**:
- Events longer than `BLINK_MAX_DUR` (default 0.6 seconds) are excluded (likely noise)

**Output**: Boolean mask indicating blink frames

### Step 4: Detect Fixations

**Function**: `fixation_detection(gx, gy, t, maxdist, mindur)`

Detects fixations based on gaze stability:

**Detection criteria**:
1. **Dispersion threshold**: Maximum distance from mean gaze position < `FIXATION_MAX_DIST` (default 0.02 normalized units)
2. **Duration threshold**: Fixation lasts ≥ `FIXATION_MIN_DUR` (default 0.2 seconds)

**Algorithm**:
- Computes rolling mean of gaze position
- Calculates distance from mean for each sample
- Identifies contiguous segments below distance threshold
- Filters by minimum duration

**Output**: Boolean mask indicating fixation frames

### Step 5: Detect Saccades

**Function**: `saccade_detection(gx, gy, t, min_len, vel_thr, acc_thr)`

Detects saccades using velocity and acceleration thresholds:

**Detection criteria**:
1. **Velocity threshold**: Gaze velocity > `SACCADE_VEL_THRESH` (default 0.5)
2. **Acceleration threshold**: Gaze acceleration > `SACCADE_ACC_THRESH` (default 5.0)
3. **Minimum length**: Saccade lasts ≥ `SACCADE_MIN_LEN` samples (default 2)

**Algorithm**:
- Computes gaze velocity from position using gradient
- Computes gaze acceleration from velocity using gradient
- Identifies frames exceeding both thresholds
- Filters by minimum duration

**Output**: Boolean mask indicating saccade frames

### Step 6: Extract Windowed Metrics

**Function**: `extract_eye_metrics(df, participant_id, condition, sample_rate, win_sec, overlap_fr, missing_max, trim_sec, ds_factor)`

Extracts statistical metrics in sliding windows:

**Windowing**:
- Window size: `WINDOW_SECONDS` seconds (default 60s)
- Overlap: `WINDOW_OVERLAP` (default 0.5 = 50%)
- Hop size: (1 - overlap) × window_size (default 30s)

**Preprocessing**:
- Optionally trim first `trim_sec` seconds (default 0)
- Optionally downsample by factor `ds_factor` (default 1)
- Interpolate missing gaze data (linear interpolation)
- Apply median filter with outlier clipping

**Metrics extracted per window**:

**Event counts**:
- `blink_count`: Number of blink events detected
- `fixation_count`: Number of fixation events detected
- `saccade_count`: Number of saccade events detected
- `blink_frac`: Proportion of window in blink state
- `fixation_frac`: Proportion of window in fixation state
- `saccade_frac`: Proportion of window in saccade state

**Gaze statistics**:
- `mean_x`, `mean_y`: Mean gaze position
- `std_x`, `std_y`: Gaze position standard deviation
- `dispersion`: RMS dispersion (√(std_x² + std_y²))
- `mean_vel`: Mean gaze velocity
- `max_vel`: Maximum gaze velocity
- `mean_acc`: Mean gaze acceleration

**Pupil statistics**:
- `mean_pupil`: Mean pupil size (average of both eyes)

**Window metadata**:
- `participant`: Participant ID
- `condition`: Condition code (L, M, H)
- `window_index`: Sequential window number
- `start_time`: Window start time (milliseconds)
- `end_time`: Window end time (milliseconds)

**Missing data handling**:
- Windows with > `MISSING_MAX` (default 25%) missing data are excluded
- Missing data calculated separately for gaze and pupil metrics

## Feature Descriptions

### Event Features

- **blink_count**: Number of distinct blink events in window
- **blink_frac**: Proportion of window frames marked as blink (0-1)
- **fixation_count**: Number of distinct fixation events
- **fixation_frac**: Proportion of window frames in fixation state
- **saccade_count**: Number of distinct saccade events
- **saccade_frac**: Proportion of window frames in saccade state

### Gaze Features

- **mean_x, mean_y**: Average gaze position in normalized coordinates
- **std_x, std_y**: Standard deviation of gaze position (spatial variability)
- **dispersion**: Overall gaze dispersion (RMS of std_x and std_y)
- **mean_vel**: Average gaze velocity (normalized units/second)
- **max_vel**: Maximum gaze velocity
- **mean_acc**: Average gaze acceleration (normalized units/second²)

### Pupil Features

- **mean_pupil**: Average pupil diameter (mean of left and right eyes)

## Command-Line Options

```
python process_eye_data.py [--overwrite]
```

### --overwrite

Force reprocessing of all files, ignoring existing output files.

**Without flag**: Pipeline skips files that already have output in the metrics directory. This saves processing time when adding new data files.

**With flag**: All files are reprocessed regardless of existing outputs.

## Output File Details

### Normalized Data (`normalized/<pid>_<cond>_eyegaze_normalized.csv`)

Contains frame-by-frame normalized gaze coordinates and event markers (if `SAVE_NORMALIZED=True`):

**Columns**:
- Original columns from EyeLink CSV
- `R Gaze X Norm`, `R Gaze Y Norm`: Right eye normalized coordinates
- `L Gaze X Norm`, `L Gaze Y Norm`: Left eye normalized coordinates
- `Blink`: Boolean blink mask
- `Fixation`: Boolean fixation mask
- `Saccade`: Boolean saccade mask

### Metrics Data (`metrics/<pid>_<cond>_eyegaze_metrics.csv`)

Contains windowed metrics with participant and condition labels:

**Columns**: participant, condition, window_index, start_time, end_time, blink_count, blink_frac, fixation_count, fixation_frac, saccade_count, saccade_frac, mean_x, mean_y, std_x, std_y, dispersion, mean_vel, max_vel, mean_acc, mean_pupil

**Format**: Compatible with pose pipeline linear metrics for random forest modeling

### Combined Data (`combined/eyegaze_metrics_all.csv`)

Concatenation of all individual metrics files:
- Sorted by participant, condition, window_index
- Includes data from all participants and conditions
- Ready for statistical analysis and machine learning

### Events Data (`events/<pid>_<cond>_eyegaze_events.csv`)

Frame-by-frame event markers (if `SAVE_EVENTS=True`):

**Columns**: Time Stamp, Blink, Fixation, Saccade

## Integration with Pose Pipeline

The eye tracking pipeline integrates with the pose pipeline for consistent data organization:

**Shared resources**:
- Uses pose's `participant_info.csv` for condition mapping
- Imports `create_condition_mapping()` from pose utilities
- Imports `load_participant_info_file()` for participant data

**Consistent output format**:
- Same filename convention: `<pid>_<cond>_<type>.csv`
- Same condition codes: L, M, H
- Compatible CSV structure for combined modeling

**Session-to-condition mapping**:
1. Parse session number from filename (session01 → 1)
2. Look up condition in participant_info.csv (participant 3105, trial 1 → L)
3. Use condition code in output filenames (3105_L_eyegaze_metrics.csv)

## Analysis Workflow

1. **Run pipeline**: `python process_eye_data.py`
2. **Check outputs**: Verify files created in `data/processed/`
3. **Load combined metrics**: Use `data/processed/combined/eyegaze_metrics_all.csv` for analysis
4. **Statistical analysis**: Use `eye_gaze_analysis.ipynb` for visualization and statistical tests
5. **Multimodal integration**: Combine with pose pipeline linear metrics for comprehensive analysis

## Dependencies

**Required**:
- Python 3.8+
- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7 (for signal processing)

**Optional**:
- python-dotenv >= 0.19 (for .env file support)
- jupyter >= 1.0 (for notebook analysis)

Install all dependencies:
```bash
pip install numpy pandas scipy python-dotenv jupyter
```

## Troubleshooting

### "Raw data directory not found"
- Check that `EYELINK_RAW_DIR` in `.env` points to correct directory
- Verify directory exists and contains CSV files

### "Missing required columns"
- Verify EyeLink CSV has columns: R Gaze X, R Gaze Y, L Gaze X, L Gaze Y, R Pupil Size, L Pupil Size, Time Stamp
- Check for typos or extra spaces in column names

### "No condition found for participant"
- Verify `participant_info.csv` exists in project root
- Check participant ID matches between CSV filename and participant info
- Verify participant has condition mapping for the trial number

### "WARNING: scipy not available"
- Install scipy: `pip install scipy`
- Required for median filtering and interpolation operations

## Notes

- Eye tracking data is processed at the original sampling rate (default 1000 Hz)
- Blink detection requires both eyes to meet criteria simultaneously for robust detection
- Fixation and saccade detection use right eye by default (can be modified in code)
- Window-level metrics aggregate across entire window, dropping windows with excessive missing data
- Output files are compatible with random forest modeling pipeline used for pose data
