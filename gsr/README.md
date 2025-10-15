# GSR Processing Pipeline

Pipeline for processing Shimmer GSR data, decomposing into phasic (SCR) and tonic (SCL) components, detecting SCR peaks, and extracting electrodermal activity (EDA) features for cognitive workload analysis.

## Overview

This pipeline processes raw Shimmer GSR CSV files through the following steps:

1. **Load raw data** - Import Shimmer GSR waveform data
2. **Clean signal** - Remove noise and smooth the EDA signal
3. **Decompose** - Separate into phasic (SCR) and tonic (SCL) components
4. **Detect SCR peaks** - Identify skin conductance responses
5. **Extract features** - Compute interval-related EDA metrics

The pipeline integrates with the pose pipeline's condition mapping system and produces output compatible with random forest modeling.

**Output**: Cleaned EDA signals, SCR/SCL components, peak locations, and comprehensive EDA features with participant and condition labels.

## Directory Structure

```
gsr/
├── process_gsr_data.py             # Main processing script
│
├── utils/
│   ├── config.py                   # Configuration parameters and processing flags
│   ├── gsr_utils.py                # Core GSR processing functions
│   └── __init__.py
│
├── data/
│   ├── gsr_data/                   # Raw Shimmer CSV files (input)
│   │   ├── 3208_session01.csv      # Participant 3208, session 1
│   │   ├── 3208_session02.csv      # Participant 3208, session 2
│   │   └── ...
│   │
│   └── processed/                  # Pipeline outputs (see structure below)
│       ├── signals/                # Cleaned EDA, SCR, SCL per file
│       ├── features/               # EDA features per file
│       └── combined/               # Combined features across all files
│
└── gsr_analysis.ipynb              # Statistical analysis and visualization notebook
```

## Data Format

### Input Files

**Location**: `data/gsr_data/`

**Naming convention**: `<participantID>_session<number>.csv`

**Example**: `3208_session01.csv` (participant 3208, session 1)

**Required columns**:
- `Shimmer_AD66_GSR_Skin_Conductance_CAL`: Skin conductance (microsiemens)
- `Shimmer_AD66_Timestamp_FormattedUnix_CAL`: Timestamps
- `elapsed_seconds`: Time in seconds from start

**Sampling rate**: 20 Hz (Shimmer device standard)

**Session-to-Trial Mapping**: Session numbers (session01, session02, session03) are mapped to trial numbers (1, 2, 3), which are then mapped to condition codes (L, M, H) using the participant info file.

### Participant Info File

**Location**: Project root `participant_info.csv`

**Required columns**:
- `Participant ID`: Participant ID (e.g., 3208)
- `session01`, `session02`, `session03`: Condition codes for each session

**Condition codes**: L (Low), M (Moderate), H (High)

**Example**:
```
Participant ID,session01,session02,session03
3208,L,M,H
3209,M,H,L
```

This file maps session numbers to experimental conditions. The pipeline uses this to generate condition-based output filenames (e.g., `3208_L_gsr_features.csv` instead of `3208_session01_gsr_features.csv`).

## Output Structure

```
data/processed/
├── signals/                        # Cleaned EDA signals and components
│   └── <pid>_<cond>_gsr_signals.csv
│
├── features/                       # EDA features
│   └── <pid>_<cond>_gsr_features.csv
│
└── combined/                       # Combined datasets
    └── gsr_features_all.csv        # All participants and conditions
```

**Filename convention**: `<pid>_<cond>_gsr_<type>.csv`
- `<pid>`: Participant ID (e.g., 3208)
- `<cond>`: Condition letter (L, M, or H)
- `<type>`: Output type (signals, features)

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas neurokit2 python-dotenv
```

**Required packages**:
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `neurokit2`: EDA signal processing and feature extraction
- `python-dotenv`: Environment variable management (optional)

### 2. Configure Data Paths

**Option A: Use environment variables (recommended for development)**

Create or update the `.env` file in the project root:

```bash
# .env file
GSR_RAW_DIR=/path/to/your/data/gsr_data
GSR_OUT_BASE=/path/to/your/output/processed
PARTICIPANT_INFO_FILE=participant_info.csv
```

The `.env` file is not committed to version control and allows each developer to use custom paths.

**Option B: Use default paths (recommended for published data)**

Ensure your data is in the standard location:
```
gsr/
└── data/
    └── gsr_data/                   # Raw Shimmer CSVs here
```

And participant info in project root:
```
MATB-Workload-Classification-Facial-Pose-Data/
└── participant_info.csv            # Participant metadata here
```

No configuration needed - the pipeline will use these default paths.

### 3. Run the Pipeline

**Process new files only** (skip files with existing output):
```bash
cd gsr
python process_gsr_data.py
```

**Force reprocessing** (overwrite existing files):
```bash
python process_gsr_data.py --overwrite
```

### 4. Monitor Progress

The pipeline displays progress messages for each file:
- Loading and validation status
- Session-to-condition mapping confirmation
- SCR peak detection results
- Output file creation
- Processing summary (successful/skipped/failed)

## Configuration

### Key Parameters

Edit `utils/config.py` to modify processing parameters:

```python
# GSR Processing Parameters
SAMPLE_RATE = 20                    # Shimmer device sampling rate (Hz)

# EDA signal cleaning method
# Options: 'neurokit', 'biosppy'
CLEANING_METHOD = "neurokit"

# Phasic/tonic decomposition method
# Options: 'highpass', 'cvxEDA', 'smoothmedian'
PHASIC_METHOD = "highpass"

# SCR peak detection method
# Options: 'neurokit', 'gamboa2008', 'kim2004', 'vanhalem2020', 'nabian2018'
PEAK_METHOD = "neurokit"

# Window parameters (for optional windowed analysis)
WINDOW_SECONDS = 60                 # Window size in seconds
WINDOW_OVERLAP = 0.5                # Window overlap fraction (0.5 = 50%)
```

### Processing Flags

Control which outputs are generated:

```python
# Output control
SAVE_SIGNALS = True                 # Save cleaned signals, SCR, SCL
SAVE_FEATURES = True                # Save EDA features
PLOT_SIGNALS = False                # Plot during processing (slow)
```

## Processing Details

### Step 1: Load and Validate Data

**Function**: `import_shimmer_eda_data(data_directory, participant_id, session_num)`

Reads Shimmer GSR CSV file and performs validation:
- Checks filename format and file existence
- Validates presence of required columns
- Parses timestamps to datetime objects
- Returns GSR DataFrame

**Validation errors**: Raises `FileNotFoundError` or `ValueError` if validation fails.

### Step 2: Clean EDA Signal

**Function**: `nk.eda_clean(eda_signal, sampling_rate, method)`

**NeuroKit2 cleaning pipeline**:
- Removes noise and artifacts
- Smooths the signal
- Preserves SCR morphology

**Method**: Configurable via `CLEANING_METHOD` (default: "neurokit")

**Output**: Cleaned EDA signal ready for decomposition

### Step 3: Decompose into Phasic and Tonic Components

**Function**: `nk.eda_phasic(eda_cleaned, sampling_rate, method)`

**Component separation**:
- **Phasic (SCR)**: Fast-changing responses to stimuli
- **Tonic (SCL)**: Slow baseline skin conductance level

**Methods**:
- `highpass`: High-pass filtering (default, fast)
- `cvxEDA`: Convex optimization (slower, more accurate)
- `smoothmedian`: Median smoothing

**Output**: DataFrame with EDA_Phasic and EDA_Tonic columns

### Step 4: Detect SCR Peaks

**Function**: `nk.eda_peaks(eda_phasic, sampling_rate, method)`

**SCR peak detection**:
- Identifies skin conductance responses in phasic component
- Returns peak locations, onsets, amplitudes, rise times, recovery times

**Methods**:
- `neurokit`: Default NeuroKit2 algorithm
- `gamboa2008`: Gamboa et al. (2008) algorithm
- `kim2004`: Kim et al. (2004) algorithm

**Output**:
- Peak indices
- Peak characteristics (amplitude, rise time, etc.)

### Step 5: Extract EDA Features

**Function**: `nk.eda_intervalrelated(epochs, sampling_rate)`

**Interval-related features** (> 10 seconds):

**SCR Features**:
- **SCR_Peaks_N**: Number of SCR occurrences
- **SCR_Peaks_Amplitude_Mean**: Mean amplitude of SCR peaks

**Tonic Features**:
- **EDA_Tonic_SD**: Standard deviation of tonic component

**Sympathetic Activity**:
- **EDA_Sympathetic**: Sympathetic activity index (requires > 64s)
- **EDA_SympatheticN**: Normalized sympathetic activity

**Signal Characteristics**:
- **EDA_Autocorrelation**: Signal autocorrelation (requires > 30s)

**Total**: 6 EDA features extracted per session

**Output**: Single-row DataFrame with all EDA features

## Feature Descriptions

### SCR Features

**Number of Responses**:
- **SCR_Peaks_N**: Count of skin conductance responses
  - Indicates arousal frequency
  - Higher values suggest more frequent responses to stimuli

**Amplitude**:
- **SCR_Peaks_Amplitude_Mean**: Average SCR peak amplitude (microsiemens)
  - Indicates strength of responses
  - Larger amplitudes suggest stronger arousal

### Tonic Features

**Baseline Variability**:
- **EDA_Tonic_SD**: Standard deviation of tonic (SCL) component
  - Indicates baseline stability
  - Higher values suggest more variable baseline arousal

### Sympathetic Activity

**Arousal Indices**:
- **EDA_Sympathetic**: Sympathetic activity index
  - Derived from spectral analysis
  - Higher values indicate increased sympathetic activation

- **EDA_SympatheticN**: Normalized sympathetic activity (0-1 range)
  - Normalized version for cross-session comparison

### Signal Characteristics

**Temporal Structure**:
- **EDA_Autocorrelation**: Autocorrelation of EDA signal
  - Indicates temporal regularity
  - Values near 1 suggest highly regular patterns

## Command-Line Options

```
python process_gsr_data.py [--overwrite]
```

### --overwrite

Force reprocessing of all files, ignoring existing output files.

**Without flag**: Pipeline skips files that already have output in the features directory. This saves processing time when adding new data files.

**With flag**: All files are reprocessed regardless of existing outputs.

## Output File Details

### Signals Data (`signals/<pid>_<cond>_gsr_signals.csv`)

Contains sample-by-sample processed EDA data (if `SAVE_SIGNALS=True`):

**Columns**:
- `EDA_Raw`: Original raw GSR signal
- `EDA_Clean`: Cleaned EDA signal
- `EDA_Tonic`: Tonic component (SCL)
- `EDA_Phasic`: Phasic component (SCR)
- `SCR_Onsets`: Binary mask of SCR onset locations
- `SCR_Peaks`: Binary mask of SCR peak locations
- `SCR_Height`: SCR amplitude (including tonic)
- `SCR_Amplitude`: SCR amplitude (excluding tonic)
- `SCR_RiseTime`: Rise time for each SCR
- `SCR_Recovery`: Binary mask of recovery points
- `SCR_RecoveryTime`: Recovery time for each SCR
- `participant`: Participant ID
- `condition`: Condition code (L, M, H)

**Note**: These files can be useful for visual inspection but are not required for analysis

### Features Data (`features/<pid>_<cond>_gsr_features.csv`)

Contains EDA features computed from entire session:

**Columns**: 6 EDA features plus:
- `participant`: Participant ID
- `condition`: Condition code (L, M, H)
- `filename`: Original GSR filename

**Format**: One row per file (session), compatible with pose/ECG pipelines for random forest modeling

### Combined Data (`combined/gsr_features_all.csv`)

Concatenation of all individual feature files:
- Sorted by participant and condition
- Includes data from all participants and conditions
- Ready for statistical analysis and machine learning
- Compatible with mixed effects models (see `gsr_analysis.ipynb`)

## Integration with Pose/ECG/Eye Tracking Pipelines

The GSR pipeline integrates with other pipelines for consistent data organization:

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
2. Look up condition in participant_info.csv (participant 3208, session01 → L)
3. Use condition code in output filenames (3208_L_gsr_features.csv)

## Analysis Workflow

1. **Run pipeline**: `python process_gsr_data.py`
2. **Check outputs**: Verify files created in `data/processed/`
3. **Load combined features**: Use `data/processed/combined/gsr_features_all.csv` for analysis
4. **Statistical analysis**: Use `gsr_analysis.ipynb` for visualization and statistical tests
5. **Multimodal integration**: Combine with pose, ECG, and eye tracking features for comprehensive workload modeling

## Dependencies

**Required**:
- Python 3.8+
- numpy >= 1.20
- pandas >= 1.3
- neurokit2 >= 0.2.0 (EDA processing and feature extraction)

**Optional**:
- python-dotenv >= 0.19 (for .env file support)
- jupyter >= 1.0 (for notebook analysis)
- rpy2 >= 3.5 (for R-based statistical analysis in notebook)

Install all dependencies:
```bash
pip install numpy pandas neurokit2 python-dotenv jupyter
```

## Troubleshooting

### "Raw data directory not found"
- Check that `GSR_RAW_DIR` in `.env` points to correct directory
- Verify directory exists and contains CSV files

### "Missing GSR column in data"
- Verify Shimmer CSV has column: `Shimmer_AD66_GSR_Skin_Conductance_CAL`
- Check for typos or extra spaces in column names
- Ensure you're loading the correct Shimmer device file

### "No condition found for participant"
- Verify `participant_info.csv` exists in project root
- Check participant ID matches between CSV filename and participant info
- Verify participant has condition mapping for the session number
- Check column names in participant_info.csv (should be: session01, session02, session03)

### "WARNING: NeuroKit2 not available"
- Install NeuroKit2: `pip install neurokit2`
- Required for all EDA signal processing and feature extraction

### "EDA feature extraction failed"
- Signal may be too short (< 10 seconds recommended for interval features)
- Too few SCR peaks detected (check signal quality)
- Try different `PEAK_METHOD` in config.py for better peak detection

### SCR peak detection issues
- Adjust `PEAK_METHOD` in config.py (try 'gamboa2008', 'kim2004', or 'neurokit')
- Check signal quality - excessive noise or poor contact may reduce detection
- Verify sampling rate is correct (20 Hz for Shimmer)

## Notes

- GSR data is processed at the original sampling rate (20 Hz)
- EDA features are computed from the entire session (no windowing by default)
- Sympathetic activity features require at least ~64 seconds of clean data
- Autocorrelation features require at least ~30 seconds of data
- SCR detection is performed on the phasic component only
- Output files are compatible with random forest modeling pipeline used for pose/ECG/eye tracking data
- For windowed EDA analysis, signals can be segmented in post-processing

## References

**NeuroKit2**:
- Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., ... & Najjar, R. P. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*, 53(4), 1689-1696.

**EDA/GSR Methods**:
- Boucsein, W. (2012). *Electrodermal activity* (2nd ed.). Springer Science & Business Media.
- Dawson, M. E., Schell, A. M., & Filion, D. L. (2007). The electrodermal system. *Handbook of psychophysiology*, 3, 159-181.

**Peak Detection Algorithms**:
- Gamboa, H. (2008). Multi-modal behavioral biometrics based on HCI and electrophysiology. *PhD Thesis*, Universidade Técnica de Lisboa.
- Kim, K. H., Bang, S. W., & Kim, S. R. (2004). Emotion recognition system using short-term monitoring of physiological signals. *Medical and biological engineering and computing*, 42(3), 419-427.
