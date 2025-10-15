# ECG Processing Pipeline

Pipeline for processing Zephyr BioHarness ECG data, detecting R-peaks, and extracting heart rate variability (HRV) features for cognitive workload analysis.

## Overview

This pipeline processes raw Zephyr ECG CSV files through the following steps:

1. **Load raw data** - Import Zephyr ECG waveform and summary data
2. **Clean signal** - Apply bandpass filtering and detrending to remove noise
3. **Detect R-peaks** - Identify heartbeat peaks using configurable algorithms
4. **Correct artifacts** - Apply Kubios method to fix ectopic beats and artifacts
5. **Calculate heart rate** - Interpolate HR between R-peaks using spline methods
6. **Apply windowing** - Segment signals into 60-second windows with 50% overlap
7. **Extract HRV features** - Compute time, frequency, and non-linear domain metrics per window

The pipeline integrates with the pose pipeline's condition mapping system and produces output compatible with random forest modeling.

**Output**: Cleaned ECG signals, R-peak locations, heart rate time series, and windowed HRV features with participant and condition labels (~15 windows per 480-second session).

## Directory Structure

```
ecg/
├── process_ecg_data.py             # Main processing script
│
├── utils/
│   ├── config.py                   # Configuration parameters and processing flags
│   ├── ecg_utils.py                # Core ECG processing functions
│   └── __init__.py
│
├── data/
│   ├── ecg_data/                   # Raw Zephyr CSV files (input)
│   │   ├── 3105_ecg_session01.csv      # Participant 3105, ECG waveform
│   │   ├── 3105_summary_session01.csv  # Participant 3105, HR/BR/activity summary
│   │   └── ...
│   │
│   └── processed/                  # Pipeline outputs (see structure below)
│       ├── signals/                # Cleaned ECG signals and R-peaks (optional)
│       ├── features/               # HRV features per file
│       └── combined/               # Combined features across all files
│
└── ecg_analysis.ipynb              # Statistical analysis and visualization notebook
```

## Data Format

### Input Files

**Location**: `data/ecg_data/`

**Naming convention**:
- ECG waveform: `<participantID>_ecg_session<number>.csv`
- Summary data: `<participantID>_summary_session<number>.csv`

**Example**:
- `3105_ecg_session01.csv` (participant 3105, ECG waveform, session 1)
- `3105_summary_session01.csv` (participant 3105, HR/BR/activity, session 1)

**Required columns in ECG file**:
- `EcgWaveform`: Raw ECG signal (mV)
- `Time`: Timestamp in format `dd/mm/yyyy HH:MM:SS.fff`

**Summary file columns** (optional, for contextual data):
- `HeartRate`: Device-computed heart rate (bpm)
- `BreathingRate`: Breathing rate (breaths/min)
- `Posture`: Body posture code
- `Activity`: Physical activity level
- `Time`: Timestamp

**Sampling rate**: 250 Hz (Zephyr BioHarness standard)

**Session-to-Trial Mapping**: Session numbers (session01, session02, session03) are mapped to trial numbers (1, 2, 3), which are then mapped to condition codes (L, M, H) using the participant info file.

### Participant Info File

**Location**: Project root `participant_info.csv`

**Required columns**:
- `Participant ID`: Participant ID (e.g., 3105)
- `session01`, `session02`, `session03`: Condition codes for each session

**Condition codes**: L (Low), M (Moderate), H (High)

**Example**:
```
Participant ID,session01,session02,session03
3105,L,M,H
3106,M,H,L
```

This file maps session numbers to experimental conditions. The pipeline uses this to generate condition-based output filenames (e.g., `3105_L_ecg_features.csv` instead of `3105_session01_ecg_features.csv`).

## Output Structure

```
data/processed/
├── signals/                        # Cleaned signals and R-peaks (optional)
│   └── <pid>_<cond>_ecg_signals.csv
│
├── features/                       # HRV features
│   └── <pid>_<cond>_ecg_features.csv
│
└── combined/                       # Combined datasets
    └── ecg_features_all.csv        # All participants and conditions
```

**Filename convention**: `<pid>_<cond>_ecg_<type>.csv`
- `<pid>`: Participant ID (e.g., 3105)
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
- `neurokit2`: ECG signal processing and HRV analysis
- `python-dotenv`: Environment variable management (optional)

### 2. Configure Data Paths

**Option A: Use environment variables (recommended for development)**

Create or update the `.env` file in the project root:

```bash
# .env file
ECG_RAW_DIR=/path/to/your/data/ecg_data
ECG_OUT_BASE=/path/to/your/output/processed
PARTICIPANT_INFO_FILE=participant_info.csv
```

The `.env` file is not committed to version control and allows each developer to use custom paths.

**Option B: Use default paths (recommended for published data)**

Ensure your data is in the standard location:
```
ecg/
└── data/
    └── ecg_data/                   # Raw Zephyr CSVs here
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
cd ecg
python process_ecg_data.py
```

**Force reprocessing** (overwrite existing files):
```bash
python process_ecg_data.py --overwrite
```

### 4. Monitor Progress

The pipeline displays progress messages for each file:
- Loading and validation status
- Session-to-condition mapping confirmation
- R-peak detection results
- HRV feature extraction
- Output file creation
- Processing summary (successful/skipped/failed)

## Configuration

### Key Parameters

Edit `utils/config.py` to modify processing parameters:

```python
# ECG Processing Parameters
SAMPLE_RATE = 250                   # Zephyr BioHarness sampling rate (Hz)

# Signal cleaning method
# Options: 'neurokit', 'biosppy', 'pantompkins1985', 'hamilton2002',
#          'elgendi2010', 'engzeemod2012', 'vg'
CLEANING_METHOD = "engzeemod2012"

# R-peak detection method
# Options: 'neurokit', 'pantompkins1985', 'hamilton2002', 'zong2003',
#          'martinez2004', 'christov2004', 'gamboa2008', 'elgendi2010',
#          'engzeemod2012', 'manikandan2012', 'kalidas2017', 'nabian2018',
#          'rodrigues2021', 'promac'
PEAK_METHOD = "engzeemod2012"

# Signal quality assessment method
# Options: 'averageQRS', 'zhao2018'
QUALITY_METHOD = "averageQRS"

# Signal quality approach
# Options: 'simple', 'fuzzy'
QUALITY_APPROACH = "fuzzy"

# Heart rate interpolation method
# Options: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
#          'previous', 'next', 'monotone_cubic'
INTERPOLATION_METHOD = "monotone_cubic"

# Windowing parameters (for windowed HRV feature extraction)
WINDOW_SECONDS = 60                 # Window duration in seconds
WINDOW_OVERLAP = 0.5                # Window overlap fraction (0.5 = 50% overlap)
```

### Processing Flags

Control which outputs are generated:

```python
# Output control
SAVE_SIGNALS = True                 # Save cleaned signals, R-peaks, HR
SAVE_FEATURES = True                # Save HRV features
PLOT_SIGNALS = False                # Plot during processing (slow)
BASELINE_CORRECTION = False         # Apply baseline correction
```

## Processing Details

### Step 1: Load and Validate Data

**Function**: `import_zephyr_ecg_data(data_directory, participant_id, session_num)`

Reads Zephyr ECG and summary CSV files and performs validation:
- Checks filename format and file existence
- Validates presence of required columns (`EcgWaveform`)
- Parses timestamps to datetime objects
- Drops unnecessary columns (battery, temperature, etc.)
- Returns both ECG waveform and summary DataFrames

**Validation errors**: Raises `FileNotFoundError` or `ValueError` if validation fails.

### Step 2: Clean ECG Signal

**Function**: `nk.ecg_clean(ecg_signal, sampling_rate, method)`

**NeuroKit2 cleaning pipeline**:
- Removes baseline wander (detrending)
- Applies bandpass filtering (typically 0.5-40 Hz)
- Removes high-frequency noise
- Preserves QRS complex morphology

**Method**: Configurable via `CLEANING_METHOD` (default: "engzeemod2012")

**Output**: Cleaned ECG signal ready for R-peak detection

### Step 3: Assess Signal Quality

**Function**: `nk.ecg_quality(ecg_cleaned, sampling_rate, method, approach)`

**Quality assessment**:
- Evaluates signal quality on per-sample or per-segment basis
- Returns quality scores (0 = poor, 1 = good)
- Used to exclude low-quality segments from analysis

**Methods**:
- `averageQRS`: Compares QRS morphology to average template
- `zhao2018`: Machine learning-based quality assessment

**Output**: Quality scores for each sample

### Step 4: Detect R-Peaks

**Function**: `nk.ecg_peaks(ecg_cleaned, sampling_rate, method)`

**R-peak detection algorithms**:
- Multiple configurable algorithms available
- Default: "engzeemod2012" (robust to noise and baseline wander)
- Returns sample indices of detected R-peaks

**Output**:
- `instant_peaks`: DataFrame with R-peak binary mask
- `rpeaks`: Dictionary with R-peak indices

### Step 5: Correct R-Peaks

**Function**: `nk.signal_fixpeaks(rpeaks, sampling_rate, method="Kubios")`

**Artifact correction** (Kubios method):
- Detects ectopic beats (premature or delayed)
- Identifies missing beats
- Corrects long/short RR intervals
- Uses iterative approach for robust correction

**Output**:
- Corrected R-peak indices
- Original (uncorrected) R-peak indices for comparison

### Step 6: Calculate Heart Rate

**Function**: `nk.ecg_rate(rpeaks, sampling_rate, interpolation_method)`

**Heart rate interpolation**:
- Computes instantaneous HR from RR intervals (HR = 60000 / RR_ms)
- Interpolates HR between beats using splines
- Returns HR time series at original sampling rate

**Interpolation method**: Configurable (default: "monotone_cubic" for smooth, physiologically plausible HR)

**Output**: Heart rate time series aligned with ECG signal

### Step 6: Apply Windowing

**Function**: `extract_windowed_hrv_features(signals, rpeaks, window_seconds, overlap, sr)`

**Windowing approach**:
- Segments signals into **60-second windows** with **50% overlap**
- Window size: 60 seconds = 15,000 samples (at 250 Hz)
- Step size: 30 seconds (50% overlap) = 7,500 samples
- Example: For 480-second session → 15 overlapping windows

**Per-window processing**:
- Extracts R-peaks within each window
- Requires minimum 5 R-peaks per window for valid HRV analysis
- Converts R-peak indices to window-local coordinates
- Computes HRV features independently for each window

**Rationale**: 60-second windows provide:
- Adequate data for time-domain HRV metrics
- Sufficient resolution for basic frequency analysis (though limited for VLF)
- Temporal tracking of HRV dynamics during task
- Consistency with pose and eye tracking feature extraction

**Output**: DataFrame with window metadata (window_index, t_start_sec, t_end_sec) plus HRV features

### Step 7: Extract HRV Features Per Window

**Function**: `nk.hrv_time()`, `nk.hrv_frequency()`, `nk.hrv_nonlinear()`

**HRV feature domains**:

**Time domain** (18 features):
- `HRV_MeanNN`: Mean NN interval (ms)
- `HRV_SDNN`: Standard deviation of NN intervals
- `HRV_RMSSD`: Root mean square of successive differences
- `HRV_pNN50`: Percentage of successive NNs differing by > 50ms
- Plus: median, range, CVNN, CVSD, etc.

**Frequency domain** (16 features):
- `HRV_LF`: Low frequency power (0.04-0.15 Hz)
- `HRV_HF`: High frequency power (0.15-0.4 Hz)
- `HRV_LFHF`: LF/HF ratio (sympathovagal balance)
- `HRV_VLF`: Very low frequency power
- Plus: normalized powers, peak frequencies, etc.

**Non-linear domain** (20+ features):
- `HRV_SD1`, `HRV_SD2`: Poincaré plot measures
- `HRV_ApEn`: Approximate entropy
- `HRV_SampEn`: Sample entropy
- `HRV_DFA_alpha1`, `HRV_DFA_alpha2`: Detrended fluctuation analysis
- Plus: correlation dimension, recurrence measures, fractal dimensions

**Total**: ~60 HRV features extracted per file

**Output**: Single-row DataFrame with all HRV features

## Feature Descriptions

### Time Domain Features

**NN interval statistics**:
- **HRV_MeanNN**: Mean NN interval (ms) - baseline heart period
- **HRV_SDNN**: Standard deviation of NN intervals - overall HRV
- **HRV_RMSSD**: Root mean square of successive differences - short-term HRV
- **HRV_pNN50**: Percentage of successive NNs differing > 50ms - parasympathetic activity

**Derived metrics**:
- **HRV_CVNN**: Coefficient of variation of NN intervals (SDNN/MeanNN)
- **HRV_MedianNN**: Median NN interval
- **HRV_MadNN**: Median absolute deviation of NN intervals
- **HRV_MCVNN**: Mean coefficient of variation

### Frequency Domain Features

**Power spectral density**:
- **HRV_LF**: Low frequency power (0.04-0.15 Hz) - mixed sympathetic/parasympathetic
- **HRV_HF**: High frequency power (0.15-0.4 Hz) - parasympathetic (respiratory sinus arrhythmia)
- **HRV_VLF**: Very low frequency power (< 0.04 Hz) - long-term regulation
- **HRV_LFHF**: LF/HF ratio - sympathovagal balance indicator

**Normalized powers**:
- **HRV_LFn**: Normalized LF (LF / (LF + HF))
- **HRV_HFn**: Normalized HF (HF / (LF + HF))

**Peak frequencies**:
- **HRV_LFpeak**: Peak frequency in LF band
- **HRV_HFpeak**: Peak frequency in HF band (respiratory rate estimate)

### Non-Linear Features

**Poincaré plot**:
- **HRV_SD1**: Short-term variability (beat-to-beat)
- **HRV_SD2**: Long-term variability
- **HRV_SD1SD2**: SD1/SD2 ratio

**Entropy measures** (complexity/regularity):
- **HRV_ApEn**: Approximate entropy
- **HRV_SampEn**: Sample entropy
- **HRV_FuzzyEn**: Fuzzy entropy

**Fractal analysis**:
- **HRV_DFA_alpha1**: Short-term fractal scaling exponent
- **HRV_DFA_alpha2**: Long-term fractal scaling exponent

**Other**:
- **HRV_CorrDim**: Correlation dimension
- **HRV_RecurrenceRate**: Recurrence rate from recurrence plot

## Command-Line Options

```
python process_ecg_data.py [--overwrite]
```

### --overwrite

Force reprocessing of all files, ignoring existing output files.

**Without flag**: Pipeline skips files that already have output in the features directory. This saves processing time when adding new data files.

**With flag**: All files are reprocessed regardless of existing outputs.

## Output File Details

### Signals Data (`signals/<pid>_<cond>_ecg_signals.csv`)

Contains sample-by-sample processed ECG data (if `SAVE_SIGNALS=True`):

**Columns**:
- `ECG_Raw`: Original raw ECG signal
- `ECG_Clean`: Cleaned ECG signal
- `ECG_Rate`: Interpolated heart rate time series
- `ECG_Quality`: Signal quality scores
- `ECG_Quality_RPeaks`: Quality with corrected R-peaks
- `ECG_Quality_RPeaksUncorrected`: Quality with original R-peaks
- `ECG_R_Peaks`: Binary mask of R-peak locations
- `participant`: Participant ID
- `condition`: Condition code (L, M, H)

**Note**: These files can be large (250 samples/second × session duration)

### Features Data (`features/<pid>_<cond>_ecg_features.csv`)

Contains windowed HRV features computed from 60-second overlapping windows:

**Columns**: ~60 HRV features plus:
- `window_index`: Window number (0, 1, 2, ...)
- `t_start_sec`: Window start time in seconds
- `t_end_sec`: Window end time in seconds
- `participant`: Participant ID
- `condition`: Condition code (L, M, H)
- `filename`: Original ECG filename

**Format**: Multiple rows per file (typically ~15 windows for 480-second session), compatible with pose and eye tracking pipelines for random forest modeling

**Example**:
```
window_index,t_start_sec,t_end_sec,HRV_MeanNN,HRV_SDNN,...,participant,condition,filename
0,0.0,60.0,577.62,11.57,...,3105,L,3105_ecg_session01.csv
1,30.0,90.0,580.45,12.23,...,3105,L,3105_ecg_session01.csv
2,60.0,120.0,582.11,13.01,...,3105,L,3105_ecg_session01.csv
```

### Combined Data (`combined/ecg_features_all.csv`)

Concatenation of all individual feature files:
- Sorted by participant and condition
- Includes data from all participants and conditions
- Ready for statistical analysis and machine learning
- Compatible with mixed effects models (see `ecg_analysis.ipynb`)

## Integration with Pose Pipeline

The ECG pipeline integrates with the pose pipeline for consistent data organization:

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
2. Look up condition in participant_info.csv (participant 3105, session01 → L)
3. Use condition code in output filenames (3105_L_ecg_features.csv)

## Analysis Workflow

1. **Run pipeline**: `python process_ecg_data.py`
2. **Check outputs**: Verify files created in `data/processed/`
3. **Load combined features**: Use `data/processed/combined/ecg_features_all.csv` for analysis
4. **Statistical analysis**: Use `ecg_analysis.ipynb` for visualization and statistical tests
5. **Multimodal integration**: Combine with pose and eye tracking features for comprehensive workload modeling

## Dependencies

**Required**:
- Python 3.8+
- numpy >= 1.20
- pandas >= 1.3
- neurokit2 >= 0.2.0 (ECG processing and HRV analysis)

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
- Check that `ECG_RAW_DIR` in `.env` points to correct directory
- Verify directory exists and contains CSV files

### "Missing EcgWaveform column in data"
- Verify Zephyr ECG CSV has column named `EcgWaveform`
- Check for typos or extra spaces in column names
- Ensure you're loading the `*_ecg_session*.csv` file, not `*_summary_session*.csv`

### "No condition found for participant"
- Verify `participant_info.csv` exists in project root
- Check participant ID matches between CSV filename and participant info
- Verify participant has condition mapping for the session number
- Check column names in participant_info.csv (should be: session01, session02, session03)

### "WARNING: NeuroKit2 not available"
- Install NeuroKit2: `pip install neurokit2`
- Required for all ECG signal processing and HRV analysis

### "HRV feature extraction failed"
- Signal may be too short (< 5 minutes recommended for frequency domain features)
- Too few R-peaks detected (check signal quality)
- Try different `PEAK_METHOD` in config.py for better R-peak detection

### R-peak detection issues
- Adjust `PEAK_METHOD` in config.py (try 'neurokit', 'pantompkins1985', or 'hamilton2002')
- Check signal quality - excessive noise may require better cleaning
- Verify sampling rate is correct (250 Hz for Zephyr)

## Notes

- ECG data is processed at the original sampling rate (250 Hz)
- HRV features are computed from 60-second windows with 50% overlap (30-second step)
- Each 480-second session generates approximately 15 windowed feature records
- Time-domain features work well with 60-second windows
- Frequency-domain features are limited by window duration (VLF analysis requires longer windows)
- Non-linear features (e.g., DFA_alpha2) may not be calculable for short windows and will be omitted
- R-peak correction (Kubios method) is essential for accurate HRV metrics
- Missing or ectopic beats are automatically corrected in feature extraction
- Windowed format ensures consistency with pose and eye tracking pipelines for multimodal analysis
- Output files are compatible with random forest modeling pipeline used across all modalities

## References

**NeuroKit2**:
- Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., ... & Najjar, R. P. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*, 53(4), 1689-1696.

**HRV Standards**:
- Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043-1065.

**Peak Detection Algorithms**:
- Engelse, W. A., & Zeelenberg, C. (1979). A single scan algorithm for QRS-detection and feature extraction. *Computers in Cardiology*, 6(1979), 37-42.
- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, 3, 230-236.
