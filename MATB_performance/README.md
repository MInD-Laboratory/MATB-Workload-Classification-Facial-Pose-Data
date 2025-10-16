# MATB Performance Data Processing Pipeline

This pipeline processes Multi-Attribute Task Battery (MATB) performance data and extracts windowed performance metrics for all four sub-tasks.

## Overview

The MATB is a complex multitasking environment consisting of four concurrent sub-tasks:
1. **System Monitoring (SYSMON)** - Detecting and responding to indicator light and gauge failures
2. **Communications (COMMS)** - Responding to radio call-signs
3. **Tracking (TRACK)** - Manual joystick control to keep cursor within target
4. **Resource Management (RESMAN)** - Monitoring and managing fuel tank levels

This pipeline extracts windowed performance metrics for each sub-task, including failure rates, reaction times, and aggregate accuracy measures.

## Data Description

### Input Data
- **Location**: `PNAS-MATB/matb_outputs/` (external shared folder)
- **Format**: CSV files from MATB task
- **Naming**: `{participant_id}_session{number}.csv` (e.g., `3105_session01.csv`)
- **Content**: Event logs and performance measurements for all MATB sub-tasks

### Output Data

#### Individual Files
**Location**: `data/processed/metrics/`
**Format**: `{participant}_{condition}_performance_metrics.csv`

**Columns**:
- `participant`: Participant ID
- `condition`: Workload condition (L/M/H)
- `window_index`: Window number (0-14)
- `start_time`, `end_time`: Window time range in seconds
- **System Monitoring metrics**:
  - `sysmon_failure_rate`: Percentage of missed events
  - `sysmon_average_reaction_times`: Mean RT in milliseconds
- **Communications metrics**:
  - `comms_failure_rate`: Percentage of missed events
  - `comms_events`: Total radio events
  - `comms_own_events`: Own call-sign events
  - `comms_average_reaction_times`: Mean RT in milliseconds
- **Tracking metrics**:
  - `track_failure_rate`: Percentage of time cursor out of target
- **Resource Management metrics**:
  - `resman_failure_rate`: Percentage of time tanks out of tolerance
- **Aggregate metrics**:
  - `average_accuracy`: Mean accuracy across all four sub-tasks (%)
  - `average_reaction_time`: Mean RT across SYSMON and COMMS (seconds)

#### Combined File
**Location**: `data/processed/combined/performance_metrics_all.csv`
**Content**: All individual files concatenated and sorted by participant, condition, window

## Processing Pipeline

### 1. Configuration (`utils/config.py`)

Centralized configuration for all processing parameters:

```python
from MATB_performance.utils.config import CFG

# Path configuration
CFG.RAW_DIR          # Raw MATB CSV files location
CFG.OUT_BASE         # Processed output directory

# Processing parameters
CFG.WINDOW_SECONDS   # 60 (window size)
CFG.WINDOW_OVERLAP   # 0.5 (50% overlap)
CFG.TOTAL_TIME       # 480 (8 minutes per session)
```

**Environment Variables** (`.env`):
```bash
MATB_RAW_DIR=/path/to/PNAS-MATB/matb_outputs
MATB_OUT_BASE=/path/to/MATB-Workload-Classification-Facial-Pose-Data/MATB_performance/data/processed
```

### 2. Utility Functions (`utils/performance_utils.py`)

Core metric extraction functions:
- `sysmon_measures()` - Extract system monitoring metrics
- `comms_measures()` - Extract communications metrics
- `track_measures()` - Extract tracking performance
- `resman_measures()` - Extract resource management performance

### 3. Processing Script (`process_performance_data.py`)

Main command-line script for batch processing:

```bash
# Process all files
python process_performance_data.py

# Overwrite existing output
python process_performance_data.py --overwrite
```

**Workflow**:
1. Load raw MATB CSV files from `RAW_DIR`
2. Parse participant ID and session from filename
3. Map session to condition using `participant_info.csv`
4. Extract windowed metrics for all four sub-tasks
5. Calculate aggregate accuracy and reaction time
6. Save individual and combined metric files

### 4. Analysis Notebook (`performance_analysis.ipynb`)

Interactive notebook for visualization and statistical analysis:
- Loads pre-processed metrics
- Adds session order information
- Runs linear mixed-effects models (R via rpy2)
- Generates publication-ready figures

## Usage

### Initial Setup

1. **Configure paths** in `.env`:
   ```bash
   MATB_RAW_DIR=/path/to/PNAS-MATB/matb_outputs
   MATB_OUT_BASE=./MATB_performance/data/processed
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy python-dotenv

   # For statistical analysis in notebook
   pip install rpy2 matplotlib
   ```

3. **Ensure R packages** are installed (for notebook):
   ```r
   install.packages(c("lmerTest", "emmeans"))
   ```

### Processing Data

```bash
cd MATB_performance
python process_performance_data.py
```

**Output**:
```
======================================================================
MATB Performance Processing Pipeline
======================================================================

Input directory: /path/to/PNAS-MATB/matb_outputs
Output directory: ./data/processed
Overwrite mode: False

Found 129 CSV files to process

Processing: 3105_session01.csv
  Loaded 161089 rows
  Participant: 3105, Condition: L
  Saved metrics: 3105_L_performance_metrics.csv
  Extracted 15 windows

...

======================================================================
Processing Summary
======================================================================
Total files:       129
Successful:        129
Skipped (exists):  0
Failed:            0
======================================================================

Processing complete!
```

### Analysis and Visualization

Open and run `performance_analysis.ipynb` to:
- Explore processed metrics
- Run statistical analysis
- Generate figures

## File Structure

```
MATB_performance/
├── utils/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   └── performance_utils.py     # Metric extraction functions
├── data/
│   └── processed/
│       ├── metrics/             # Individual participant files
│       │   ├── 3105_L_performance_metrics.csv
│       │   ├── 3105_M_performance_metrics.csv
│       │   └── ...
│       └── combined/            # Combined files
│           ├── performance_metrics_all.csv
│           └── processing_summary.json
├── process_performance_data.py  # Main processing script
├── performance_analysis.ipynb   # Analysis and visualization
└── README.md                    # This file
```

## Window Parameters

- **Window Size**: 60 seconds
- **Overlap**: 50% (30-second step)
- **Total Duration**: 480 seconds (8 minutes)
- **Windows per Session**: 15 windows

**Window Times**:
```
Window 0:  0-60s
Window 1:  30-90s
Window 2:  60-120s
...
Window 14: 420-480s
```

## Metric Calculation Details

### System Monitoring (SYSMON)
- **Events**: 6 indicators (2 lights, 4 scales) that can fail
- **Failure Rate**: (Misses / Total Events) × 100
- **Reaction Time**: Time from failure onset to correct response

### Communications (COMMS)
- **Events**: Radio call-signs (both "own" and "other")
- **Failure Rate**: (Misses + Bad Radio Responses) / Own Events × 100
- **Reaction Time**: Time from call-sign to correct response

### Tracking (TRACK)
- **Measurement**: Continuous cursor position sampling
- **Failure Rate**: (Time Out of Target / Total Time) × 100

### Resource Management (RESMAN)
- **Measurement**: Fuel tank A and B levels sampled continuously
- **Failure Rate**: (Time Out of Tolerance / Total Time) × 100

### Aggregate Metrics
- **Average Accuracy**: 100 - mean(failure rates across all 4 sub-tasks)
- **Average Reaction Time**: mean(SYSMON RT, COMMS RT) converted to seconds

## Integration with Other Pipelines

This pipeline follows the same architecture as:
- **Pose**: Facial landmark tracking
- **Eye Tracking**: Gaze and pupil metrics
- **ECG**: Heart rate variability
- **GSR**: Electrodermal activity

All pipelines share:
- Same windowing parameters (60s, 50% overlap)
- Same condition mapping via `participant_info.csv`
- Same output naming convention
- Compatible output formats for multimodal analysis

## Troubleshooting

### File not found errors
- Check `.env` paths point to correct OneDrive location
- Ensure `PNAS-MATB/matb_outputs/` folder exists
- Verify `participant_info.csv` is in `PNAS-MATB/` folder

### Import errors
```bash
pip install pandas numpy python-dotenv
```

### Processing errors
- Check MATB CSV files have expected columns
- Verify filename format: `{participant}_session{number}.csv`
- Check participant IDs match those in `participant_info.csv`

## Output Validation

Verify processing success:

```bash
# Check individual metrics
ls data/processed/metrics/ | wc -l    # Should be ~129 files

# Check combined file
wc -l data/processed/combined/performance_metrics_all.csv  # Should be ~1936 lines (129 files × 15 windows + header)

# View processing summary
cat data/processed/combined/processing_summary.json
```

## Citation

If using this pipeline, please cite the original MATB reference:

> Santiago-Espada, Y., Myer, R. R., Latorella, K. A., & Comstock Jr, J. R. (2011).
> *The Multi-Attribute Task Battery II (MATB-II) Software for Human Performance and Workload Research:
> A User's Guide* (NASA/TM–2011-217164). NASA Langley Research Center.

## Contact

For questions or issues, please refer to the main project documentation.
