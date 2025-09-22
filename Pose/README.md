# Facial Pose Analysis Pipeline

A modular pipeline for processing and analyzing facial pose data from OpenPose landmark detection. This codebase provides tools for quality control, feature extraction, and statistical analysis of facial pose data.

## Overview

This pipeline transforms raw OpenPose facial landmark data into analysis-ready features. The code is modular and documented for reproducibility.

### Key Features

- **Quality Control**: Automated detection and masking of unreliable data periods
- **Coordinate Normalization**: Procrustes alignment and original stabilization methods
- **Feature Extraction**: Individual functions for each facial feature (blinks, mouth movement, head pose)
- **Temporal Filtering**: Butterworth and other filters for noise reduction
- **Statistical Analysis**: Comprehensive statistics and normalization options
- **Visualization**: Research-quality plots and figures
- **Reproducibility**: Detailed documentation and parameter tracking

## Directory Structure

```
Pose/
├── pose_analysis.ipynb          # Main analysis notebook
├── utils/                       # Modular utility functions
│   ├── __init__.py
│   ├── landmark_config.py       # Landmark definitions and configurations
│   ├── quality_control.py       # QC analysis and bad window detection
│   ├── coordinate_normalization.py  # Procrustes alignment and stabilization
│   ├── feature_extraction.py    # Individual feature extraction functions
│   ├── temporal_filtering.py    # Temporal smoothing and filtering
│   ├── masking.py              # Quality-based data masking
│   ├── statistical_analysis.py # Statistical analysis and normalization
│   ├── plotting.py             # Visualization functions
│   ├── data_loading.py         # Experimental condition mapping
│   └── pipeline.py             # Complete processing pipeline
├── feature_data/               # Output: Processed feature files
├── output/                     # Output: Analysis results and figures
└── _old/                       # Original code for reference
```

## Usage

Run the complete analysis by executing `pose_analysis.ipynb` or use the pipeline functions directly:

```python
from utils.pipeline import run_complete_pose_pipeline

output_paths = run_complete_pose_pipeline(
    raw_input_dir="data/raw_pose",
    output_base_dir="data/processed",
    coordinate_system="procrustes",
    apply_temporal_filter=True,
    cutoff_frequency=10.0
)
```

## Features Extracted

- **Eye Aspect Ratio**: `blink_dist` - Eye closure measurement
- **Mouth Opening**: `mouth_dist` - Lip separation distance
- **Head Rotation**: `head_rotation_angle` - Head orientation
- **Face Center**: `center_face_x`, `center_face_y`, `center_face_magnitude`
- **Eye Regions**: `left_eye_*`, `right_eye_*` coordinates
- **Pupil/Gaze**: `avg_pupil_*` - Gaze position and deviation
- **Quality Indicators**: Landmark detection confidence scores

## Processing Steps

1. **Quality Control**: Window-based analysis with confidence thresholding
2. **Coordinate Normalization**: Procrustes alignment or eye-corner stabilization
3. **Feature Extraction**: Facial expression and movement features
4. **Temporal Filtering**: Butterworth low-pass filtering
5. **Statistical Analysis**: Group comparisons and visualization

## Function Documentation

### Quality Control Functions
- `run_quality_control_batch()`: Analyze data quality across files
- `analyze_file_quality()`: Single file quality analysis
- `summarize_quality_control()`: Generate quality summaries

### Feature Extraction Functions
- `extract_all_features()`: Extract all facial features
- `extract_eye_aspect_ratio()`: Eye closure measurements
- `extract_mouth_opening()`: Mouth movement features
- `extract_head_rotation()`: Head pose features
- `extract_pupil_features()`: Gaze tracking features

### Coordinate Normalization Functions
- `compute_procrustes_alignment()`: Procrustes shape alignment
- `compute_original_alignment()`: Eye-corner stabilization

### Statistical Analysis Functions
- `calculate_summary_statistics()`: Descriptive statistics
- `compare_groups_statistical()`: Group comparisons
- `perform_feature_analysis()`: Complete statistical analysis
- `apply_z_score_normalization()`: Z-score normalization

### Data Loading Functions
- `load_pose_data_with_conditions()`: Load data with experimental conditions
- `parse_condition_mapping()`: Map participant trials to conditions
- `prepare_data_for_statistical_analysis()`: Prepare data for analysis

### Visualization Functions
- `plot_qc_summary()`: Quality control visualizations
- `plot_feature_timeseries()`: Feature time series plots
- `plot_statistical_bars()`: Statistical comparison plots
- `plot_correlation_matrix()`: Feature correlation plots

## Output Files

- **Feature Data**: `feature_data/*.csv` - Extracted features per participant/trial
- **Quality Control**: `quality_control/*.csv` - Data quality assessments
- **Reports**: `reports/*.csv` - Processing summaries and logs
- **Figures**: `output/figures/*.png` - Statistical plots and visualizations