# Random Forest Modeling for Workload Classification

This directory contains Random Forest classification models for predicting workload levels from multimodal physiological and behavioral data collected during the Multi-Attribute Task Battery (MATB).

## Overview

The modeling pipeline implements **four different cross-validation strategies** to assess model performance and generalization:

1. **Random Split** - Stratified 80/20 split across all windows (within-task generalization)
2. **Participant Split** - Random ~20% of participants held out (cross-participant generalization)
3. **Leave-One-Participant-Out (LOPO)** - Each participant held out once (strictest generalization test)
4. **Participant-Specific Learning Curves** - Individual models per participant showing calibration requirements

Each of the first three strategies runs across **31 experiments** covering:
- 5 individual modalities (pose, performance, eye-tracking, GSR, ECG)
- 10 two-way combinations
- 10 three-way combinations
- 5 four-way combinations
- 1 five-way combination (all modalities)

The participant-specific strategy trains separate models for each participant at multiple training sizes to assess how much calibration data is needed for individual-level prediction.

## Directory Structure

```
modeling/
├── utils/
│   ├── __init__.py              # Module exports
│   ├── config.py                # Feature paths and hyperparameters
│   └── pipeline_utils.py        # Shared functions for data loading, modeling, evaluation
├── model_output/                # All results saved here
│   ├── random_split/
│   │   ├── procrustes_global/   # Results using global Procrustes normalization
│   │   ├── procrustes_participant/
│   │   └── original/
│   ├── participant_split/
│   │   ├── procrustes_global/
│   │   ├── procrustes_participant/
│   │   └── original/
│   ├── lopo/
│   │   ├── procrustes_global/
│   │   ├── procrustes_participant/
│   │   └── original/
│   └── participant_specific/
│       ├── procrustes_global/
│       ├── procrustes_participant/
│       └── original/
├── run_rf_random_split.py       # Random split strategy
├── run_rf_participant_split.py  # Participant split strategy
├── run_rf_lopo.py               # Leave-one-participant-out strategy
├── run_rf_participant_specific.py  # Participant-specific learning curves
└── README.md                    # This file
```

## Feature Modalities

### 1. Pose (Facial Landmarks)
- **Path**: `../Pose/data/processed/linear_metrics/`
- **Variants**:
  - `original_linear.csv` - No Procrustes normalization
  - `procrustes_participant_linear.csv` - Participant-specific Procrustes alignment
  - `procrustes_global_linear.csv` - Global Procrustes alignment
- **Features**: Head position (x, y, z), head rotation (pitch, yaw, roll), blink aperture, mouth movements
- **Optional**: Velocity and acceleration derivatives can be included via `use_pose_derivatives` flag

### 2. Performance (MATB Task Metrics)
- **Path**: `../MATB_performance/data/processed/combined/performance_metrics_all.csv`
- **Features**: Task accuracy, reaction times, failure rates for all MATB sub-tasks (tracking, system monitoring, communication, resource management)

### 3. Eye Tracking (Gaze & Pupil)
- **Path**: `../eye_tracking/data/processed/combined/eyegaze_metrics_all.csv`
- **Features**: Fixation duration, saccade metrics, pupil diameter, blink rate, gaze dispersion

### 4. GSR (Galvanic Skin Response)
- **Path**: `../gsr/data/processed/combined/gsr_features_all.csv`
- **Features**: Skin conductance level (SCL), phasic/tonic components, response amplitudes, rise/recovery times

### 5. ECG (Heart Rate Variability)
- **Path**: `../ecg/data/processed/combined/ecg_features_all.csv`
- **Features**: Heart rate (HR), RMSSD, pNN50, SDNN, LF/HF ratio, time and frequency domain HRV metrics

## Class Configurations

The pipeline supports **six different class configurations** for flexible analysis:

### Three-Class (Default)
- **all**: L vs M vs H (Low, Medium, High workload)

### Binary Comparisons (Exclude Middle Class)
- **L_vs_H**: Low vs High (exclude Medium)
- **L_vs_M**: Low vs Medium (exclude High)
- **M_vs_H**: Medium vs High (exclude Low)

### Binary Comparisons (Merge Classes)
- **LM_vs_H**: Low+Medium vs High
- **L_vs_MH**: Low vs Medium+High

These configurations are controlled via the `class_config` parameter in `DEFAULT_MODEL_CONFIG`. Each configuration filters or merges the data appropriately and is reflected in the output directory naming with suffixes like `_LvH`, `_LMvH`, etc.

## Usage

### Basic Usage

Each script supports the same core command-line interface:

```bash
# Run with default settings (procrustes_global pose variant, 3-class)
python run_rf_random_split.py

# Run with original pose variant
python run_rf_random_split.py --pose-variant original

# Run with participant-specific procrustes normalization
python run_rf_participant_split.py --pose-variant procrustes_participant

# Overwrite existing results
python run_rf_lopo.py --pose-variant procrustes_global --overwrite

# Dry run (see what would execute without running)
python run_rf_random_split.py --dry-run
```

### Participant-Specific Learning Curves

The participant-specific script trains separate models for each participant using varying amounts of training data:

```bash
# Default: stratified sampling (random N windows per condition)
python run_rf_participant_specific.py

# Temporal-stratified sampling (first N windows per condition with buffer)
python run_rf_participant_specific.py --strategy temporal_stratified

# With different pose variant
python run_rf_participant_specific.py --pose-variant original

# Overwrite existing results
python run_rf_participant_specific.py --overwrite
```

**Training Sizes**: The script tests [1, 2, 3, 5, 7, 9, 11] windows per condition (L, M, H), meaning training sizes range from 3 total windows (1 per condition) up to 33 windows (11 per condition).

**Sampling Strategies**:
- **stratified**: Randomly samples N windows from each condition (balanced representation)
- **temporal_stratified**: Takes the first N windows (in time) from each condition with a 1-window buffer to prevent data leakage due to 50% overlap between consecutive windows

**Role of Random Seeds**:

The `n_seeds` parameter (default: 10) controls different aspects depending on the sampling strategy:

- **stratified**: Random seeds control both:
  1. Which windows are randomly sampled for training/test (data selection)
  2. RandomForest model initialization (bootstrap sampling, feature subsets, tie-breaking)

  Result: Different seeds produce different train/test splits AND different model initializations

- **temporal_stratified**: Random seeds only control:
  1. RandomForest model initialization (bootstrap sampling, feature subsets, tie-breaking)

  The train/test split is deterministic (always first N windows for training, window N+1 as buffer, windows N+2+ for test)

  Result: Different seeds produce the same train/test split but different model training outcomes

**Why Multiple Seeds Matter**: Even with deterministic data splitting (temporal_stratified), running multiple seeds provides:
- Confidence intervals around accuracy estimates
- Robustness against RandomForest initialization effects
- Variance estimates reflecting model training randomness rather than data sampling randomness

For both strategies, results are aggregated across seeds to provide mean and standard deviation metrics for each participant at each training size.

### Command-Line Arguments

**All Scripts**:
- `--pose-variant`: Pose normalization method
  - `original`: No Procrustes normalization
  - `procrustes_participant`: Participant-specific Procrustes alignment
  - `procrustes_global`: Global Procrustes alignment (default)
- `--overwrite`: Overwrite existing results (default: skip completed experiments)
- `--dry-run`: Show what would run without executing

**Random & Participant Split Only**:
- `--resume`: Resume incomplete experiments at seed level

**Participant-Specific Only**:
- `--strategy`: Sampling strategy for learning curves
  - `stratified`: Random N windows from each condition (default)
  - `temporal_stratified`: First N windows from each condition with 1-window buffer

### Recommended Workflow

**Step 1**: Run random split to compare pose variants and modality combinations (fastest, ~2-4 hours)
```bash
python run_rf_random_split.py --pose-variant procrustes_global
```

**Step 2**: Run participant split for best-performing configuration (~2-4 hours)
```bash
python run_rf_participant_split.py --pose-variant procrustes_global
```

**Step 3**: Run LOPO for strictest generalization assessment (~10-15 hours)
```bash
python run_rf_lopo.py --pose-variant procrustes_global
```

**Step 4**: Run participant-specific to assess individual calibration requirements (~8-12 hours)
```bash
python run_rf_participant_specific.py --pose-variant procrustes_global --strategy temporal_stratified
```

## Model Configuration

### Random Forest Hyperparameters

Defined in `utils/config.py`:

```python
RF_PARAMS = {
    "n_estimators": 300,        # Number of trees
    "max_depth": None,          # Unlimited tree depth
    "class_weight": "balanced", # Handle class imbalance
    "n_jobs": -1,               # Use all CPU cores
}
```

### Evaluation Settings

```python
DEFAULT_MODEL_CONFIG = {
    "n_seeds": 10,                    # Random seeds for reliability (10 for all strategies)
    "feature_selection": "backward",  # Backward elimination using permutation importance
    "use_pca": False,                 # Apply PCA dimensionality reduction
    "pca_variance": 0.95,             # Variance to retain if using PCA
    "write_cm": True,                 # Save confusion matrices
    "tune_hyperparameters": False,    # Run RandomizedSearchCV for hyperparameter tuning
    "tune_n_iter": 30,                # Number of tuning iterations
    "tune_cv_folds": 5,               # CV folds for tuning
    "use_pose_derivatives": True,     # Include velocity/acceleration features from pose
    "use_time_features": False,       # Include temporal position features (normalized time within condition)
    "class_config": "all",            # Class configuration (see Class Configurations section)
    "include_order": False,           # Include condition order (LMH vs LHM) as binary feature
}
```

### Feature Selection

When `feature_selection` is set to `"backward"`:
1. For participant-specific models: Features are selected using backward elimination on pooled data from the largest training size (11 windows per condition)
2. For other strategies: Features are selected using backward elimination with permutation importance
3. The same feature set is used across all random seeds for consistency
4. Selected features are saved in the output JSON files

### Condition Order Features

When `include_order` is True (default), the model includes a binary feature indicating whether the participant experienced conditions in LMH or LHM order. This captures potential order effects in the experimental design.

### Pose Derivatives

When `use_pose_derivatives` is True, the pipeline includes velocity (first derivative) and acceleration (second derivative) features computed from the pose time series. This captures dynamic aspects of head and facial movements.

### Time Features

When `use_time_features` is True, the pipeline includes normalized temporal position within each condition as features. This captures potential time-on-task effects.

## Output Format

### Random, Participant, and LOPO Splits

Each experiment produces a JSON file (e.g., `pose_perf.json`):

```json
{
  "name": "pose_perf",
  "config": {
    "split_strategy": "random",
    "n_seeds": 10,
    "feature_selection": "backward",
    "use_pose_derivatives": true,
    "use_time_features": false,
    "class_config": "all",
    "include_order": false,
    ...
  },
  "metrics": {
    "test_bal_acc_mean": 0.75,
    "test_bal_acc_std": 0.03,
    "test_f1_mean": 0.74,
    "test_f1_std": 0.03,
    "test_kappa_mean": 0.62,
    "test_kappa_std": 0.04,
    "test_precision_L_mean": 0.80,
    "test_recall_L_mean": 0.78,
    ...
  },
  "confusion_matrix": [[...], [...], [...]],
  "selected_features": [...],
  "n_features": 97,
  "n_seeds": 10,
  "timestamp": "2025-10-17T10:30:00"
}
```

### Participant-Specific Learning Curves

Each experiment produces:
1. **JSON file** with full results across all training sizes
2. **CSV file** (`*_learning_curve.csv`) for easy plotting

JSON structure:
```json
{
  "name": "pose_perf",
  "config": {
    "split_strategy": "participant_specific",
    "sampling_strategy": "temporal_stratified",
    "n_seeds": 10,
    ...
  },
  "results_by_size": {
    "1": {
      "participant_results": [
        {"participant": "3105", "test_bal_acc_mean": 0.55, "test_bal_acc_std": 0.08, ...},
        ...
      ],
      "aggregate_mean": 0.58,
      "aggregate_std": 0.12
    },
    "2": { ... },
    ...
    "11": { ... }
  },
  "learning_curve_summary": {
    "training_sizes": [1, 2, 3, 5, 7, 9, 11],
    "mean_accuracies": [0.58, 0.64, 0.68, 0.72, 0.75, 0.76, 0.77],
    "std_accuracies": [0.12, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06]
  }
}
```

### Experiment Log CSV

Quick summary of all experiments (`experiment_log.csv`):

```csv
experiment_name,split_strategy,n_features,n_seeds,test_bal_acc_mean,test_bal_acc_std,...
pose,random,85,10,0.68,0.04,...
performance,random,12,10,0.72,0.03,...
pose_perf,random,97,10,0.75,0.03,...
...
```

### LOPO Output

LOPO experiments run each fold with multiple random seeds and include per-participant results with confidence intervals:

```json
{
  "participant_results": [
    {
      "participant": "3105",
      "n_seeds": 10,
      "test_bal_acc_mean": 0.72,
      "test_bal_acc_std": 0.03,
      "test_f1_mean": 0.70,
      "test_f1_std": 0.02,
      ...
    },
    {
      "participant": "3206",
      "n_seeds": 10,
      "test_bal_acc_mean": 0.68,
      "test_bal_acc_std": 0.04,
      "test_f1_mean": 0.66,
      "test_f1_std": 0.03,
      ...
    },
    ...
  ],
  "n_participants": 49,
  "n_seeds": 10,
  "metrics": {
    "test_bal_acc_mean": 0.70,
    "test_bal_acc_std": 0.08,
    ...
  }
}
```

Each participant fold is evaluated with 10 different random seeds (RF initialization), and results are aggregated to provide mean and standard deviation. The overall metrics aggregate the participant-level means across all participants.

## Interpreting Results

### Key Metrics

- **Balanced Accuracy**: Primary metric (handles class imbalance by averaging recall across classes)
- **F1 Score**: Harmonic mean of precision and recall (macro-averaged for multi-class)
- **Cohen's Kappa**: Agreement adjusted for chance (measures beyond-chance classification)
- **Per-class metrics**: Precision, recall, F1 for each workload level individually

### Comparing Split Strategies

1. **Random Split**: Upper bound (easiest, same participants in train/test)
2. **Participant Split**: Moderate difficulty (unseen participants, some participant overlap possible)
3. **LOPO**: Lower bound (strictest test, each participant completely held out)
4. **Participant-Specific**: Individual-level performance showing calibration needs

Expected performance: Random ≥ Participant ≥ LOPO

The gap between random and LOPO indicates how much performance depends on participant-specific patterns vs. generalizable workload signatures.

### Comparing Pose Variants

- **Original**: Raw normalized coordinates (interocular distance normalization)
- **Procrustes Participant**: Aligned to participant-specific mean template (removes individual baseline differences)
- **Procrustes Global**: Aligned to grand mean template across all participants (removes global position/rotation variability)

Best variant depends on whether individual differences or common patterns are more predictive. Procrustes normalization typically improves generalization by removing irrelevant geometric variation.

### Participant-Specific Learning Curves

The learning curves show how balanced accuracy increases with more training windows per condition. Key insights:

- **Plateau point**: Training size where accuracy stops improving significantly
- **Inter-participant variability**: Standard deviation across participants at each training size
- **Minimum viable calibration**: Smallest training size achieving acceptable performance

These curves inform practical deployment decisions about how much individual calibration data is needed.

## Expected Runtime

| Strategy | Pose Variants | Total Experiments | Estimated Time |
|----------|---------------|-------------------|----------------|
| Random Split | 1 | 31 | 2-4 hours |
| Random Split | 3 (all) | 93 | 6-12 hours |
| Participant Split | 1 | 31 | 2-4 hours |
| LOPO | 1 | 31 | 10-15 hours |
| Participant-Specific | 1 | 31 × 7 sizes | 8-12 hours |

**Total for all strategies and variants**: ~35-50 hours

Can be parallelized by running different scripts or pose variants simultaneously in separate terminal sessions.

## Experiment Tracking

### Checking Progress

```bash
# View summary of configured experiments
python run_rf_random_split.py --dry-run

# View participant-specific training plan
python run_rf_participant_specific.py --dry-run

# Check experiment log
cat model_output/random_split/procrustes_global/experiment_log.csv

# Check specific experiment results
cat model_output/lopo/procrustes_global/pose_perf.json | python -m json.tool
```

### Resuming Interrupted Runs

```bash
# Random and participant split support resume at seed level
python run_rf_random_split.py --resume

# Participant-specific supports resume at experiment/size level
python run_rf_participant_specific.py

# LOPO runs all participants atomically (no resume, but skips completed experiments)
python run_rf_lopo.py
```

The pipeline automatically detects and skips completed experiments unless `--overwrite` is specified. This allows safe re-running of scripts after interruptions.

## Dependencies

Required packages:
```bash
pip install numpy pandas scikit-learn tqdm scipy
```

All scripts use scikit-learn's RandomForestClassifier and standard evaluation metrics. No deep learning frameworks required.

## Citation

If using this modeling pipeline, please cite:

> Richardson, M. et al. (2025). Multimodal Classification of Cognitive Workload in Complex Tasks.
> [Publication details pending]
