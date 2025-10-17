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
│   ├── participant_split/
│   ├── lopo/
│   └── participant_specific/
├── run_logs/                    # Configuration combination run logs
├── run_rf_random_split.py       # Random split strategy
├── run_rf_participant_split.py  # Participant split strategy
├── run_rf_lopo.py               # Leave-one-participant-out strategy
├── run_rf_participant_specific.py  # Participant-specific learning curves
├── run_config_combinations.py   # Automated configuration testing
├── check_config.py              # Verify current configuration
├── diagnose_config_issue.py     # Diagnostic tool for configuration problems
├── verify_fix.py                # Verify derivative filtering is working correctly
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
- **Features**: Task accuracy, reaction times, failure rates for all MATB sub-tasks

### 3. Eye Tracking (Gaze & Pupil)
- **Path**: `../eye_tracking/data/processed/combined/eyegaze_metrics_all.csv`
- **Features**: Fixation duration, saccade metrics, pupil diameter, blink rate, gaze dispersion

### 4. GSR (Galvanic Skin Response)
- **Path**: `../gsr/data/processed/combined/gsr_features_all.csv`
- **Features**: Skin conductance level, phasic/tonic components, response amplitudes, rise/recovery times

### 5. ECG (Heart Rate Variability)
- **Path**: `../ecg/data/processed/combined/ecg_features_all.csv`
- **Features**: Heart rate, RMSSD, pNN50, SDNN, LF/HF ratio, time and frequency domain HRV metrics

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

These configurations are controlled via the `class_config` parameter in `DEFAULT_MODEL_CONFIG` in `utils/config.py`.

## Basic Usage

### Running Individual Modeling Strategies

Each script supports the same core command-line interface:

```bash
# Run with default settings (procrustes_global pose variant, 3-class)
python run_rf_random_split.py

# Run with specific pose variant
python run_rf_random_split.py --pose-variant original

# Overwrite existing results
python run_rf_lopo.py --pose-variant procrustes_global --overwrite

# Dry run to see what would execute without running
python run_rf_random_split.py --dry-run

# Resume interrupted runs (random and participant split only)
python run_rf_random_split.py --resume
```

### Participant-Specific Learning Curves

```bash
# Default: stratified sampling (random N windows per condition)
python run_rf_participant_specific.py

# Temporal-stratified sampling (first N windows per condition with buffer)
python run_rf_participant_specific.py --strategy temporal_stratified

# With specific pose variant
python run_rf_participant_specific.py --pose-variant original --overwrite
```

**Training Sizes**: Tests [1, 2, 3, 5, 7, 9, 11] windows per condition (L, M, H).

**Sampling Strategies**:
- **stratified**: Randomly samples N windows from each condition (balanced representation)
- **temporal_stratified**: Takes first N windows (in time) from each condition with 1-window buffer to prevent data leakage

### Running Configuration Combinations

The `run_config_combinations.py` script automates testing multiple model configurations by systematically varying specified parameters. It updates `utils/config.py`, clears Python cache, runs the model, and repeats for all combinations.

```bash
# Test derivatives on/off with normalization on/off (4 combinations)
python run_config_combinations.py \
    --method random \
    --vary use_pose_derivatives normalize_features

# Test all feature selection options (3 combinations)
python run_config_combinations.py \
    --method lopo \
    --vary feature_selection

# Test PCA with different seed count
python run_config_combinations.py \
    --method random \
    --vary use_pca \
    --n-seeds 10

# Preview combinations without running
python run_config_combinations.py \
    --method lopo \
    --vary use_pose_derivatives normalize_features \
    --dry-run
```

**Available Methods**:
- `random` - Random split (30-60 min per combination)
- `participant` - Participant split (30-60 min per combination)
- `lopo` - Leave-one-participant-out (4-8 hours per combination)
- `specific` - Participant-specific (2-4 hours per combination)

**Available Parameters to Vary**:

Boolean (True/False):
- `use_pose_derivatives` - Include velocity/acceleration features
- `normalize_features` - Z-score normalization
- `use_pca` - PCA dimensionality reduction
- `tune_hyperparameters` - Hyperparameter tuning
- `use_time_features` - Temporal position features
- `include_order` - Condition order features

Categorical:
- `feature_selection` - backward, forward, or None
- `class_config` - all, L_vs_H, L_vs_M, M_vs_H, LM_vs_H, L_vs_MH

**How It Works**:
1. Generates all combinations of specified parameters
2. For each combination:
   - Updates `utils/config.py` with new parameter values
   - Clears Python cache to ensure changes load
   - Runs the specified modeling script with `--overwrite`
   - Logs success/failure
3. Restores original configuration when complete
4. Saves detailed log to `run_logs/run_METHOD_TIMESTAMP.json`

**Example**: Testing 2 parameters with 2 values each generates 4 combinations and runs the full modeling pipeline 4 times sequentially.

### Diagnostic and Verification Tools

```bash
# Check current configuration settings
python check_config.py

# Diagnose configuration issues (check if config changes are being applied)
python diagnose_config_issue.py

# Verify derivative filtering is working correctly
python verify_fix.py
```

These tools help verify that configuration changes are properly applied and affecting model behavior.

## Model Configuration

### Random Forest Hyperparameters

Defined in `utils/config.py`:

```python
RF_PARAMS = {
    "n_estimators": 500,        # Number of trees
    "max_depth": 30,            # Maximum tree depth
    "class_weight": "balanced", # Handle class imbalance
    "min_samples_split": 5,     # Minimum samples to split node
    "min_samples_leaf": 2,      # Minimum samples in leaf
    "max_features": "log2",     # Features per split
    "n_jobs": -1,               # Use all CPU cores
}
```

### Model Pipeline Settings

```python
DEFAULT_MODEL_CONFIG = {
    "n_seeds": 20,                    # Random seeds for reliability
    "feature_selection": "backward",  # Options: "backward", "forward", None
    "use_pose_derivatives": True,     # Include velocity/acceleration features
    "use_time_features": False,       # Include temporal position features
    "include_order": False,           # Include condition order as feature
    "normalize_features": True,       # Z-score normalization
    "class_config": "all",            # Class configuration
    "use_pca": False,                 # Apply PCA dimensionality reduction
    "pca_variance": 0.90,             # Variance to retain if using PCA
    "write_cm": True,                 # Save confusion matrices
    "tune_hyperparameters": False,    # Run RandomizedSearchCV
    "tune_n_iter": 30,                # Number of tuning iterations
    "tune_cv_folds": 5,               # CV folds for tuning
}
```

### Feature Processing

**Feature Selection**: When `feature_selection` is set to `"backward"`, the pipeline uses backward elimination with permutation importance. Features are selected once (using training data) and applied consistently across all random seeds.

**Normalization**: When `normalize_features` is True, features are z-score normalized using training set statistics (mean and standard deviation). The same statistics are applied to test data to prevent data leakage.

**PCA**: When `use_pca` is True, Principal Component Analysis is applied after feature selection to reduce dimensionality while retaining `pca_variance` (default 90%) of variance.

**Pose Derivatives**: When `use_pose_derivatives` is True, velocity (first derivative) and acceleration (second derivative) features from pose time series are included.

**Time Features**: When `use_time_features` is True, normalized temporal position within each condition is included as features.

**Condition Order**: When `include_order` is True, a binary feature indicating LMH vs LHM condition order is included.

### Role of Random Seeds

The `n_seeds` parameter (default: 20) controls:

**For Random/Participant Split**:
- Different train/test splits
- Different RandomForest initialization (bootstrap sampling, feature subsets)

**For LOPO**:
- Different RandomForest initialization for each participant fold
- Results aggregated across seeds to provide confidence intervals

**For Participant-Specific (stratified)**:
- Different random training/test window selections
- Different RandomForest initialization

**For Participant-Specific (temporal_stratified)**:
- Only RandomForest initialization (data split is deterministic)

Multiple seeds provide confidence intervals and robustness against initialization effects. Results are reported as mean and standard deviation across seeds.

## Output Format

### Directory Structure

Results are organized by method and configuration:

```
model_output/
├── random_split/
│   ├── procrustes_global_all_backward_zscore/     # With derivatives, normalized
│   │   ├── settings.json                           # Configuration used
│   │   ├── experiment_log.csv                      # Summary of all experiments
│   │   ├── pose.json                               # Individual experiment results
│   │   ├── pose_perf.json
│   │   └── ...
│   └── procrustes_global_all_backward_noderiv_zscore/  # Without derivatives, normalized
└── lopo/
    └── procrustes_global_all_backward_zscore/
```

Directory names encode the configuration:
- Pose variant (e.g., `procrustes_global`)
- Class configuration (e.g., `all`, `LvH`)
- Feature selection method (e.g., `backward`, `none`)
- Optional modifiers (e.g., `noderiv`, `zscore`, `nonorm`, `pca`, `hyp`)

### JSON Output Files

Each experiment produces a JSON file with the following structure:

```json
{
  "name": "pose_perf",
  "config": {
    "split_strategy": "random",
    "n_seeds": 20,
    "feature_selection": "backward",
    "use_pose_derivatives": true,
    "normalize_features": true,
    "use_time_features": false,
    "class_config": "all",
    "use_pca": false,
    "tune_hyperparameters": false
  },
  "metrics": {
    "test_bal_acc_mean": 0.75,
    "test_bal_acc_std": 0.03,
    "test_f1_mean": 0.74,
    "test_f1_std": 0.03,
    "test_kappa_mean": 0.62,
    "test_kappa_std": 0.04,
    "test_precision_L_mean": 0.80,
    "test_precision_L_std": 0.02,
    "test_recall_L_mean": 0.78,
    "test_recall_L_std": 0.03,
    "test_f1_L_mean": 0.79,
    "test_f1_L_std": 0.02
  },
  "confusion_matrix": [[85.2, 10.5, 4.3], [12.1, 79.8, 8.1], [5.7, 15.2, 79.1]],
  "selected_features": ["feature1", "feature2", ...],
  "n_features": 97,
  "n_seeds": 20,
  "timestamp": "2025-10-18T10:30:00"
}
```

### LOPO Output

LOPO experiments include per-participant results with multiple random seeds per fold:

```json
{
  "participant_results": [
    {
      "participant": "3105",
      "n_seeds": 20,
      "test_bal_acc_mean": 0.72,
      "test_bal_acc_std": 0.03,
      "test_f1_mean": 0.70,
      "test_f1_std": 0.02,
      "confusion_matrix": [[...], [...], [...]]
    },
    ...
  ],
  "n_participants": 49,
  "n_seeds": 20,
  "metrics": {
    "test_bal_acc_mean": 0.70,
    "test_bal_acc_std": 0.08
  }
}
```

### Participant-Specific Output

Produces two files per experiment:

1. **JSON file** with complete results across all training sizes
2. **CSV file** (`*_learning_curve.csv`) for plotting

JSON structure:
```json
{
  "name": "pose_perf",
  "config": {
    "split_strategy": "participant_specific",
    "sampling_strategy": "temporal_stratified",
    "n_seeds": 20
  },
  "learning_curve": [
    {
      "aggregated": {
        "train_size": 1,
        "n_participants": 49,
        "n_seeds": 20,
        "test_bal_acc_mean": 0.58,
        "test_bal_acc_std": 0.12,
        "confusion_matrix": [[...], [...], [...]]
      },
      "per_participant": [...]
    },
    ...
  ],
  "training_sizes": [1, 2, 3, 5, 7, 9, 11]
}
```

### Configuration Combination Logs

When using `run_config_combinations.py`, logs are saved to `run_logs/`:

```json
{
  "timestamp": "2025-10-18T10:30:00",
  "method": "lopo",
  "pose_variant": "procrustes_global",
  "total_combinations": 4,
  "successful": 4,
  "failed": 0,
  "combinations": [
    {
      "config": {
        "use_pose_derivatives": true,
        "normalize_features": true,
        "n_seeds": 20
      },
      "success": true,
      "description": "normalize_features=True, use_pose_derivatives=True"
    },
    ...
  ]
}
```

## Interpreting Results

### Key Metrics

- **Balanced Accuracy**: Primary metric (averages recall across classes, handles class imbalance)
- **F1 Score**: Harmonic mean of precision and recall (weighted average for multi-class)
- **Cohen's Kappa**: Agreement adjusted for chance
- **Per-class Metrics**: Precision, recall, F1 for each workload level (L, M, H)
- **Confusion Matrix**: Row-normalized percentages showing prediction distribution

### Comparing Split Strategies

Expected performance hierarchy: Random >= Participant >= LOPO

- **Random Split**: Upper bound (within-task generalization, same participants in train/test)
- **Participant Split**: Cross-participant generalization (~20% participants held out)
- **LOPO**: Lower bound (strictest test, each participant completely held out)

The gap between random and LOPO indicates how much performance depends on participant-specific patterns versus generalizable workload signatures.

### Comparing Pose Variants

- **Original**: Raw normalized coordinates (interocular distance normalization)
- **Procrustes Participant**: Aligned to participant-specific mean template
- **Procrustes Global**: Aligned to grand mean template across all participants

Procrustes normalization typically improves generalization by removing irrelevant geometric variation while preserving workload-relevant movements.

### Participant-Specific Learning Curves

Learning curves show balanced accuracy versus training size (windows per condition). Key insights:

- **Plateau Point**: Training size where accuracy stops improving
- **Inter-participant Variability**: Standard deviation across participants
- **Minimum Calibration**: Smallest training size achieving acceptable performance

These inform deployment decisions about individual calibration requirements.

## Expected Runtime

| Strategy | Experiments | Time (n_seeds=20) |
|----------|-------------|-------------------|
| Random Split | 31 | 3-6 hours |
| Participant Split | 31 | 3-6 hours |
| LOPO | 31 | 15-25 hours |
| Participant-Specific | 31 × 7 sizes | 12-20 hours |

Runtime varies with:
- Number of features (affects feature selection time)
- Hyperparameter tuning (adds 2-3x time if enabled)
- Number of seeds (linear scaling)
- Hardware (CPU cores for n_jobs=-1)

## Recommended Workflow

**Step 1**: Run random split to compare configurations (fastest initial test)
```bash
python run_rf_random_split.py --pose-variant procrustes_global
```

**Step 2**: Test configuration variations using automated combinations
```bash
python run_config_combinations.py \
    --method random \
    --vary use_pose_derivatives normalize_features \
    --n-seeds 10
```

**Step 3**: Run participant split for cross-participant validation
```bash
python run_rf_participant_split.py --pose-variant procrustes_global
```

**Step 4**: Run LOPO for strictest generalization assessment
```bash
python run_rf_lopo.py --pose-variant procrustes_global
```

**Step 5**: Run participant-specific to assess calibration requirements
```bash
python run_rf_participant_specific.py \
    --pose-variant procrustes_global \
    --strategy temporal_stratified
```

## Verifying Configuration

Configuration changes in `utils/config.py` require Python cache to be cleared to take effect:

```bash
# Clear Python bytecode cache
rm -rf utils/__pycache__ __pycache__

# Verify current configuration
python check_config.py

# Diagnose configuration issues
python diagnose_config_issue.py
```

When running models with modified configurations, always use the `--overwrite` flag to ensure existing results are regenerated with the new settings:

```bash
python run_rf_random_split.py --pose-variant procrustes_global --overwrite
```

## Dependencies

Required Python packages:
```
numpy>=1.20
pandas>=1.3
scikit-learn>=1.0
scipy>=1.7
tqdm>=4.60
```

Install via:
```bash
pip install numpy pandas scikit-learn scipy tqdm
```

All modeling uses scikit-learn's `RandomForestClassifier` and standard evaluation metrics. No deep learning frameworks required.

## Citation

If using this modeling pipeline, please cite:

> Richardson, M. et al. (2025). Multimodal Classification of Cognitive Workload in Complex Tasks.
> [Publication details pending]
