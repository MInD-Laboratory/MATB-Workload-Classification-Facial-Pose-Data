# Statistical Analysis Utilities

Shared utilities for running statistical analyses and creating publication-ready figures across all data modalities.

## What This Does (Simple Explanation)

This module provides functions used by all the other analysis pipelines to:
- Run statistical tests comparing Low, Medium, and High workload conditions
- Calculate which differences are statistically significant
- Create standardized bar plots showing results
- Handle repeated measures from the same participants correctly

Instead of duplicating statistical code in each pipeline, they all use these shared functions for consistency.

## Quick Start

```python
# Import the functions
from stats_utils.stats_figures import run_rpy2_lmer, barplot_ax
import matplotlib.pyplot as plt

# Run statistical analysis
pairwise_p, means, cis = run_rpy2_lmer(
    df=your_data,  # DataFrame with columns: participant, condition, window_index, your_metric
    dv='your_metric',  # Name of the column to analyze
    feature_label='Your Metric Name'
)

# Create a figure
fig, ax = plt.subplots(figsize=(6, 5))

# Calculate standard errors from confidence intervals
sems = {cond: (cis[cond][1] - cis[cond][0]) / (2 * 1.96) for cond in ['L', 'M', 'H']}

# Plot the results
barplot_ax(ax, means, sems, pairwise_p, ylabel='Your Metric', metric_name='Your Metric')
plt.tight_layout()
plt.savefig('your_metric.png', dpi=300)
```

## Overview

This module provides reusable functions for statistical analysis and visualization used throughout the project. It includes tools for mixed-effects modeling, pairwise comparisons, and standardized plotting functions for workload condition comparisons.

## Contents

```
stats_utils/
├── __init__.py                         # Module initialization
├── stats_figures.py                    # Statistical analysis and plotting functions
├── metric_comparison_figures.ipynb     # Interactive notebook for metric comparisons
└── README.md                           # This file
```

## Core Functions

### Statistical Analysis

#### `print_means(df, dv, group='condition')`
Print and return means of a dependent variable grouped by condition.

**Parameters:**
- `df`: DataFrame containing the data
- `dv`: Dependent variable column name
- `group`: Grouping variable (default: 'condition')

**Returns:**
- Series of means by group

**Example:**
```python
from stats_utils.stats_figures import print_means

means = print_means(df, 'blink_aperture_mean', group='condition')
```

#### `run_rpy2_lmer(df, dv, feature_label)`
Fit a linear mixed-effects model using R's lmerTest package via rpy2.

**Parameters:**
- `df`: DataFrame with columns: participant, condition, session_order_numeric, window_index, and the dependent variable
- `dv`: Dependent variable column name
- `feature_label`: Label for printing/output

**Returns:**
- `pairwise_p`: Dictionary of p-values for pairwise comparisons (L-M, L-H, M-H)
- `means`: Dictionary of estimated marginal means for each condition (L, M, H)
- `cis`: Dictionary of 95% confidence intervals for each condition

**Model Formula:**
```r
dv ~ condition + session_order_numeric + window_index + (1|participant_id)
```

**Example:**
```python
from stats_utils.stats_figures import run_rpy2_lmer

pairwise_p, means, cis = run_rpy2_lmer(
    df=data,
    dv='blink_aperture_mean',
    feature_label='Blink Aperture'
)
```

### Plotting Functions

#### `barplot_ax(ax, means, sems, pvals, ylabel, metric_name, ...)`
Create a bar plot with error bars and significance annotations on a given matplotlib axis.

**Parameters:**
- `ax`: Matplotlib axis object
- `means`: Dictionary of means for each condition (L, M, H)
- `sems`: Dictionary of standard errors for each condition
- `pvals`: Dictionary of pairwise comparison p-values
- `ylabel`: Y-axis label
- `metric_name`: Name of the metric for annotations
- `colors`: Optional custom colors for bars (default: gray scale)
- `bar_width`: Width of bars (default: 0.80)
- `ylim_padding`: Tuple of (bottom, top) padding fractions for y-limits

**Features:**
- Automatically adds significance stars (*, **, ***)
- Draws significance brackets between conditions
- Handles multiple comparison brackets with proper spacing
- Customizable colors and styling

**Significance Levels:**
- `***`: p < 0.001
- `**`: p < 0.01
- `*`: p < 0.05
- `ns`: p ≥ 0.05

**Example:**
```python
import matplotlib.pyplot as plt
from stats_utils.stats_figures import barplot_ax

fig, ax = plt.subplots(figsize=(6, 5))
barplot_ax(
    ax=ax,
    means={'L': 0.25, 'M': 0.22, 'H': 0.18},
    sems={'L': 0.01, 'M': 0.01, 'H': 0.01},
    pvals={('L', 'M'): 0.03, ('L', 'H'): 0.001, ('M', 'H'): 0.02},
    ylabel='Blink Aperture (normalized)',
    metric_name='Blink Aperture'
)
plt.tight_layout()
plt.savefig('blink_aperture.png', dpi=300)
```

## Usage Workflow

### 1. Prepare Data

Ensure your DataFrame contains the required columns:
- `participant` or `participant_id`: Participant identifier
- `condition`: Workload condition (L, M, H)
- `session_order_numeric`: Numeric session order variable
- `window_index`: Window/trial index
- Dependent variable(s) of interest

### 2. Run Statistical Analysis

```python
from stats_utils.stats_figures import run_rpy2_lmer, print_means

# Print descriptive statistics
means = print_means(df, 'your_metric')

# Run mixed-effects model and get pairwise comparisons
pairwise_p, emm_means, cis = run_rpy2_lmer(
    df=df,
    dv='your_metric',
    feature_label='Your Metric Name'
)
```

### 3. Create Publication Figure

```python
import matplotlib.pyplot as plt
from stats_utils.stats_figures import barplot_ax

# Calculate standard errors from confidence intervals
sems = {cond: (cis[cond][1] - cis[cond][0]) / (2 * 1.96)
        for cond in ['L', 'M', 'H']}

# Create figure
fig, ax = plt.subplots(figsize=(6, 5))
barplot_ax(
    ax=ax,
    means=emm_means,
    sems=sems,
    pvals=pairwise_p,
    ylabel='Your Metric',
    metric_name='Your Metric'
)
plt.tight_layout()
plt.savefig('your_metric_comparison.png', dpi=300)
```

## Dependencies

**Required:**
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.20.0` - Numerical operations
- `matplotlib>=3.4.0` - Plotting
- `rpy2>=3.4.5` - R integration for mixed-effects models

**R packages** (installed via R):
- `lmerTest` - Linear mixed-effects models with p-values
- `emmeans` - Estimated marginal means and contrasts

Install R packages:
```r
install.packages(c("lmerTest", "emmeans"))
```

## Interactive Notebook

The `metric_comparison_figures.ipynb` notebook provides examples of:
- Loading and preparing data from multiple modalities
- Running statistical analyses across different metrics
- Creating publication-ready comparison figures
- Combining results from pose, ECG, GSR, and eye-tracking data

## Integration with Analysis Pipelines

This module is imported by analysis notebooks in each data modality:
- `Pose/pose_stats_figures.ipynb` - Facial pose metric comparisons
- `ecg/ecg_analysis.ipynb` - Heart rate variability comparisons
- `gsr/gsr_analysis.ipynb` - Electrodermal activity comparisons
- `eye_tracking/eye_gaze_analysis.ipynb` - Eye tracking metric comparisons
- `MATB_performance/performance_metrics_analysis.ipynb` - Task performance comparisons

## Statistical Model Details

The mixed-effects model accounts for:
- **Fixed effects:**
  - `condition`: Workload condition (L, M, H) - primary variable of interest
  - `session_order_numeric`: Session order to control for learning/fatigue
  - `window_index`: Temporal position within session

- **Random effects:**
  - `(1|participant_id)`: Random intercept for each participant

This model structure:
- Controls for individual differences via random intercepts
- Accounts for temporal effects within sessions
- Controls for session order effects across participants
- Provides robust estimates of condition effects on dependent variables

## Notes

- All functions assume data has been preprocessed and windowed appropriately
- Mixed-effects models require sufficient data (typically >20 participants)
- P-values from pairwise comparisons are NOT adjusted for multiple comparisons by default
- For publication, consider applying Bonferroni or FDR correction if testing many metrics
- Plotting functions use grayscale by default for publication compatibility
- Custom colors can be specified for presentations or color figures

## Troubleshooting

### "R package not found" errors
```bash
# Install R packages
R -e "install.packages(c('lmerTest', 'emmeans'), repos='https://cloud.r-project.org')"
```

### "rpy2 not available" errors
```bash
pip install rpy2
```

### Convergence warnings from lmer
- May indicate insufficient data or overly complex model
- Try simplifying random effects structure
- Ensure sufficient observations per participant/condition

### Column not found errors
- Verify DataFrame has required columns: participant (or participant_id), condition, session_order_numeric, window_index
- Check dependent variable column name matches `dv` parameter
