# MATB Workload Classification: Multi-Modal Analysis

**Research Project:** Code, Data Analysis and Modelling comparing Workload Classification and Prediction from Facial Pose, Physiological, and Eye-Tracking Measures.

## What This Project Does

This project analyzes how different measures of human behavior and physiology change during tasks with different levels of mental workload (Low, Medium, High).

We process data from:
- **Facial movements** - How your face moves and expressions change
- **Heart activity** - Heart rate patterns and variability
- **Skin conductance** - Electrical activity on the skin (related to stress/arousal)
- **Eye movements** - Where you look and how your pupils respond
- **Task performance** - How well you complete the tasks

All of this data is then used to train machine learning models that can automatically detect workload levels.

## Project Overview

This repository contains analysis pipelines for workload classification using multiple physiological and behavioral measures collected during Multi-Attribute Task Battery (MATB) experiments:

- **Facial Pose Analysis** - Extract behavioral features from facial landmarks
- **ECG Analysis** - Heart rate variability and cardiac measures
- **GSR Analysis** - Galvanic skin response and electrodermal activity
- **Eye-Tracking Analysis** - Gaze patterns and pupil responses
- **MATB Performance** - Task accuracy and reaction times
- **Multi-Modal Modeling** - Machine learning classification and prediction

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Raw Data Collection                        │
│  (Pose, ECG, GSR, Eye-Tracking, Task Performance during MATB)  │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──> Pose Pipeline ──────────> Facial Features
             ├──> ECG Pipeline ───────────> HRV Features
             ├──> GSR Pipeline ───────────> EDA Features
             ├──> Eye Pipeline ───────────> Gaze/Pupil Features
             └──> Performance Pipeline ──> Task Metrics
                         │
                         ▼
             ┌───────────────────────┐
             │  Feature Extraction   │
             │  (60s windows, 50%    │
             │   overlap)            │
             └───────────┬───────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │  Statistical Analysis │
             │  (Mixed-effects       │
             │   models)             │
             └───────────────────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │  Machine Learning     │
             │  (Random Forest       │
             │   classification)     │
             └───────────────────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │  Workload Prediction  │
             │  (Low/Medium/High)    │
             └───────────────────────┘
```

## Quick Start

### Prerequisites

- **Anaconda** or **Miniconda** installed on your system
  - Download from: https://www.anaconda.com/products/distribution
  - Or Miniconda: https://docs.conda.io/en/latest/miniconda.html
- **Git** (to clone this repository)
- **R** (for advanced statistical analysis - will be installed via conda)

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
# Clone the repository
git clone [REPOSITORY_URL]

# Navigate to the project directory
cd MATB-Workload-Classification-Facial-Pose-Data
```

#### 2. Create and Activate Conda Environment

```bash
# Create a new environment named 'matb-analysis' with Python 3.11
conda create --name matb-analysis python=3.11

# Activate the environment (do this every time you work on the project)
conda activate matb-analysis
```

#### 3. Install Core Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install R and R packages via conda-forge (for statistical analysis)
conda install -c conda-forge r-base rpy2

# Install required R packages for mixed-effects modeling
R -e "install.packages(c('lmerTest', 'emmeans'), repos='https://cloud.r-project.org')"
```

#### 4. Verify Installation

```bash
# Test Python packages
python -c "import numpy, pandas, matplotlib, seaborn, scipy; print('All Python packages installed!')"

# Test R integration
python -c "import rpy2.robjects as ro; print('R integration working!')"
```

## Project Structure

```
MATB-Workload-Classification-Facial-Pose-Data/
├── README.md                           # This file - project overview and setup
├── requirements.txt                    # Python package dependencies
│
├── pose/                               # FACIAL POSE ANALYSIS (COMPLETE)
│   ├── pose_processing_pipeline.py    # Main processing pipeline
│   ├── pose_processing_visualisation.ipynb # Interactive analysis notebook
│   ├── stats_figures.ipynb            # Statistical analysis and plots
│   ├── utils/                          # Modular analysis utilities
│   │   ├── config.py                  # Configuration settings
│   │   ├── io_utils.py                # File I/O operations
│   │   ├── preprocessing_utils.py     # Data preprocessing
│   │   ├── signal_utils.py            # Signal processing
│   │   ├── normalize_utils.py         # Coordinate normalization
│   │   ├── features_utils.py          # Feature extraction
│   │   ├── window_utils.py            # Windowing and segmentation
│   │   └── stats_utils.py             # Statistical analysis
│   ├── data/                          # Data directory (user-provided)
│   │   ├── pose_data/                 # Raw OpenPose CSV files
│   │   └── processed/                 # Processed outputs (generated)
│   └── figs/                          # Generated statistical plots
│
├── ecg/                              # Heart rate analysis
├── gsr/                              # Galvanic skin response
├── eye_tracking/                     # Gaze analysis
├── MATB_performance/                 # Task performance metrics
├── modeling/                         # ML classification
└── stats_utils/                      # Statistical analysis utilities
```

## Facial Pose Analysis

The facial pose analysis pipeline includes:

### Processing Pipeline
- **8-step automated processing** from raw OpenPose data to analysis-ready features
- **3 normalization approaches**: Original, Procrustes Global, Procrustes Participant
- **Smart skip logic** - automatically detects and skips completed steps
- **Robust error handling** and progress tracking

### Key Features
- **Behavioral metrics**: Head rotation, blink patterns, mouth movements, head displacement
- **Motion statistics**: Velocity, acceleration, RMS for all features
- **Statistical analysis**: Mixed-effects models with R integration
- **Publication-ready plots** with significance testing

### Quick Run Example - POSE Analysis
```bash
# Navigate to pose directory
cd Pose

# Run the complete processing pipeline
python process_pose_data.py

# Visualise the analysis pipline using interactive Jupyter notebook
jupyter notebook pose_processing_visualisation.ipynb
```

## Documentation

Each analysis module contains its own detailed README:
- **[Pose Analysis](Pose/README.md)** - Facial landmark processing and feature extraction
- **[ECG Analysis](ecg/README.md)** - Heart rate variability analysis
- **[GSR Analysis](gsr/README.md)** - Galvanic skin response processing
- **[Eye-Tracking Analysis](eye_tracking/README.md)** - Gaze pattern analysis
- **[MATB Performance](MATB_performance/README.md)** - Task performance metrics
- **[Modeling](modeling/README.md)** - Machine learning classification
- **[Stats Utilities](stats_utils/README.md)** - Statistical analysis utilities

### Interactive Notebooks
- **[Pose Processing & Visualization](Pose/pose_processing_visualisation.ipynb)** - Complete pipeline walkthrough with visualizations
- **[Pose Statistical Analysis](Pose/pose_stats_figures.ipynb)** - Publication-ready statistical plots
- **[ECG Analysis](ecg/ecg_analysis.ipynb)** - ECG processing and HRV analysis
- **[GSR Analysis](gsr/gsr_analysis.ipynb)** - GSR signal processing and features
- **[Eye Tracking Analysis](eye_tracking/eye_gaze_analysis.ipynb)** - Eye tracking metrics and visualization
- **[Performance Analysis](MATB_performance/performance_metrics_analysis.ipynb)** - MATB task performance
- **[Modeling Visualization](modeling/modeling_figures.ipynb)** - Model results and comparisons

## Data Structure

Data files are organized within each module's `data/` directory. The project expects:

```
data/
└── participant_info.csv          # Participant metadata and condition mapping

Pose/data/
└── pose_data/
    ├── 3101_01_pose.csv          # Participant 3101, Session 1
    ├── 3101_02_pose.csv          # Participant 3101, Session 2
    ├── 3101_03_pose.csv          # Participant 3101, Session 3
    └── ...                       # Additional participants

eye_tracking/data/
└── eyelink_data/
    ├── 3105_session01.csv        # Participant 3105, Session 1
    └── ...

ecg/data/
└── ecg_data/
    ├── 3105_ecg_session01.csv    # ECG waveform
    ├── 3105_summary_session01.csv # Summary data
    └── ...

gsr/data/
└── gsr_data/
    ├── 3208_session01.csv        # Participant 3208, Session 1
    └── ...
```

The data can be downloaded from the associated OSF repository. 

## Citation

If you use this code in your research, please cite the associated publication.

## License

This project is licensed for academic and research use.