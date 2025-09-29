# MATB Workload Classification: Multi-Modal Analysis

**Research Project:** Code, Data Analysis and Modelling comparing Workload Classification and Prediction from Facial Pose, Physiological, and Eye-Tracking Measures.

## Project Overview

This repository contains a comprehensive analysis pipeline for workload classification using multiple physiological and behavioral modalities collected during Multi-Attribute Task Battery (MATB) experiments:

- **Facial Pose Analysis** (COMPLETE) - Extract behavioral features from OpenPose facial landmarks
- **ECG Analysis** (Planned) - Heart rate variability and cardiac measures
- **GSR Analysis** (Planned) - Galvanic skin response and stress indicators
- **Eye-Tracking Analysis** (Planned) - Gaze patterns and attention measures
- **Multi-Modal Modeling** (Planned) - Machine learning classification and prediction

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
git clone https://github.com/yourusername/MATB-Workload-Classification-Facial-Pose-Data.git

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
│   │   ├── raw_data/                  # Raw OpenPose CSV files
│   │   └── processed/                 # Processed outputs (generated)
│   └── figs/                          # Generated statistical plots
│
├── docs/                              # Comprehensive documentation
│   ├── POSE.md                        # Detailed pose analysis guide
│   ├── ECG.md                         # ECG analysis documentation
│   ├── GSR.md                         # GSR analysis documentation
│   └── EYE_TRACKING.md               # Eye-tracking documentation
│
├── ecg/                              # Heart rate analysis (PLANNED)
├── gsr/                              # Galvanic skin response (PLANNED)
├── eye_tracking/                     # Gaze analysis (PLANNED)
└── modeling/                         # ML classification (PLANNED)
```

## Facial Pose Analysis (Ready to Use!)

The facial pose analysis pipeline is fully implemented and ready for use:

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
cd pose

# Run the complete processing pipeline
python pose_processing_pipeline.py

# Visualise the analysis pipline using interactive Jupyter notebook
jupyter notebook pose_processing_visualisation.ipynb
```

## Documentation

### Core Documentation
- **[Main README](README.md)** - This file (project overview and setup)

### Module-Specific Documentation
- **[Pose Analysis Guide](docs/POSE.md)** - Complete guide to facial pose analysis
  - Data requirements and formats
  - Processing pipeline details
  - Feature extraction methods
  - Statistical analysis approaches
  - Troubleshooting guide

- **[ECG Documentation](docs/ECG.md)** - Heart rate variability analysis (Planned)
- **[GSR Documentation](docs/GSR.md)** - Galvanic skin response analysis (Planned)
- **[Eye-Tracking Documentation](docs/EYE_TRACKING.md)** - Gaze pattern analysis (Planned)

### Interactive Notebooks
- **[Pose Processing & Visualization](pose/pose_processing_visualisation.ipynb)** - Complete pipeline walkthrough with visualizations
- **[Statistical Analysis](pose/stats_figures.ipynb)** - Publication-ready statistical plots

## Data Download

**Data download links will be added here after data hosting is set up.**

Expected data structure:
```
pose/data/raw_data/
├── participant_info.csv          # Participant metadata and conditions
├── 3101_01_pose.csv              # Participant 3101, Trial 1
├── 3101_02_pose.csv              # Participant 3101, Trial 2
├── 3101_03_pose.csv              # Participant 3101, Trial 3
└── ...                           # Additional participants
```

## Citation

If you use this code in your research, please cite the associated publication (details to be added upon publication).

## License

This project is licensed for academic and research use. See license details (to be added).