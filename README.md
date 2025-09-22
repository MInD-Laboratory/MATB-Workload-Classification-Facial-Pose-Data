# MATB Workload Classification: Multi-Modal Analysis

**Research Project:** Code, Data Analysis and Modelling comparing Workload Classification and Prediction from Facial Pose, Physiological, and Eye-Tracking Measures.

## Project Overview

This repository contains a comprehensive analysis pipeline for workload classification using multiple physiological and behavioral modalities:

- **Facial Pose Analysis** (✅ Complete) - Extract behavioral features from OpenPose facial landmarks
- **ECG Analysis** (🚧 Planned) - Heart rate variability and cardiac measures
- **GSR Analysis** (🚧 Planned) - Galvanic skin response and stress indicators
- **Eye-Tracking Analysis** (🚧 Planned) - Gaze patterns and attention measures
- **Multi-Modal Modeling** (🚧 Planned) - Machine learning classification and prediction

## Getting Started

### Prerequisites

- **Anaconda** or **Miniconda** installed on your system
  - Download from: https://www.anaconda.com/products/distribution
  - Or Miniconda: https://docs.conda.io/en/latest/miniconda.html
- **Python 3.8+** (will be installed automatically with conda)
- **Git** (to clone this repository)

### Quick Setup (For Beginners)

Follow these step-by-step instructions to set up your analysis environment:

#### Step 1: Clone the Repository

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

```bash
# Clone the repository
git clone https://github.com/yourusername/MATB-Workload-Classification-Facial-Pose-Data.git

# Navigate to the project directory
cd MATB-Workload-Classification-Facial-Pose-Data
```

#### Step 2: Create a Conda Environment

Create a new conda environment specifically for this project:

```bash
# Create a new environment named 'matb-analysis' with Python 3.9
conda create --name matb-analysis python=3.9

# Activate the environment
conda activate matb-analysis
```

**Note:** You'll need to activate this environment every time you work on the project.

#### Step 3: Install Required Packages

Install all necessary Python packages using the requirements file:

```bash
# Install packages from requirements.txt
pip install -r requirements.txt
```

This will install all packages needed for the facial pose analysis pipeline.

#### Step 4: Verify Installation

Test that everything is working correctly:

```bash
# Start Python and test imports
python -c "import numpy, pandas, matplotlib, seaborn; print('✅ All packages installed successfully!')"
```

#### Step 5: Start Jupyter Notebook

Launch Jupyter to begin analysis:

```bash
# Start Jupyter Notebook
jupyter notebook
```

This will open your web browser with the Jupyter interface. Navigate to the `Pose/` folder and open `pose_analysis.ipynb` to begin.

### Alternative: One-Command Setup

For experienced users, you can set up everything in one go:

```bash
# Create environment and install packages
conda create --name matb-analysis python=3.9 -y && \
conda activate matb-analysis && \
pip install -r requirements.txt
```

## Project Structure

```
MATB-Workload-Classification-Facial-Pose-Data/
├── README.md                    # This file - setup and overview
├── requirements.txt             # Python package dependencies
├──
├── Pose/                        # 🟢 Facial pose analysis (COMPLETE)
│   ├── README.md               # Detailed pose analysis documentation
│   ├── pose_analysis.ipynb     # Main analysis notebook
│   ├── utils/                  # Modular analysis utilities
│   ├── feature_data/           # Processed features (generated)
│   ├── output/                 # Analysis results (generated)
│   └── _old/                   # Legacy code for reference
│
├── ECG/                        # 🔵 Heart rate analysis (PLANNED)
│   └── (coming soon)
│
├── GSR/                        # 🔵 Galvanic skin response (PLANNED)
│   └── (coming soon)
│
├── EyeTracking/                # 🔵 Gaze analysis (PLANNED)
│   └── (coming soon)
│
└── Modeling/                   # 🔵 ML classification (PLANNED)
    └── (coming soon)
```

## Current Status: Facial Pose Analysis

The facial pose analysis pipeline is **complete and ready to use**. It provides:

### Key Features
- **Automated Quality Control** - Identifies unreliable tracking periods
- **Flexible Feature Extraction** - Per-feature coordinate normalization control
- **Nose-Relative Gaze Tracking** - Separates eye movement from head movement
- **Research-Grade Processing** - Publication-ready analysis pipeline
- **Comprehensive Documentation** - Detailed technical documentation

### Quick Start for Pose Analysis

1. **Prepare your data**: Place OpenPose CSV files in `Pose/data/raw_pose/`
2. **Run analysis**: Open and execute `Pose/pose_analysis.ipynb`
3. **Review results**: Check `Pose/feature_data/` for processed features

### Example Usage

```python
from Pose.utils.pipeline import run_complete_pose_pipeline

# Run complete facial pose analysis
output_paths = run_complete_pose_pipeline(
    raw_input_dir="Pose/data/raw_pose",
    output_base_dir="Pose/data/processed",
    coordinate_system="procrustes",
    apply_temporal_filter=True
)
```

## Troubleshooting

### Common Issues

**Issue**: `conda: command not found`
- **Solution**: Install Anaconda/Miniconda and restart your terminal

**Issue**: `pip install` fails
- **Solution**: Make sure your conda environment is activated: `conda activate matb-analysis`

**Issue**: Import errors in Python
- **Solution**: Verify you're in the correct environment: `conda info --envs`

**Issue**: Jupyter notebook won't start
- **Solution**: Install jupyter in your environment: `pip install jupyter`

### Environment Management

```bash
# List all conda environments
conda env list

# Activate the project environment
conda activate matb-analysis

# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove --name matb-analysis
```

## Documentation

- **Pose Analysis**: See `Pose/README.md` for detailed usage instructions
- **Technical Details**: See `docs/POSE.md` for comprehensive technical documentation
- **API Reference**: All functions are thoroughly documented with docstrings

## Data Requirements

### Facial Pose Data
- **Format**: OpenPose CSV files with facial landmarks
- **Structure**: `x0,y0,prob0,x1,y1,prob1,...,x69,y69,prob69`
- **Sampling Rate**: 30-60 fps recommended
- **Landmarks**: 68 facial + 2 pupil landmarks (70 total)

## Contributing

This is a research project. For modifications:
1. Maintain modular structure
2. Add comprehensive documentation
3. Include parameter validation
4. Test with sample data

## Future Development

**Planned Modules:**
- **ECG Analysis**: Heart rate variability, R-peak detection, frequency domain analysis
- **GSR Analysis**: Skin conductance features, stress response quantification
- **Eye-Tracking**: Saccade detection, fixation analysis, attention metrics
- **Multi-Modal Modeling**: Machine learning classification, feature fusion, prediction models

## Support

For questions about:
- **Setup Issues**: Check troubleshooting section above
- **Pose Analysis**: See `Pose/README.md` and `docs/POSE.md`
- **Technical Details**: Review function docstrings and technical documentation

## License

[Specify your license here]

---

**Next Steps:**
1. Complete the setup steps above
2. Navigate to `Pose/` folder for facial pose analysis
3. Review `Pose/README.md` for detailed usage instructions
4. Execute `pose_analysis.ipynb` to begin analysis
