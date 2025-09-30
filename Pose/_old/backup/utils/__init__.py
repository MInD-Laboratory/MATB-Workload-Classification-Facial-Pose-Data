"""
Facial Pose Analysis Utilities

A comprehensive toolkit for processing and analyzing facial pose data from OpenPose.

This package provides modular utilities for:
- Quality control and bad window detection
- Coordinate normalization (Procrustes and original methods)
- Feature extraction from facial landmarks
- Temporal filtering and noise reduction
- Statistical analysis and visualization
- Complete processing pipelines

Author: Pose Analysis Pipeline
Date: 2024
"""

# Import main pipeline functions for easy access
from .pipeline import run_complete_pose_pipeline, process_single_file

# Import key analysis functions
from .quality_control import run_quality_control_batch, summarize_quality_control
from .feature_extraction import extract_all_features
from .statistical_analysis import calculate_summary_statistics, calculate_correlation_matrix
from .plotting import plot_qc_summary, plot_feature_timeseries

__version__ = "1.0.0"
__author__ = "Pose Analysis Pipeline"

# Define what gets imported with "from utils import *"
__all__ = [
    'run_complete_pose_pipeline',
    'process_single_file',
    'run_quality_control_batch',
    'summarize_quality_control',
    'extract_all_features',
    'calculate_summary_statistics',
    'calculate_correlation_matrix',
    'plot_qc_summary',
    'plot_feature_timeseries'
]