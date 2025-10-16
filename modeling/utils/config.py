#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration module for Random Forest modeling pipeline.

Provides centralized configuration for:
- Feature file paths (all modalities)
- Model hyperparameters
- Experiment settings
"""

from pathlib import Path
import pandas as pd

# ============================================================================
# POSE FEATURE VARIANTS
# ============================================================================

POSE_VARIANTS = {
    "original": {
        "name": "pose_original",
        "path": "../Pose/data/processed/linear_metrics/original_linear.csv",
        "description": "No Procrustes normalization"
    },
    "procrustes_participant": {
        "name": "pose_procrustes_participant",
        "path": "../Pose/data/processed/linear_metrics/procrustes_participant_linear.csv",
        "description": "Procrustes with participant-specific normalization"
    },
    "procrustes_global": {
        "name": "pose_procrustes_global",
        "path": "../Pose/data/processed/linear_metrics/procrustes_global_linear.csv",
        "description": "Procrustes with global normalization"
    }
}

# ============================================================================
# OTHER MODALITY FEATURE PATHS (fixed)
# ============================================================================

OTHER_MODALITIES = {
    "performance": {
        "path": "../MATB_performance/data/processed/combined/performance_metrics_all.csv",
        "description": "MATB task performance metrics"
    },
    "eye_tracking": {
        "path": "../eye_tracking/data/processed/combined/eyegaze_metrics_all.csv",
        "description": "Eye gaze and pupil metrics"
    },
    "gsr": {
        "path": "../gsr/data/processed/combined/gsr_features_all.csv",
        "description": "Galvanic skin response features"
    },
    "ecg": {
        "path": "../ecg/data/processed/combined/ecg_features_all.csv",
        "description": "Heart rate variability features"
    }
}

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUTPUT_BASE = Path("model_output")

# ============================================================================
# RANDOM FOREST HYPERPARAMETERS
# ============================================================================

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "class_weight": "balanced",
    "n_jobs": -1,
}

# ============================================================================
# DEFAULT MODEL CONFIGURATION
# ============================================================================

DEFAULT_MODEL_CONFIG = {
    "n_seeds": 2,                     # Number of random seeds for reliability (random/participant splits and participant-specific sampling)
    "feature_selection": "none",  # Options: "backward", "forward", None
    "use_pca": False,                 # Apply PCA after feature selection
    "pca_variance": 0.95,             # Variance to retain if using PCA
    "write_cm": True,                 # Save confusion matrices
    "tune_hyperparameters": False,    # Tune RF hyperparameters
    "tune_n_iter": 30,                # Number of tuning iterations
    "tune_cv_folds": 5,               # CV folds for tuning
    "use_pose_derivatives": False,    # Include velocity/acceleration features from pose
    "use_time_features": False,       # Include temporal position features (normalized time)
    "class_config": "all",            # Class configuration: "all", "L_vs_H", "L_vs_M", "M_vs_H", "LM_vs_H", "L_vs_MH"
    "include_order": True,            # Include condition order (LMH vs LHM) as feature
}

# ============================================================================
# CLASS CONFIGURATIONS
# ============================================================================

# Define all possible class comparison configurations
CLASS_CONFIGS = {
    "all": {
        "description": "3-class: L vs M vs H",
        "labels": ["L", "M", "H"],
        "filter_fn": None,  # Keep all data
        "suffix": ""  # No suffix for default 3-class
    },
    "L_vs_H": {
        "description": "Binary: Low vs High (exclude Medium)",
        "labels": ["L", "H"],
        "filter_fn": lambda df: df[df["condition"].isin(["L", "H"])].copy(),
        "suffix": "_LvH"
    },
    "L_vs_M": {
        "description": "Binary: Low vs Medium (exclude High)",
        "labels": ["L", "M"],
        "filter_fn": lambda df: df[df["condition"].isin(["L", "M"])].copy(),
        "suffix": "_LvM"
    },
    "M_vs_H": {
        "description": "Binary: Medium vs High (exclude Low)",
        "labels": ["M", "H"],
        "filter_fn": lambda df: df[df["condition"].isin(["M", "H"])].copy(),
        "suffix": "_MvH"
    },
    "LM_vs_H": {
        "description": "Binary: Low+Medium vs High (merge L,M)",
        "labels": ["LM", "H"],
        "filter_fn": lambda df: _merge_classes(df, {"LM": ["L", "M"], "H": ["H"]}),
        "suffix": "_LMvH"
    },
    "L_vs_MH": {
        "description": "Binary: Low vs Medium+High (merge M,H)",
        "labels": ["L", "MH"],
        "filter_fn": lambda df: _merge_classes(df, {"L": ["L"], "MH": ["M", "H"]}),
        "suffix": "_LvMH"
    },
}


def _merge_classes(df, merge_map):
    """
    Merge multiple classes into single categories.

    Args:
        df (pd.DataFrame): Input dataframe with 'condition' column
        merge_map (dict): Mapping new_label -> [old_labels]

    Returns:
        pd.DataFrame: Dataframe with merged condition labels
    """
    df_out = df.copy()

    for new_label, old_labels in merge_map.items():
        mask = df_out["condition"].isin(old_labels)
        df_out.loc[mask, "condition"] = new_label

    return df_out


# ============================================================================
# TARGET LABELS (Default)
# ============================================================================

LABELS = ["L", "M", "H"]  # Workload conditions: Low, Medium, High (default)

# ============================================================================
# NON-FEATURE COLUMNS (to exclude from modeling)
# ============================================================================

ID_COLS = {
    "condition", "participant", "window_index",
    "window_start", "window_end", "minute",
    "window_start_s", "window_end_s",
    "source", "t_start_frame", "t_end_frame",
    "start_time", "end_time",  # From performance data
}


def get_feature_groups(pose_variant="original"):
    """
    Generate feature group mappings for selected pose variant.

    Args:
        pose_variant: One of 'original', 'procrustes_participant', 'procrustes_global'

    Returns:
        dict: Mapping of feature group names to (filepath, phase) tuples
    """
    if pose_variant not in POSE_VARIANTS:
        raise ValueError(
            f"Unknown pose variant: '{pose_variant}'. "
            f"Must be one of: {list(POSE_VARIANTS.keys())}"
        )

    pose_config = POSE_VARIANTS[pose_variant]

    return {
        # Pose (selected variant)
        "pose": (pose_config["path"], "main"),

        # Other modalities (fixed)
        "performance": (OTHER_MODALITIES["performance"]["path"], "main"),
        "eye_tracking": (OTHER_MODALITIES["eye_tracking"]["path"], "main"),
        "gsr": (OTHER_MODALITIES["gsr"]["path"], "main"),
        "ecg": (OTHER_MODALITIES["ecg"]["path"], "main"),
    }
