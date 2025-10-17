#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline utilities for Random Forest modeling.

This module provides all shared functions for:
- Data loading and preprocessing
- Feature selection
- Train/test splitting (random, participant, LOPO)
- Model training and evaluation
- Results management

Adapted from reference code in .forcomparison/Modeling/
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance

# Import configuration
from .config import RF_PARAMS, LABELS, ID_COLS, CLASS_CONFIGS

# Quiet warnings
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

def get_all_model_configs(experiments, feature_groups, default_config):
    """
    Expand experiment definitions into fully-resolved model configs.

    Args:
        experiments (list): List of experiment dicts with 'name' and 'feature_groups'
        feature_groups (dict): Mapping group_name -> (filepath, phase) tuples
        default_config (dict): Global defaults for models

    Returns:
        dict: {model_name: config dict with resolved file list}
    """
    all_models = {}

    for exp in experiments:
        name = exp["name"]

        # Merge default config with experiment overrides
        config = default_config.copy()
        config.update(exp)

        # Resolve feature group names to actual file tuples
        config["files"] = []
        for group_name in exp["feature_groups"]:
            if group_name not in feature_groups:
                raise ValueError(
                    f"Unknown feature group '{group_name}' in experiment '{name}'. "
                    f"Available groups: {list(feature_groups.keys())}"
                )
            config["files"].append(feature_groups[group_name])

        all_models[name] = config

    return all_models


def get_config_dirname(pose_variant, config):
    """
    Generate a unique directory name for a specific configuration.

    Args:
        pose_variant (str): Pose variant name (e.g., "procrustes_global")
        config (dict): Model configuration dictionary

    Returns:
        str: Directory name like "procrustes_global_all_backward" or "original_LvH_none"
    """
    parts = [pose_variant]

    # Class configuration
    class_config = config.get("class_config", "all")
    parts.append(class_config)

    # Feature selection method
    feat_sel = config.get("feature_selection", None)
    parts.append(str(feat_sel) if feat_sel else "none")

    # Optional: pose derivatives
    if not config.get("use_pose_derivatives", True):
        parts.append("noderiv")

    # Optional: time features
    if config.get("use_time_features", False):
        parts.append("time")

    # Optional: normalization - add suffix to distinguish from older runs
    if config.get("normalize_features", True):
        parts.append("zscore")
    else:
        parts.append("nonorm")

    # Optional: condition order
    if config.get("include_order", False):
        parts.append("order")

    # Optional: sampling strategy (for participant-specific models)
    if config.get("sampling_strategy"):
        parts.append(config["sampling_strategy"])

    # Optional: hyperparameter tuning
    if config.get("tune_hyperparameters", False):
        parts.append("hyp")

    # Optional: PCA
    if config.get("use_pca", False):
        parts.append("pca")

    return "_".join(parts)


def get_config_suffix(config):
    """
    Generate a suffix for output filenames based on model configuration.

    Args:
        config (dict): Model configuration dictionary

    Returns:
        str: Suffix string like "_backward_hyp" or "_none_noderiv_LvH"
    """
    if not config:
        return ""

    parts = []

    # Feature selection method
    feat_sel = config.get("feature_selection", None)
    if feat_sel:
        parts.append(str(feat_sel))
    else:
        parts.append("none")

    # Hyperparameter tuning
    if config.get("tune_hyperparameters", False):
        parts.append("hyp")

    # PCA
    if config.get("use_pca", False):
        parts.append("pca")

    # Pose derivatives (only add suffix if disabled)
    if not config.get("use_pose_derivatives", True):
        parts.append("noderiv")

    # Time features (only add suffix if enabled)
    if config.get("use_time_features", False):
        parts.append("time")

    # Normalization - add suffix to distinguish from older runs
    if config.get("normalize_features", True):
        parts.append("zscore")
    else:
        parts.append("nonorm")

    # Condition order (only add suffix if enabled)
    if config.get("include_order", False):
        parts.append("order")

    # Class configuration (add suffix if not default "all")
    class_config = config.get("class_config", "all")
    if class_config in CLASS_CONFIGS and class_config != "all":
        parts.append(CLASS_CONFIGS[class_config]["suffix"].lstrip("_"))

    return "_" + "_".join(parts) if parts else ""


def save_run_settings(output_dir, pose_variant, config, split_strategy):
    """
    Save configuration settings to settings.json in output directory.

    Args:
        output_dir (Path): Output directory for this run
        pose_variant (str): Pose variant used
        config (dict): Model configuration
        split_strategy (str): Split strategy used
    """
    settings = {
        "pose_variant": pose_variant,
        "split_strategy": split_strategy,
        "class_config": config.get("class_config", "all"),
        "feature_selection": config.get("feature_selection", None),
        "use_pose_derivatives": config.get("use_pose_derivatives", True),
        "use_time_features": config.get("use_time_features", False),
        "normalize_features": config.get("normalize_features", True),
        "include_order": config.get("include_order", False),
        "tune_hyperparameters": config.get("tune_hyperparameters", False),
        "use_pca": config.get("use_pca", False),
        "n_seeds": config.get("n_seeds", 10),
        "timestamp": datetime.now().isoformat(),
    }

    settings_path = output_dir / "settings.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)


def find_matching_run(base_output_dir, pose_variant, config, split_strategy):
    """
    Search for existing run with matching configuration.

    Args:
        base_output_dir (Path): Base output directory (e.g., model_output/random_split)
        pose_variant (str): Pose variant
        config (dict): Model configuration
        split_strategy (str): Split strategy

    Returns:
        Path or None: Path to matching run directory, or None if not found
    """
    if not base_output_dir.exists():
        return None

    # Key settings to match
    target_settings = {
        "pose_variant": pose_variant,
        "split_strategy": split_strategy,
        "class_config": config.get("class_config", "all"),
        "feature_selection": config.get("feature_selection", None),
        "use_pose_derivatives": config.get("use_pose_derivatives", True),
        "use_time_features": config.get("use_time_features", False),
        "normalize_features": config.get("normalize_features", True),
        "include_order": config.get("include_order", False),
        "tune_hyperparameters": config.get("tune_hyperparameters", False),
        "use_pca": config.get("use_pca", False),
    }

    # Add sampling_strategy if present (for participant-specific models)
    if config.get("sampling_strategy"):
        target_settings["sampling_strategy"] = config["sampling_strategy"]

    # Search all subdirectories for matching settings.json
    for subdir in base_output_dir.iterdir():
        if not subdir.is_dir():
            continue

        settings_path = subdir / "settings.json"
        if not settings_path.exists():
            continue

        try:
            with open(settings_path, "r") as f:
                existing_settings = json.load(f)

            # Check if all key settings match
            if all(existing_settings.get(k) == v for k, v in target_settings.items()):
                return subdir
        except (json.JSONDecodeError, IOError):
            continue

    return None


def check_model_complete(model_name, output_dir, config=None):
    """
    Check if an experiment's JSON output already exists.

    Args:
        model_name (str): Unique experiment name
        output_dir (str | Path): Directory where JSON results are saved
        config (dict, optional): Model configuration to generate filename suffix

    Returns:
        bool: True if {model_name}{suffix}.json exists
    """
    suffix = get_config_suffix(config) if config else ""
    output_path = Path(output_dir) / f"{model_name}{suffix}.json"
    return output_path.exists()


def prompt_user_action():
    """
    CLI prompt for how to handle existing results.

    Returns:
        str: One of {'overwrite','continue','skip','cancel'}
    """
    print("\nWhat would you like to do?")
    print("  [o] Overwrite all existing results")
    print("  [c] Continue incomplete experiments (resume)")
    print("  [s] Skip existing results, only run new experiments")
    print("  [x] Cancel and exit")

    while True:
        choice = input("\nChoice [o/c/s/x]: ").strip().lower()
        if choice == 'o':
            return 'overwrite'
        elif choice == 'c':
            return 'continue'
        elif choice == 's':
            return 'skip'
        elif choice == 'x':
            return 'cancel'
        else:
            print("Invalid choice. Please enter o, c, s, or x.")


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def apply_class_config(df, class_config="all"):
    """
    Apply class filtering/merging based on configuration.

    Args:
        df (pd.DataFrame): Input dataframe with 'condition' column
        class_config (str): Class configuration name (e.g., "all", "L_vs_H", "LM_vs_H")

    Returns:
        tuple: (filtered_df, labels_for_this_config)

    Raises:
        ValueError: If class_config is not recognized
    """
    if class_config not in CLASS_CONFIGS:
        raise ValueError(
            f"Unknown class_config: '{class_config}'. "
            f"Must be one of: {list(CLASS_CONFIGS.keys())}"
        )

    config = CLASS_CONFIGS[class_config]

    # Apply filtering/merging function if provided
    if config["filter_fn"] is not None:
        df_filtered = config["filter_fn"](df)
    else:
        df_filtered = df.copy()

    return df_filtered, config["labels"]


def normalize_window_columns(df):
    """
    Standardize window boundary column names.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Same data with normalized column names
    """
    rename_map = {}
    if "window_start_s" in df.columns and "window_start" not in df.columns:
        rename_map["window_start_s"] = "window_start"
    if "window_end_s" in df.columns and "window_end" not in df.columns:
        rename_map["window_end_s"] = "window_end"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def load_participant_order_info(participant_info_path="../data/participant_info.csv"):
    """
    Load participant condition order information.

    Args:
        participant_info_path (str): Path to participant_info.csv

    Returns:
        pd.DataFrame: DataFrame with columns ['participant', 'condition_order']
                     where condition_order is 'LMH' or 'LHM'
    """
    info_path = Path(participant_info_path)
    if not info_path.exists():
        print(f"Warning: Participant info file not found at {info_path}")
        print("Condition order features will not be included.")
        return None

    # Read the file
    df = pd.read_csv(info_path)

    # Clean up BOM if present and strip whitespace
    df.columns = [col.lstrip('\ufeff').strip() for col in df.columns]

    # Extract participant ID and determine order
    order_info = []
    for _, row in df.iterrows():
        participant = str(row['Participant ID']).strip()

        # Get session order
        sessions = [
            str(row.get('session01', '')).strip(),
            str(row.get('session02', '')).strip(),
            str(row.get('session03', '')).strip()
        ]

        # Create order string (e.g., 'LMH' or 'LHM')
        condition_order = ''.join(sessions)

        # Skip invalid entries
        if '-' in condition_order or len(condition_order) != 3:
            continue

        order_info.append({
            'participant': participant,
            'condition_order': condition_order
        })

    return pd.DataFrame(order_info)


def add_condition_order_features(df, participant_order_df):
    """
    Add condition order as one-hot encoded features.

    Args:
        df (pd.DataFrame): Input dataframe with 'participant' column
        participant_order_df (pd.DataFrame): Participant order info from load_participant_order_info()

    Returns:
        pd.DataFrame: Dataframe with added order features
    """
    if participant_order_df is None:
        return df

    # Merge order information
    df_out = df.merge(
        participant_order_df,
        on='participant',
        how='left'
    )

    # One-hot encode condition_order
    if 'condition_order' in df_out.columns:
        # Get unique orders
        unique_orders = df_out['condition_order'].dropna().unique()

        for order in unique_orders:
            col_name = f'order_{order}'
            df_out[col_name] = (df_out['condition_order'] == order).astype(int)

        # Drop the original condition_order column
        df_out = df_out.drop(columns=['condition_order'])

        print(f"  Added condition order features: {[f'order_{o}' for o in unique_orders]}")

    return df_out


def load_and_merge_features(file_list, skip_every=None, include_order=False):
    """
    Load multiple feature CSVs and merge them on key identifiers.

    Args:
        file_list (list): List of (filepath, phase) tuples
        skip_every (int | None): Keep only every Nth window when 'window_index' exists
        include_order (bool): If True, add condition order features from participant_info.csv

    Returns:
        pd.DataFrame: Merged wide-format dataset

    Raises:
        FileNotFoundError: If any CSV path is missing
    """
    dfs_to_merge = []

    for filepath, phase in file_list:
        if not Path(filepath).exists():
            raise FileNotFoundError(
                f"Feature file not found: {filepath}\n"
                f"Please generate feature files before running pipeline."
            )

        df = pd.read_csv(filepath)
        df = normalize_window_columns(df)

        # Optional downsampling
        if skip_every is not None and "window_index" in df.columns:
            df = df[df["window_index"] % skip_every == 0].copy()

        # Ensure join keys are string-typed
        if "participant" in df.columns:
            df["participant"] = df["participant"].astype(str)
        if "condition" in df.columns:
            df["condition"] = df["condition"].astype(str)

        # Derive 'minute' for coarser alignment
        if "window_start" in df.columns:
            df["minute"] = (df["window_start"] / 60).round().astype(int)

        dfs_to_merge.append(df)

    # Merge logic
    if len(dfs_to_merge) == 1:
        merged = dfs_to_merge[0]
    else:
        # Compute intersection of columns to find viable merge keys
        common_cols = set(dfs_to_merge[0].columns)
        for df in dfs_to_merge[1:]:
            common_cols = common_cols & set(df.columns)

        # Prefer exact alignment on window_index; fallback to 'minute'
        if "window_index" in common_cols:
            merge_keys = ["participant", "condition", "window_index"]
        elif "minute" in common_cols:
            merge_keys = ["participant", "condition", "minute"]
        else:
            # Derive 'minute' from 'window_start'
            for i, df in enumerate(dfs_to_merge):
                if "window_start" in df.columns and "minute" not in df.columns:
                    dfs_to_merge[i]["minute"] = (df["window_start"] / 60).round().astype(int)
            merge_keys = ["participant", "condition", "minute"]

        # Start merge from the first df
        merged = dfs_to_merge[0]
        timing_cols = [c for c in ["window_start", "window_end", "window_index"]
                       if c in merged.columns]

        for df in dfs_to_merge[1:]:
            # Drop duplicate timing columns from other dfs
            cols_to_drop = [c for c in df.columns
                            if c in timing_cols and c not in merge_keys]
            df_clean = df.drop(columns=cols_to_drop, errors="ignore")

            merged = pd.merge(
                merged,
                df_clean,
                on=merge_keys,
                how="inner",
                suffixes=("", "_dup")
            )

            # Clean any duplicate columns
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            merged = merged.drop(columns=dup_cols, errors="ignore")

    # Ensure both 'window_start' and 'minute' exist
    if "window_start" not in merged.columns and "minute" in merged.columns:
        merged["window_start"] = merged["minute"] * 60
    if "minute" not in merged.columns and "window_start" in merged.columns:
        merged["minute"] = (merged["window_start"] / 60).round().astype(int)

    # Optional: Add condition order features
    if include_order:
        participant_order_df = load_participant_order_info()
        if participant_order_df is not None:
            merged = add_condition_order_features(merged, participant_order_df)

    return merged.reset_index(drop=True)


def add_time_features(df):
    """
    Add normalized temporal position features.

    Creates features that capture position within trial/session:
    - time_norm: Normalized time (0-1) within participant-condition combination
    - time_norm_global: Normalized time (0-1) across entire dataset

    Args:
        df (pd.DataFrame): Input dataframe with window_index or window_start

    Returns:
        pd.DataFrame: Dataframe with added time features
    """
    df_out = df.copy()

    # Normalize time within each participant-condition trial
    if "participant" in df.columns and "condition" in df.columns:
        if "window_index" in df.columns:
            # Use window_index for normalization (preferred)
            df_out["time_norm"] = df.groupby(["participant", "condition"])["window_index"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
            )
        elif "window_start" in df.columns:
            # Fallback to window_start
            df_out["time_norm"] = df.groupby(["participant", "condition"])["window_start"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
            )
        else:
            # No time info available, use global index
            df_out["time_norm"] = np.linspace(0, 1, len(df))

    # Global normalized time (across all data)
    if "window_index" in df.columns:
        df_out["time_norm_global"] = (df["window_index"] - df["window_index"].min()) / \
                                      (df["window_index"].max() - df["window_index"].min()) \
                                      if df["window_index"].max() > df["window_index"].min() else 0.5
    else:
        df_out["time_norm_global"] = np.linspace(0, 1, len(df))

    return df_out


def drop_identifier_columns(df, use_pose_derivatives=True, use_time_features=False):
    """
    Remove non-feature columns prior to modeling.

    Drops both known ID columns and any non-numeric columns.
    Also replaces inf/NaN values with 0.

    Args:
        df (pd.DataFrame): Input dataframe
        use_pose_derivatives (bool): If False, drop pose velocity/acceleration columns
        use_time_features (bool): If True, add normalized temporal position features

    Returns:
        pd.DataFrame: Dataframe with only numeric feature columns, cleaned
    """
    # Optional: add time features before dropping ID columns
    if use_time_features:
        df = add_time_features(df)
        # print(f"  Including temporal features (time_norm, time_norm_global)")

    # First drop known ID columns
    cols_to_drop = [c for c in ID_COLS if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")

    # Also drop any remaining non-numeric columns (object/string dtype)
    # This catches any unexpected metadata columns like filenames
    non_numeric_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        # print(f"  Dropping {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols[:5]}...")
        df_clean = df_clean.drop(columns=non_numeric_cols, errors="ignore")

    # Optional: drop pose velocity and acceleration features
    if not use_pose_derivatives:
        derivative_cols = [c for c in df_clean.columns if '_vel_' in c or '_acc_' in c]
        if derivative_cols:
            print(f"  Excluding {len(derivative_cols)} pose derivative features (vel/acc)")
            df_clean = df_clean.drop(columns=derivative_cols, errors="ignore")

    # Replace inf values with NaN, then fill NaN with 0
    # This handles edge cases from feature calculations (e.g., division by zero)
    inf_count = np.isinf(df_clean.values).sum()
    nan_count = np.isnan(df_clean.values).sum()

    if inf_count > 0 or nan_count > 0:
        # print(f"  Cleaning data: {inf_count} inf values, {nan_count} NaN values -> replacing with 0")
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_clean


# ============================================================================
# TRAIN/TEST SPLITTING
# ============================================================================

def make_train_test_split(df, split_strategy, random_state=0):
    """
    Split dataset into train/test according to strategy.

    Args:
        df (pd.DataFrame): Full merged dataset with 'condition' and 'participant'
        split_strategy (str): 'random' or 'participant'
        random_state (int): Seed for reproducibility

    Returns:
        tuple: (train_df, test_df)

    Raises:
        ValueError: For unsupported split_strategy
    """
    y = df["condition"].values

    if split_strategy == "random":
        # Stratified on label to preserve class ratios
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=0.2,
            stratify=y,
            random_state=random_state
        )

    elif split_strategy == "participant":
        # Leave-participant-out split: hold out ~20% of participants
        participants = df["participant"].unique()
        rng = np.random.default_rng(random_state)
        rng.shuffle(participants)

        n_test = max(1, int(0.2 * len(participants)))
        test_participants = set(participants[:n_test])

        is_test = df["participant"].isin(test_participants)
        train_idx = df.index[~is_test]
        test_idx = df.index[is_test]

    else:
        raise ValueError(
            f"Unknown split_strategy: '{split_strategy}'. "
            f"Must be 'random' or 'participant'."
        )

    return df.loc[train_idx], df.loc[test_idx]


def make_lopo_splits(df):
    """
    Generate leave-one-participant-out splits.

    Args:
        df (pd.DataFrame): Full merged dataset

    Yields:
        tuple: (train_df, test_df, participant_id) for each held-out participant
    """
    participants = sorted(df["participant"].unique())

    for participant in participants:
        is_test = df["participant"] == participant
        train_df = df[~is_test].copy()
        test_df = df[is_test].copy()

        yield train_df, test_df, participant


# ============================================================================
# PARTICIPANT-SPECIFIC LEARNING CURVES
# ============================================================================

def sample_participant_data_stratified(df_participant, n_windows_per_condition, random_state=0):
    """
    Sample N windows from each condition for stratified learning curve.

    Args:
        df_participant (pd.DataFrame): Data for single participant
        n_windows_per_condition (int): Number of windows to sample per condition
        random_state (int): Random seed

    Returns:
        tuple: (train_df, test_df) or (None, None) if insufficient data
    """
    rng = np.random.default_rng(random_state)
    train_indices = []

    conditions = df_participant['condition'].unique()

    # Check if we have enough data in each condition
    for condition in conditions:
        condition_data = df_participant[df_participant['condition'] == condition]
        if len(condition_data) < n_windows_per_condition + 1:  # Need at least 1 for test
            return None, None

    # Sample from each condition
    for condition in conditions:
        condition_indices = df_participant[df_participant['condition'] == condition].index
        sampled_indices = rng.choice(condition_indices, size=n_windows_per_condition, replace=False)
        train_indices.extend(sampled_indices)

    # Test set is everything not in training
    test_indices = df_participant.index.difference(train_indices)

    if len(test_indices) == 0:
        return None, None

    train_df = df_participant.loc[train_indices]
    test_df = df_participant.loc[test_indices]

    return train_df, test_df


def sample_participant_data_temporal(df_participant, n_windows_total, random_state=0):
    """
    Sample first N windows in temporal order for temporal learning curve.

    Args:
        df_participant (pd.DataFrame): Data for single participant
        n_windows_total (int): Total number of windows for training
        random_state (int): Not used, included for API consistency

    Returns:
        tuple: (train_df, test_df) or (None, None) if insufficient data
    """
    # Sort by temporal order
    if 'window_index' in df_participant.columns:
        df_sorted = df_participant.sort_values('window_index')
    elif 'window_start' in df_participant.columns:
        df_sorted = df_participant.sort_values('window_start')
    else:
        # Fallback: use existing order
        df_sorted = df_participant

    # Check if we have enough data
    if len(df_sorted) < n_windows_total + 1:  # Need at least 1 for test
        return None, None

    # Take first N windows for training
    train_df = df_sorted.iloc[:n_windows_total]
    test_df = df_sorted.iloc[n_windows_total:]

    if len(test_df) == 0:
        return None, None

    return train_df, test_df


def sample_participant_data_temporal_stratified(df_participant, n_windows_per_condition, random_state=0):
    """
    Sample first N windows (in temporal order) from EACH condition.

    Excludes immediately adjacent windows from test set to avoid data leakage
    due to 50% overlapping windows.

    For each condition:
    - Training: First N windows (in time)
    - Buffer: Window N+1 (excluded due to overlap with window N)
    - Test: Windows N+2 onwards

    Args:
        df_participant (pd.DataFrame): Data for single participant
        n_windows_per_condition (int): Number of windows to use from each condition
        random_state (int): Not used, included for API consistency

    Returns:
        tuple: (train_df, test_df) or (None, None) if insufficient data
    """
    train_indices = []
    test_indices = []

    conditions = df_participant['condition'].unique()

    # Check if we have enough data in each condition
    # Need N for training + 1 buffer + at least 1 for test = N+2 minimum
    for condition in conditions:
        condition_data = df_participant[df_participant['condition'] == condition]
        if len(condition_data) < n_windows_per_condition + 2:  # Need buffer + test
            return None, None

    # Process each condition separately
    for condition in conditions:
        # Get data for this condition and sort by temporal order
        condition_mask = df_participant['condition'] == condition
        condition_data = df_participant[condition_mask].copy()

        if 'window_index' in condition_data.columns:
            condition_data = condition_data.sort_values('window_index')
        elif 'window_start' in condition_data.columns:
            condition_data = condition_data.sort_values('window_start')

        # Take first N windows for training
        train_windows = condition_data.iloc[:n_windows_per_condition]
        train_indices.extend(train_windows.index.tolist())

        # Exclude window N+1 (buffer to avoid leakage from 50% overlap)
        # Take windows N+2 onwards for test
        if len(condition_data) > n_windows_per_condition + 1:
            test_windows = condition_data.iloc[n_windows_per_condition + 1:]  # Skip buffer window
            test_indices.extend(test_windows.index.tolist())

    if len(test_indices) == 0:
        return None, None

    train_df = df_participant.loc[train_indices]
    test_df = df_participant.loc[test_indices]

    return train_df, test_df


def sample_participant_data_shuffled(df_participant, n_windows_total, random_state=0):
    """
    Random sampling within participant for shuffled learning curve.

    Args:
        df_participant (pd.DataFrame): Data for single participant
        n_windows_total (int): Total number of windows for training
        random_state (int): Random seed

    Returns:
        tuple: (train_df, test_df) or (None, None) if insufficient data
    """
    rng = np.random.default_rng(random_state)

    # Check if we have enough data
    if len(df_participant) < n_windows_total + 1:  # Need at least 1 for test
        return None, None

    # Random sampling
    train_indices = rng.choice(df_participant.index, size=n_windows_total, replace=False)
    test_indices = df_participant.index.difference(train_indices)

    if len(test_indices) == 0:
        return None, None

    train_df = df_participant.loc[train_indices]
    test_df = df_participant.loc[test_indices]

    return train_df, test_df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def prefilter_low_variance_and_corr(X, var_thresh=1e-8, corr_thresh=0.95):
    """
    Prefilter: drop near-constant features and highly correlated pairs.

    Args:
        X (pd.DataFrame): Feature matrix
        var_thresh: Variance threshold
        corr_thresh: Correlation threshold

    Returns:
        pd.DataFrame: Filtered features
    """
    # Remove near-zero variance features
    vt = VarianceThreshold(threshold=var_thresh)
    X_filtered = pd.DataFrame(
        vt.fit_transform(X),
        index=X.index,
        columns=X.columns[vt.get_support()]
    )

    if X_filtered.shape[1] <= 1:
        return X_filtered

    # Remove highly correlated features
    corr = X_filtered.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] >= corr_thresh].tolist()
        to_drop.update(high_corr)

    keep_cols = [c for c in X_filtered.columns if c not in to_drop]
    return X_filtered[keep_cols]


def backward_elimination_permutation(
    X, y,
    cv_folds=5,
    n_repeats=5,
    threshold_percentile=25,
    min_features=5,
    random_state=0
):
    """
    Permutation importance-based backward elimination.

    Args:
        X (pd.DataFrame): Feature matrix
        y: Target labels
        cv_folds: Number of CV folds
        n_repeats: Number of permutation repeats
        threshold_percentile: Remove features below this importance percentile
        min_features: Minimum features to retain
        random_state: Random seed

    Returns:
        tuple: (selected_features, best_score)
    """
    # Prefilter
    X_filtered = prefilter_low_variance_and_corr(X, corr_thresh=0.95)
    features = list(X_filtered.columns)

    if len(features) <= min_features:
        return features, 0.0

    rf = RandomForestClassifier(**RF_PARAMS, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def cv_score(feat_list):
        return cross_val_score(
            rf, X_filtered[feat_list], y, cv=cv,
            scoring="balanced_accuracy", n_jobs=-1
        ).mean()

    best_score = cv_score(features)
    best_features = features.copy()

    iteration = 0
    while len(features) > min_features:
        iteration += 1

        # Compute permutation importance across CV folds
        importances = []
        for train_idx, val_idx in cv.split(X_filtered[features], y):
            X_train = X_filtered[features].iloc[train_idx]
            X_val = X_filtered[features].iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            rf.set_params(random_state=random_state + iteration)
            rf.fit(X_train, y_train)

            perm_imp = permutation_importance(
                rf, X_val, y_val,
                scoring="balanced_accuracy",
                n_repeats=n_repeats,
                random_state=random_state + iteration,
                n_jobs=-1
            )
            importances.append(perm_imp.importances_mean)

        # Average importances across folds
        mean_importance = np.mean(importances, axis=0)
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': mean_importance
        }).sort_values('importance')

        # Remove bottom percentile
        n_to_remove = max(1, int(len(features) * (threshold_percentile / 100)))
        n_to_remove = min(n_to_remove, len(features) - min_features)

        if n_to_remove == 0:
            break

        features_to_remove = importance_df['feature'].iloc[:n_to_remove].tolist()
        candidate_features = [f for f in features if f not in features_to_remove]

        # Evaluate
        candidate_score = cv_score(candidate_features)

        print(f"  Iteration {iteration}: {len(features)} -> {len(candidate_features)} features, "
              f"score: {candidate_score:.4f}")

        # Update best if improved
        if candidate_score >= best_score - 0.005:  # Allow small drops
            if candidate_score > best_score:
                best_score = candidate_score
                best_features = candidate_features.copy()
            features = candidate_features
        else:
            # Performance degraded significantly, stop
            break

    return best_features, best_score


def tune_rf_hyperparameters(X, y, n_iter=50, cv_folds=3, random_state=0):
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV.

    Args:
        X (pd.DataFrame): Feature matrix
        y: Target labels
        n_iter: Number of parameter settings sampled
        cv_folds: Number of CV folds
        random_state: Random seed

    Returns:
        dict: Best hyperparameters found
    """
    from scipy.stats import randint

    # Define hyperparameter search space
    param_distributions = {
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_depth': [None, 10, 20, 30, 50],
        'n_estimators': randint(100, 500),
    }

    # Base RF with fixed parameters
    rf_base = RandomForestClassifier(
        class_weight='balanced',
        n_jobs=-1,
        random_state=random_state
    )

    # Randomized search
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )

    print(f"  Tuning RF hyperparameters ({n_iter} iterations, {cv_folds}-fold CV)...")
    search.fit(X, y)

    print(f"  Best CV score: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    return search.best_params_


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

def fit_and_evaluate(df, split_strategy, seed, config, labels, selected_features=None, tuned_rf_params=None):
    """
    Train and evaluate a single RF pipeline for a given seed and split.

    Args:
        df (pd.DataFrame): Full merged dataset
        split_strategy (str): 'random' or 'participant'
        seed (int): Random seed
        config (dict): Model config controlling scaler/PCA usage
        labels (list): Class labels for this experiment (e.g., ["L", "H"] or ["L", "M", "H"])
        selected_features (list | None): Pre-selected feature names
        tuned_rf_params (dict | None): Tuned RF hyperparameters

    Returns:
        dict: {'metrics': {...}, 'cm': confusion_matrix}
    """
    # Split once per seed
    train_df, test_df = make_train_test_split(df, split_strategy, random_state=seed)

    # Extract targets
    y_train = train_df["condition"].values
    y_test = test_df["condition"].values

    # Extract features (respecting config settings)
    use_pose_deriv = config.get("use_pose_derivatives", True)
    use_time_feat = config.get("use_time_features", False)
    X_train = drop_identifier_columns(train_df, use_pose_derivatives=use_pose_deriv, use_time_features=use_time_feat).drop(columns=["condition"], errors="ignore")
    X_test = drop_identifier_columns(test_df, use_pose_derivatives=use_pose_deriv, use_time_features=use_time_feat).drop(columns=["condition"], errors="ignore")

    # Enforce pre-selected feature set if provided
    if selected_features is not None:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    # Build pipeline components
    pipeline_steps = []

    # Z-score normalization: fit on training data, apply to both train and test
    if config.get("normalize_features", True):
        pipeline_steps.append(("scaler", StandardScaler()))

    if config.get("use_pca", False):
        pipeline_steps.append(("imputer", SimpleImputer(strategy='median')))
        pca_variance = config.get("pca_variance", 0.95)
        pipeline_steps.append(("pca", PCA(n_components=pca_variance)))

    # RF config per-seed
    rf_kwargs = dict(RF_PARAMS)

    # Apply tuned hyperparameters if provided
    if tuned_rf_params:
        rf_kwargs.update(tuned_rf_params)

    rf_kwargs["random_state"] = seed
    rf_kwargs.setdefault("n_jobs", -1)
    pipeline_steps.append(("rf", RandomForestClassifier(**rf_kwargs)))

    pipe = Pipeline(pipeline_steps)

    # Fit and predict
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Core metrics
    metrics = {
        "test_acc": accuracy_score(y_test, y_pred),
        "test_bal_acc": balanced_accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred, labels=labels, average="weighted"),
        "test_kappa": cohen_kappa_score(y_test, y_pred, labels=labels),
    }

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average=None, zero_division=0
    )

    for i, label in enumerate(labels):
        metrics[f"test_precision_{label}"] = precision[i]
        metrics[f"test_recall_{label}"] = recall[i]
        metrics[f"test_f1_{label}"] = f1[i]

    # Confusion matrix (normalized by true label counts, in %)
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true") * 100.0

    return {"metrics": metrics, "cm": cm}


def run_single_model(name, config, output_dir, force=False, resume=False):
    """
    Run a single experiment (random or participant split).

    Args:
        name (str): Experiment name
        config (dict): Experiment config (includes 'files', 'split_strategy', etc.)
        output_dir (str | Path): Output directory for JSON + log CSV
        force (bool): If True, overwrite existing JSON
        resume (bool): If True, skip completed models unless incomplete
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = get_config_suffix(config)
    output_path = output_dir / f"{name}{suffix}.json"

    # Respect existing results unless explicitly forcing/resuming
    if output_path.exists() and not force and not resume:
        print(f"[SKIP] {name}: already complete")
        return

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Features: {' + '.join([Path(f[0]).stem for f in config['files']])}")
    print(f"Split: {config['split_strategy']}")
    print(f"{'='*60}")

    # Load and merge declared feature groups
    merged = load_and_merge_features(
        config["files"],
        skip_every=config.get("skip_every"),
        include_order=config.get("include_order", False)
    )
    print(f"Loaded data: {merged.shape}")

    # -----------------------------
    # APPLY CLASS CONFIGURATION
    # -----------------------------
    class_config = config.get("class_config", "all")
    merged, class_labels = apply_class_config(merged, class_config)

    if class_config != "all":
        class_desc = CLASS_CONFIGS[class_config]["description"]
        print(f"Class configuration: {class_desc}")
        print(f"  Filtered data: {merged.shape}")

    # -----------------------------
    # OPTIONAL HYPERPARAMETER TUNING
    # -----------------------------
    tuned_rf_params = {}
    if config.get("tune_hyperparameters", False):
        print("\nPerforming hyperparameter tuning (once for all seeds)...")
        train_df, _ = make_train_test_split(merged, config["split_strategy"], random_state=0)
        y_train = train_df["condition"].values
        use_pose_deriv = config.get("use_pose_derivatives", True)
        use_time_feat = config.get("use_time_features", False)
        X_train = drop_identifier_columns(train_df, use_pose_derivatives=use_pose_deriv, use_time_features=use_time_feat).drop(columns=["condition"], errors="ignore")

        tuned_rf_params = tune_rf_hyperparameters(
            X_train, y_train,
            n_iter=config.get("tune_n_iter", 50),
            cv_folds=config.get("tune_cv_folds", 3),
            random_state=0
        )

        print(f"\n  Using tuned parameters for all seeds in this experiment.")

    # -----------------------------
    # ONE-TIME FEATURE SELECTION
    # -----------------------------
    selected_features = None
    if config.get("feature_selection") == "backward":
        print("Performing feature selection (once for all seeds)...")
        train_df, _ = make_train_test_split(merged, config["split_strategy"], random_state=0)
        y_train = train_df["condition"].values
        use_pose_deriv = config.get("use_pose_derivatives", True)
        use_time_feat = config.get("use_time_features", False)
        X_train = drop_identifier_columns(train_df, use_pose_derivatives=use_pose_deriv, use_time_features=use_time_feat).drop(columns=["condition"], errors="ignore")

        selected_features, score = backward_elimination_permutation(
            X_train, y_train,
            n_repeats=3,
            threshold_percentile=20,
            min_features=5,
            random_state=0
        )
        print(f"  Selected {len(selected_features)}/{len(X_train.columns)} features (score: {score:.4f})")

    # Evaluate across multiple seeds with the fixed feature subset
    n_seeds = config.get("n_seeds", 10)
    all_metrics = []
    all_cms = []

    for seed in tqdm(range(n_seeds), desc=f"{name}"):
        result = fit_and_evaluate(
            merged,
            split_strategy=config["split_strategy"],
            seed=seed,
            config=config,
            labels=class_labels,
            selected_features=selected_features,
            tuned_rf_params=tuned_rf_params if tuned_rf_params else None
        )

        all_metrics.append(result["metrics"])
        all_cms.append(result["cm"])

    # Aggregate metrics across seeds
    metrics_df = pd.DataFrame(all_metrics)

    aggregated_metrics = {
        "test_acc_mean": float(metrics_df["test_acc"].mean()),
        "test_acc_std": float(metrics_df["test_acc"].std(ddof=1)),
        "test_bal_acc_mean": float(metrics_df["test_bal_acc"].mean()),
        "test_bal_acc_std": float(metrics_df["test_bal_acc"].std(ddof=1)),
        "test_f1_mean": float(metrics_df["test_f1"].mean()),
        "test_f1_std": float(metrics_df["test_f1"].std(ddof=1)),
        "test_kappa_mean": float(metrics_df["test_kappa"].mean()),
        "test_kappa_std": float(metrics_df["test_kappa"].std(ddof=1)),
    }

    # Aggregate per-class metrics
    for label in class_labels:
        aggregated_metrics[f"test_precision_{label}_mean"] = float(metrics_df[f"test_precision_{label}"].mean())
        aggregated_metrics[f"test_precision_{label}_std"] = float(metrics_df[f"test_precision_{label}"].std(ddof=1))
        aggregated_metrics[f"test_recall_{label}_mean"] = float(metrics_df[f"test_recall_{label}"].mean())
        aggregated_metrics[f"test_recall_{label}_std"] = float(metrics_df[f"test_recall_{label}"].std(ddof=1))
        aggregated_metrics[f"test_f1_{label}_mean"] = float(metrics_df[f"test_f1_{label}"].mean())
        aggregated_metrics[f"test_f1_{label}_std"] = float(metrics_df[f"test_f1_{label}"].std(ddof=1))

    # Average confusion matrices
    cm_avg = np.mean(np.stack(all_cms, axis=0), axis=0)

    # Persist results
    results = {
        "name": name,
        "config": {k: v for k, v in config.items() if k != "files"},
        "metrics": aggregated_metrics,
        "confusion_matrix": cm_avg.tolist(),
        "selected_features": selected_features if selected_features else [],
        "tuned_rf_params": tuned_rf_params if tuned_rf_params else {},
        "n_features": len(selected_features) if selected_features else len(merged.columns) - len(ID_COLS),
        "n_seeds": n_seeds,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Completed: {name}")
    print(f"  Balanced Accuracy: {aggregated_metrics['test_bal_acc_mean']:.4f} "
          f"± {aggregated_metrics['test_bal_acc_std']:.4f}")
    print(f"  Features used: {results['n_features']}")

    # Also append to CSV log
    log_to_csv(name, results, output_dir)


# ============================================================================
# LOPO-SPECIFIC MODEL RUNNER
# ============================================================================

def run_lopo_model(name, config, output_dir, force=False):
    """
    Run a single LOPO experiment (leave-one-participant-out).

    Args:
        name (str): Experiment name
        config (dict): Experiment config
        output_dir (str | Path): Output directory
        force (bool): If True, overwrite existing JSON
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = get_config_suffix(config)
    output_path = output_dir / f"{name}{suffix}.json"

    # Respect existing results unless forcing
    if output_path.exists() and not force:
        print(f"[SKIP] {name}: already complete")
        return

    print(f"\n{'='*60}")
    print(f"Running LOPO: {name}")
    print(f"Features: {' + '.join([Path(f[0]).stem for f in config['files']])}")
    print(f"{'='*60}")

    # Load and merge features
    merged = load_and_merge_features(
        config["files"],
        include_order=config.get("include_order", False)
    )
    print(f"Loaded data: {merged.shape}")

    # -----------------------------
    # APPLY CLASS CONFIGURATION
    # -----------------------------
    class_config = config.get("class_config", "all")
    merged, class_labels = apply_class_config(merged, class_config)

    if class_config != "all":
        class_desc = CLASS_CONFIGS[class_config]["description"]
        print(f"Class configuration: {class_desc}")
        print(f"  Filtered data: {merged.shape}")

    participants = sorted(merged["participant"].unique())
    print(f"Participants: {len(participants)}")

    # Get number of random seeds
    n_seeds = config.get("n_seeds", 10)
    print(f"Random seeds per fold: {n_seeds}")

    # -----------------------------
    # OPTIONAL FEATURE SELECTION (once, using first LOPO fold)
    # -----------------------------
    selected_features = None
    if config.get("feature_selection") == "backward":
        print("\nPerforming feature selection (once using first LOPO fold)...")
        print("Note: This will add significant computation time to LOPO")

        # Use first participant as test set for feature selection
        first_participant = participants[0]
        train_df = merged[merged["participant"] != first_participant].copy()
        y_train = train_df["condition"].values
        use_pose_deriv = config.get("use_pose_derivatives", True)
        use_time_feat = config.get("use_time_features", False)
        X_train = drop_identifier_columns(train_df, use_pose_derivatives=use_pose_deriv, use_time_features=use_time_feat).drop(columns=["condition"], errors="ignore")

        selected_features, score = backward_elimination_permutation(
            X_train, y_train,
            n_repeats=3,
            threshold_percentile=20,
            min_features=5,
            random_state=0
        )
        print(f"  Selected {len(selected_features)}/{len(X_train.columns)} features (score: {score:.4f})")

    # Store per-participant results
    participant_results = []

    for train_df, test_df, participant in tqdm(make_lopo_splits(merged),
                                                total=len(participants),
                                                desc=f"LOPO {name}"):
        # Extract targets
        y_train = train_df["condition"].values
        y_test = test_df["condition"].values

        # Extract features (respecting config settings)
        use_pose_deriv = config.get("use_pose_derivatives", True)
        use_time_feat = config.get("use_time_features", False)
        X_train = drop_identifier_columns(train_df, use_pose_derivatives=use_pose_deriv, use_time_features=use_time_feat).drop(columns=["condition"], errors="ignore")
        X_test = drop_identifier_columns(test_df, use_pose_derivatives=use_pose_deriv, use_time_features=use_time_feat).drop(columns=["condition"], errors="ignore")

        # Apply feature selection if performed
        if selected_features is not None:
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]

        # Run with multiple seeds for this participant fold
        seed_metrics = []
        seed_cms = []  # Store confusion matrices for each seed
        for seed in range(n_seeds):
            # Train RF model (different seed each time)
            rf = RandomForestClassifier(**RF_PARAMS, random_state=seed)

            # Z-score normalization: fit on training data, apply to both train and test
            if config.get("normalize_features", True):
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train.values if hasattr(X_train, 'values') else X_train
                X_test_scaled = X_test.values if hasattr(X_test, 'values') else X_test

            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_test_scaled)

            # Compute metrics for this seed
            metrics = {
                "test_acc": accuracy_score(y_test, y_pred),
                "test_bal_acc": balanced_accuracy_score(y_test, y_pred),
                "test_f1": f1_score(y_test, y_pred, labels=class_labels, average="weighted"),
                "test_kappa": cohen_kappa_score(y_test, y_pred, labels=class_labels),
            }
            seed_metrics.append(metrics)

            # Compute confusion matrix for this seed (normalized as percentages)
            cm = confusion_matrix(y_test, y_pred, labels=class_labels, normalize="true") * 100.0
            seed_cms.append(cm)

        # Aggregate across seeds for this participant
        seed_df = pd.DataFrame(seed_metrics)

        # Average confusion matrices across seeds
        cm_avg = np.mean(np.stack(seed_cms, axis=0), axis=0)

        participant_metrics = {
            "participant": participant,
            "n_seeds": n_seeds,
            "test_acc_mean": float(seed_df["test_acc"].mean()),
            "test_acc_std": float(seed_df["test_acc"].std(ddof=1)) if n_seeds > 1 else 0.0,
            "test_bal_acc_mean": float(seed_df["test_bal_acc"].mean()),
            "test_bal_acc_std": float(seed_df["test_bal_acc"].std(ddof=1)) if n_seeds > 1 else 0.0,
            "test_f1_mean": float(seed_df["test_f1"].mean()),
            "test_f1_std": float(seed_df["test_f1"].std(ddof=1)) if n_seeds > 1 else 0.0,
            "test_kappa_mean": float(seed_df["test_kappa"].mean()),
            "test_kappa_std": float(seed_df["test_kappa"].std(ddof=1)) if n_seeds > 1 else 0.0,
            "confusion_matrix": cm_avg.tolist(),  # Add per-participant confusion matrix
        }

        participant_results.append(participant_metrics)

    # Aggregate across all participants
    results_df = pd.DataFrame(participant_results)

    # Aggregate participant-level means across all participants
    aggregated_metrics = {
        "test_acc_mean": float(results_df["test_acc_mean"].mean()),
        "test_acc_std": float(results_df["test_acc_mean"].std(ddof=1)),
        "test_bal_acc_mean": float(results_df["test_bal_acc_mean"].mean()),
        "test_bal_acc_std": float(results_df["test_bal_acc_mean"].std(ddof=1)),
        "test_f1_mean": float(results_df["test_f1_mean"].mean()),
        "test_f1_std": float(results_df["test_f1_mean"].std(ddof=1)),
        "test_kappa_mean": float(results_df["test_kappa_mean"].mean()),
        "test_kappa_std": float(results_df["test_kappa_mean"].std(ddof=1)),
    }

    # Aggregate confusion matrices across all participants
    all_participant_cms = [p["confusion_matrix"] for p in participant_results]
    overall_cm = np.mean(all_participant_cms, axis=0)

    # Persist results
    results = {
        "name": name,
        "config": {k: v for k, v in config.items() if k != "files"},
        "metrics": aggregated_metrics,
        "confusion_matrix": overall_cm.tolist(),
        "participant_results": participant_results,
        "selected_features": selected_features if selected_features else [],
        "n_participants": len(participants),
        "n_seeds": n_seeds,
        "n_features": len(selected_features) if selected_features else len(X_train.columns),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Completed LOPO: {name}")
    print(f"  Balanced Accuracy: {aggregated_metrics['test_bal_acc_mean']:.4f} "
          f"± {aggregated_metrics['test_bal_acc_std']:.4f}")

    # Also append to CSV log
    log_to_csv(name, results, output_dir)


# ============================================================================
# RESULTS MANAGEMENT
# ============================================================================

def log_to_csv(name, results, output_dir):
    """
    Append/update a compact CSV log for quick experiment comparisons.

    Args:
        name (str): Experiment name
        results (dict): Results payload saved to JSON
        output_dir (str | Path): Folder where 'experiment_log.csv' lives
    """
    log_path = Path(output_dir) / "experiment_log.csv"

    # Flatten metrics into one row
    row = {
        "experiment_name": name,
        "split_strategy": results["config"].get("split_strategy", "lopo"),
        "n_features": results.get("n_features", 0),
        "n_seeds": results.get("n_seeds", results.get("n_participants", 0)),
        "test_bal_acc_mean": results["metrics"]["test_bal_acc_mean"],
        "test_bal_acc_std": results["metrics"]["test_bal_acc_std"],
        "test_f1_mean": results["metrics"]["test_f1_mean"],
        "test_f1_std": results["metrics"]["test_f1_std"],
        "test_kappa_mean": results["metrics"]["test_kappa_mean"],
        "test_kappa_std": results["metrics"]["test_kappa_std"],
        "timestamp": results.get("timestamp", ""),
    }

    df = pd.DataFrame([row])

    if log_path.exists():
        existing = pd.read_csv(log_path)
        # De-duplicate by experiment_name
        existing = existing[existing["experiment_name"] != name]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(log_path, index=False)


def print_summary(output_dir):
    """
    Pretty-print a quick summary of all experiments from the CSV log.

    Args:
        output_dir (str | Path): Directory containing 'experiment_log.csv'
    """
    log_path = Path(output_dir) / "experiment_log.csv"

    if not log_path.exists():
        print("No experiments logged yet.")
        return

    df = pd.read_csv(log_path)

    print("\nEXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<40} {'Split':<15} {'Bal Acc':<20}")
    print("-" * 80)

    for _, row in df.iterrows():
        bal_acc = f"{row['test_bal_acc_mean']*100:.1f}% ± {row['test_bal_acc_std']*100:.1f}"
        print(f"{row['experiment_name']:<40} {row['split_strategy']:<15} {bal_acc:<20}")

    print("=" * 80)


# ============================================================================
# PARTICIPANT-SPECIFIC MODEL RUNNER
# ============================================================================

def run_participant_specific_model(name, config, output_dir, training_sizes, sampling_strategy, force=False):
    """
    Run participant-specific learning curve experiment with multiple random seeds.

    Trains separate models for each participant at different training sizes,
    creating a learning curve showing how performance improves with more data.

    For each participant and training size, runs multiple random seeds for the
    sampling process and aggregates results to provide robust estimates with
    confidence intervals.

    Args:
        name (str): Experiment name
        config (dict): Experiment config (uses config['n_seeds'] for number of random seeds, default 5)
        output_dir (str | Path): Output directory
        training_sizes (list): List of training sizes to test
        sampling_strategy (str): 'stratified', 'temporal', or 'shuffled'
        force (bool): If True, overwrite existing JSON
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = get_config_suffix(config)
    output_path = output_dir / f"{name}{suffix}.json"

    # Respect existing results unless forcing
    if output_path.exists() and not force:
        print(f"[SKIP] {name}: already complete")
        return

    print(f"\n{'='*60}")
    print(f"Running Participant-Specific: {name}")
    print(f"Features: {' + '.join([Path(f[0]).stem for f in config['files']])}")
    print(f"Sampling strategy: {sampling_strategy}")
    print(f"{'='*60}")

    # Load and merge features
    merged = load_and_merge_features(
        config["files"],
        include_order=config.get("include_order", False)
    )
    print(f"Loaded data: {merged.shape}")

    # -----------------------------
    # APPLY CLASS CONFIGURATION
    # -----------------------------
    class_config = config.get("class_config", "all")
    merged, class_labels = apply_class_config(merged, class_config)

    if class_config != "all":
        class_desc = CLASS_CONFIGS[class_config]["description"]
        print(f"Class configuration: {class_desc}")
        print(f"  Filtered data: {merged.shape}")

    participants = sorted(merged["participant"].unique())
    print(f"Participants: {len(participants)}")
    print(f"Training sizes: {training_sizes}")

    # Get number of random seeds to use
    n_seeds = config.get("n_seeds", 10)
    print(f"Random seeds per participant: {n_seeds}")

    # Choose sampling function based on strategy
    if sampling_strategy == "stratified":
        sample_fn = sample_participant_data_stratified
        use_per_condition = True  # stratified uses windows per condition
    elif sampling_strategy == "temporal_stratified":
        sample_fn = sample_participant_data_temporal_stratified
        use_per_condition = True  # temporal_stratified uses windows per condition
        print("  Note: Using temporal ordering with 1-window buffer to avoid data leakage")
    elif sampling_strategy == "temporal":
        sample_fn = sample_participant_data_temporal
        use_per_condition = False  # temporal uses total windows
    elif sampling_strategy == "shuffled":
        sample_fn = sample_participant_data_shuffled
        use_per_condition = False  # shuffled uses total windows
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    # -----------------------------
    # OPTIONAL FEATURE SELECTION (once, using largest training size and pooled data)
    # -----------------------------
    selected_features = None
    if config.get("feature_selection") == "backward":
        print("\nPerforming feature selection (once using largest training size)...")
        print("Note: This will add significant computation time")

        # Use largest training size for most stable feature selection
        largest_size = max(training_sizes)
        print(f"  Sampling at largest training size: {largest_size} windows per condition")

        # Pool training data from ALL participants at this training size
        all_train_data = []
        for participant in participants:
            df_participant = merged[merged["participant"] == participant].copy()

            # Sample using the chosen strategy
            if use_per_condition:
                train_df, _ = sample_fn(df_participant, largest_size, random_state=0)
            else:
                train_df, _ = sample_fn(df_participant, largest_size, random_state=0)

            if train_df is not None:
                # Check if all classes are present
                train_classes = set(train_df["condition"].unique())
                if len(train_classes) == len(class_labels):
                    all_train_data.append(train_df)

        if len(all_train_data) > 0:
            # Pool data across participants
            pooled_train = pd.concat(all_train_data, ignore_index=True)
            print(f"  Pooled training data: {len(pooled_train)} samples from {len(all_train_data)} participants")

            # Extract features
            y_train = pooled_train["condition"].values
            use_pose_deriv = config.get("use_pose_derivatives", True)
            use_time_feat = config.get("use_time_features", False)
            X_train = drop_identifier_columns(pooled_train, use_pose_derivatives=use_pose_deriv, use_time_features=use_time_feat).drop(columns=["condition"], errors="ignore")

            # Run backward elimination
            selected_features, score = backward_elimination_permutation(
                X_train, y_train,
                n_repeats=3,
                threshold_percentile=20,
                min_features=5,
                random_state=0
            )
            print(f"  Selected {len(selected_features)}/{len(X_train.columns)} features (score: {score:.4f})")
        else:
            print(f"  Warning: Could not pool sufficient training data for feature selection")
            print(f"  Proceeding without feature selection")

    # Store learning curve results
    learning_curve_results = []

    # For each training size
    for train_size in tqdm(training_sizes, desc=f"Training sizes for {name}"):
        print(f"\n--- Training size: {train_size} ---")

        participant_results = []

        # For each participant
        for participant in participants:
            # Get participant's data
            df_participant = merged[merged["participant"] == participant].copy()

            # Store results across seeds for this participant
            seed_results = []
            seed_cms = []  # Store confusion matrices for each seed

            # Run multiple random seeds for this participant
            for seed in range(n_seeds):
                # Sample train/test split according to strategy with different random seed
                if use_per_condition:
                    # Stratified: train_size is windows per condition
                    train_df, test_df = sample_fn(df_participant, train_size, random_state=seed)
                else:
                    # Temporal/Shuffled: train_size is total windows
                    train_df, test_df = sample_fn(df_participant, train_size, random_state=seed)

                # Skip if insufficient data
                if train_df is None or test_df is None:
                    continue

                # Extract targets
                y_train = train_df["condition"].values
                y_test = test_df["condition"].values

                # Check if all classes are present in training data
                train_classes = set(y_train)
                if len(train_classes) < len(class_labels):
                    # Skip if not all classes present in training
                    continue

                # Extract features (respecting config settings)
                use_pose_deriv = config.get("use_pose_derivatives", True)
                use_time_feat = config.get("use_time_features", False)
                X_train = drop_identifier_columns(train_df, use_pose_derivatives=use_pose_deriv, use_time_features=use_time_feat).drop(columns=["condition"], errors="ignore")
                X_test = drop_identifier_columns(test_df, use_pose_derivatives=use_pose_deriv, use_time_features=use_time_feat).drop(columns=["condition"], errors="ignore")

                # Apply feature selection if performed
                if selected_features is not None:
                    X_train = X_train[selected_features]
                    X_test = X_test[selected_features]

                # Train RF model (use seed for RF too)
                rf = RandomForestClassifier(**RF_PARAMS, random_state=seed)

                # Z-score normalization: fit on training data, apply to both train and test
                if config.get("normalize_features", True):
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_train_scaled = X_train.values if hasattr(X_train, 'values') else X_train
                    X_test_scaled = X_test.values if hasattr(X_test, 'values') else X_test

                rf.fit(X_train_scaled, y_train)
                y_pred = rf.predict(X_test_scaled)

                # Compute metrics
                metrics = {
                    "seed": seed,
                    "n_train_samples": len(train_df),
                    "n_test_samples": len(test_df),
                    "test_acc": accuracy_score(y_test, y_pred),
                    "test_bal_acc": balanced_accuracy_score(y_test, y_pred),
                    "test_f1": f1_score(y_test, y_pred, labels=class_labels, average="weighted", zero_division=0),
                    "test_kappa": cohen_kappa_score(y_test, y_pred, labels=class_labels),
                }

                seed_results.append(metrics)

                # Compute confusion matrix for this seed (normalized as percentages)
                cm = confusion_matrix(y_test, y_pred, labels=class_labels, normalize="true") * 100.0
                seed_cms.append(cm)

            # Aggregate across seeds for this participant
            if seed_results:
                # Convert to dataframe for easy aggregation
                seed_df = pd.DataFrame(seed_results)

                # Average confusion matrices across seeds
                if seed_cms:
                    cm_avg = np.mean(np.stack(seed_cms, axis=0), axis=0)
                else:
                    # If no confusion matrices were collected, create a zero matrix
                    cm_avg = np.zeros((len(class_labels), len(class_labels)))

                # Compute mean and std across seeds
                participant_metrics = {
                    "participant": participant,
                    "train_size": train_size,
                    "n_seeds": len(seed_results),
                    "n_train_samples_mean": float(seed_df["n_train_samples"].mean()),
                    "n_test_samples_mean": float(seed_df["n_test_samples"].mean()),
                    "test_acc_mean": float(seed_df["test_acc"].mean()),
                    "test_acc_std": float(seed_df["test_acc"].std(ddof=1)) if len(seed_df) > 1 else 0.0,
                    "test_bal_acc_mean": float(seed_df["test_bal_acc"].mean()),
                    "test_bal_acc_std": float(seed_df["test_bal_acc"].std(ddof=1)) if len(seed_df) > 1 else 0.0,
                    "test_f1_mean": float(seed_df["test_f1"].mean()),
                    "test_f1_std": float(seed_df["test_f1"].std(ddof=1)) if len(seed_df) > 1 else 0.0,
                    "test_kappa_mean": float(seed_df["test_kappa"].mean()),
                    "test_kappa_std": float(seed_df["test_kappa"].std(ddof=1)) if len(seed_df) > 1 else 0.0,
                    "confusion_matrix": cm_avg.tolist(),  # Add per-participant confusion matrix
                }

                participant_results.append(participant_metrics)

        # Aggregate across participants for this training size
        if participant_results:
            results_df = pd.DataFrame(participant_results)

            # Aggregate confusion matrices across all participants for this training size
            all_participant_cms = [p["confusion_matrix"] for p in participant_results]
            overall_cm = np.mean(all_participant_cms, axis=0)

            # Aggregate participant-level means (averaged over seeds) across all participants
            aggregated = {
                "train_size": train_size,
                "n_participants": len(participant_results),
                "n_seeds": n_seeds,
                "test_acc_mean": float(results_df["test_acc_mean"].mean()),
                "test_acc_std": float(results_df["test_acc_mean"].std(ddof=1)) if len(results_df) > 1 else 0.0,
                "test_bal_acc_mean": float(results_df["test_bal_acc_mean"].mean()),
                "test_bal_acc_std": float(results_df["test_bal_acc_mean"].std(ddof=1)) if len(results_df) > 1 else 0.0,
                "test_f1_mean": float(results_df["test_f1_mean"].mean()),
                "test_f1_std": float(results_df["test_f1_mean"].std(ddof=1)) if len(results_df) > 1 else 0.0,
                "test_kappa_mean": float(results_df["test_kappa_mean"].mean()),
                "test_kappa_std": float(results_df["test_kappa_mean"].std(ddof=1)) if len(results_df) > 1 else 0.0,
                "confusion_matrix": overall_cm.tolist(),  # Add aggregated confusion matrix
            }

            learning_curve_results.append({
                "aggregated": aggregated,
                "per_participant": participant_results
            })

            print(f"  Participants: {len(participant_results)}/{len(participants)}")
            print(f"  Balanced Accuracy: {aggregated['test_bal_acc_mean']:.4f} ± {aggregated['test_bal_acc_std']:.4f}")

    # Persist results
    results = {
        "name": name,
        "config": {k: v for k, v in config.items() if k != "files"},
        "sampling_strategy": sampling_strategy,
        "n_seeds": n_seeds,
        "learning_curve": learning_curve_results,
        "n_participants": len(participants),
        "training_sizes": training_sizes,
        "selected_features": selected_features if selected_features else [],
        "n_features": len(selected_features) if selected_features else None,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Completed Participant-Specific: {name}")
    if selected_features:
        print(f"  Features used: {len(selected_features)}")

    # Create summary CSV for easy plotting
    summary_rows = []
    for lc_result in learning_curve_results:
        agg = lc_result["aggregated"]
        summary_rows.append({
            "experiment_name": name,
            "train_size": agg["train_size"],
            "n_participants": agg["n_participants"],
            "test_bal_acc_mean": agg["test_bal_acc_mean"],
            "test_bal_acc_std": agg["test_bal_acc_std"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / f"{name}{suffix}_learning_curve.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"  Learning curve saved to: {summary_csv_path}")
