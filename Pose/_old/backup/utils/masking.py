"""
================================================================================
DATA MASKING MODULE FOR QUALITY CONTROL
================================================================================

This module provides functions for masking unreliable data based on quality
control results. It applies masks to feature data by setting bad windows to NaN,
allowing downstream processing to handle or interpolate these gaps appropriately.

Key functions:
- Load QC results and convert to mask format
- Apply masks to feature DataFrames
- Generate masking statistics and reports

Author: Pose Analysis Pipeline
Date: 2024
================================================================================
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, List

from .landmark_config import QC_TO_FEATURE_COLUMNS


# ============================================================================
# QC RESULT LOADING
# ============================================================================

def load_bad_windows(bad_window_csv_path: str,
                    qc_window_frames: int,
                    qc_overlap: float) -> Tuple[Dict[str, Dict[str, List[Tuple[int, int]]]], int]:
    """
    Load quality control results and convert to mask format.

    Loads the CSV file containing bad window indices and organizes them
    into a nested dictionary for efficient lookup during masking.

    Parameters
    ----------
    bad_window_csv_path : str
        Path to the metric_bad_window_indices.csv file from QC
    qc_window_frames : int
        Number of frames per QC window (must match QC analysis)
    qc_overlap : float
        Overlap fraction used in QC (must match QC analysis)

    Returns
    -------
    tuple
        bad_windows_map : dict mapping filename -> metric -> list of (start, end) ranges
        step_frames : int - number of frames between window starts

    Examples
    --------
    >>> bad_map, step = load_bad_windows('qc/metric_bad_window_indices.csv', 1800, 0.0)
    >>> bad_map['participant_001.csv']['eyes']
    [(0, 1800), (3600, 5400)]
    """
    # Check if file exists
    if not os.path.isfile(bad_window_csv_path):
        raise FileNotFoundError(f"QC bad window file not found: {bad_window_csv_path}")

    # Load the CSV - handle empty file case
    try:
        df = pd.read_csv(bad_window_csv_path)
    except pd.errors.EmptyDataError:
        # No bad windows found - this is actually good!
        print("  No bad windows detected in quality control (excellent data quality!)")
        # Return empty dictionary and step frames
        step_frames = int(round(qc_window_frames * (1.0 - qc_overlap)))
        return {}, step_frames

    # Check if dataframe is empty
    if df.empty:
        print("  No bad windows detected in quality control (excellent data quality!)")
        step_frames = int(round(qc_window_frames * (1.0 - qc_overlap)))
        return {}, step_frames

    # Verify required columns exist
    required_columns = {'file', 'metric'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"QC CSV missing required columns: {missing}")

    # Normalize filenames and metric names for consistent matching
    df['file'] = df['file'].astype(str).map(lambda s: os.path.basename(s).strip())
    df['metric'] = df['metric'].astype(str).str.strip().str.lower()

    # Check what format the data is in
    has_explicit_ranges = {'start_frame', 'end_frame_exclusive'}.issubset(df.columns)
    has_window_index = 'window_index' in df.columns

    # Calculate step size between windows
    step_frames = int(round(qc_window_frames * (1.0 - qc_overlap)))

    # If we only have window indices, calculate frame ranges
    if not has_explicit_ranges and has_window_index:
        df = df.copy()
        df['window_index'] = df['window_index'].astype(int)
        df['start_frame'] = (df['window_index'] * step_frames).astype(int)
        df['end_frame_exclusive'] = (df['start_frame'] + qc_window_frames).astype(int)
    elif not has_explicit_ranges:
        raise ValueError("QC CSV must have either window_index or start_frame/end_frame_exclusive columns")

    # Build nested dictionary: filename -> metric -> list of bad frame ranges
    bad_windows_map = {}

    for (filename, metric), group in df.groupby(['file', 'metric']):
        # Extract frame ranges as tuples
        frame_ranges = list(zip(
            group['start_frame'].astype(int),
            group['end_frame_exclusive'].astype(int)
        ))

        # Add to nested dictionary
        if filename not in bad_windows_map:
            bad_windows_map[filename] = {}
        bad_windows_map[filename][metric] = frame_ranges

    # Print summary
    total_bad_windows = sum(len(ranges) for file_dict in bad_windows_map.values()
                          for ranges in file_dict.values())
    print(f"Loaded {total_bad_windows} bad window(s) from {bad_window_csv_path}")
    print(f"  Window size: {qc_window_frames} frames, Step: {step_frames} frames")
    print(f"  Files with bad windows: {len(bad_windows_map)}")

    return bad_windows_map, step_frames


# ============================================================================
# MASK APPLICATION
# ============================================================================

def apply_bad_window_masks(filename: str,
                          df_features: pd.DataFrame,
                          bad_windows_map: Dict[str, Dict[str, List[Tuple[int, int]]]],
                          qc_to_columns_map: Dict[str, List[str]] = None) -> Tuple[pd.DataFrame, Dict, int]:
    """
    Apply quality control masks to feature data.

    Sets feature values to NaN during time windows identified as unreliable
    by the quality control analysis. Different facial regions can have
    different bad windows.

    Parameters
    ----------
    filename : str
        Name of the file being processed (for lookup in bad_windows_map)
    df_features : pd.DataFrame
        DataFrame containing extracted features to mask
    bad_windows_map : dict
        Nested dictionary from load_bad_windows()
    qc_to_columns_map : dict, optional
        Mapping of QC metrics to feature columns. If None, uses default

    Returns
    -------
    tuple
        df_masked : DataFrame with masks applied
        stats : Dictionary with masking statistics per metric
        total_masked : Total number of frames masked

    Examples
    --------
    >>> df_masked, stats, total = apply_bad_window_masks(
    ...     'participant_001.csv', df_features, bad_windows_map
    ... )
    >>> print(f"Masked {total} frames total")
    Masked 450 frames total
    """
    # Use default QC to column mapping if not provided
    if qc_to_columns_map is None:
        qc_to_columns_map = QC_TO_FEATURE_COLUMNS

    # Make a copy to avoid modifying original
    df_masked = df_features.copy()
    n_frames = len(df_masked)

    # Get bad windows for this specific file
    file_bad_windows = bad_windows_map.get(filename, {})

    # Initialize statistics tracking
    stats = {}
    total_masked_frames = 0

    # Create boolean mask for each QC metric
    metric_masks = {metric: np.zeros(n_frames, dtype=bool)
                   for metric in qc_to_columns_map.keys()}

    # Build masks from bad window ranges
    for qc_metric_raw, window_ranges in file_bad_windows.items():
        # Normalize metric name
        qc_metric = qc_metric_raw.lower()

        if qc_metric not in metric_masks:
            print(f"Warning: Unknown QC metric '{qc_metric}' in bad windows, skipping")
            continue

        # Mark all frames in bad windows as True
        for start_frame, end_frame in window_ranges:
            # Ensure we don't exceed DataFrame bounds
            start_frame = max(0, int(start_frame))
            end_frame = min(n_frames, int(end_frame))

            if start_frame < end_frame:
                metric_masks[qc_metric][start_frame:end_frame] = True

    # Apply masks to corresponding feature columns
    for qc_metric, mask in metric_masks.items():
        frames_masked = int(mask.sum())

        # Get list of columns affected by this metric
        affected_columns = qc_to_columns_map.get(qc_metric, [])

        # Apply mask to each affected column
        columns_masked = []
        for col in affected_columns:
            if col in df_masked.columns:
                # Set masked frames to NaN
                df_masked.loc[mask, col] = np.nan
                columns_masked.append(col)

        # Record statistics
        stats[qc_metric] = {
            "frames_total": n_frames,
            "frames_masked": frames_masked,
            "pct_masked": (frames_masked / n_frames * 100) if n_frames > 0 else np.nan,
            "windows_masked": len(file_bad_windows.get(qc_metric, [])),
            "columns_affected": columns_masked
        }
        total_masked_frames += frames_masked

    return df_masked, stats, total_masked_frames


# ============================================================================
# MASK STATISTICS AND REPORTING
# ============================================================================

def generate_masking_report(masking_stats: Dict[str, Dict]) -> pd.DataFrame:
    """
    Generate a summary report of masking statistics.

    Creates a DataFrame summarizing how much data was masked for each
    metric and file.

    Parameters
    ----------
    masking_stats : dict
        Dictionary of masking statistics from apply_bad_window_masks()

    Returns
    -------
    pd.DataFrame
        Summary report with statistics per metric

    Examples
    --------
    >>> report = generate_masking_report(stats)
    >>> print(report.to_string())
    """
    rows = []

    for metric, metric_stats in masking_stats.items():
        rows.append({
            'metric': metric,
            'frames_total': metric_stats['frames_total'],
            'frames_masked': metric_stats['frames_masked'],
            'percent_masked': metric_stats['pct_masked'],
            'windows_masked': metric_stats['windows_masked'],
            'columns_affected': len(metric_stats['columns_affected'])
        })

    return pd.DataFrame(rows)


def create_mask_visualization_data(df: pd.DataFrame,
                                  mask_info: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a DataFrame suitable for visualizing masked regions.

    Adds binary columns indicating which frames are masked for each metric,
    useful for creating timeline visualizations.

    Parameters
    ----------
    df : pd.DataFrame
        Original feature DataFrame
    mask_info : dict
        Masking statistics from apply_bad_window_masks()

    Returns
    -------
    pd.DataFrame
        DataFrame with added mask indicator columns

    Examples
    --------
    >>> viz_df = create_mask_visualization_data(df_features, stats)
    >>> viz_df['mask_eyes'].sum()  # Number of frames with eyes masked
    150
    """
    viz_df = df.copy()

    # Add binary mask columns
    for metric, metric_stats in mask_info.items():
        # Create mask column name
        mask_col = f'mask_{metric}'

        # Initialize as all False
        viz_df[mask_col] = False

        # Could reconstruct mask from stats if needed
        # This is a placeholder for more complex visualization needs

    return viz_df


# ============================================================================
# INTERPOLATION AFTER MASKING
# ============================================================================

def interpolate_masked_regions(df: pd.DataFrame,
                              method: str = 'linear',
                              limit: int = None,
                              columns: List[str] = None) -> pd.DataFrame:
    """
    Interpolate NaN values created by masking.

    Fills gaps in the data using various interpolation methods. This is
    useful when continuous signals are needed for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with masked (NaN) values
    method : str, optional
        Interpolation method ('linear', 'polynomial', 'spline'), default: 'linear'
    limit : int, optional
        Maximum number of consecutive NaNs to fill. None = no limit
    columns : list of str, optional
        Specific columns to interpolate. None = all numeric columns

    Returns
    -------
    pd.DataFrame
        DataFrame with interpolated values

    Examples
    --------
    >>> df_filled = interpolate_masked_regions(df_masked, method='linear', limit=30)
    """
    df_interpolated = df.copy()

    # Determine columns to interpolate
    if columns is None:
        columns = [col for col in df.columns
                  if pd.api.types.is_numeric_dtype(df[col])]

    # Apply interpolation to each column
    for col in columns:
        if col not in df.columns:
            continue

        if method == 'linear':
            df_interpolated[col] = df[col].interpolate(
                method='linear',
                limit=limit,
                limit_direction='both'
            )
        elif method == 'polynomial':
            df_interpolated[col] = df[col].interpolate(
                method='polynomial',
                order=3,
                limit=limit,
                limit_direction='both'
            )
        elif method == 'spline':
            # Only use spline if we have enough non-NaN values
            non_nan_count = df[col].notna().sum()
            if non_nan_count > 3:  # Minimum for spline
                df_interpolated[col] = df[col].interpolate(
                    method='spline',
                    order=3,
                    limit=limit,
                    limit_direction='both'
                )
            else:
                # Fall back to linear
                df_interpolated[col] = df[col].interpolate(
                    method='linear',
                    limit=limit,
                    limit_direction='both'
                )
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    return df_interpolated


# ============================================================================
# QUALITY METRICS AFTER MASKING
# ============================================================================

def calculate_data_completeness(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate completeness statistics for masked data.

    Provides metrics on how much valid data remains after masking.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potentially masked (NaN) values

    Returns
    -------
    dict
        Dictionary with completeness statistics

    Examples
    --------
    >>> completeness = calculate_data_completeness(df_masked)
    >>> print(f"Overall completeness: {completeness['overall']:.1%}")
    Overall completeness: 95.3%
    """
    stats = {}

    # Overall completeness
    total_values = df.size
    non_nan_values = df.notna().sum().sum()
    stats['overall'] = non_nan_values / total_values if total_values > 0 else 0

    # Per-column completeness
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_total = len(df[col])
            col_valid = df[col].notna().sum()
            stats[f'column_{col}'] = col_valid / col_total if col_total > 0 else 0

    # Calculate consecutive gap statistics
    gap_stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Find gaps (consecutive NaNs)
            is_nan = df[col].isna()
            gaps = []
            current_gap = 0

            for val in is_nan:
                if val:
                    current_gap += 1
                else:
                    if current_gap > 0:
                        gaps.append(current_gap)
                    current_gap = 0

            # Add last gap if exists
            if current_gap > 0:
                gaps.append(current_gap)

            if gaps:
                gap_stats[col] = {
                    'max_gap': max(gaps),
                    'mean_gap': np.mean(gaps),
                    'num_gaps': len(gaps)
                }
            else:
                gap_stats[col] = {
                    'max_gap': 0,
                    'mean_gap': 0,
                    'num_gaps': 0
                }

    stats['gap_statistics'] = gap_stats

    return stats