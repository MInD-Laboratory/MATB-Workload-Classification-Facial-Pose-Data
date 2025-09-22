"""
================================================================================
QUALITY CONTROL MODULE FOR FACIAL POSE DATA
================================================================================

This module provides functions for identifying and tracking poor-quality landmark
data in facial pose recordings. It divides data into time windows and checks
each window for excessive missing or low-confidence landmarks.

The quality control process:
1. Divides each recording into overlapping time windows
2. Checks each landmark in each window for data quality
3. Marks windows as "bad" if key landmarks are missing too often
4. Groups results by facial region (eyes, mouth, etc.)
5. Outputs detailed reports for use in downstream processing

Author: Pose Analysis Pipeline
Date: 2024
================================================================================
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Dict
from tqdm import tqdm

from .landmark_config import (
    METRIC_KEYPOINTS,
    RELEVANT_KEYPOINTS,
    get_column_name
)


# ============================================================================
# WINDOW MANAGEMENT FUNCTIONS
# ============================================================================

def calculate_window_ranges(n_rows: int, window_size: int, overlap: float) -> List[Tuple[int, int]]:
    """
    Calculate sliding window positions for quality control analysis.

    Divides data into overlapping segments to check quality in each segment
    separately. This allows detection of brief periods of poor tracking.

    Parameters
    ----------
    n_rows : int
        Total number of rows (frames) in the data
    window_size : int
        Number of frames per window
    overlap : float
        Fraction of overlap between consecutive windows (0.0 to 1.0)
        0.0 = no overlap, 0.5 = 50% overlap, etc.

    Returns
    -------
    list of tuple
        List of (start, end) frame indices for each window
        End index is exclusive (Python slice convention)

    Examples
    --------
    >>> calculate_window_ranges(100, 30, 0.0)
    [(0, 30), (30, 60), (60, 90)]
    >>> calculate_window_ranges(100, 30, 0.5)
    [(0, 30), (15, 45), (30, 60), (45, 75), (60, 90)]
    """
    # Handle edge case: file too short for even one window
    if n_rows < window_size:
        return []

    # Calculate step size between window starts
    # With 0% overlap, step = window_size
    # With 50% overlap, step = window_size * 0.5
    step = max(1, int(round(window_size * (1 - overlap))))

    # Generate window ranges
    windows = []
    for start in range(0, n_rows - window_size + 1, step):
        end = start + window_size
        windows.append((start, end))

    return windows


# ============================================================================
# MISSING DATA DETECTION FUNCTIONS
# ============================================================================

def find_longest_consecutive_true(series: pd.Series) -> int:
    """
    Find the longest consecutive sequence of True values in a boolean series.

    Used to detect the longest gap of missing data in a window.

    Parameters
    ----------
    series : pd.Series
        Boolean series where True indicates missing/bad data

    Returns
    -------
    int
        Length of the longest consecutive True sequence

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([True, True, False, True, False])
    >>> find_longest_consecutive_true(s)
    2
    """
    # Convert to numpy for faster processing
    values = series.to_numpy()

    # Track current and maximum run lengths
    current_run = 0
    max_run = 0

    # Iterate through values counting consecutive Trues
    for value in values:
        if value:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run


def check_landmark_missing(df_window: pd.DataFrame, landmark_idx: int,
                          conf_threshold: float) -> pd.Series:
    """
    Determine which frames have missing or low-confidence data for a landmark.

    A landmark is considered "missing" if:
    - Its x or y coordinate is NaN
    - Its confidence/probability is below the threshold

    Parameters
    ----------
    df_window : pd.DataFrame
        DataFrame containing landmark data for a time window
    landmark_idx : int
        Index of the landmark to check (0-70 for OpenPose)
    conf_threshold : float
        Minimum confidence value for a landmark to be considered valid

    Returns
    -------
    pd.Series
        Boolean series where True indicates missing/bad landmark data

    Examples
    --------
    >>> df = pd.DataFrame({'x37': [1, 2, np.nan], 'y37': [1, 2, 3],
    ...                    'prob37': [0.9, 0.1, 0.8]})
    >>> missing = check_landmark_missing(df, 37, 0.3)
    >>> missing.tolist()
    [False, True, True]  # Low confidence and NaN coordinate
    """
    # Get column names for this landmark
    x_col = get_column_name(landmark_idx, 'x')
    y_col = get_column_name(landmark_idx, 'y')
    prob_col = get_column_name(landmark_idx, 'prob')

    # If columns don't exist, consider all frames as missing
    if (x_col not in df_window.columns or
        y_col not in df_window.columns or
        prob_col not in df_window.columns):
        return pd.Series(True, index=df_window.index)

    # A landmark is "good" if it has:
    # 1. High confidence (>= threshold)
    # 2. Valid x coordinate (not NaN)
    # 3. Valid y coordinate (not NaN)
    good_confidence = df_window[prob_col] >= conf_threshold
    valid_x = df_window[x_col].notna()
    valid_y = df_window[y_col].notna()

    good_landmark = good_confidence & valid_x & valid_y

    # Return True for missing/bad landmarks
    return ~good_landmark


# ============================================================================
# FILE-LEVEL QUALITY CONTROL ANALYSIS
# ============================================================================

def analyze_file_quality(filepath: str, window_size: int, overlap: float,
                        conf_threshold: float, max_interpolation: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform quality control analysis on a single CSV file.

    This function:
    1. Divides the data into overlapping windows
    2. Checks each landmark in each window for excessive missing data
    3. Marks windows as "bad" if key landmarks are missing too often
    4. Groups landmarks by facial region (eyes, mouth, etc.)

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing landmark data
    window_size : int
        Number of frames per window
    overlap : float
        Fraction of overlap between windows (0.0 to 1.0)
    conf_threshold : float
        Minimum confidence for valid landmarks
    max_interpolation : int
        Maximum consecutive missing frames that can be interpolated
        Windows with gaps larger than this are marked as "bad"

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        keypoint_summary : Summary statistics per landmark
        metric_summary : Summary statistics per facial region
        bad_window_details : Detailed list of bad windows with frame ranges

    Examples
    --------
    >>> kp_df, met_df, details_df = analyze_file_quality(
    ...     'participant_001.csv', window_size=1800, overlap=0.0,
    ...     conf_threshold=0.3, max_interpolation=60
    ... )
    """
    # Load the data file
    df = pd.read_csv(filepath)
    filename = os.path.basename(filepath)

    # Calculate window positions
    window_ranges = calculate_window_ranges(len(df), window_size, overlap)
    total_windows = len(window_ranges)

    # Handle files too short for analysis
    if total_windows == 0:
        # Return empty results with proper structure
        kp_rows = [{
            "file": filename,
            "keypoint": kp,
            "bad_windows": 0,
            "total_windows": 0,
            "pct_bad": np.nan
        } for kp in RELEVANT_KEYPOINTS]

        met_rows = [{
            "file": filename,
            "metric": metric,
            "bad_windows": 0,
            "total_windows": 0,
            "pct_bad": np.nan
        } for metric in METRIC_KEYPOINTS.keys()]

        return pd.DataFrame(kp_rows), pd.DataFrame(met_rows), pd.DataFrame([])

    # Initialize counters for bad windows
    keypoint_bad_counts = {kp: 0 for kp in RELEVANT_KEYPOINTS}
    metric_bad_counts = {metric: 0 for metric in METRIC_KEYPOINTS}
    metric_bad_details = {metric: [] for metric in METRIC_KEYPOINTS}

    # Analyze each window
    for window_idx, (start, end) in enumerate(window_ranges):
        # Extract window data
        df_window = df.iloc[start:end].reset_index(drop=True)

        # Check each individual landmark in this window
        keypoint_is_bad = {}
        for kp_idx in RELEVANT_KEYPOINTS:
            # Check if landmark is missing
            missing_series = check_landmark_missing(df_window, kp_idx, conf_threshold)

            # Find longest gap of missing data
            longest_gap = find_longest_consecutive_true(missing_series)

            # Window is bad if gap exceeds interpolation limit
            is_bad = (longest_gap > max_interpolation)
            keypoint_is_bad[kp_idx] = is_bad

            # Update counter
            if is_bad:
                keypoint_bad_counts[kp_idx] += 1

        # Check each facial region (metric)
        # A region is bad if ANY of its landmarks are bad
        for metric, landmark_list in METRIC_KEYPOINTS.items():
            metric_is_bad = any(keypoint_is_bad.get(kp, True) for kp in landmark_list)

            if metric_is_bad:
                metric_bad_counts[metric] += 1
                metric_bad_details[metric].append({
                    "window_index": window_idx,
                    "start_frame": start,
                    "end_frame": end  # Exclusive end
                })

    # Build summary DataFrames
    # 1. Per-keypoint summary
    keypoint_rows = []
    for kp_idx in RELEVANT_KEYPOINTS:
        bad_count = keypoint_bad_counts[kp_idx]
        keypoint_rows.append({
            "file": filename,
            "keypoint": kp_idx,
            "bad_windows": bad_count,
            "total_windows": total_windows,
            "pct_bad": (bad_count / total_windows) * 100 if total_windows > 0 else np.nan
        })

    # 2. Per-metric summary
    metric_rows = []
    for metric in METRIC_KEYPOINTS:
        bad_count = metric_bad_counts[metric]
        metric_rows.append({
            "file": filename,
            "metric": metric,
            "bad_windows": bad_count,
            "total_windows": total_windows,
            "pct_bad": (bad_count / total_windows) * 100 if total_windows > 0 else np.nan
        })

    # 3. Detailed bad window list
    detail_rows = []
    for metric, window_list in metric_bad_details.items():
        for window_info in window_list:
            detail_rows.append({
                "file": filename,
                "metric": metric,
                "window_index": window_info["window_index"],
                "start_frame": window_info["start_frame"],
                "end_frame_exclusive": window_info["end_frame"]
            })

    return pd.DataFrame(keypoint_rows), pd.DataFrame(metric_rows), pd.DataFrame(detail_rows)


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def run_quality_control_batch(input_dir: str, output_dir: str, window_size: int,
                             overlap: float, conf_threshold: float,
                             max_interpolation: int) -> Tuple[str, str, str]:
    """
    Run quality control analysis on all CSV files in a directory.

    Processes all CSV files in the input directory and saves three output files:
    1. keypoint_bad_windows.csv - Statistics per landmark
    2. metric_bad_windows.csv - Statistics per facial region
    3. metric_bad_window_indices.csv - Detailed bad window locations

    Parameters
    ----------
    input_dir : str
        Directory containing input CSV files
    output_dir : str
        Directory to save QC results
    window_size : int
        Number of frames per window
    overlap : float
        Fraction of overlap between windows
    conf_threshold : float
        Minimum confidence for valid landmarks
    max_interpolation : int
        Maximum consecutive missing frames

    Returns
    -------
    tuple of (str, str, str)
        Paths to the three output CSV files

    Examples
    --------
    >>> kp_path, met_path, detail_path = run_quality_control_batch(
    ...     'data/raw', 'data/qc', window_size=1800, overlap=0.0,
    ...     conf_threshold=0.3, max_interpolation=60
    ... )
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Get list of CSV files to process
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    # Initialize lists to collect results
    all_keypoint_dfs = []
    all_metric_dfs = []
    all_detail_dfs = []

    # Process each file with progress bar
    for filename in tqdm(csv_files, desc="QC Analysis"):
        filepath = os.path.join(input_dir, filename)

        try:
            # Analyze this file
            kp_df, met_df, detail_df = analyze_file_quality(
                filepath, window_size, overlap,
                conf_threshold, max_interpolation
            )

        except Exception as e:
            # If analysis fails, create error entries
            print(f"Error processing {filename}: {e}")

            kp_df = pd.DataFrame([{
                "file": filename,
                "keypoint": None,
                "bad_windows": None,
                "total_windows": None,
                "pct_bad": None,
                "error": str(e)
            }])

            met_df = pd.DataFrame([{
                "file": filename,
                "metric": None,
                "bad_windows": None,
                "total_windows": None,
                "pct_bad": None,
                "error": str(e)
            }])

            detail_df = pd.DataFrame([{
                "file": filename,
                "metric": None,
                "window_index": None,
                "start_frame": None,
                "end_frame_exclusive": None,
                "error": str(e)
            }])

        # Collect results
        all_keypoint_dfs.append(kp_df)
        all_metric_dfs.append(met_df)
        all_detail_dfs.append(detail_df)

    # Combine all results
    combined_keypoints = pd.concat(all_keypoint_dfs, ignore_index=True)
    combined_metrics = pd.concat(all_metric_dfs, ignore_index=True)
    combined_details = pd.concat(all_detail_dfs, ignore_index=True)

    # Define output paths
    keypoint_path = os.path.join(output_dir, "keypoint_bad_windows.csv")
    metric_path = os.path.join(output_dir, "metric_bad_windows.csv")
    detail_path = os.path.join(output_dir, "metric_bad_window_indices.csv")

    # Save results
    combined_keypoints.to_csv(keypoint_path, index=False)
    combined_metrics.to_csv(metric_path, index=False)

    # For details, ensure we write headers even if empty
    if combined_details.empty:
        # Write empty dataframe with proper headers
        empty_df = pd.DataFrame(columns=['file', 'metric', 'window_index', 'start_frame', 'end_frame_exclusive'])
        empty_df.to_csv(detail_path, index=False)
    else:
        combined_details.to_csv(detail_path, index=False)

    # Print summary
    print("\nQuality Control Results Saved:")
    print(f"  Keypoint statistics: {keypoint_path}")
    print(f"  Metric statistics: {metric_path}")
    print(f"  Bad window details: {detail_path}")

    return keypoint_path, metric_path, detail_path


def summarize_quality_control(metric_path: str) -> Tuple[int, int, float]:
    """
    Generate quick summary statistics from QC results.

    Parameters
    ----------
    metric_path : str
        Path to the metric_bad_windows.csv file

    Returns
    -------
    tuple of (int, int, float)
        total_bad_windows : Total number of bad windows across all files
        total_windows : Total number of windows analyzed
        percent_bad : Percentage of windows that are bad

    Examples
    --------
    >>> bad, total, pct = summarize_quality_control('qc/metric_bad_windows.csv')
    >>> print(f"QC Summary: {bad}/{total} bad windows ({pct:.2f}%)")
    QC Summary: 45/1200 bad windows (3.75%)
    """
    # Load the metric summary file
    df = pd.read_csv(metric_path)

    # Calculate totals
    total_bad = df['bad_windows'].sum()
    total_windows = df['total_windows'].sum()

    # Calculate percentage
    if total_windows > 0:
        percent_bad = (total_bad / total_windows) * 100
    else:
        percent_bad = float('nan')

    return total_bad, total_windows, percent_bad