"""
================================================================================
TEMPORAL FILTERING MODULE FOR FACIAL POSE DATA
================================================================================

This module provides functions for applying temporal filters to smooth noisy
facial landmark data. The primary filter is a Butterworth low-pass filter that
removes high-frequency noise while preserving the underlying signal.

Key features:
- Zero-phase filtering (no temporal shift)
- Preserves NaN patterns exactly
- Handles edge cases gracefully
- Configurable filter parameters

Author: Pose Analysis Pipeline
Date: 2024
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from typing import Optional, List, Tuple


# ============================================================================
# BUTTERWORTH LOW-PASS FILTER
# ============================================================================

def apply_butterworth_filter(df: pd.DataFrame,
                            sampling_rate: float = 60.0,
                            cutoff_frequency: float = 10.0,
                            filter_order: int = 4,
                            columns: Optional[List[str]] = None,
                            verbose: bool = True) -> pd.DataFrame:
    """
    Apply Butterworth low-pass filter to smooth temporal data.

    This filter removes high-frequency noise while preserving underlying
    signal trends. Uses zero-phase filtering to avoid temporal shift.

    Key features:
    - Preserves original NaN locations exactly
    - Internally fills gaps for filter stability only
    - Handles short sequences gracefully
    - Applies to all numeric columns by default

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series data to filter
    sampling_rate : float, optional
        Sampling rate in Hz (frames per second), default: 60.0
    cutoff_frequency : float, optional
        Cutoff frequency in Hz (frequencies above this are attenuated), default: 10.0
    filter_order : int, optional
        Filter order (higher = steeper rolloff), default: 4
    columns : list of str, optional
        Specific columns to filter. If None, filters all numeric columns
    verbose : bool, optional
        Whether to print filtering statistics, default: True

    Returns
    -------
    pd.DataFrame
        DataFrame with filtered data, original NaN patterns preserved

    Examples
    --------
    >>> df = pd.read_csv('features.csv')
    >>> df_smooth = apply_butterworth_filter(df, cutoff_frequency=5.0)
    >>> # Only filter specific columns
    >>> df_smooth = apply_butterworth_filter(df, columns=['center_face_x', 'center_face_y'])
    """
    # Make a copy to avoid modifying original
    df_filtered = df.copy()

    # Determine which columns to filter
    if columns is None:
        # Filter all numeric columns
        numeric_cols = [col for col in df.columns
                       if pd.api.types.is_numeric_dtype(df[col])]
    else:
        # Filter specified columns (verify they're numeric)
        numeric_cols = [col for col in columns
                       if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        if verbose and len(numeric_cols) < len(columns):
            missing = set(columns) - set(numeric_cols)
            print(f"Warning: Skipping non-numeric or missing columns: {missing}")

    if not numeric_cols:
        if verbose:
            print("Warning: No numeric columns to filter")
        return df_filtered

    # Design Butterworth filter
    # Normalize frequency to Nyquist frequency (half the sampling rate)
    nyquist_freq = sampling_rate / 2.0
    normalized_cutoff = cutoff_frequency / nyquist_freq

    # Clamp to valid range (must be between 0 and 1)
    normalized_cutoff = min(max(normalized_cutoff, 1e-6), 0.999999)

    # Create second-order sections representation (more numerically stable)
    sos = butter(filter_order, normalized_cutoff, btype='low', output='sos')

    # Track filtering results for reporting
    stats = {
        'all_nan': [],      # Columns with all NaN values
        'too_short': [],    # Columns too short to filter
        'failed': [],       # Columns where filtering failed
        'filtered': []      # Successfully filtered columns
    }

    # Process each column
    for col in numeric_cols:
        # Get column data as numpy array
        data = df_filtered[col].to_numpy(dtype=float, copy=True)

        # Track which values are finite (not NaN or inf)
        finite_mask = np.isfinite(data)

        # Skip if no valid data
        if not finite_mask.any():
            stats['all_nan'].append(col)
            continue

        # Check minimum length requirement
        # Need enough points for stable filtering
        min_length = max(25, 4 * filter_order + 5)
        if len(data) < min_length:
            stats['too_short'].append(col)
            continue

        # Create filled version for filtering
        # (Filter can't handle NaN values internally)
        data_filled = data.copy()

        if not finite_mask.all():
            # Need to fill gaps for filtering
            indices = np.arange(len(data))
            finite_indices = indices[finite_mask]
            finite_values = data[finite_mask]

            # Linear interpolation for internal gaps
            data_filled[~finite_mask] = np.interp(
                indices[~finite_mask],
                finite_indices,
                finite_values
            )

            # Constant extrapolation at edges
            first_valid = finite_indices[0]
            last_valid = finite_indices[-1]
            data_filled[:first_valid] = data_filled[first_valid]
            data_filled[last_valid + 1:] = data_filled[last_valid]

        # Apply zero-phase filtering
        try:
            # sosfiltfilt applies the filter twice (forward and backward)
            # This results in zero phase shift but doubles the filter order
            filtered_data = sosfiltfilt(sos, data_filled, padtype='odd')

            # Restore original NaN locations
            filtered_data[~finite_mask] = np.nan

            # Update DataFrame
            df_filtered[col] = filtered_data
            stats['filtered'].append(col)

        except Exception as e:
            # Filtering failed, keep original data
            stats['failed'].append((col, str(e)))
            if verbose:
                print(f"Warning: Failed to filter {col}: {e}")

    # Print summary if verbose
    if verbose:
        print(f"\nButterworth Filter Summary:")
        print(f"  Parameters: fs={sampling_rate}Hz, fc={cutoff_frequency}Hz, order={filter_order}")
        print(f"  Successfully filtered: {len(stats['filtered'])} columns")

        if stats['all_nan']:
            print(f"  All NaN (unchanged): {len(stats['all_nan'])} columns")
        if stats['too_short']:
            print(f"  Too short (unchanged): {len(stats['too_short'])} columns")
        if stats['failed']:
            print(f"  Failed (unchanged): {len(stats['failed'])} columns")

    return df_filtered


# ============================================================================
# MOVING AVERAGE FILTER
# ============================================================================

def apply_moving_average(df: pd.DataFrame,
                        window_size: int = 5,
                        columns: Optional[List[str]] = None,
                        center: bool = True,
                        min_periods: int = 1) -> pd.DataFrame:
    """
    Apply simple moving average filter to smooth data.

    A simpler alternative to Butterworth filtering. Uses a sliding window
    to average values, reducing noise but with less control than frequency
    domain filtering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series data to smooth
    window_size : int, optional
        Size of the moving average window in frames, default: 5
    columns : list of str, optional
        Specific columns to filter. If None, filters all numeric columns
    center : bool, optional
        Whether to center the window (True) or use trailing window (False), default: True
    min_periods : int, optional
        Minimum number of valid values required in window, default: 1

    Returns
    -------
    pd.DataFrame
        DataFrame with smoothed data

    Examples
    --------
    >>> df = pd.read_csv('features.csv')
    >>> df_smooth = apply_moving_average(df, window_size=7)
    """
    # Make a copy to avoid modifying original
    df_filtered = df.copy()

    # Determine which columns to filter
    if columns is None:
        numeric_cols = [col for col in df.columns
                       if pd.api.types.is_numeric_dtype(df[col])]
    else:
        numeric_cols = [col for col in columns
                       if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    # Apply rolling average to each column
    for col in numeric_cols:
        df_filtered[col] = df[col].rolling(
            window=window_size,
            center=center,
            min_periods=min_periods
        ).mean()

    return df_filtered


# ============================================================================
# MEDIAN FILTER
# ============================================================================

def apply_median_filter(df: pd.DataFrame,
                       window_size: int = 5,
                       columns: Optional[List[str]] = None,
                       center: bool = True,
                       min_periods: int = 1) -> pd.DataFrame:
    """
    Apply median filter to remove impulse noise.

    Median filtering is particularly effective at removing spike noise
    while preserving edges better than mean filters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series data to filter
    window_size : int, optional
        Size of the median filter window in frames, default: 5
    columns : list of str, optional
        Specific columns to filter. If None, filters all numeric columns
    center : bool, optional
        Whether to center the window (True) or use trailing window (False), default: True
    min_periods : int, optional
        Minimum number of valid values required in window, default: 1

    Returns
    -------
    pd.DataFrame
        DataFrame with filtered data

    Examples
    --------
    >>> df = pd.read_csv('features.csv')
    >>> df_clean = apply_median_filter(df, window_size=3)
    """
    # Make a copy to avoid modifying original
    df_filtered = df.copy()

    # Determine which columns to filter
    if columns is None:
        numeric_cols = [col for col in df.columns
                       if pd.api.types.is_numeric_dtype(df[col])]
    else:
        numeric_cols = [col for col in columns
                       if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    # Apply rolling median to each column
    for col in numeric_cols:
        df_filtered[col] = df[col].rolling(
            window=window_size,
            center=center,
            min_periods=min_periods
        ).median()

    return df_filtered


# ============================================================================
# FILTER COMPARISON AND SELECTION
# ============================================================================

def compare_filter_effects(df: pd.DataFrame,
                          column: str,
                          sampling_rate: float = 60.0) -> dict:
    """
    Compare the effects of different filters on a single column.

    Useful for determining which filter and parameters work best for your data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    column : str
        Column name to analyze
    sampling_rate : float, optional
        Sampling rate in Hz, default: 60.0

    Returns
    -------
    dict
        Dictionary with filtered versions and statistics

    Examples
    --------
    >>> comparison = compare_filter_effects(df, 'center_face_x')
    >>> for filter_name, stats in comparison.items():
    ...     print(f"{filter_name}: SNR={stats['snr']:.2f}")
    """
    results = {}

    # Original signal
    original = df[column].copy()
    results['original'] = {
        'data': original,
        'std': original.std(),
        'mean': original.mean()
    }

    # Butterworth with different cutoffs
    for cutoff in [5.0, 10.0, 15.0]:
        filtered = apply_butterworth_filter(
            df[[column]],
            sampling_rate=sampling_rate,
            cutoff_frequency=cutoff,
            verbose=False
        )[column]

        noise = original - filtered
        snr = 10 * np.log10(filtered.var() / noise.var()) if noise.var() > 0 else np.inf

        results[f'butterworth_{cutoff}Hz'] = {
            'data': filtered,
            'std': filtered.std(),
            'mean': filtered.mean(),
            'snr': snr
        }

    # Moving average with different windows
    for window in [3, 5, 7]:
        filtered = apply_moving_average(
            df[[column]],
            window_size=window
        )[column]

        noise = original - filtered
        snr = 10 * np.log10(filtered.var() / noise.var()) if noise.var() > 0 else np.inf

        results[f'moving_avg_{window}'] = {
            'data': filtered,
            'std': filtered.std(),
            'mean': filtered.mean(),
            'snr': snr
        }

    # Median filter
    for window in [3, 5]:
        filtered = apply_median_filter(
            df[[column]],
            window_size=window
        )[column]

        noise = original - filtered
        snr = 10 * np.log10(filtered.var() / noise.var()) if noise.var() > 0 else np.inf

        results[f'median_{window}'] = {
            'data': filtered,
            'std': filtered.std(),
            'mean': filtered.mean(),
            'snr': snr
        }

    return results


# ============================================================================
# ADAPTIVE FILTERING
# ============================================================================

def apply_adaptive_filter(df: pd.DataFrame,
                         columns: Optional[List[str]] = None,
                         noise_threshold: float = 2.0) -> pd.DataFrame:
    """
    Apply adaptive filtering based on local signal characteristics.

    This filter adapts its behavior based on local signal properties,
    applying stronger filtering in noisy regions and preserving details
    in stable regions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time series data
    columns : list of str, optional
        Columns to filter. If None, filters all numeric columns
    noise_threshold : float, optional
        Threshold for detecting noisy regions (in standard deviations), default: 2.0

    Returns
    -------
    pd.DataFrame
        DataFrame with adaptively filtered data

    Examples
    --------
    >>> df = pd.read_csv('features.csv')
    >>> df_adaptive = apply_adaptive_filter(df)
    """
    df_filtered = df.copy()

    # Determine columns to filter
    if columns is None:
        numeric_cols = [col for col in df.columns
                       if pd.api.types.is_numeric_dtype(df[col])]
    else:
        numeric_cols = [col for col in columns
                       if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    for col in numeric_cols:
        data = df[col].copy()

        # Calculate local variance using rolling window
        local_std = data.rolling(window=15, center=True, min_periods=1).std()
        global_std = data.std()

        # Identify noisy regions (high local variance)
        is_noisy = local_std > noise_threshold * global_std

        # Apply different filtering based on noise level
        smooth_data = data.copy()

        # Strong filtering for noisy regions
        if is_noisy.any():
            noisy_filtered = apply_median_filter(
                pd.DataFrame({col: data}),
                window_size=5,
                columns=[col]
            )[col]
            smooth_data[is_noisy] = noisy_filtered[is_noisy]

        # Light filtering for stable regions
        stable_mask = ~is_noisy
        if stable_mask.any():
            stable_filtered = apply_moving_average(
                pd.DataFrame({col: data}),
                window_size=3,
                columns=[col]
            )[col]
            smooth_data[stable_mask] = stable_filtered[stable_mask]

        df_filtered[col] = smooth_data

    return df_filtered