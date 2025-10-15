"""
GSR processing utilities using NeuroKit2.

This module provides functions for:
- Loading Shimmer GSR data
- Cleaning EDA signals and detecting SCR peaks
- Extracting electrodermal activity (EDA) features
- Mapping GSR files to experimental conditions
"""

import numpy as np
import pandas as pd
import os
import re
import neurokit2 as nk
from pathlib import Path
from typing import Tuple, Optional

from .config import CFG


# --- Filename parsing functions ---

def parse_gsr_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse participant ID and session number from GSR filename.

    Args:
        filename: GSR filename (e.g., '3208_session01.csv')

    Returns:
        Tuple of (participant_id, session_number) or (None, None) if invalid

    Examples:
        >>> parse_gsr_filename('3208_session01.csv')
        ('3208', 1)
        >>> parse_gsr_filename('3208_session02.csv')
        ('3208', 2)
    """
    # Match pattern: <participant>_session<number>.csv
    pattern = r'^(\d+)_session(\d+)\.csv$'
    match = re.match(pattern, filename)

    if not match:
        return None, None

    participant_id = match.group(1)
    session_num = int(match.group(2))

    return participant_id, session_num


def map_session_to_condition(
    session_str: str,
    participant_id: str,
    condition_map: dict
) -> Optional[str]:
    """Map session string to condition using participant info.

    Args:
        session_str: Session string (e.g., "session01", "session02", "session03")
        participant_id: Participant ID
        condition_map: Nested dict from create_condition_mapping()
                      {participant_id: {trial_num: condition}}

    Returns:
        Condition code ('L', 'M', 'H') or None if not found

    Note:
        This is identical to the eye tracking and ECG implementation.
    """
    # Extract trial number from session string (session01 -> 1)
    trial_num = int(session_str.replace('session', '').lstrip('0') or '0')

    if participant_id not in condition_map:
        return None

    trial_map = condition_map[participant_id]

    if trial_num not in trial_map:
        return None

    return trial_map[trial_num]


# --- Data loading functions ---

def import_shimmer_eda_data(
    data_directory: str,
    participant_id: str = None,
    session_num: int = None,
    filename: str = None
) -> pd.DataFrame:
    """Import GSR data from Shimmer device.

    Loads EDA/GSR waveform data from Shimmer sensor.

    Args:
        data_directory: Directory containing Shimmer data files
        participant_id: Participant ID (e.g., '3208')
        session_num: Session number (1, 2, or 3)
        filename: Explicit filename (overrides participant/session)

    Returns:
        DataFrame with Shimmer GSR data

    Raises:
        FileNotFoundError: If required file is not found
        ValueError: If data format is invalid

    Examples:
        >>> import_shimmer_eda_data('/path/to/data', '3208', 1)
        # Loads 3208_session01.csv
    """
    # Determine filename
    if filename is None:
        if participant_id is None or session_num is None:
            raise ValueError("Must provide either (participant_id, session_num) or filename")
        filename = f"{participant_id}_session{session_num:02d}.csv"

    # Construct full path
    filepath = Path(data_directory) / filename

    # Check file exists
    if not filepath.exists():
        raise FileNotFoundError(f"GSR file not found: {filepath}")

    # Load GSR data
    # Note: Shimmer files have simple single-level headers
    shimmer_eda_df = pd.read_csv(filepath)

    # Validate required columns
    required_col = 'Shimmer_AD66_GSR_Skin_Conductance_CAL'
    if required_col not in shimmer_eda_df.columns:
        available_cols = [c for c in shimmer_eda_df.columns if 'GSR' in c or 'EDA' in c]
        raise ValueError(
            f"Missing required GSR column: {required_col}\n"
            f"Available GSR columns: {available_cols}"
        )

    # Parse timestamp column
    timestamp_col = 'Shimmer_AD66_Timestamp_FormattedUnix_CAL'
    if timestamp_col in shimmer_eda_df.columns:
        shimmer_eda_df['Time'] = pd.to_datetime(
            shimmer_eda_df[timestamp_col],
            format='%Y/%m/%d %H:%M:%S.%f'
        )
        shimmer_eda_df['Timestamp'] = shimmer_eda_df['Time'].dt.time

    return shimmer_eda_df


# --- EDA signal processing functions ---

def processing_eda_signal(
    eda_signal: np.ndarray,
    sampling_rate: int = None,
    method_clean: str = None,
    method_phasic: str = None,
    method_peak: str = None,
    plot_signal: bool = None
) -> Tuple[pd.DataFrame, dict]:
    """Process EDA signal: clean, decompose into SCR/SCL, detect peaks.

    Re-creates built-in nk.eda_process() function with custom mid-level functions
    that can be optimized with customized methods.

    Args:
        eda_signal: Raw EDA signal array
        sampling_rate: Sampling rate in Hz (default: from config)
        method_clean: Cleaning method (default: from config)
        method_phasic: Phasic/tonic decomposition method (default: from config)
        method_peak: Peak detection method (default: from config)
        plot_signal: Whether to plot signal (default: from config)

    Returns:
        Tuple of (signals DataFrame, info dict)

        signals DataFrame contains:
            - EDA_Raw: the raw signal
            - EDA_Clean: the cleaned signal
            - EDA_Tonic: the tonic component (SCL)
            - EDA_Phasic: the phasic component (SCR)
            - SCR_Onsets: peak onsets marked as "1" in list of zeros
            - SCR_Peaks: peaks marked as "1" in list of zeros
            - SCR_Height: SCR amplitude including tonic component
            - SCR_Amplitude: SCR amplitude excluding tonic component
            - SCR_RiseTime: time from onset to peak
            - SCR_Recovery: recovery points marked as "1"

        info dict contains:
            - SCR_Peaks: peak indices
            - SCR_Onsets: onset indices
            - sampling_rate: signals' sampling rate

    Processing Steps:
        1. CLEANING [nk.eda_clean]: Remove noise and smooth
        2. DECOMPOSE [nk.eda_phasic]: Separate SCR (phasic) and SCL (tonic)
        3. PEAK DETECTION [nk.eda_peaks]: Identify SCR peaks
    """
    # Use config defaults if not specified
    if sampling_rate is None:
        sampling_rate = CFG.SAMPLE_RATE
    if method_clean is None:
        method_clean = CFG.CLEANING_METHOD
    if method_phasic is None:
        method_phasic = CFG.PHASIC_METHOD
    if method_peak is None:
        method_peak = CFG.PEAK_METHOD
    if plot_signal is None:
        plot_signal = CFG.PLOT_SIGNALS

    # PROCESS STEPS: CLEAN, DECOMPOSE, PEAK DETECTION

    # CLEANING [eda_clean()]: detrend and filter
    eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=sampling_rate, method=method_clean)

    # DECOMPOSE [eda_phasic()]: Decompose EDA into Phasic (SCR) and Tonic (SCL)
    eda_decompose = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate, method=method_phasic)

    # DETECT SCR PEAKS [eda_peaks()]: Identify Skin Conductance Response peaks
    instant_peaks, rpeaks = nk.eda_peaks(
        eda_decompose['EDA_Phasic'],
        sampling_rate=sampling_rate,
        method=method_peak
    )

    # Store Output
    signals = pd.DataFrame({
        "EDA_Raw": np.squeeze(eda_signal),
        "EDA_Clean": eda_cleaned
    })

    signals = pd.concat([signals, eda_decompose, instant_peaks], axis=1)
    infos = rpeaks
    infos["sampling_rate"] = sampling_rate

    return signals, infos


# --- EDA feature extraction ---

def eda_feature_extraction(
    signals: pd.DataFrame,
    rpeaks: dict,
    sr: int = None,
    save_output_folder: str = ''
) -> pd.DataFrame:
    """Extract EDA features from processed signals.

    Uses NeuroKit2 to compute event-related and interval-related EDA features.

    Args:
        signals: DataFrame with processed EDA signals (from processing_eda_signal)
        rpeaks: Dictionary with peak information (from processing_eda_signal)
        sr: Sampling rate in Hz (default: from config)
        save_output_folder: Folder to save intermediate outputs (default: '')

    Returns:
        DataFrame with EDA features

    Features extracted:
        Event-related (< 10 seconds):
            - EDA_SCR: Whether SCR occurs (1 or 0)
            - SCR_Peak_Amplitude: Peak amplitude of first SCR
            - SCR_Peak_Amplitude_Time: Time of first SCR peak
            - SCR_RiseTime: Time from onset to peak
            - SCR_RecoveryTime: Time to half amplitude

        Interval-related (> 10 seconds):
            - SCR_Peaks_N: Number of SCR occurrences
            - SCR_Peaks_Amplitude_Mean: Mean amplitude of SCR peaks
            - EDA_Tonic_SD: Standard deviation of tonic component
            - EDA_Sympathetic: Sympathetic activity index (if duration > 64s)
            - EDA_Autocorrelation: Autocorrelation (if duration > 30s)

    See NeuroKit2 documentation for complete feature descriptions:
    https://neuropsychology.github.io/NeuroKit/functions/eda.html
    """
    # Use config defaults if not specified
    if sr is None:
        sr = CFG.SAMPLE_RATE

    if save_output_folder and not os.path.exists(save_output_folder):
        os.makedirs(save_output_folder)

    try:
        # Compute EDA features directly from signals
        # Use nk.eda_intervalrelated with properly formatted data

        # Create a properly formatted epochs structure
        # NeuroKit2 expects a 'Label' column for eda_intervalrelated
        data = signals.copy()
        data['Label'] = '1'  # Add Label column (required by NeuroKit2)

        # Create epochs dict (treating entire session as one epoch)
        epochs = {'1': data}

        # Compute Interval-related Features (> 10 seconds)
        # This is appropriate for session-level GSR data
        interval_features = nk.eda_intervalrelated(epochs, sampling_rate=sr)

        # Drop columns that are all NaN
        interval_features = interval_features.dropna(axis=1, how='all')

        # Remove 'Label' column (artifact of epoch processing)
        if 'Label' in interval_features.columns:
            interval_features = interval_features.drop('Label', axis=1)

        return interval_features

    except Exception as e:
        # If EDA analysis fails, return empty DataFrame
        print(f"Warning: EDA feature extraction failed: {e}")
        return pd.DataFrame()


# --- Windowing utilities ---

def windows_indices(n: int, win: int, hop: int) -> list:
    """Generate sliding window indices for time series analysis.

    Creates overlapping or non-overlapping windows across a sequence.

    Args:
        n: Total length of the sequence (number of samples)
        win: Window size in samples
        hop: Step size between windows in samples (hop < win creates overlap)

    Returns:
        List of tuples: (start_index, end_index, window_index)

    Example:
        >>> windows_indices(1200, 600, 300)  # 50% overlap
        [(0, 600, 0), (300, 900, 1), (600, 1200, 2)]
    """
    out = []
    w = 0
    start = 0

    while start + win <= n:
        out.append((start, start + win, w))
        start += hop
        w += 1

    return out


def extract_windowed_eda_features(
    signals: pd.DataFrame,
    window_seconds: int = None,
    overlap: float = None,
    sr: int = None
) -> pd.DataFrame:
    """Extract EDA features from windowed signal segments.

    Applies sliding windows to EDA signals and computes interval-related
    features for each window.

    Args:
        signals: DataFrame with processed EDA signals (from processing_eda_signal)
        window_seconds: Window size in seconds (default: from config)
        overlap: Window overlap fraction 0-1 (default: from config)
        sr: Sampling rate in Hz (default: from config)

    Returns:
        DataFrame with one row per window containing:
        - window_index: Window number
        - t_start_sec: Start time in seconds
        - t_end_sec: End time in seconds
        - All EDA interval-related features

    Example:
        For 60-second windows with 50% overlap on 200-second signal:
        - Window 0: 0-60s
        - Window 1: 30-90s
        - Window 2: 60-120s
        - etc.
    """
    # Use config defaults if not specified
    if sr is None:
        sr = CFG.SAMPLE_RATE
    if window_seconds is None:
        window_seconds = CFG.WINDOW_SECONDS
    if overlap is None:
        overlap = CFG.WINDOW_OVERLAP

    # Calculate window parameters in samples
    win_samples = int(window_seconds * sr)
    hop_samples = int(win_samples * (1 - overlap))

    # Generate window indices
    n_samples = len(signals)
    windows = windows_indices(n_samples, win_samples, hop_samples)

    if len(windows) == 0:
        print(f"Warning: Signal too short for windowing (need {win_samples} samples, have {n_samples})")
        return pd.DataFrame()

    # Extract features for each window
    window_features = []

    for start, end, widx in windows:
        # Extract window segment
        window_data = signals.iloc[start:end].copy()

        # Skip windows with insufficient data
        if len(window_data) < win_samples * 0.9:  # Allow 10% tolerance
            continue

        # Check for missing data in critical columns
        if window_data[['EDA_Clean', 'EDA_Phasic', 'EDA_Tonic']].isnull().any().any():
            continue

        try:
            # Create epoch structure for NeuroKit2
            window_data['Label'] = '1'
            epochs = {'1': window_data}

            # Extract interval-related features for this window
            features = nk.eda_intervalrelated(epochs, sampling_rate=sr)

            # Drop all-NaN columns
            features = features.dropna(axis=1, how='all')

            # Remove Label column if present
            if 'Label' in features.columns:
                features = features.drop('Label', axis=1)

            # Add window metadata
            features['window_index'] = widx
            features['t_start_sec'] = start / sr
            features['t_end_sec'] = end / sr

            window_features.append(features)

        except Exception as e:
            print(f"Warning: Feature extraction failed for window {widx}: {e}")
            continue

    if len(window_features) == 0:
        return pd.DataFrame()

    # Combine all windows
    result = pd.concat(window_features, ignore_index=True)

    # Reorder columns: metadata first, then features
    metadata_cols = ['window_index', 't_start_sec', 't_end_sec']
    feature_cols = [c for c in result.columns if c not in metadata_cols]
    result = result[metadata_cols + feature_cols]

    return result
