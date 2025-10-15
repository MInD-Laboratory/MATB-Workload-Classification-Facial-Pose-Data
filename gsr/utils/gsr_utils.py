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
