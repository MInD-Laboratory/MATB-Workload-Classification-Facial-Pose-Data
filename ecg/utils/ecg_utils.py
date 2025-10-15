"""
ECG processing utilities using NeuroKit2.

This module provides functions for:
- Loading Zephyr ECG data
- Cleaning ECG signals and detecting R-peaks
- Extracting heart rate variability (HRV) features
- Mapping ECG files to experimental conditions
"""

import numpy as np
import pandas as pd
import os
import re
import warnings
import neurokit2 as nk
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional

from .config import CFG

# Suppress NeuroKit2 warnings about DFA_alpha2 (expected for short windows)
warnings.filterwarnings('ignore', message='.*DFA_alpha2.*')


# --- Utility functions ---

def remove_brackets(val):
    """Remove brackets from a string representation of a list or value."""
    if isinstance(val, str):
        return val.replace('[', '').replace(']', '')
    return val


def find_files_with_substring(folder, substring):
    """Return a list of file paths in 'folder' containing 'substring' in the filename."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if substring in f]


# --- Filename parsing functions ---

def parse_ecg_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse participant ID and session number from ECG filename.

    Args:
        filename: ECG filename (e.g., '3208_ecg_session01.csv' or '3208_summary_session01.csv')

    Returns:
        Tuple of (participant_id, session_number) or (None, None) if invalid

    Examples:
        >>> parse_ecg_filename('3208_ecg_session01.csv')
        ('3208', 1)
        >>> parse_ecg_filename('3208_summary_session02.csv')
        ('3208', 2)
    """
    # Match pattern: <participant>_<ecg|summary>_session<number>.csv
    pattern = r'^(\d+)_(?:ecg|summary)_session(\d+)\.csv$'
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
        This is identical to the eye tracking implementation.
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

def import_zephyr_ecg_data(
    data_directory: str,
    participant_id: str = None,
    session_num: int = None,
    ecg_filename: str = None,
    summary_filename: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Import ECG data from Zephyr BioHarness device.

    Loads both the ECG waveform data and summary data (HR, BR, posture, activity).

    Args:
        data_directory: Directory containing Zephyr data files
        participant_id: Participant ID (e.g., '3208')
        session_num: Session number (1, 2, or 3)
        ecg_filename: Explicit ECG filename (overrides participant/session)
        summary_filename: Explicit summary filename (overrides participant/session)

    Returns:
        Tuple of (ecg_dataframe, summary_dataframe)

    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If data format is invalid

    Examples:
        >>> import_zephyr_ecg_data('/path/to/data', '3208', 1)
        # Loads 3208_ecg_session01.csv and 3208_summary_session01.csv
    """
    # Determine filenames
    if ecg_filename is None or summary_filename is None:
        if participant_id is None or session_num is None:
            raise ValueError("Must provide either (participant_id, session_num) or (ecg_filename, summary_filename)")
        ecg_filename = f"{participant_id}_ecg_session{session_num:02d}.csv"
        summary_filename = f"{participant_id}_summary_session{session_num:02d}.csv"

    # Construct full paths
    ecg_path = Path(data_directory) / ecg_filename
    summary_path = Path(data_directory) / summary_filename

    # Check files exist
    if not ecg_path.exists():
        raise FileNotFoundError(f"ECG file not found: {ecg_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    # Load summary data
    zephyr_summary_df = pd.read_csv(summary_path, index_col='Time')
    zephyr_summary_df['Time'] = pd.to_datetime(zephyr_summary_df.index, format='%d/%m/%Y %H:%M:%S.%f')
    zephyr_summary_df['Timestamp'] = zephyr_summary_df['Time'].dt.time
    zephyr_summary_df.index = pd.to_datetime(zephyr_summary_df.index, dayfirst=True)

    # Drop columns not needed for analysis (if they exist)
    cols_to_drop = ['SkinTemp', 'GSR', 'BatteryVolts', 'ROGState', 'ROGTime',
                    'LinkQuality', 'RSSI', 'TxPower', 'Ext.Status',
                    'BatteryLevel', 'AuxADC1', 'AuxADC2', 'AuxADC3']
    cols_to_drop = [col for col in cols_to_drop if col in zephyr_summary_df.columns]
    if cols_to_drop:
        zephyr_summary_df = zephyr_summary_df.drop(cols_to_drop, axis=1)

    # Load ECG waveform
    zephyr_ecg_df = pd.read_csv(ecg_path, index_col='Time')
    zephyr_ecg_df['Time'] = pd.to_datetime(zephyr_ecg_df.index, format='%d/%m/%Y %H:%M:%S.%f', dayfirst=True)
    zephyr_ecg_df['Timestamp'] = zephyr_ecg_df['Time'].dt.time
    zephyr_ecg_df.index = pd.to_datetime(zephyr_ecg_df.index, format='%d/%m/%Y %H:%M:%S.%f', dayfirst=True)

    return zephyr_ecg_df, zephyr_summary_df


# --- ECG signal processing functions ---

def processing_ecg_signal(
    ecg_signal: np.ndarray,
    sampling_rate: int = None,
    method_peak: str = None,
    method_clean: str = None,
    method_quality: str = None,
    approach_quality: str = None,
    interpolation_method: str = None,
    plot_signal: bool = None,
    plot_fix: bool = False,
    output_folder: str = ''
) -> Tuple[pd.DataFrame, dict]:
    """Process ECG signal: clean, detect R-peaks, calculate HR, assess quality.

    Re-creates built-in nk.ecg_process() function with custom mid-level functions
    that can be optimized with customized methods.

    Args:
        ecg_signal: Raw ECG signal array
        sampling_rate: Sampling rate in Hz (default: from config)
        method_peak: R-peak detection method (default: from config)
        method_clean: Cleaning method (default: from config)
        method_quality: Quality assessment method (default: from config)
        approach_quality: Quality approach (default: from config)
        interpolation_method: HR interpolation method (default: from config)
        plot_signal: Whether to plot signal (default: from config)
        plot_fix: Whether to plot peak fixing (default: False)
        output_folder: Folder to save plots (default: '')

    Returns:
        Tuple of (signals DataFrame, rpeaks dict)

        signals DataFrame contains:
            - ECG_Raw: the raw signal
            - ECG_Clean: the cleaned signal
            - ECG_Rate: heart rate interpolated between R-peaks
            - ECG_Quality: the quality of the cleaned signal
            - ECG_R_Peaks: the R-peaks marked as "1" in a list of zeros
            - ECG_Quality_RPeaks: the quality of the rpeaks
            - ECG_Quality_RPeaksUncorrected: the quality of the uncorrected rpeaks

        rpeaks dict contains:
            - ECG_R_Peaks: corrected R-peak locations
            - ECG_R_Peaks_Uncorrected: original R-peak locations
            - sampling_rate: signals' sampling rate

    Processing Steps:
        1. CLEANING [nk.ecg_clean]: Detrend and filter
        2. SIGNAL QUALITY [nk.ecg_quality]: Quality of the cleaned signal
        3. R-PEAK DETECTION [nk.ecg_peaks]: Find R-peaks
        4. FIX PEAKS [nk.signal_fixpeaks]: Correct R-peaks (Kubios method)
        5. HEART RATE EXTRACTION [nk.ecg_rate]: Calculate HR from peaks
    """
    # Use config defaults if not specified
    if sampling_rate is None:
        sampling_rate = CFG.SAMPLE_RATE
    if method_peak is None:
        method_peak = CFG.PEAK_METHOD
    if method_clean is None:
        method_clean = CFG.CLEANING_METHOD
    if method_quality is None:
        method_quality = CFG.QUALITY_METHOD
    if approach_quality is None:
        approach_quality = CFG.QUALITY_APPROACH
    if interpolation_method is None:
        interpolation_method = CFG.INTERPOLATION_METHOD
    if plot_signal is None:
        plot_signal = CFG.PLOT_SIGNALS

    # PROCESS STEPS: CLEAN, PEAK DETECTION, HEART-RATE, SIGNAL QUALITY ASSESSMENT

    # CLEANING [ecg_clean()]: detrend and filter
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method=method_clean)

    # SIGNAL QUALITY [ecg_quality]: Quality of the cleaned signal
    quality = nk.ecg_quality(ecg_cleaned, sampling_rate=sampling_rate,
                             method=method_quality, approach=approach_quality)

    # Store Output of cleaning
    signals = pd.DataFrame({"ECG_Raw": ecg_signal,
                            "ECG_Clean": ecg_cleaned,
                            "ECG_Quality": quality})
    signals = pd.concat([signals], axis=1)
    infos = pd.DataFrame([])
    infos["sampling_rate"] = sampling_rate

    # Detect R peaks and extract features
    if method_peak is not None:
        # R-PEAK DETECTION [ecg_peaks]
        instant_peaks, rpeaks, = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate,
                                              method=method_peak, show=False)

        # FIX-PEAKS [signal_fixpeaks]: Correct R-Peaks
        info_correct, rpeaks_corrected = nk.signal_fixpeaks(
            rpeaks, sampling_rate=sampling_rate, iterative=True, method="Kubios", show=plot_fix)
        rpeaks['ECG_R_Peaks_Uncorrected'] = rpeaks['ECG_R_Peaks']
        rpeaks['ECG_R_Peaks'] = rpeaks_corrected

        # HEART RATE EXTRACTION [ecg_rate]
        rate = nk.ecg_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned),
                           interpolation_method=interpolation_method, show=False)

        # Store Output features
        try:
            quality_rpeak = nk.ecg_quality(ecg_cleaned, sampling_rate=sampling_rate,
                                          rpeaks=rpeaks['ECG_R_Peaks'],
                                          method=method_quality, approach=approach_quality)
        except ValueError:
            quality_rpeak = np.nan

        try:
            quality_rpeak_uncorrected = nk.ecg_quality(ecg_cleaned, sampling_rate=sampling_rate,
                                                       rpeaks=rpeaks['ECG_R_Peaks_Uncorrected'],
                                                       method=method_quality, approach=approach_quality)
        except ValueError:
            quality_rpeak_uncorrected = np.nan

        signals = pd.DataFrame({"ECG_Raw": ecg_signal,
                                "ECG_Clean": ecg_cleaned,
                                "ECG_Rate": rate,
                                "ECG_Quality": quality,
                                "ECG_Quality_RPeaks": quality_rpeak,
                                "ECG_Quality_RPeaksUncorrected": quality_rpeak_uncorrected})
        signals = pd.concat([signals, instant_peaks], axis=1)
        infos = rpeaks
        infos.update(info_correct)
        infos["sampling_rate"] = sampling_rate

    return signals, infos


def ecg_feature_extraction(
    signals: pd.DataFrame,
    rpeaks: dict,
    sr: int = None,
    save_output_folder: str = '',
    baseline_correction: bool = None
) -> pd.DataFrame:
    """Extract heart rate variability (HRV) features from ECG signals.

    Uses NeuroKit2 to compute HRV indices in time, frequency, and non-linear domains.

    Args:
        signals: DataFrame with processed ECG signals (from processing_ecg_signal)
        rpeaks: Dictionary with R-peak information (from processing_ecg_signal)
        sr: Sampling rate in Hz (default: from config)
        save_output_folder: Folder to save intermediate outputs (default: '')
        baseline_correction: Apply baseline correction (default: from config)

    Returns:
        DataFrame with HRV features

    Features extracted:
        - Time domain: MeanNN, SDNN, RMSSD, pNN50, etc.
        - Frequency domain: LF, HF, LFHF ratio, VLF, etc.
        - Non-linear: SD1, SD2, entropy measures, fractal dimensions, etc.

    See NeuroKit2 documentation for complete feature descriptions:
    https://neuropsychology.github.io/NeuroKit/functions/hrv.html
    """
    # Use config defaults if not specified
    if sr is None:
        sr = CFG.SAMPLE_RATE
    if baseline_correction is None:
        baseline_correction = CFG.BASELINE_CORRECTION

    if save_output_folder and not os.path.exists(save_output_folder):
        os.makedirs(save_output_folder)

    # Compute HRV features using individual functions
    # This approach is more robust than using nk.hrv() directly
    try:
        # Extract R-peak indices
        peaks = rpeaks.get('ECG_R_Peaks', [])
        if len(peaks) == 0:
            print("Warning: No R-peaks found")
            return pd.DataFrame()

        # Compute individual HRV domain features
        # Time domain features
        hrv_time = nk.hrv_time(peaks, sampling_rate=sr, show=False)

        # Frequency domain features
        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sr, show=False)

        # Non-linear domain features
        hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=sr, show=False)

        # Combine all features into single DataFrame
        hrv_features = pd.concat([hrv_time, hrv_freq, hrv_nonlinear], axis=1)

        # Clean up the output
        # Remove any list/array columns that might have brackets
        for col in hrv_features.columns:
            if hrv_features[col].dtype == object:
                hrv_features[col] = hrv_features[col].apply(remove_brackets)

        # Drop columns that are all NaN
        hrv_features = hrv_features.dropna(axis=1, how='all')

        return hrv_features

    except Exception as e:
        # If HRV analysis fails (e.g., too few peaks), return empty DataFrame
        print(f"Warning: HRV feature extraction failed: {e}")
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
        >>> windows_indices(15360, 15360, 7680)  # 60s windows with 50% overlap at 256 Hz
        [(0, 15360, 0), (7680, 23040, 1), ...]
    """
    out = []
    w = 0
    start = 0

    while start + win <= n:
        out.append((start, start + win, w))
        start += hop
        w += 1

    return out


def extract_windowed_hrv_features(
    signals: pd.DataFrame,
    rpeaks: dict,
    window_seconds: int = None,
    overlap: float = None,
    sr: int = None
) -> pd.DataFrame:
    """Extract HRV features from windowed ECG signal segments.

    Applies sliding windows to ECG signals and computes HRV features
    for each window.

    Args:
        signals: DataFrame with processed ECG signals (from processing_ecg_signal)
        rpeaks: Dictionary with R-peak information (from processing_ecg_signal)
        window_seconds: Window size in seconds (default: from config)
        overlap: Window overlap fraction 0-1 (default: from config)
        sr: Sampling rate in Hz (default: from config)

    Returns:
        DataFrame with one row per window containing:
        - window_index: Window number
        - t_start_sec: Start time in seconds
        - t_end_sec: End time in seconds
        - All HRV features (time, frequency, non-linear domains)

    Example:
        For 60-second windows with 50% overlap on 300-second signal:
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

    # Get all R-peak locations
    all_rpeaks = rpeaks.get('ECG_R_Peaks', [])
    if len(all_rpeaks) == 0:
        print("Warning: No R-peaks found in signal")
        return pd.DataFrame()

    # Extract features for each window
    window_features = []

    for start, end, widx in windows:
        # Find R-peaks within this window
        window_rpeaks = [r for r in all_rpeaks if start <= r < end]

        # Need at least 5 R-peaks for meaningful HRV analysis
        if len(window_rpeaks) < 5:
            continue

        # Convert to local indices (relative to window start)
        window_rpeaks_local = [r - start for r in window_rpeaks]

        try:
            # Extract HRV features for this window
            # Time domain
            hrv_time = nk.hrv_time(window_rpeaks_local, sampling_rate=sr, show=False)

            # Frequency domain
            hrv_freq = nk.hrv_frequency(window_rpeaks_local, sampling_rate=sr, show=False)

            # Non-linear domain
            hrv_nonlinear = nk.hrv_nonlinear(window_rpeaks_local, sampling_rate=sr, show=False)

            # Combine all features
            features = pd.concat([hrv_time, hrv_freq, hrv_nonlinear], axis=1)

            # Clean up object columns
            for col in features.columns:
                if features[col].dtype == object:
                    features[col] = features[col].apply(remove_brackets)

            # Drop all-NaN columns
            features = features.dropna(axis=1, how='all')

            # Add window metadata
            features['window_index'] = widx
            features['t_start_sec'] = start / sr
            features['t_end_sec'] = end / sr

            window_features.append(features)

        except Exception as e:
            print(f"Warning: HRV extraction failed for window {widx}: {e}")
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
