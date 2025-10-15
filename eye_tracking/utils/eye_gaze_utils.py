"""Eye tracking processing utilities.

Functions for loading, normalizing, and extracting metrics from eye tracking data.
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt, resample_poly
import re
from pathlib import Path
import sys

# Add parent directory to path for importing pose utilities
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import config
from .config import CFG, SCIPY_AVAILABLE

# Import pose condition mapping utilities
from Pose.utils.preprocessing_utils import (
    load_participant_info,
    create_condition_mapping
)
from Pose.utils.io_utils import load_participant_info_file


def load_file_data(directory, filename):
    """
    Load a single eye-tracking CSV file and extract participant/session info from the filename.

    Args:
        directory: Directory containing the CSV file
        filename: Name of the CSV file

    Returns:
        Dictionary with metadata and DataFrame, or None if file invalid
        Keys: file_name, participant_id, session_number, data

    Expected filename format: <participantID>_session<number>.csv
    Example: 3105_session01.csv → participant=3105, session=01
    """
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print(f'File {filename} not found in directory {directory}')
        return None

    if not filename.endswith('.csv'):
        return None

    # Parse filename: 3105_session01.csv → participant=3105, session=01
    parts = filename.replace('.csv', '').split('_')
    if len(parts) < 2:
        print(f"Invalid filename format: {filename}. Expected: <participantID>_session<number>.csv")
        return None

    participant_id = parts[0]
    session_str = parts[1]

    # Extract session number: "session01" → "01"
    if session_str.startswith('session'):
        session_number = session_str.replace('session', '')
    else:
        session_number = session_str

    # Load data
    df = pd.read_csv(filepath)
    if df.empty or df.shape[0] == 0:
        print(f"Empty data in {filename}")
        return None

    # Validate required columns
    required_columns = ['R Gaze X', 'R Gaze Y', 'L Gaze X', 'L Gaze Y',
                       'R Pupil Size', 'L Pupil Size', 'Time Stamp']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns in {filename}: {missing_cols}")
        return None

    # Convert to numeric
    columns_of_interest = ['R Gaze X', 'R Gaze Y', 'L Gaze X', 'L Gaze Y',
                          'R Pupil Size', 'L Pupil Size']
    for col in columns_of_interest:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    file_data = {
        'file_name': filename,
        'participant_id': participant_id,
        'session_number': session_number,
        'data': df
    }
    return file_data

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a Butterworth lowpass filter to a 1D numpy array.

    Args:
        data: Input signal array
        cutoff: Cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order (default: 4)

    Returns:
        Filtered array
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def median_clip(arr, k=None, nsd=None):
    """
    Apply a median filter and clip outliers beyond nsd standard deviations.

    Args:
        arr: Input array
        k: Kernel size for median filter (default: from config)
        nsd: Number of standard deviations for clipping (default: from config)

    Returns:
        Filtered array with outliers clipped
    """
    if k is None:
        k = CFG.MEDIAN_FILTER_KERNEL
    if nsd is None:
        nsd = CFG.OUTLIER_N_SD

    arr = medfilt(arr, kernel_size=k)
    m, s = np.nanmean(arr), np.nanstd(arr)
    np.clip(arr, m - nsd*s, m + nsd*s, out=arr)
    return arr

def normalize_gaze_to_screen(df, screen_width=None, screen_height=None):
    """
    Normalize gaze coordinates to [0,1] screen space and set out-of-bounds or bad pupil data to NaN.

    Args:
        df: DataFrame with eye tracking data
        screen_width: Screen width in pixels (default: from config)
        screen_height: Screen height in pixels (default: from config)

    Returns:
        Modified DataFrame with normalized coordinates
    """
    if screen_width is None:
        screen_width = CFG.SCREEN_WIDTH
    if screen_height is None:
        screen_height = CFG.SCREEN_HEIGHT

    # Pupil size outlier detection using config threshold
    for pupil_col in ['R Pupil Size', 'L Pupil Size']:
        if pupil_col in df.columns:
            threshold = CFG.PUPIL_OUTLIER_THRESH * df[pupil_col].std()
            df.loc[df[pupil_col].sub(df[pupil_col].mean()).abs() > threshold, pupil_col] = 0

    # Mask gaze when pupil is 0 (blink or bad tracking)
    df.loc[df['R Pupil Size'] == 0, ['R Gaze X', 'R Gaze Y']] = np.nan
    df.loc[df['L Pupil Size'] == 0, ['L Gaze X', 'L Gaze Y']] = np.nan

    # Mask out-of-bounds gaze coordinates
    df.loc[df['R Gaze X'] < 0, ['R Gaze X', 'R Gaze Y']] = np.nan
    df.loc[df['L Gaze X'] < 0, ['L Gaze X', 'L Gaze Y']] = np.nan
    df.loc[df['R Gaze Y'] < 0, ['R Gaze X', 'R Gaze Y']] = np.nan
    df.loc[df['L Gaze Y'] < 0, ['L Gaze X', 'L Gaze Y']] = np.nan
    df.loc[df['R Gaze X'] > screen_width, ['R Gaze X', 'R Gaze Y']] = np.nan
    df.loc[df['L Gaze X'] > screen_width, ['L Gaze X', 'L Gaze Y']] = np.nan
    df.loc[df['R Gaze Y'] > screen_height, ['R Gaze X', 'R Gaze Y']] = np.nan
    df.loc[df['L Gaze Y'] > screen_height, ['L Gaze X', 'L Gaze Y']] = np.nan

    # Normalize to [0, 1] range
    df['R Gaze X'] = df['R Gaze X'] / screen_width
    df['R Gaze Y'] = df['R Gaze Y'] / screen_height
    df['L Gaze X'] = df['L Gaze X'] / screen_width
    df['L Gaze Y'] = df['L Gaze Y'] / screen_height

    return df

def pupil_blink_detection(pr, pl, t, z_thresh=None, raw_floor=None, max_dur=None):
    """
    Detect blinks based on z-scored pupil size and a raw floor threshold.

    Args:
        pr: Right pupil size array
        pl: Left pupil size array
        t: Time array
        z_thresh: Z-score threshold for blink (default: from config)
        raw_floor: Minimum pupil size for valid data (default: from config)
        max_dur: Maximum blink duration in seconds (default: from config)

    Returns:
        Tuple of (blink_starts, blink_ends)
        - blink_starts: List of [start_time]
        - blink_ends: List of [start_time, end_time, duration]
    """
    if z_thresh is None:
        z_thresh = CFG.BLINK_Z_THRESH
    if raw_floor is None:
        raw_floor = CFG.BLINK_RAW_FLOOR
    if max_dur is None:
        max_dur = CFG.BLINK_MAX_DUR

    pup = (pr + pl) / 2
    z   = (pup - np.nanmean(pup)) / np.nanstd(pup)
    blink = (z < z_thresh) | (pup < raw_floor)
    blink = pd.Series(blink).rolling(5, center=True, min_periods=1).median().astype(bool)
    diff  = np.diff(blink.astype(int))
    starts, ends = np.where(diff == 1)[0] + 1, np.where(diff == -1)[0] + 1
    if blink.iloc[0]:  starts = np.insert(starts, 0, 0)
    if blink.iloc[-1]: ends   = np.append(ends, len(blink)-1)
    Sblk, Eblk = [], []
    for s, e in zip(starts, ends):
        dur = t[e] - t[s]
        if dur <= max_dur:
            Sblk.append([t[s]])
            Eblk.append([t[s], t[e], dur])
    return Sblk, Eblk

def remove_missing(x, y, t):
    """
    Remove samples where x or y is NaN, keeping t in sync.

    Args:
        x: X coordinates array
        y: Y coordinates array
        t: Time array

    Returns:
        Tuple of (x_filtered, y_filtered, t_filtered)
    """
    ok = ~(np.isnan(x) | np.isnan(y))
    return x[ok], y[ok], t[ok]

def fixation_detection(x, y, t, maxdist=None, mindur=None):
    """
    Detect fixations using a distance threshold and minimum duration.

    Args:
        x: Gaze X coordinates array
        y: Gaze Y coordinates array
        t: Time array
        maxdist: Maximum allowed distance for fixation in normalized units (default: from config)
        mindur: Minimum fixation duration in seconds (default: from config)

    Returns:
        Tuple of (fixation_starts, fixation_ends)
        - fixation_starts: List of [start_time]
        - fixation_ends: List of [start_time, end_time, duration, x_pos, y_pos]
    """
    if maxdist is None:
        maxdist = CFG.FIXATION_MAX_DIST
    if mindur is None:
        mindur = CFG.FIXATION_MIN_DUR

    x, y, t = remove_missing(x, y, t)
    Sfix, Efix, anchor, in_fix = [], [], 0, False
    for i in range(1, len(x)):
        dist = np.hypot(x[anchor] - x[i], y[anchor] - y[i])
        if dist <= maxdist and not in_fix:
            anchor, in_fix = i, True
            Sfix.append([t[i]])
        elif dist > maxdist and in_fix:
            in_fix = False
            if t[i-1] - Sfix[-1][0] >= mindur:
                Efix.append([Sfix[-1][0], t[i-1], t[i-1]-Sfix[-1][0], x[anchor], y[anchor]])
            else:
                Sfix.pop()
            anchor = i
        elif not in_fix:
            anchor += 1
    if len(Sfix) > len(Efix):
        Efix.append([Sfix[-1][0], t[-1], t[-1]-Sfix[-1][0], x[anchor], y[anchor]])
    return Sfix, Efix

def saccade_detection(x, y, t, minlen=None, vel_thr=None, acc_thr=None):
    """
    Detect saccades based on velocity and acceleration thresholds.

    Args:
        x: Gaze X coordinates array
        y: Gaze Y coordinates array
        t: Time array
        minlen: Minimum saccade length in samples (default: from config)
        vel_thr: Velocity threshold (default: from config)
        acc_thr: Acceleration threshold (default: from config)

    Returns:
        Tuple of (saccade_starts, saccade_ends, velocity, acceleration)
    """
    if minlen is None:
        minlen = CFG.SACCADE_MIN_LEN
    if vel_thr is None:
        vel_thr = CFG.SACCADE_VEL_THRESH
    if acc_thr is None:
        acc_thr = CFG.SACCADE_ACC_THRESH

    x, y, t = remove_missing(x, y, t)
    dt  = np.diff(t); vel = np.hypot(np.diff(x), np.diff(y)) / dt
    acc = np.diff(vel) / dt[1:]
    sacc = (vel[1:] > vel_thr) & (acc > acc_thr)
    sacc = np.insert(sacc, [0,0], [False, False])
    diff = np.diff(sacc.astype(int))
    starts, ends = np.where(diff == 1)[0] + 1, np.where(diff == -1)[0] + 1
    if sacc[0]:  starts = np.insert(starts, 0, 0)
    if sacc[-1]: ends   = np.append(ends, len(sacc)-1)
    Ssac, Esac = [], []
    for s, e in zip(starts, ends):
        if e - s >= minlen:
            Ssac.append([t[s]])
            Esac.append([t[s], t[e], t[e]-t[s]])
    return Ssac, Esac, vel, acc

def extract_eye_metrics(df, participant_id, condition,
                       sample_rate=None, win_sec=None, overlap_fr=None,
                       missing_max=None, trim_sec=0, ds_factor=1):
    """
    Extract windowed eye-tracking metrics (blinks, fixations, saccades, etc.) from a DataFrame.
    Sliding windows are used; metrics are computed for each window.

    Args:
        df: DataFrame with normalized eye tracking data
        participant_id: Participant ID string
        condition: Condition code (L, M, or H)
        sample_rate: Sampling rate in Hz (default: from config)
        win_sec: Window size in seconds (default: from config)
        overlap_fr: Window overlap fraction 0-1 (default: from config)
        missing_max: Maximum proportion of missing data per window (default: from config)
        trim_sec: Seconds to trim from start (default: 0)
        ds_factor: Downsampling factor (default: 1, no downsampling)

    Returns:
        DataFrame of metrics per window with columns:
        - participant: Participant ID
        - condition: Condition code
        - window_index: Window number
        - start_time: Window start time in seconds
        - end_time: Window end time in seconds
        - pct_missing: Percentage of missing data
        - mean_vel: Mean gaze velocity
        - max_vel: Maximum gaze velocity
        - mean_acc: Mean gaze acceleration
        - rms_disp: RMS dispersion
        - fix_count: Number of fixations
        - fix_mean_dur: Mean fixation duration
        - fix_rate: Fixation rate (per second)
        - blink_count: Number of blinks
        - blink_mean_dur: Mean blink duration
        - blink_rate: Blink rate (per second)
        - sac_count: Number of saccades
        - sac_mean_dur: Mean saccade duration
        - sac_rate: Saccade rate (per second)
    """
    # Use config values if not specified
    if sample_rate is None:
        sample_rate = CFG.SAMPLE_RATE
    if win_sec is None:
        win_sec = CFG.WINDOW_SECONDS
    if overlap_fr is None:
        overlap_fr = CFG.WINDOW_OVERLAP
    if missing_max is None:
        missing_max = CFG.MISSING_MAX

    metrics = []
    FPS = sample_rate
    WIN_SAMPLES = win_sec * FPS
    STEP = int(WIN_SAMPLES * (1 - overlap_fr))
    TRIM_SAMPLES = trim_sec * FPS

    # Extract data arrays
    Rx = df['R Gaze X'].to_numpy()
    Ry = df['R Gaze Y'].to_numpy()
    Lx = df['L Gaze X'].to_numpy()
    Ly = df['L Gaze Y'].to_numpy()
    pr = df['R Pupil Size'].to_numpy()
    pl = df['L Pupil Size'].to_numpy()
    ts = df['Time Stamp'].to_numpy()

    # Downsample if ds_factor > 1
    if ds_factor > 1:
        Rx = resample_poly(Rx, up=1, down=ds_factor)
        Ry = resample_poly(Ry, up=1, down=ds_factor)
        Lx = resample_poly(Lx, up=1, down=ds_factor)
        Ly = resample_poly(Ly, up=1, down=ds_factor)
        pr = resample_poly(pr, up=1, down=ds_factor)
        pl = resample_poly(pl, up=1, down=ds_factor)
        ts = resample_poly(ts, up=1, down=ds_factor)
        FPS = int(FPS / ds_factor)
        win_samples = win_sec * FPS
        step = int(win_samples * (1 - overlap_fr))

    # Valid eye logic
    valid_L = pl != 0
    valid_R = pr != 0
    both_ok = valid_L & valid_R

    gx = np.where(both_ok, 0.5*(Lx+Rx),
                  np.where(valid_L, Lx,
                           np.where(valid_R, Rx, np.nan)))
    gy = np.where(both_ok, 0.5*(Ly+Ry),
                  np.where(valid_L, Ly,
                           np.where(valid_R, Ry, np.nan)))

    # Step 1: Blink detection (before filtering)
    Sblk_all, Eblk_all = pupil_blink_detection(pr, pl, ts)

    # Step 2: Fill missing gaze
    gx_filled = pd.Series(gx).interpolate(limit_direction='both').to_numpy()
    gy_filled = pd.Series(gy).interpolate(limit_direction='both').to_numpy()

    # Step 3: Filter (median clip) after interpolation
    gx_clean = median_clip(gx_filled.copy())
    gy_clean = median_clip(gy_filled.copy())

    # Step 4: Sliding Window Metrics
    win_starts = np.arange(0, ts[-1] - win_sec + 1, win_sec * (1 - overlap_fr))
    w_idx = 0
    for win_start in win_starts:
        win_end = win_start + win_sec
        idx = np.where((ts >= win_start) & (ts < win_end))[0]
        if len(idx) == 0:
            w_idx += 1
            continue
        t_win  = ts[idx]
        pl_win = pl[idx]; pr_win = pr[idx]
        gx_win = gx_clean[idx].copy(); gy_win = gy_clean[idx].copy()

        # QC: missing proportion
        missing = (~valid_L[idx]) & (~valid_R[idx])
        if missing.mean() > missing_max:
            w_idx += 1
            continue

        # Blink events in this window
        Sblk = [b for b in Sblk_all if t_win[0] <= b[0] <= t_win[-1]]
        Eblk = [b for b in Eblk_all if t_win[0] <= b[0] <= t_win[-1]]

        # Velocity & acceleration
        dt  = np.diff(t_win)
        vel = np.hypot(np.diff(gx_win), np.diff(gy_win)) / dt if len(dt) > 0 else np.array([])
        acc = np.diff(vel) / dt[1:] if len(dt) > 1 else np.array([])

        # Dispersion
        gx_raw = gx_clean[idx]; gy_raw = gy_clean[idx]
        rms_disp = np.sqrt(np.nanmean((gx_raw-np.nanmean(gx_raw))**2 +
                                      (gy_raw-np.nanmean(gy_raw))**2))

        # Fixations & saccades
        _, Efix          = fixation_detection(gx_win, gy_win, t_win)
        Ssac, Esac, _, _ = saccade_detection(gx_win, gy_win, t_win)

        dur = t_win[-1] - t_win[0] if len(t_win) > 1 else 0

        metrics.append({
            'participant'    : participant_id,
            'condition'      : condition,
            'window_index'   : w_idx,
            'start_time'     : t_win[0],
            'end_time'       : t_win[-1],
            'pct_missing'    : 100 * missing.mean(),
            'mean_vel'       : np.nanmean(vel) if len(vel) > 0 else np.nan,
            'max_vel'        : np.nanmax(vel) if len(vel) > 0 else np.nan,
            'mean_acc'       : np.nanmean(acc) if len(acc) > 0 else np.nan,
            'rms_disp'       : rms_disp,
            'fix_count'      : len(Efix),
            'fix_mean_dur'   : np.nanmean([f[2] for f in Efix]) if Efix else np.nan,
            'fix_rate'       : len(Efix) / dur if dur > 0 else np.nan,
            'blink_count'    : len(Eblk),
            'blink_mean_dur' : np.nanmean([b[2] for b in Eblk]) if Eblk else np.nan,
            'blink_rate'     : len(Eblk) / dur if dur > 0 else np.nan,
            'sac_count'      : len(Esac),
            'sac_mean_dur'   : np.nanmean([s[2] for s in Esac]) if Esac else np.nan,
            'sac_rate'       : len(Esac) / dur if dur > 0 else np.nan,
        })
        w_idx += 1
    return pd.DataFrame(metrics)

def plot_data(original_df, filtered_df, columns, down_sample=10, sample_rate=1000, filename='data', dir='dir', save=False):
    """
    Plot original and filtered gaze/pupil data for visual inspection.

    Args:
        original_df: Original DataFrame
        filtered_df: Filtered DataFrame
        columns: List of columns to plot
        down_sample: Downsampling factor for filtered data
        sample_rate: Data sample rate in Hz
        filename: Filename for saving if save=True
        dir: Directory for saving if save=True
        save: Whether to save the plot
    """
    import matplotlib.pyplot as plt
    print('Plotting data...')
    original_time = np.arange(len(original_df)) / sample_rate
    filtered_time = np.arange(len(filtered_df)) * down_sample / sample_rate
    plt.figure(figsize=(10, 8))
    for i, column in enumerate(columns, 1):
        plt.subplot(len(columns), 2, 2*i-1)
        plt.plot(original_time, original_df[column], label='Original')
        plt.xlabel('Time (s)')
        plt.ylabel(column)
        plt.title(f'{column} - Original')
        plt.legend()
        plt.subplot(len(columns), 2, 2*i)
        plt.plot(filtered_time, filtered_df[column], label='Filtered', linestyle='-')
        plt.xlabel('Time (s)')
        plt.ylabel(column)
        plt.title(f'{column} - Filtered')
        plt.legend()
        if 'Gaze' in column:
            plt.ylim(0, 1)
        if save:
            plt.savefig( dir + '/ts_plots/' + filename.split('.')[0] + '.jpeg')
    plt.tight_layout()
    plt.show()
