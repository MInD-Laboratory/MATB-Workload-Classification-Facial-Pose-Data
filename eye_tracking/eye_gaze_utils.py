import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt, resample_poly
import re

def load_file_data(directory, filename):
    """
    Load a single eye-tracking CSV file and extract participant/session info from the filename.
    Returns a dict with metadata and the loaded DataFrame, or None if file not found or empty.
    """
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print(f'File {filename} not found in directory {directory}')
        return None
    if filename.endswith('.csv'):
        parts = filename.replace('.csv', '').split('_')
        participant_id = parts[0]
        session_number = parts[1].replace('session', '')
        condition = parts[2] if len(parts) > 2 else None
        df = pd.read_csv(filepath)
        if df.empty or df.shape[0] == 0:
            return None
        columns_of_interest = ['R Gaze X', 'R Gaze Y', 'L Gaze X', 'L Gaze Y', 'R Pupil Size', 'L Pupil Size']
        for col in columns_of_interest:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        file_data = {
            'file_name': filename,
            'participant_id': participant_id,
            'session_number': session_number,
            'condition': condition,
            'data': df
        }
    return file_data

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a Butterworth lowpass filter to a 1D numpy array.
    cutoff: cutoff frequency (Hz)
    fs: sampling frequency (Hz)
    order: filter order
    Returns filtered array.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def median_clip(arr, k=5, nsd=5):
    """
    Apply a median filter and clip outliers beyond nsd standard deviations.
    k: kernel size for median filter
    nsd: number of standard deviations for clipping
    Returns filtered array.
    """
    arr = medfilt(arr, kernel_size=k)
    m, s = np.nanmean(arr), np.nanstd(arr)
    np.clip(arr, m - nsd*s, m + nsd*s, out=arr)
    return arr

def normalize_gaze_to_screen(df,screen_width=2560,screen_height=1440):
    """
    Normalize gaze coordinates to [0,1] screen space and set out-of-bounds or bad pupil data to NaN.
    Modifies and returns the input DataFrame.
    """
    for pupil_col in ['R Pupil Size', 'L Pupil Size']:
        if pupil_col in df.columns:
            df.loc[df[pupil_col].sub(df[pupil_col].mean()).abs() > 3 * df[pupil_col].std(), pupil_col] = 0
    df.loc[df['R Pupil Size'] == 0, ['R Gaze X', 'R Gaze Y']] = np.nan
    df.loc[df['L Pupil Size'] == 0, ['L Gaze X', 'L Gaze Y']] = np.nan
    df.loc[df['R Gaze X'] < 0, ['R Gaze X', 'R Gaze Y']] = np.nan
    df.loc[df['L Gaze X'] < 0, ['L Gaze X', 'L Gaze Y']] = np.nan
    df.loc[df['R Gaze Y'] < 0, ['R Gaze X', 'R Gaze Y']] = np.nan
    df.loc[df['L Gaze Y'] < 0, ['L Gaze X', 'L Gaze Y']] = np.nan
    df.loc[df['R Gaze X'] > screen_width, ['R Gaze X', 'R Gaze Y']] = np.nan
    df.loc[df['L Gaze X'] > screen_width, ['L Gaze X', 'L Gaze Y']] = np.nan
    df.loc[df['R Gaze Y'] > screen_height, ['R Gaze X', 'R Gaze Y']] = np.nan
    df.loc[df['L Gaze Y'] > screen_height, ['L Gaze X', 'L Gaze Y']] = np.nan
    df['R Gaze X'] = df['R Gaze X'] / screen_width
    df['R Gaze Y'] = df['R Gaze Y'] / screen_height
    df['L Gaze X'] = df['L Gaze X'] / screen_width
    df['L Gaze Y'] = df['L Gaze Y'] / screen_height
    return df

def pupil_blink_detection(pr, pl, t, z_thresh=-2, raw_floor=30, max_dur=0.6):
    """
    Detect blinks based on z-scored pupil size and a raw floor threshold.
    pr, pl: right/left pupil size arrays
    t: time array
    z_thresh: z-score threshold for blink
    raw_floor: minimum pupil size for valid data
    max_dur: maximum blink duration (seconds)
    Returns lists of blink start and end events.
    """
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
    Returns filtered x, y, t arrays.
    """
    ok = ~(np.isnan(x) | np.isnan(y))
    return x[ok], y[ok], t[ok]

def fixation_detection(x, y, t, maxdist=1.0, mindur=0.20):
    """
    Detect fixations using a distance threshold and minimum duration.
    x, y, t: gaze coordinates and time arrays
    maxdist: maximum allowed distance for fixation (in normalized units)
    mindur: minimum fixation duration (seconds)
    Returns lists of fixation start and end events.
    """
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

def saccade_detection(x, y, t, minlen=15, vel_thr=5, acc_thr=50):
    """
    Detect saccades based on velocity and acceleration thresholds.
    x, y, t: gaze coordinates and time arrays
    minlen: minimum saccade length (samples)
    vel_thr: velocity threshold
    acc_thr: acceleration threshold
    Returns lists of saccade start/end events, velocity, and acceleration arrays.
    """
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

def extract_eye_metrics(df, sample_rate=1000, win_sec=60, overlap_fr=0.5, missing_max=0.25, trim_sec=0, ds_factor=1, butter_cutoff=50):
    """
    Extract windowed eye-tracking metrics (blinks, fixations, saccades, etc.) from a DataFrame.
    Sliding windows are used; metrics are computed for each window.
    Returns a DataFrame of metrics per window.
    """
    metrics = []
    FPS = sample_rate
    WIN_SAMPLES = win_sec * FPS
    STEP = int(WIN_SAMPLES * (1 - overlap_fr))
    TRIM_SAMPLES = trim_sec * FPS

    # Update to new column names
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
        # Adjust FPS for windowing
        FPS = int(FPS / ds_factor)
        win_samples = win_sec * FPS
        step = int(win_samples * (1 - overlap_fr))

    # valid‐eye logic
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

    # Step 2: Fill missing gaze (excluding blink NaNs if needed)
    gx_filled = pd.Series(gx).interpolate(limit_direction='both').to_numpy()
    gy_filled = pd.Series(gy).interpolate(limit_direction='both').to_numpy()

    # Step 3: Filter (median clip) after interpolation
    gx_clean = median_clip(gx_filled.copy())
    gy_clean = median_clip(gy_filled.copy())

    # Step 4: Sliding Window Metrics (using time stamp)
    # Windowing: all windows are 60s long, 50% overlap, first and last part only overlap once
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

        # QC: missing proportion (both eyes missing)
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
        _, Efix          = fixation_detection(gx_win, gy_win, t_win, maxdist=0.02, mindur=0.20)
        Ssac, Esac, _, _ = saccade_detection(gx_win, gy_win, t_win, minlen=2, vel_thr=0.5, acc_thr=5)

        dur = t_win[-1] - t_win[0] if len(t_win) > 1 else 0

        metrics.append({
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
    columns: list of columns to plot
    down_sample: downsampling factor for filtered data
    sample_rate: data sample rate (Hz)
    filename, dir: for saving plots if save=True
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