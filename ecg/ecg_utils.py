
import numpy as np
import pandas as pd
import os
import neurokit2 as nk
import matplotlib.pyplot as plt

# --- Utility functions ---
def remove_brackets(val):
    """Remove brackets from a string representation of a list or value."""
    if isinstance(val, str):
        return val.replace('[', '').replace(']', '')
    return val

def find_files_with_substring(folder, substring):
    """Return a list of file paths in 'folder' containing 'substring' in the filename."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if substring in f]


# Function to import ecg data
def import_zephyr_ecg_data(subfolder_Zephyr_data):
    # Load Zephyr ECG summary data
    substring = '_Summary.csv'
    zephyr_summary_filename = find_files_with_substring(subfolder_Zephyr_data, substring)[0]
    zephyr_summary_df = pd.read_csv(zephyr_summary_filename, index_col='Time')
    zephyr_summary_df['Time'] = pd.to_datetime(zephyr_summary_df.index, format='%d/%m/%Y %H:%M:%S.%f')
    zephyr_summary_df['Timestamp'] = zephyr_summary_df['Time'].dt.time
    zephyr_summary_df.index = pd.to_datetime(zephyr_summary_df.index, dayfirst=True)  # Convert timestamp columns to datetime
    zephyr_summary_df = zephyr_summary_df.drop(['SkinTemp', 'GSR', 'BatteryVolts', 'ROGState', 'ROGTime',
                                                'LinkQuality', 'RSSI', 'TxPower', 'Ext.Status',
                                                'BatteryLevel', 'AuxADC1', 'AuxADC2', 'AuxADC3'], axis=1)
    # ECG waveform
    substring = '_ECG.csv'
    zephyr_ecg_filename = find_files_with_substring(subfolder_Zephyr_data, substring)[-1]
    zephyr_ecg_df = pd.read_csv(zephyr_ecg_filename, index_col='Time')
    zephyr_ecg_df['Time'] = pd.to_datetime(zephyr_ecg_df.index, format='%d/%m/%Y %H:%M:%S.%f',dayfirst=True)
    zephyr_ecg_df['Timestamp'] = zephyr_ecg_df['Time'].dt.time
    zephyr_ecg_df.index = pd.to_datetime(zephyr_ecg_df.index, format='%d/%m/%Y %H:%M:%S.%f',dayfirst=True)  # Convert timestamp columns to datetime

    return zephyr_ecg_df, zephyr_summary_df


# function to process ecg signal: clean, detect R peaks, and generate heart rate signal
def processing_ecg_signal(ecg_signal, sampling_rate=250, method_peak='engzeemod2012', method_clean="engzeemod2012",
                          method_quality="averageQRS", approach_quality="fuzzy", interpolation_method='monotone_cubic',
                          plot_signal=True, plot_fix=False, output_folder=''):
    """

 Re-create built-in nk.ecg_process() function with a custom process fucntion that calls “mid-level functions” such as:
 ecg_clean(), ecg_peaks(), ecg_rate() ect.
 Each function can be optimised with customised methods


    Procedure to process ECG signal: clean, quality metric, detect R peaks, HR
Returns:
        signals (DataFrame) – A DataFrame of the same length as the ecg_signal containing the following columns:
            "ECG_Raw": the raw signal.
            "ECG_Clean": the cleaned signal.
            "ECG_Rate": heart rate interpolated between R-peaks.
            "ECG_Quality": the quality of the cleaned signal
            "ECG_R_Peaks": the R-peaks marked as “1” in a list of zeros.
            "ECG_Quality_RPeaks": the quality of the rpeaks
            "ECG_Quality_RPeaksUncorrected": the quality of the rpeaks uncorrected

        rpeaks (dict) – A dictionary containing the samples at which the R-peaks occur:
                        "ECG_R_Peaks":
                        "ECG_R_Peaks_Uncorrected"
                        "sampling_rate" :signals’ sampling rate.

PROCESSES:

    *CLEANING [nk.ecg_clean]:
        methods: 'Neurokit', 'biosppy', 'pantompkins1985', 'hamilton2002', 'elgendi2010', 'engzeemod2012', 'vg'


    *R-PEAK DETECTION [nk.ecg_peaks]:
        Find R-peaks in the ECG Signal with ecg_peaks()
        ecg_peaks(ecg_cleaned, sampling_rate=1000, method='neurokit', correct_artifacts=False, show=False, **kwargs)

        R-peak detection methods:
            - neurokit (default), pantompkins1985, hamilton2002, zong2003, martinez2004, christov2004, gamboa2008,
              elgendi2010, engzeemod2012, manikandan2012, kalidas2017, nabian2018, rodrigues2021, promac

        'correct_artifacts' (bool) – identify and fix artifacts, using the method by Lipponen & Tarvainen (2019).


    *FIX PEAKS [nk.signal_fixpeaks]:
        methods: "Kubios" or "neurokit"
        [Note: "Kubios" only for peaks in ECG or PPG. "neurokit" can be used with peaks in ECG, PPG, or respiratory data.]


    *SIGNAL QUALITY [nk.ecg_quality]:
        methods: "averageQRS" (default) or "zhao2018" and approach: "simple" or "fuzzy"


    *HEART RATE EXTRACTION [nk.ecg_rate]:
        ecg_rate(peaks, sampling_rate=1000, desired_length=None, interpolation_method='monotone_cubic', show=False)

        Calculate signal rate (per minute) from a series of peaks.

        'desired_length' (int) – If left at the default None, the returned rated will have the same number of elements as peaks
        To interpolate the rate over the entire duration of the signal, set desired_length to the number of samples in the signal.

        'method' - "monotone_cubic" is chosen as the default interpolation method since it ensures monotone interpolation
        between data points (i.e., it prevents physiologically implausible “overshoots” or “undershoots” in the y-direction).
        iterpolation methods:
        "linear", "nearest", "zero", "slinear", "quadratic", "cubic", "previous", "next", "monotone_cubic"

    """

    # PROCESS STEPS: CLEAN, PEAK DETECTION, HEART-RATE, SIGNAL QUALITY ASSESMENT
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
        instant_peaks, rpeaks, = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method=method_peak, show=False)
        # FIX-PEAKS [signal_fixpeaks]: Correct R-Peaks [used in nk.ecg_peaks with correct_artifacts = True)
        info_correct, rpeaks_corrected = nk.signal_fixpeaks(
            rpeaks, sampling_rate=sampling_rate, iterative=True, method="Kubios", show=plot_fix)
        rpeaks['ECG_R_Peaks_Uncorrected'] = rpeaks['ECG_R_Peaks']
        rpeaks['ECG_R_Peaks'] = rpeaks_corrected

        # HEART RATE EXTRACTION [ecg_rate]
        rate = nk.ecg_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned),
                           interpolation_method=interpolation_method, show=False)

        # Store Output features
        try:
            quality_rpeak = nk.ecg_quality(ecg_cleaned, sampling_rate=sampling_rate, rpeaks=rpeaks['ECG_R_Peaks'],
                                           method=method_quality, approach=approach_quality)
        except ValueError:
            quality_rpeak = np.NaN
        try:
            quality_rpeak_uncorrected = nk.ecg_quality(ecg_cleaned, sampling_rate=sampling_rate,
                                                       rpeaks=rpeaks['ECG_R_Peaks_Uncorrected'],
                                                       method=method_quality, approach=approach_quality)
        except ValueError:
            quality_rpeak_uncorrected = np.NaN
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


def ecg_feature_extraction(epochs, sr, save_output_folder, baseline_correction):
    """NeuroKit2 to compute heart rate variability (HRV) indices in the time-, frequency-, and non-linear domain.
    https://url.au.m.mimecastprotect.com/s/j5sECwV1jpSG8p92PHVfOsJXEcP?domain=neuropsychology.github.io
    https://url.au.m.mimecastprotect.com/s/ARYmCxngGkf1MOwzBHvhksy9WoP?domain=neuropsychology.github.io
    HRV is the temporal variation between consecutive heartbeats (RR intervals).
    use peaks (heartbeat peaks occurences), as the input to the HRV functions to extract the indices.

        *HRV ANALYSIS [nk.hrv]:

        hrv(peaks, sampling_rate=1000, show=False, **kwargs)

        outputs: DataFrame – Contains HRV indices in a DataFrame.
                 If RSP data was provided (e.g., output of bio_process()), RSA indices will also be included.

        This function computes all HRV indices available in NeuroKit.
        It is essentially a convenience function that aggregates results from the time domain, frequency domain, and non-linear domain.

            [sub-domains]

            Time-domain features [hrv_time()]:
            Reflect the total variability of HR. The time-domain indices can be categorized into deviation-based and
            calculated directly from the normal beat-to-beat intervals  (normal RR intervals or NN intervals) and
            difference-based indices derived from the difference between successive NN intervals.

            DataFrame – Contains time domain HRV metrics:
                MeanNN: The mean of the RR intervals.
                SDNN: The standard deviation of the RR intervals.
                SDANN1, SDANN2, SDANN5: The standard deviation of average RR intervals extracted from n-minute segments of time series data (1, 2 and 5 by default). Note that these indices require a minimal duration of signal to be computed (3, 6 and 15 minutes respectively) and will be silently skipped if the data provided is too short.
                SDNNI1, SDNNI2, SDNNI5: The mean of the standard deviations of RR intervals extracted from n-minute segments of time series data (1, 2 and 5 by default). Note that these indices require a minimal duration of signal to be computed (3, 6 and 15 minutes respectively) and will be silently skipped if the data provided is too short.
                RMSSD: The square root of the mean of the squared successive differences between adjacent RR intervals. It is equivalent (although on another scale) to SD1, and therefore it is redundant to report correlations with both (Ciccone, 2017).
                SDSD: The standard deviation of the successive differences between RR intervals.
                CVNN: The standard deviation of the RR intervals (SDNN) divided by the mean of the RR intervals (MeanNN).
                CVSD: The root mean square of successive differences (RMSSD) divided by the mean of the RR intervals (MeanNN).
                MedianNN: The median of the RR intervals.
                MadNN: The median absolute deviation of the RR intervals.
                MCVNN: The median absolute deviation of the RR intervals (MadNN) divided by the median of the RR intervals (MedianNN).
                IQRNN: The interquartile range (IQR) of the RR intervals.
                SDRMSSD: SDNN / RMSSD, a time-domain equivalent for the low Frequency-to-High Frequency (LF/HF) Ratio (Sollers et al., 2007).
                Prc20NN: The 20th percentile of the RR intervals (Han, 2017; Hovsepian, 2015).
                Prc80NN: The 80th percentile of the RR intervals (Han, 2017; Hovsepian, 2015).
                pNN50: The proportion of RR intervals greater than 50ms, out of the total number of RR intervals.
                pNN20: The proportion of RR intervals greater than 20ms, out of the total number of RR intervals.
                MinNN: The minimum of the RR intervals (Parent, 2019; Subramaniam, 2022).
                MaxNN: The maximum of the RR intervals (Parent, 2019; Subramaniam, 2022).
                TINN: A geometrical parameter of the HRV, or more specifically, the baseline width of the RR intervals distribution obtained by triangular interpolation, where the error of least squares determines the triangle. It is an approximation of the RR interval distribution.
                HTI: The HRV triangular index, measuring the total number of RR intervals divided by the height of the RR intervals histogram.

            Frequency-domain features [hrv_frequency()]:
            Involve extracting for example the spectral power density pertaining to different frequency bands. Again, you can visualize the power across frequency bands by specifying show=True in hrv_frequency().
                Computes frequency domain HRV metrics, such as the power in different frequency bands.
                ULF: The spectral power of ultra low frequencies (by default, .0 to .0033 Hz). Very long signals are required for this to index to be extracted, otherwise, will return NaN.
                VLF: The spectral power of very low frequencies (by default, .0033 to .04 Hz).
                LF: The spectral power of low frequencies (by default, .04 to .15 Hz).
                HF: The spectral power of high frequencies (by default, .15 to .4 Hz).
                VHF: The spectral power of very high frequencies (by default, .4 to .5 Hz).
                TP: The total spectral power.
                LFHF: The ratio obtained by dividing the low frequency power by the high frequency power.
                LFn: The normalized low frequency, obtained by dividing the low frequency power by the total power.
                HFn: The normalized high frequency, obtained by dividing the low frequency power by the total power.
                LnHF: The log transformed HF.

                Note that a minimum duration of the signal containing the peaks is recommended for some HRV indices to be meaningful.
                For instance, 1, 2 and 5 minutes of high quality signal are the recommended minima for HF, LF and LF/HF, respectively.

            Non-linear features  [hrv_nonlinear()]:
            This function computes non-linear indices, which include features derived from the Poincaré plot,
            as well as other complexity() indices corresponding to entropy or fractal dimension.

                The Poincaré plot is a graphical representation of each NN interval plotted against its preceding NN interval. The ellipse that emerges is a visual quantification of the correlation between successive NN intervals.

                Basic indices derived from the Poincaré plot analysis include:
                SD1: Standard deviation perpendicular to the line of identity. It is an index of short-term RR interval fluctuations, i.e., beat-to-beat variability. It is equivalent (although on another scale) to RMSSD, and therefore it is redundant to report correlation with both.
                SD2: Standard deviation along the identity line. Index of long-term HRV changes.
                SD1/SD2: ratio of SD1 to SD2. Describes the ratio of short term to long term variations in HRV.
                S: Area of ellipse described by SD1 and SD2 (pi * SD1 * SD2). It is proportional to SD1SD2.
                CSI: The Cardiac Sympathetic Index (Toichi, 1997) is a measure of cardiac sympathetic function independent of vagal activity, calculated by dividing the longitudinal variability of the Poincaré plot (4*SD2) by its transverse variability (4*SD1).
                CVI: The Cardiac Vagal Index (Toichi, 1997) is an index of cardiac parasympathetic function (vagal activity unaffected by sympathetic activity), and is equal equal to the logarithm of the product of longitudinal (4*SD2) and transverse variability (4*SD1).
                CSI_Modified: The modified CSI (Jeppesen, 2014) obtained by dividing the square of the longitudinal variability by its transverse variability.

                Indices of Heart Rate Asymmetry (HRA), i.e., asymmetry of the Poincaré plot (Yan, 2017), include:
                GI: Guzik’s Index, defined as the distance of points above line of identity (LI) to LI divided by the distance of all points in Poincaré plot to LI except those that are located on LI.
                SI: Slope Index, defined as the phase angle of points above LI divided by the phase angle of all points in Poincaré plot except those that are located on LI.
                AI: Area Index, defined as the cumulative area of the sectors corresponding to the points that are located above LI divided by the cumulative area of sectors corresponding to all points in the Poincaré plot except those that are located on LI.
                PI: Porta’s Index, defined as the number of points below LI divided by the total number of points in Poincaré plot except those that are located on LI.
                SD1d and SD1a: short-term variance of contributions of decelerations (prolongations of RR intervals) and accelerations (shortenings of RR intervals), respectively (Piskorski, 2011)
                C1d and C1a: the contributions of heart rate decelerations and accelerations to short-term HRV, respectively (Piskorski, 2011).
                SD2d and SD2a: long-term variance of contributions of decelerations (prolongations of RR intervals) and accelerations (shortenings of RR intervals), respectively (Piskorski, 2011).
                C2d and C2a: the contributions of heart rate decelerations and accelerations to long-term HRV, respectively (Piskorski, 2011).
                SDNNd and SDNNa: total variance of contributions of decelerations (prolongations of RR intervals) and accelerations (shortenings of RR intervals), respectively (Piskorski, 2011).
                Cd and Ca: the total contributions of heart rate decelerations and accelerations to HRV.
                Indices of Heart Rate Fragmentation (Costa, 2017) include:
                PIP: Percentage of inflection points of the RR intervals series.
                IALS: Inverse of the average length of the acceleration/deceleration segments.
                PSS: Percentage of short segments.
                PAS: Percentage of NN intervals in alternation segments.

                Indices of Complexity and Fractal Physiology include:
                ApEn: See entropy_approximate().
                SampEn: See entropy_sample().
                ShanEn: See entropy_shannon().
                FuzzyEn: See entropy_fuzzy().
                MSEn: See entropy_multiscale().
                CMSEn: See entropy_multiscale().
                RCMSEn: See entropy_multiscale().
                CD: See fractal_correlation().
                HFD: See fractal_higuchi() (with kmax set to "default").
                KFD: See fractal_katz().
                LZC: See complexity_lempelziv().
                DFA_alpha1: The monofractal detrended fluctuation analysis of the HR signal, corresponding to short-term correlations. See fractal_dfa().
                DFA_alpha2: The monofractal detrended fluctuation analysis of the HR signal, corresponding to long-term correlations. See fractal_dfa().
                MFDFA indices: Indices related to the multifractal spectrum.

    """

    if not os.path.exists(f'{save_output_folder}'):
        os.makedirs(f'{save_output_folder}')

    # Compute Event-Related Features [<10seconds]
    # ------------------------------------------------------------------------------------------------------------------
    '''
    EVENT-RELATED FEATURES --------------------- typically for <10 second analysis
    [ecg_eventrelated()]: ECG analysis  will return a dataframe of the features
        - *ECG_Rate_Max: Max. HR 
        - *ECG_Rate_Min: Min. HR
        - *ECG_Rate_Mean: Mean HR
        - *ECG_Rate_SD: SD of HR
        - *ECG_Rate_Max_Time: Time Max. HR occurs
        - *ECG_Rate_Min_Time: Time Min. HR occurs
        - ECG_Phase_Atrial: Indicates whether onset of even concurs with respiratory systole(1) or diastole(0) (atrium is top of L/R heart chambers)
        - ECG_Phase_Ventricular: Indicates whether onset of event concurs with respiratory systole(1) or diastole(0) (ventricle is bottom of L/R heart chambers)
        - ECG_Phase_Atrial_Completion: Indicates the stage of current cardiac (atrial) phase (0 to 1) at the onset of the event
        - ECG_Phase_Ventricular_Completion: Indicates the stage of the current cardiac (ventricular) phase (0 to 1) at the onset of the event
        It also provides features relating to models...
        - ECG_Rate_Trend_Linear: Linear trend parameter
        - ECG_Rate_Trend_Quadratic: Quadratic curvature parameter
        - ECG_Rate_Trend_R2: Quality of the quadratic model (low = less meaningful/reliable)
    To be able to use ecg_eventrelated() ECG analysis, an array or list of onset of events is required.

    You can create a dictionary of events using events_create() and a list/array of onsets (optional labels and conditions)
    # events_create(event_onsets, event_durations=None, event_labels=None, event_conditions=None)

    OR

    You can find events in a continuous signal with events_find() such as with a photosensor.
    # events_find(event_channel, threshold='auto', threshold_keep='above', start_at=0, end_at=None, duration_min=1, 
                duration_max=None, inter_min=0, discard_first=0, discard_last=0, event_labels=None, event_conditions=None)

    '''

    event_features = nk.ecg_eventrelated(epochs)
    event_features = event_features.dropna(axis=1, how='all')
    plt.close('all')
    if baseline_correction is False:
        event_features["ECG_Rate_Max"] = event_features["ECG_Rate_Max"] + event_features["ECG_Rate_Baseline"]
        event_features["ECG_Rate_Min"] = event_features["ECG_Rate_Min"] + event_features["ECG_Rate_Baseline"]
        event_features["ECG_Rate_Mean"] = event_features["ECG_Rate_Mean"] + event_features["ECG_Rate_Baseline"]

    # Compute Interval-RELATED Features [>10seconds]
    # ------------------------------------------------------------------------------------------------------------------
    '''
    [nk.ecg_intervalrelated]: analyze longer periods of data (i.e., greater than 10 seconds)
    mean signal rate, variability metrices pertaining to heart rate variability (HRV) 
    https://url.au.m.mimecastprotect.com/s/HUjHCyoj8PurXnkpJFQiQsxjqN2?domain=neuropsychology.github.io
    INTERVAL-RELATED FEATURES --------------------- typically for >10 second analysis
    Output (ECG_Rate_mean & ECG_HRV [different HRV metrics]):
        'ECG_Rate_Mean', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1',
           'HRV_SDANN2', 'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD',
           'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN',
           'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN', 'HRV_Prc80NN',
           'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI',
           'HRV_TINN', 'HRV_ULF', 'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF',
           'HRV_TP', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1',
           'HRV_SD2', 'HRV_SD1SD2', 'HRV_S', 'HRV_CSI', 'HRV_CVI',
           'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS', 'HRV_PAS',
           'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a',
           'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a',
           'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_DFA_alpha1',
           'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
           'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
           'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry',
           'HRV_MFDFA_alpha1_Fluctuation', 'HRV_MFDFA_alpha1_Increment',
           'HRV_DFA_alpha2', 'HRV_MFDFA_alpha2_Width', 'HRV_MFDFA_alpha2_Peak',
           'HRV_MFDFA_alpha2_Mean', 'HRV_MFDFA_alpha2_Max',
           'HRV_MFDFA_alpha2_Delta', 'HRV_MFDFA_alpha2_Asymmetry',
           'HRV_MFDFA_alpha2_Fluctuation', 'HRV_MFDFA_alpha2_Increment',
           'HRV_ApEn', 'HRV_SampEn', 'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn',
           'HRV_CMSEn', 'HRV_RCMSEn', 'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC'
    '''
    interval_features = nk.ecg_intervalrelated(epochs, sampling_rate=sr)
    interval_features = interval_features.applymap(remove_brackets)
    interval_features = interval_features.dropna(axis=1, how='all')

    return interval_features, event_features

