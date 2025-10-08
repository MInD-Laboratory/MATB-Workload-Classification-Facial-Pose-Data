
import pandas as pd
import os
import numpy as np
import neurokit2 as nk

# Local implementation of find_csv_with_substring
def find_csv_with_substring(folder, substring):
    """
    Search for a CSV file in the given folder whose filename contains the specified substring.
    Returns the first matching filename found, or raises FileNotFoundError if none found.
    """
    for filename in os.listdir(folder):
        if filename.endswith('.csv') and substring in filename:
            return filename
    raise FileNotFoundError(f"No CSV file with substring '{substring}' found in {folder}")


# Function to import eda data
def import_shimmer_eda_data(folder_shimmer_data):
    # eda waveform
    # subfolder_shimmer_data = os.path.join(folder_shimmer_data, os.listdir(folder_shimmer_data)[0])
    subfolder_shimmer_data = folder_shimmer_data
    substring = 'Shimmer'
    shimmer_eda_filename = find_csv_with_substring(subfolder_shimmer_data, substring)  # [0]
    shimmer_eda_filename = os.path.join(subfolder_shimmer_data, shimmer_eda_filename)
    shimmer_eda_df = pd.read_csv(shimmer_eda_filename, skiprows=[0], header=[0, 1], index_col=[0])
    shimmer_eda_df['Time'] = pd.to_datetime(shimmer_eda_df.index, format='%Y/%m/%d %H:%M:%S.%f')
    shimmer_eda_df['Timestamp'] = shimmer_eda_df['Time'].dt.time
    shimmer_eda_df.index = pd.to_datetime(shimmer_eda_df.index,
                                          format='%Y/%m/%d %H:%M:%S.%f')  # Convert timestamp columns to datetime

    return shimmer_eda_df


# function to process eda signal: clean, Decompose into phasic skin conductance response (SCR) and tonic skin conductance level (SCL) signals, and detect peaks in SCR
def processing_eda_signal(eda_signal, sampling_rate=20, method_clean="neurokit",
                          method_phasic='highpass', method_peak='neurokit',
                          plot_signal=True, output_folder=''):
    """

     Re-create built-in nk.eda_process() function with a custom process() fucntion that calls “mid-level functions” such as:
     eda_clean(), eda_peaks(),  eda_phasic() ect.
     Each function can be optimised with customised methods


        Procedure to process eda signal: clean, decompose SCR and SCL, detect peaks

        Returns:
        signals (DataFrame) – A DataFrame of same length as "eda_signal" containing the following columns:

            "EDA_Raw": the raw signal.
            "EDA_Clean": the cleaned signal.
            "EDA_Tonic": the tonic component of the signal, or the Tonic Skin Conductance Level (SCL).
            "EDA_Phasic": the phasic component of the signal, or the Phasic Skin Conductance Response (SCR).
            "SCR_Onsets": the samples at which the onsets of the peaks occur, marked as “1” in a list of zeros.
            "SCR_Peaks": the samples at which the peaks occur, marked as “1” in a list of zeros.
            "SCR_Height": the SCR amplitude of the signal including the Tonic component. Note that cumulative effects of close-occurring SCRs might lead to an underestimation of the amplitude.
            "SCR_Amplitude": the SCR amplitude of the signal excluding the Tonic component.
            "SCR_RiseTime": the time taken for SCR onset to reach peak amplitude within the SCR.
            "SCR_Recovery": the samples at which SCR peaks recover (decline) to half amplitude, marked as “1” in a list of zeros.

        info (dict) – A dictionary containing the information of each SCR peak
                (see eda_findpeaks()), as well as the signals’ sampling rate.
                        "SCR_Peaks":
                        "SCR_Peaks_Uncorrected"
                        "sampling_rate" :signals’ sampling rate.

        PROCESSES:

        *CLEANING [nk.eda_clean]:
        This function cleans the EDA signal by removing noise and smoothing the signal with different methods.
        eda_clean(eda_signal, sampling_rate=1000, method='neurokit')
            methods: 'Neurokit' [Default], 'biosppy'
            returns: array – Vector containing the cleaned EDA signal.

        *DECOMPOSE SCR/SCL [nk.eda_phasic]:
        Decompose the Electrodermal Activity (EDA) into two components, namely Phasic and Tonic
        eda_phasic(eda_signal, sampling_rate=1000, method='highpass', **kwargs)
            methods: 'highpass' [Default], 'cvxEDA', 'smoothmedian'
            returns: DataFrame – DataFrame containing the "Tonic" and the "Phasic" components as columns.

        *PEAK DETECTION [nk.eda_peaks]:
        Identify Skin Conductance Responses (SCR) peaks in the phasic component of Electrodermal Activity (EDA) with different possible methods
        eda_peaks(eda_cleaned, sampling_rate=1000, method='neurokit', correct_artifacts=False, show=False, **kwargs)
            methods: 'neurokit' [Default], gamboa2008, kim2004, vanhalem2020, nabian2018
            returns: info (dict) – A dictionary containing additional information,
                                  "SCR_Amplitude", "SCR_Onsets", and "SCR_Peaks"
                                  ."sampling_rate"

        *FIX PEAKS [nk.signal_fixpeaks]:
        correct the peaks
        eda_fixpeaks(peaks, onsets=None, height=None)
            returns: info (dict) – A dictionary containing additional information,
                                  "SCR_Amplitude", "SCR_Onsets", and "SCR_Peaks"
                                  ."sampling_rate"

    """

    # PROCESS STEPS: CLEAN, PEAK DETECTION, DECOMPOSE INTO SCL AND SCR

    # CLEANING [eda_clean()]: detrend and filter
    eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=sampling_rate, method=method_clean)

    # DECOMPOSE[eda_phasic()]: Decompose EDA signal into Phasic (SCR) and Tonic (SCL)
    eda_decompose = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate, method=method_phasic)

    # DETECT SCR PEAKS [eda_peaks()]:Identify Skin Conductance Responses (SCR) peaks in the phasic component of Electrodermal Activity (EDA)
    instant_peaks, rpeaks, = nk.eda_peaks(eda_decompose['EDA_Phasic'], sampling_rate=sampling_rate, method=method_peak)

    # Store Output
    signals = pd.DataFrame({"EDA_Raw": np.squeeze(eda_signal),
                            "EDA_Clean": eda_cleaned})

    signals = pd.concat([signals, eda_decompose, instant_peaks], axis=1)
    infos = rpeaks
    infos["sampling_rate"] = sampling_rate

    return signals, infos


# analysis
def eda_feature_extraction(epochs, save_output_folder):
    """


    NeuroKit2



        eda_intervalrelated()
        eda_eventrelated()
    """

    if not os.path.exists(f'{save_output_folder}'):
        os.makedirs(f'{save_output_folder}')

    '''
    EVENT-RELATED FEATURES --------------------- typically for <10 second analysis
    [eda_eventrelated()]: eda analysis  will return a dataframe of the features
        - *
    To be able to use eda_eventrelated() eda analysis, an array or list of onset of events is required.

    You can create a dictionary of events using events_create() and a list/array of onsets (optional labels and conditions)
    # events_create(event_onsets, event_durations=None, event_labels=None, event_conditions=None)

    OR

    You can find events in a continuous signal with events_find() such as with a photosensor.
    # events_find(event_channel, threshold='auto', threshold_keep='above', start_at=0, end_at=None, duration_min=1, 
                duration_max=None, inter_min=0, discard_first=0, discard_last=0, event_labels=None, event_conditions=None)

    '''
    # Compute Event-Related Features [<10seconds]
    # ------------------------------------------------------------------------------------------------------------------
    """
    [nk.eda_eventrelated()]
    Performs event-related EDA analysis on epochs

    Returns:
    DataFrame – A dataframe containing the analyzed EDA features for each epoch, with each epoch indicated by the Label column (if not present, by the Index column). The analyzed features consist the following:

        "EDA_SCR": indication of whether Skin Conductance Response (SCR) occurs following the event (1 if an SCR onset is present and 0 if absent) and if so, its corresponding peak amplitude, time of peak, rise and recovery time. If there is no occurrence of SCR, nans are displayed for the below features.
        "EDA_Peak_Amplitude": the maximum amplitude of the phasic component of the signal.
        "SCR_Peak_Amplitude": the peak amplitude of the first SCR in each epoch.
        "SCR_Peak_Amplitude_Time": the timepoint of each first SCR peak amplitude.
        "SCR_RiseTime": the risetime of each first SCR i.e., the time it takes for SCR to reach peak amplitude from onset.
        "SCR_RecoveryTime": the half-recovery time of each first SCR i.e., the time it takes for SCR to decrease to half amplitude.

    """
    event_features = nk.eda_eventrelated(epochs)
    event_features = event_features.dropna(axis=1, how='all')

    # Compute Interval-RELATED Features [>10seconds]
    # ------------------------------------------------------------------------------------------------------------------
    '''
    [nk.eda_intervalrelated]: analyze longer periods of data (i.e., greater than 10 seconds)
    eda_intervalrelated(data, sampling_rate=1000, **kwargs)
    DataFrame – A dataframe containing the analyzed EDA features. The analyzed features consist of the following: 

            "SCR_Peaks_N": the number of occurrences of Skin Conductance Response (SCR). 
            "SCR_Peaks_Amplitude_Mean": the mean amplitude of the SCR peak occurrences. 
            "EDA_Tonic_SD": the mean amplitude of the SCR peak occurrences. 

            "EDA_Sympathetic": see eda_sympathetic() (only computed if signal duration> 64 sec).

            "EDA_Autocorrelation": see eda_autocor() (only computed if signal duration > 30 sec).
    '''
    interval_features = nk.eda_intervalrelated(epochs)
    interval_features = interval_features.dropna(axis=1, how='all')
    return interval_features, event_features

