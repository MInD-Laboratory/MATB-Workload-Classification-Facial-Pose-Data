#%%
import os
import pandas as pd
import numpy as np

def get_csv_files_with_main(folder_path):
    csv_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv') and 'main' in file_name:
            file_path = os.path.join(folder_path, file_name)
            csv_files.append(file_path)
    return csv_files

sysmon_events = {
    'lights-1-failure': 'F1',
    'lights-2-failure': 'F2',
    'scales-1-failure': 'F3',
    'scales-2-failure': 'F4',
    'scales-3-failure': 'F5',
    'scales-4-failure': 'F6'
}
def calculate_sysmon_hits_per_window(df, module, value, window_size=60, overlap=0.5, total_time=480):
    hits_per_window = []
    events_per_window = []
    average_reaction_times = []
    step = int(window_size * (1 - overlap))
    if step <= 0:
        step = 1
    n_windows = max(1, int((total_time - window_size) / step) + 1)
    for window in range(n_windows):
        window_start = window * step
        window_end = window_start + window_size
        events = df[(df['module'] == module) & (df['value'] == '1') & (df['type'] == 'event') & (df['scenario_time'].between(window_start, window_end))]
        hits = 0
        reaction_times = []
        events_per_window.append(len(events))
        for index, event in events.iterrows():
            if event['value'] == '1':
                hit_row = df[(df['module'] == module) & (df['address'] == 'name') & (df['type'] == 'performance') & (df['scenario_time'].between(event['scenario_time'], event['scenario_time']+45)) & (df['value'] == sysmon_events[event['address']])].head(1)
                if not hit_row.empty:
                    next_row_index = hit_row.index + 1
                    next_row_value = df.loc[next_row_index, 'value']
                    if next_row_value.item() == value:
                        hits += 1
                        if value == 'HIT':
                            reaction_times.append(df.loc[next_row_index+1, 'value'].astype(float).item())
        hits_per_window.append(hits)
        if len(reaction_times) != 0:
            average_reaction_time = sum(reaction_times) / len(reaction_times)
            average_reaction_times.append(average_reaction_time)
        else:
            average_reaction_times.append(0)
    return hits_per_window, events_per_window, average_reaction_times

def sysmon_measures(df, window_size=60, overlap=0.5, total_time=480):
    hits_per_window, events_per_window, average_reaction_times = calculate_sysmon_hits_per_window(df, 'sysmon', 'HIT', window_size, overlap, total_time)
    misses_per_window, _, _ = calculate_sysmon_hits_per_window(df, 'sysmon', 'MISS', window_size, overlap, total_time)
    failure_rate = [(miss / event) * 100 if event != 0 else 0 for miss, event in zip(misses_per_window, events_per_window)]
    return failure_rate, average_reaction_times, events_per_window, hits_per_window

def calculate_comms_hits_per_window(df, module, value, window_size=60, overlap=0.5, total_time=480):
    hits_per_window = []
    events_per_window = []
    own_events_per_window = []
    average_reaction_times = []
    step = int(window_size * (1 - overlap))
    if step <= 0:
        step = 1
    n_windows = max(1, int((total_time - window_size) / step) + 1)
    for window in range(n_windows):
        window_start = window * step
        window_end = window_start + window_size
        events = df[(df['module'] == module) & (df['address'] == 'radioprompt') & (df['type'] == 'event') & (df['scenario_time'].between(window_start, window_end))]
        hits = 0
        reaction_time = []
        events_per_window.append(len(events))
        own_events_per_window.append(len(events[events['value'] == 'own']))
        for index, event in events.iterrows():
            if event['value'] == 'own':
                hit_row = df[(df['module'] == module) & (df['address'] == 'sdt_value') & (df['type'] == 'performance') & (df['scenario_time'].between(event['scenario_time'], event['scenario_time']+36)) & (df['value'] == value)].head(1)
                df.loc[hit_row.index, 'value'] = 'USED'
                if not hit_row.empty:
                    hits += 1
                    reaction_time.append(df.loc[hit_row.index - 1, 'value'].astype(float).item())
        hits_per_window.append(hits)
        if len(reaction_time) != 0:
            average_reaction_time = sum(reaction_time) / len(reaction_time)
            average_reaction_times.append(average_reaction_time)
        else:
            average_reaction_times.append(0)
    return hits_per_window, events_per_window, own_events_per_window, average_reaction_times

def calculate_comms_miss_per_window(df, module, value, window_size=60, overlap=0.5, total_time=480):
    hits_per_window = []
    events_per_window = []
    own_events_per_window = []
    step = int(window_size * (1 - overlap))
    if step <= 0:
        step = 1
    n_windows = max(1, int((total_time - window_size) / step) + 1)
    for window in range(n_windows):
        window_start = window * step
        window_end = window_start + window_size
        events = df[(df['module'] == module) & (df['address'] == 'radioprompt') & (df['type'] == 'event') & (df['scenario_time'].between(window_start, window_end))]
        hits = 0
        events_per_window.append(len(events))
        own_events_per_window.append(len(events[events['value'] == 'own']))
        for index, event in events.iterrows():
            if event['value'] == 'own':
                hit_row = df[(df['module'] == module) & (df['address'] == 'sdt_value') & (df['type'] == 'performance') & (df['scenario_time'].between(event['scenario_time'], event['scenario_time']+36)) & (df['value'] == value)]
                if not hit_row.empty:
                    hits += 1
        hits_per_window.append(hits)
    return hits_per_window, events_per_window, own_events_per_window

def comms_measures(df, window_size=60, overlap=0.5, total_time=480):
    misses_per_window, events_per_window, own_events_per_window = calculate_comms_miss_per_window(df, 'communications', 'MISS', window_size, overlap, total_time)
    bad_radio_per_window, _, _ = calculate_comms_miss_per_window(df, 'communications', 'BAD_RADIO', window_size, overlap, total_time)
    miss_bad_per_window = [miss + bad for miss, bad in zip(misses_per_window, bad_radio_per_window)]
    failure_rate = [(miss / event) * 100 if event != 0 else 0 for miss, event in zip(miss_bad_per_window, events_per_window)]
    _, _, _, average_reaction_times = calculate_comms_hits_per_window(df, 'communications', 'HIT', window_size, overlap, total_time)
    return failure_rate, events_per_window, own_events_per_window, average_reaction_times

def track_measures(df, window_size=60, overlap=0.5, total_time=480):
    performance_per_window = []
    step = int(window_size * (1 - overlap))
    if step <= 0:
        step = 1
    n_windows = max(1, int((total_time - window_size) / step) + 1)
    for window in range(n_windows):
        window_start = window * step
        window_end = window_start + window_size
        total_rows = len(df[(df['address'] == 'cursor_in_target') & (df['scenario_time'].between(window_start, window_end))])
        performance_rows = len(df[(df['address'] == 'cursor_in_target') & (df['value'] == '0') & (df['scenario_time'].between(window_start, window_end))])
        if total_rows > 0:
            performance_per_window.append(100* performance_rows / total_rows)
        else:
            performance_per_window.append(np.nan)
    return performance_per_window

def resman_measures(df, window_size=60, overlap=0.5, total_time=480):
    performance_per_window = []
    step = int(window_size * (1 - overlap))
    if step <= 0:
        step = 1
    n_windows = max(1, int((total_time - window_size) / step) + 1)
    for window in range(n_windows):
        window_start = window * step
        window_end = window_start + window_size
        total_rows = len(df[((df['address'] == 'a_in_tolerance') | (df['address'] == 'b_in_tolerance')) & (df['scenario_time'].between(window_start, window_end))])
        performance_rows = len(df[((df['address'] == 'a_in_tolerance') | (df['address'] == 'b_in_tolerance')) & (df['value'] == '0') & (df['scenario_time'].between(window_start, window_end))])
        if total_rows > 0:
            performance_per_window.append(100 * performance_rows / total_rows)
        else:
            performance_per_window.append(np.nan)
    return performance_per_window
   