"""
================================================================================
DATA LOADING AND CONDITION MAPPING UTILITIES
================================================================================

This module provides functions for loading pose data and mapping experimental
conditions based on participant information and trial numbers.

Author: Pose Analysis Pipeline
Date: 2024
================================================================================
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import re


def load_participant_info(participant_info_path: str) -> pd.DataFrame:
    """
    Load participant information including condition order.

    Parameters
    ----------
    participant_info_path : str
        Path to participant_info.csv file

    Returns
    -------
    pd.DataFrame
        DataFrame with participant IDs and session conditions
    """
    df = pd.read_csv(participant_info_path)

    # Clean participant ID to ensure it's a string
    df['Participant ID'] = df['Participant ID'].astype(str)

    return df


def parse_condition_mapping(participant_info: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    """
    Create a mapping from participant ID and trial number to condition.

    Parameters
    ----------
    participant_info : pd.DataFrame
        DataFrame from load_participant_info()

    Returns
    -------
    dict
        Nested dict: {participant_id: {trial_number: condition}}

    Examples
    --------
    >>> mapping['3101'][1]  # Returns 'Low'
    >>> mapping['3101'][2]  # Returns 'Moderate' or 'High' based on counterbalancing
    """
    condition_map = {}

    for _, row in participant_info.iterrows():
        participant_id = str(row['Participant ID'])

        # Skip participants with missing data
        if row['Session1'] == '-' or pd.isna(row['Session1']):
            continue

        # Create trial mapping for this participant
        trial_map = {}

        # Parse each session (Session1, Session2, Session3)
        for session_num in [1, 2, 3]:
            session_col = f'Session{session_num}'
            if session_col in row and not pd.isna(row[session_col]) and row[session_col] != '-':
                session_value = row[session_col]

                # Extract condition (L, M, or H) and trial number
                if isinstance(session_value, str) and len(session_value) >= 2:
                    condition = session_value[0]  # First character (L, M, or H)
                    # Some entries have trial info after condition

                    # Map condition letter to full name
                    condition_full = {
                        'L': 'Low',
                        'M': 'Moderate',
                        'H': 'High'
                    }.get(condition, condition)

                    # Session number corresponds to trial number
                    trial_map[session_num] = condition_full

        condition_map[participant_id] = trial_map

    return condition_map


def parse_pose_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract participant ID and trial number from pose filename.

    Parameters
    ----------
    filename : str
        Pose filename (e.g., '3101_02_pose.csv' or '3101_02.csv')

    Returns
    -------
    tuple
        (participant_id, trial_number) or (None, None) if parsing fails

    Examples
    --------
    >>> parse_pose_filename('3101_02_pose.csv')
    ('3101', 2)
    >>> parse_pose_filename('3101_02.csv')
    ('3101', 2)
    """
    # Try to match pattern: PPPP_TT where PPPP is participant ID and TT is trial
    pattern = r'(\d{4})_(\d{2})'
    match = re.search(pattern, filename)

    if match:
        participant_id = match.group(1)
        trial_number = int(match.group(2))
        return participant_id, trial_number

    return None, None


def load_pose_data_with_conditions(
    pose_data_dir: str,
    participant_info_path: str,
    feature_files: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load pose data files and add condition labels based on participant info.

    Parameters
    ----------
    pose_data_dir : str
        Directory containing processed pose feature files
    participant_info_path : str
        Path to participant_info.csv
    feature_files : list of str, optional
        Specific files to load. If None, loads all CSV files

    Returns
    -------
    pd.DataFrame
        Combined dataframe with all participants, features, and condition labels
    """
    # Load participant info and create condition mapping
    participant_info = load_participant_info(participant_info_path)
    condition_map = parse_condition_mapping(participant_info)

    # Get list of files to process
    if feature_files is None:
        feature_files = [f for f in os.listdir(pose_data_dir) if f.endswith('.csv')]

    all_data = []

    for filename in feature_files:
        # Parse filename to get participant and trial
        participant_id, trial_number = parse_pose_filename(filename)

        if participant_id is None or trial_number is None:
            print(f"Warning: Could not parse filename: {filename}")
            continue

        # Get condition for this participant and trial
        condition = None
        if participant_id in condition_map:
            if trial_number in condition_map[participant_id]:
                condition = condition_map[participant_id][trial_number]

        if condition is None:
            print(f"Warning: No condition mapping found for {participant_id} trial {trial_number}")
            continue

        # Load the data file
        filepath = os.path.join(pose_data_dir, filename)
        try:
            df = pd.read_csv(filepath)

            # Add metadata columns
            df['participant'] = participant_id
            df['trial'] = trial_number
            df['condition'] = condition
            df['filename'] = filename

            # Add time-based columns
            if 'frame_number' not in df.columns:
                df['frame_number'] = df.index

            # Assuming 60 fps
            df['time_seconds'] = df['frame_number'] / 60.0
            df['minute'] = (df['time_seconds'] / 60.0).astype(int)

            all_data.append(df)

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    if not all_data:
        print("Warning: No data files were successfully loaded")
        return pd.DataFrame()

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"Loaded {len(all_data)} files with {len(combined_df)} total frames")
    print(f"Participants: {combined_df['participant'].nunique()}")
    print(f"Conditions: {combined_df['condition'].value_counts().to_dict()}")

    return combined_df


def summarize_data_by_participant_condition(df: pd.DataFrame,
                                           features: List[str]) -> pd.DataFrame:
    """
    Create participant-level summaries for each condition.

    Parameters
    ----------
    df : pd.DataFrame
        Data with participant, condition, and feature columns
    features : list of str
        Feature columns to summarize

    Returns
    -------
    pd.DataFrame
        Summary with one row per participant-condition combination
    """
    summary_data = []

    for (participant, condition), group_df in df.groupby(['participant', 'condition']):
        summary_row = {
            'participant': participant,
            'condition': condition,
            'n_frames': len(group_df),
            'duration_minutes': group_df['time_seconds'].max() / 60.0 if 'time_seconds' in group_df.columns else np.nan
        }

        for feature in features:
            if feature in group_df.columns:
                feature_data = group_df[feature].dropna()
                if len(feature_data) > 0:
                    summary_row[f'{feature}_mean'] = feature_data.mean()
                    summary_row[f'{feature}_std'] = feature_data.std()
                    summary_row[f'{feature}_median'] = feature_data.median()
                    summary_row[f'{feature}_q25'] = feature_data.quantile(0.25)
                    summary_row[f'{feature}_q75'] = feature_data.quantile(0.75)

        summary_data.append(summary_row)

    return pd.DataFrame(summary_data)


def prepare_data_for_statistical_analysis(
    pose_data_dir: str,
    participant_info_path: str,
    features: List[str],
    aggregate_level: str = 'participant_condition'
) -> pd.DataFrame:
    """
    Prepare pose data for statistical analysis with proper condition labels.

    Parameters
    ----------
    pose_data_dir : str
        Directory with processed pose features
    participant_info_path : str
        Path to participant_info.csv
    features : list of str
        Features to include in analysis
    aggregate_level : str
        Level of aggregation: 'frame', 'minute', 'participant_condition'

    Returns
    -------
    pd.DataFrame
        Data ready for statistical analysis
    """
    # Load all data with conditions
    df_all = load_pose_data_with_conditions(pose_data_dir, participant_info_path)

    if df_all.empty:
        return pd.DataFrame()

    if aggregate_level == 'frame':
        # Return frame-level data
        return df_all[['participant', 'condition', 'trial', 'time_seconds'] + features]

    elif aggregate_level == 'minute':
        # Aggregate by minute
        minute_data = []
        for (participant, condition, minute), group_df in df_all.groupby(['participant', 'condition', 'minute']):
            minute_row = {
                'participant': participant,
                'condition': condition,
                'minute': minute
            }

            for feature in features:
                if feature in group_df.columns:
                    minute_row[feature] = group_df[feature].mean()

            minute_data.append(minute_row)

        return pd.DataFrame(minute_data)

    elif aggregate_level == 'participant_condition':
        # Aggregate to participant-condition level
        return summarize_data_by_participant_condition(df_all, features)

    else:
        raise ValueError(f"Unknown aggregation level: {aggregate_level}")