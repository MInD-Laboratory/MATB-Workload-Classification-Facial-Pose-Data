"""
================================================================================
FEATURE EXTRACTION MODULE FOR FACIAL POSE DATA
================================================================================

This module provides functions for extracting meaningful features from facial
landmark data. Each function extracts a specific feature that can be used for
analysis of facial expressions, eye movements, and head pose.

Features include:
- Eye Aspect Ratio (EAR) for blink detection
- Mouth opening distance
- Head rotation angle
- Regional averages (eyes, face center, pupils)
- Movement magnitudes (frame-to-frame displacement)

Author: Pose Analysis Pipeline
Date: 2024
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Optional

from .landmark_config import (
    EYES, ANCHOR_L, ANCHOR_R,
    MOUTH_TOP_ALT, MOUTH_BOTTOM_ALT,
    LEFT_PUPIL, RIGHT_PUPIL,
    FACIAL_REGIONS,
    NOSE_BRIDGE_INDICES,
    get_column_name
)
from .coordinate_normalization import (
    stabilize_points_original,
    get_best_coordinate
)


# ============================================================================
# BLINK DETECTION FEATURES
# ============================================================================

def extract_eye_aspect_ratio(df: pd.DataFrame, use_procrustes: bool = False) -> pd.Series:
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.

    EAR measures how "open" the eyes are. It is calculated as the ratio of
    vertical eye distances to horizontal eye distance. Lower values indicate
    more closed eyes (potential blinks).

    Formula for each eye:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    where p1-p6 are the 6 eye contour points

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing landmark coordinates
    use_procrustes : bool, optional
        Whether to use Procrustes-aligned coordinates if available (default: False)

    Returns
    -------
    pd.Series
        Eye Aspect Ratio values (average of left and right eyes)

    Examples
    --------
    >>> df = pd.read_csv('participant_001.csv')
    >>> ear = extract_eye_aspect_ratio(df)
    >>> ear.mean()  # Average EAR across all frames
    0.25
    """
    def compute_distance(lm1: int, lm2: int) -> pd.Series:
        """Helper function to compute distance between two landmarks."""
        if use_procrustes:
            x1 = get_best_coordinate(df, lm1, 'x', prefer_procrustes=True)
            y1 = get_best_coordinate(df, lm1, 'y', prefer_procrustes=True)
            x2 = get_best_coordinate(df, lm2, 'x', prefer_procrustes=True)
            y2 = get_best_coordinate(df, lm2, 'y', prefer_procrustes=True)
        else:
            # Use original coordinates with stabilization
            needed_landmarks = list(EYES["L"]) + list(EYES["R"]) + [ANCHOR_L, ANCHOR_R]
            stabilized = stabilize_points_original(df, needed_landmarks)

            if lm1 in stabilized and lm2 in stabilized:
                x1, y1 = stabilized[lm1]
                x2, y2 = stabilized[lm2]
            else:
                # Fallback to raw coordinates
                x1 = df.get(get_column_name(lm1, 'x'), pd.Series(np.nan, index=df.index))
                y1 = df.get(get_column_name(lm1, 'y'), pd.Series(np.nan, index=df.index))
                x2 = df.get(get_column_name(lm2, 'x'), pd.Series(np.nan, index=df.index))
                y2 = df.get(get_column_name(lm2, 'y'), pd.Series(np.nan, index=df.index))

        return np.hypot(x1 - x2, y1 - y2)

    # Calculate EAR for left eye
    # Vertical distances: ||38-42|| and ||39-41||
    # Horizontal distance: ||37-40||
    left_vertical_1 = compute_distance(38, 42)
    left_vertical_2 = compute_distance(39, 41)
    left_horizontal = compute_distance(37, 40)
    left_ear = (left_vertical_1 + left_vertical_2) / (2.0 * left_horizontal)

    # Calculate EAR for right eye
    # Vertical distances: ||44-48|| and ||45-47||
    # Horizontal distance: ||43-46||
    right_vertical_1 = compute_distance(44, 48)
    right_vertical_2 = compute_distance(45, 47)
    right_horizontal = compute_distance(43, 46)
    right_ear = (right_vertical_1 + right_vertical_2) / (2.0 * right_horizontal)

    # Return average of both eyes
    return (left_ear + right_ear) / 2.0


def extract_blink_rate(ear_series: pd.Series, threshold: float = 0.2,
                      min_duration: int = 3, max_duration: int = 15) -> float:
    """
    Extract blink rate from Eye Aspect Ratio time series.

    Detects blinks as periods where EAR drops below threshold for a
    reasonable duration (not too short, not too long).

    Parameters
    ----------
    ear_series : pd.Series
        Eye Aspect Ratio time series
    threshold : float, optional
        EAR threshold below which eyes are considered closed (default: 0.2)
    min_duration : int, optional
        Minimum frames for a valid blink (default: 3)
    max_duration : int, optional
        Maximum frames for a valid blink (default: 15)

    Returns
    -------
    float
        Blink rate (blinks per minute)

    Examples
    --------
    >>> ear = extract_eye_aspect_ratio(df)
    >>> blink_rate = extract_blink_rate(ear)
    >>> print(f"Blink rate: {blink_rate:.1f} blinks/min")
    Blink rate: 15.2 blinks/min
    """
    # Detect when eyes are closed (EAR below threshold)
    eyes_closed = ear_series < threshold

    # Find blink events (consecutive closed frames)
    blink_count = 0
    current_blink_duration = 0

    for is_closed in eyes_closed:
        if is_closed:
            current_blink_duration += 1
        else:
            # Check if the closed period was a valid blink
            if min_duration <= current_blink_duration <= max_duration:
                blink_count += 1
            current_blink_duration = 0

    # Check last potential blink
    if min_duration <= current_blink_duration <= max_duration:
        blink_count += 1

    # Convert to blinks per minute (assuming 60 fps)
    duration_minutes = len(ear_series) / (60 * 60)  # frames / (fps * seconds_per_minute)
    blink_rate = blink_count / duration_minutes if duration_minutes > 0 else 0

    return blink_rate


# ============================================================================
# MOUTH FEATURES
# ============================================================================

def extract_mouth_opening_distance(df: pd.DataFrame, use_procrustes: bool = False) -> pd.Series:
    """
    Calculate mouth opening distance.

    Measures the vertical distance between top and bottom lip centers,
    normalized by head size if using original coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing landmark coordinates
    use_procrustes : bool, optional
        Whether to use Procrustes-aligned coordinates if available (default: False)

    Returns
    -------
    pd.Series
        Mouth opening distance values

    Examples
    --------
    >>> df = pd.read_csv('participant_001.csv')
    >>> mouth_dist = extract_mouth_opening_distance(df)
    >>> mouth_dist.mean()
    0.15
    """
    if use_procrustes:
        # Use Procrustes coordinates directly
        x_top = get_best_coordinate(df, MOUTH_TOP_ALT, 'x', prefer_procrustes=True)
        y_top = get_best_coordinate(df, MOUTH_TOP_ALT, 'y', prefer_procrustes=True)
        x_bottom = get_best_coordinate(df, MOUTH_BOTTOM_ALT, 'x', prefer_procrustes=True)
        y_bottom = get_best_coordinate(df, MOUTH_BOTTOM_ALT, 'y', prefer_procrustes=True)
    else:
        # Use stabilized coordinates
        needed_landmarks = [ANCHOR_L, ANCHOR_R, MOUTH_TOP_ALT, MOUTH_BOTTOM_ALT]
        stabilized = stabilize_points_original(df, needed_landmarks)

        if MOUTH_TOP_ALT in stabilized and MOUTH_BOTTOM_ALT in stabilized:
            x_top, y_top = stabilized[MOUTH_TOP_ALT]
            x_bottom, y_bottom = stabilized[MOUTH_BOTTOM_ALT]
        else:
            # Fallback to raw coordinates
            x_top = df.get(get_column_name(MOUTH_TOP_ALT, 'x'), pd.Series(np.nan, index=df.index))
            y_top = df.get(get_column_name(MOUTH_TOP_ALT, 'y'), pd.Series(np.nan, index=df.index))
            x_bottom = df.get(get_column_name(MOUTH_BOTTOM_ALT, 'x'), pd.Series(np.nan, index=df.index))
            y_bottom = df.get(get_column_name(MOUTH_BOTTOM_ALT, 'y'), pd.Series(np.nan, index=df.index))

    # Calculate Euclidean distance
    return np.hypot(x_top - x_bottom, y_top - y_bottom)


# ============================================================================
# HEAD POSE FEATURES
# ============================================================================

def extract_head_rotation_angle(df: pd.DataFrame, use_procrustes: bool = False) -> pd.Series:
    """
    Calculate head rotation angle from eye corners.

    Measures the angle of the line connecting the eye corners relative to
    horizontal. Positive angles indicate head tilted to the right.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing landmark coordinates
    use_procrustes : bool, optional
        Whether to use Procrustes-aligned coordinates if available (default: False)

    Returns
    -------
    pd.Series
        Head rotation angle in radians

    Examples
    --------
    >>> df = pd.read_csv('participant_001.csv')
    >>> rotation = extract_head_rotation_angle(df)
    >>> np.degrees(rotation.mean())  # Convert to degrees
    2.5
    """
    if use_procrustes:
        # Use Procrustes coordinates
        x_left = get_best_coordinate(df, ANCHOR_L, 'x', prefer_procrustes=True)
        y_left = get_best_coordinate(df, ANCHOR_L, 'y', prefer_procrustes=True)
        x_right = get_best_coordinate(df, ANCHOR_R, 'x', prefer_procrustes=True)
        y_right = get_best_coordinate(df, ANCHOR_R, 'y', prefer_procrustes=True)
    else:
        # Use raw coordinates
        x_left = df.get(get_column_name(ANCHOR_L, 'x'), pd.Series(np.nan, index=df.index))
        y_left = df.get(get_column_name(ANCHOR_L, 'y'), pd.Series(np.nan, index=df.index))
        x_right = df.get(get_column_name(ANCHOR_R, 'x'), pd.Series(np.nan, index=df.index))
        y_right = df.get(get_column_name(ANCHOR_R, 'y'), pd.Series(np.nan, index=df.index))

    # Calculate angle using arctan2
    dx = x_right - x_left
    dy = y_right - y_left
    return np.arctan2(dy, dx)


# ============================================================================
# REGIONAL AVERAGING FEATURES
# ============================================================================

def extract_regional_averages(df: pd.DataFrame, confidence_threshold: float = 0.3,
                             use_procrustes: bool = False) -> pd.DataFrame:
    """
    Convert individual landmarks into regional averages.

    Creates meaningful regional features by averaging landmarks within each
    facial region. Low-confidence landmarks are masked out before averaging.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing landmark coordinates
    confidence_threshold : float, optional
        Minimum confidence for including a landmark (default: 0.3)
    use_procrustes : bool, optional
        Whether to use Procrustes-aligned coordinates if available (default: False)

    Returns
    -------
    pd.DataFrame
        DataFrame with regional average features

    Examples
    --------
    >>> df = pd.read_csv('participant_001.csv')
    >>> regional = extract_regional_averages(df)
    >>> regional.columns.tolist()
    ['center_face_x', 'center_face_y', 'center_face_prob', ...]
    """
    averaged_df = pd.DataFrame()

    # Process each facial region
    for region_name, landmark_indices in FACIAL_REGIONS.items():
        # Collect coordinates for this region
        x_values = []
        y_values = []
        prob_values = []

        for landmark_idx in landmark_indices:
            if use_procrustes:
                x = get_best_coordinate(df, landmark_idx, 'x', prefer_procrustes=True)
                y = get_best_coordinate(df, landmark_idx, 'y', prefer_procrustes=True)
            else:
                x_col = get_column_name(landmark_idx, 'x')
                y_col = get_column_name(landmark_idx, 'y')
                x = df.get(x_col, pd.Series(np.nan, index=df.index))
                y = df.get(y_col, pd.Series(np.nan, index=df.index))

            # Get probability/confidence
            prob_col = get_column_name(landmark_idx, 'prob')
            prob = df.get(prob_col, pd.Series(np.nan, index=df.index))

            x_values.append(x)
            y_values.append(y)
            prob_values.append(prob)

        # Calculate averages if we have data
        if x_values:
            # Stack values and calculate mean
            x_stack = pd.concat(x_values, axis=1)
            y_stack = pd.concat(y_values, axis=1)
            prob_stack = pd.concat(prob_values, axis=1)

            # Calculate regional averages
            averaged_df[f"{region_name}_x"] = x_stack.mean(axis=1)
            averaged_df[f"{region_name}_y"] = y_stack.mean(axis=1)
            averaged_df[f"{region_name}_prob"] = prob_stack.mean(axis=1)

            # Apply confidence masking
            low_confidence_mask = averaged_df[f"{region_name}_prob"] < confidence_threshold
            averaged_df.loc[low_confidence_mask, f"{region_name}_x"] = np.nan
            averaged_df.loc[low_confidence_mask, f"{region_name}_y"] = np.nan

    # Fill gaps using linear interpolation
    averaged_df = averaged_df.interpolate(method="linear", limit_direction="both")

    return averaged_df


# ============================================================================
# MOVEMENT MAGNITUDE FEATURES
# ============================================================================

def extract_movement_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate movement magnitude (speed) for each feature.

    For every x/y coordinate pair, calculates frame-to-frame movement:
    magnitude = sqrt((x[t] - x[t-1])^2 + (y[t] - y[t-1])^2)

    This captures how much each facial region is moving between frames.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with x/y coordinate columns

    Returns
    -------
    pd.DataFrame
        DataFrame with added magnitude columns

    Examples
    --------
    >>> df = pd.DataFrame({'center_face_x': [0, 1, 1], 'center_face_y': [0, 0, 1]})
    >>> df_mag = extract_movement_magnitudes(df)
    >>> df_mag['center_face_magnitude'].tolist()
    [0.0, 1.0, 1.0]
    """
    df_result = df.copy()

    # Find all x/y coordinate pairs
    for col in df.columns:
        if col.endswith("_x"):
            # Find corresponding y column
            y_col = col.replace("_x", "_y")
            if y_col in df.columns:
                # Calculate frame-to-frame differences
                dx = df[col].diff()
                dy = df[y_col].diff()

                # Calculate Euclidean distance (magnitude)
                mag_col = col.replace("_x", "_magnitude")
                df_result[mag_col] = np.hypot(dx, dy)

                # Set first frame to 0 (no previous frame)
                if len(df_result) > 0:
                    df_result.loc[df_result.index[0], mag_col] = 0.0

    return df_result


# ============================================================================
# PUPIL/GAZE FEATURES
# ============================================================================

def extract_pupil_features(df: pd.DataFrame, use_procrustes: bool = False,
                          relative_to_nose: bool = False) -> pd.DataFrame:
    """
    Extract pupil-related features including combined pupil position.

    Creates features for individual pupils and their average position,
    useful for gaze analysis. Can calculate pupil positions relative to
    nose center to factor out head movement.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing landmark coordinates
    use_procrustes : bool, optional
        Whether to use Procrustes-aligned coordinates if available (default: False)
    relative_to_nose : bool, optional
        Whether to calculate pupil positions relative to nose center,
        which helps isolate eye movement from head movement (default: False)

    Returns
    -------
    pd.DataFrame
        DataFrame with pupil features

    Examples
    --------
    >>> df = pd.read_csv('participant_001.csv')
    >>> pupils = extract_pupil_features(df, relative_to_nose=True)
    >>> pupils.columns.tolist()
    ['left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y',
     'avg_pupil_x', 'avg_pupil_y', 'avg_pupil_magnitude',
     'left_pupil_rel_nose_x', 'left_pupil_rel_nose_y', ...]
    """
    result = pd.DataFrame()

    # Get pupil coordinates
    if use_procrustes:
        left_x = get_best_coordinate(df, LEFT_PUPIL, 'x', prefer_procrustes=True)
        left_y = get_best_coordinate(df, LEFT_PUPIL, 'y', prefer_procrustes=True)
        right_x = get_best_coordinate(df, RIGHT_PUPIL, 'x', prefer_procrustes=True)
        right_y = get_best_coordinate(df, RIGHT_PUPIL, 'y', prefer_procrustes=True)
    else:
        left_x = df.get(get_column_name(LEFT_PUPIL, 'x'), pd.Series(np.nan, index=df.index))
        left_y = df.get(get_column_name(LEFT_PUPIL, 'y'), pd.Series(np.nan, index=df.index))
        right_x = df.get(get_column_name(RIGHT_PUPIL, 'x'), pd.Series(np.nan, index=df.index))
        right_y = df.get(get_column_name(RIGHT_PUPIL, 'y'), pd.Series(np.nan, index=df.index))

    # Store individual pupil positions (absolute)
    result['left_pupil_x'] = left_x
    result['left_pupil_y'] = left_y
    result['right_pupil_x'] = right_x
    result['right_pupil_y'] = right_y

    # Calculate average pupil position (combined gaze)
    result['avg_pupil_x'] = (left_x + right_x) / 2
    result['avg_pupil_y'] = (left_y + right_y) / 2

    # Calculate magnitude from origin (useful for gaze deviation)
    result['avg_pupil_magnitude'] = np.hypot(result['avg_pupil_x'], result['avg_pupil_y'])

    # Calculate nose-relative pupil positions if requested
    if relative_to_nose:
        # Calculate nose center from nose bridge landmarks
        nose_x_coords = []
        nose_y_coords = []

        for nose_idx in NOSE_BRIDGE_INDICES:
            if use_procrustes:
                nose_x = get_best_coordinate(df, nose_idx, 'x', prefer_procrustes=True)
                nose_y = get_best_coordinate(df, nose_idx, 'y', prefer_procrustes=True)
            else:
                nose_x = df.get(get_column_name(nose_idx, 'x'), pd.Series(np.nan, index=df.index))
                nose_y = df.get(get_column_name(nose_idx, 'y'), pd.Series(np.nan, index=df.index))

            nose_x_coords.append(nose_x)
            nose_y_coords.append(nose_y)

        # Calculate average nose position
        nose_center_x = pd.concat(nose_x_coords, axis=1).mean(axis=1)
        nose_center_y = pd.concat(nose_y_coords, axis=1).mean(axis=1)

        # Calculate pupil positions relative to nose center
        result['left_pupil_rel_nose_x'] = left_x - nose_center_x
        result['left_pupil_rel_nose_y'] = left_y - nose_center_y
        result['right_pupil_rel_nose_x'] = right_x - nose_center_x
        result['right_pupil_rel_nose_y'] = right_y - nose_center_y

        # Average relative pupil position
        result['avg_pupil_rel_nose_x'] = (result['left_pupil_rel_nose_x'] +
                                         result['right_pupil_rel_nose_x']) / 2
        result['avg_pupil_rel_nose_y'] = (result['left_pupil_rel_nose_y'] +
                                         result['right_pupil_rel_nose_y']) / 2

        # Magnitude of nose-relative gaze deviation
        result['avg_pupil_rel_nose_magnitude'] = np.hypot(
            result['avg_pupil_rel_nose_x'],
            result['avg_pupil_rel_nose_y']
        )

        # Store nose center for reference
        result['nose_center_x'] = nose_center_x
        result['nose_center_y'] = nose_center_y

    return result


# ============================================================================
# COMPREHENSIVE FEATURE EXTRACTION
# ============================================================================

def extract_all_features(df: pd.DataFrame, use_procrustes: bool = False,
                        confidence_threshold: float = 0.3,
                        feature_procrustes_config: Optional[dict] = None,
                        pupil_relative_to_nose: bool = False) -> pd.DataFrame:
    """
    Extract all facial pose features from landmark data with flexible Procrustes control.

    This is a comprehensive function that extracts all available features
    and combines them into a single DataFrame. Allows per-feature control
    of whether to use Procrustes transformation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing landmark coordinates
    use_procrustes : bool, optional
        Default Procrustes setting for all features (default: False)
    confidence_threshold : float, optional
        Minimum confidence for including landmarks (default: 0.3)
    feature_procrustes_config : dict, optional
        Per-feature Procrustes settings. Keys: feature names,
        Values: True/False to override default use_procrustes.
        Example: {'blink_dist': True, 'mouth_dist': False}
    pupil_relative_to_nose : bool, optional
        Whether to calculate pupil features relative to nose center (default: False)

    Returns
    -------
    pd.DataFrame
        DataFrame with all extracted features

    Examples
    --------
    >>> df = pd.read_csv('participant_001.csv')
    >>> features = extract_all_features(df, use_procrustes=True)
    >>> print(f"Extracted {len(features.columns)} features")
    Extracted 25 features

    >>> # Use Procrustes for some features but not others
    >>> config = {'blink_dist': True, 'mouth_dist': False, 'pupils': True}
    >>> features = extract_all_features(df, use_procrustes=False,
    ...                               feature_procrustes_config=config,
    ...                               pupil_relative_to_nose=True)
    """
    # Set up default configuration
    if feature_procrustes_config is None:
        feature_procrustes_config = {}

    # Helper function to get per-feature Procrustes setting
    def get_procrustes_setting(feature_name: str) -> bool:
        return feature_procrustes_config.get(feature_name, use_procrustes)

    # Extract regional averages
    regional_procrustes = get_procrustes_setting('regional_averages')
    features = extract_regional_averages(df, confidence_threshold, regional_procrustes)

    # Add movement magnitudes (operates on the regional features)
    features = extract_movement_magnitudes(features)

    # Add pupil features with flexible configuration
    pupil_procrustes = get_procrustes_setting('pupils')
    pupil_features = extract_pupil_features(df, pupil_procrustes, pupil_relative_to_nose)
    for col in pupil_features.columns:
        features[col] = pupil_features[col]

    # Add specific facial features with individual Procrustes control
    blink_procrustes = get_procrustes_setting('blink_dist')
    features['blink_dist'] = extract_eye_aspect_ratio(df, blink_procrustes)

    mouth_procrustes = get_procrustes_setting('mouth_dist')
    features['mouth_dist'] = extract_mouth_opening_distance(df, mouth_procrustes)

    head_rotation_procrustes = get_procrustes_setting('head_rotation_angle')
    features['head_rotation_angle'] = extract_head_rotation_angle(df, head_rotation_procrustes)

    return features


def extract_features_flexible(df: pd.DataFrame,
                             feature_list: list,
                             procrustes_config: Optional[dict] = None,
                             confidence_threshold: float = 0.3,
                             pupil_relative_to_nose: bool = False) -> pd.DataFrame:
    """
    Extract only specified features with fine-grained Procrustes control.

    This function allows researchers to extract only the features they need
    with precise control over coordinate normalization for each feature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing landmark coordinates
    feature_list : list
        List of features to extract. Options:
        - 'blink_dist': Eye aspect ratio
        - 'mouth_dist': Mouth opening distance
        - 'head_rotation_angle': Head rotation angle
        - 'pupils': Pupil positions and gaze features
        - 'regional_averages': Facial region center positions
        - 'movement_magnitudes': Frame-to-frame movement
    procrustes_config : dict, optional
        Per-feature Procrustes settings. Example:
        {'blink_dist': True, 'mouth_dist': False, 'pupils': True}
    confidence_threshold : float, optional
        Minimum confidence for including landmarks (default: 0.3)
    pupil_relative_to_nose : bool, optional
        Whether to calculate pupil features relative to nose center (default: False)

    Returns
    -------
    pd.DataFrame
        DataFrame with only the requested features

    Examples
    --------
    >>> # Extract only blinks and mouth features
    >>> features = extract_features_flexible(
    ...     df,
    ...     feature_list=['blink_dist', 'mouth_dist'],
    ...     procrustes_config={'blink_dist': True, 'mouth_dist': False}
    ... )

    >>> # Extract pupils relative to nose with Procrustes alignment
    >>> gaze_features = extract_features_flexible(
    ...     df,
    ...     feature_list=['pupils'],
    ...     procrustes_config={'pupils': True},
    ...     pupil_relative_to_nose=True
    ... )
    """
    if procrustes_config is None:
        procrustes_config = {}

    features = pd.DataFrame()

    # Helper function to get Procrustes setting for a feature
    def get_procrustes_setting(feature_name: str) -> bool:
        return procrustes_config.get(feature_name, False)

    # Extract requested features
    if 'regional_averages' in feature_list:
        use_proc = get_procrustes_setting('regional_averages')
        regional_features = extract_regional_averages(df, confidence_threshold, use_proc)
        features = pd.concat([features, regional_features], axis=1)

    if 'movement_magnitudes' in feature_list:
        # Movement magnitudes require coordinate features to already exist
        if features.empty:
            # Need some coordinate features first
            regional_features = extract_regional_averages(df, confidence_threshold, False)
            features = pd.concat([features, regional_features], axis=1)
        features = extract_movement_magnitudes(features)

    if 'pupils' in feature_list:
        use_proc = get_procrustes_setting('pupils')
        pupil_features = extract_pupil_features(df, use_proc, pupil_relative_to_nose)
        features = pd.concat([features, pupil_features], axis=1)

    if 'blink_dist' in feature_list:
        use_proc = get_procrustes_setting('blink_dist')
        features['blink_dist'] = extract_eye_aspect_ratio(df, use_proc)

    if 'mouth_dist' in feature_list:
        use_proc = get_procrustes_setting('mouth_dist')
        features['mouth_dist'] = extract_mouth_opening_distance(df, use_proc)

    if 'head_rotation_angle' in feature_list:
        use_proc = get_procrustes_setting('head_rotation_angle')
        features['head_rotation_angle'] = extract_head_rotation_angle(df, use_proc)

    return features