"""
================================================================================
FACIAL LANDMARK CONFIGURATION MODULE
================================================================================

This module defines the configuration for facial landmark indices used throughout
the pose analysis pipeline. It centralizes all landmark definitions to ensure
consistency across different processing steps.

The landmark indices correspond to OpenPose facial keypoints (70 total points):
- Points 0-16: Jaw line
- Points 17-26: Eyebrows
- Points 27-35: Nose bridge
- Points 36-41: Left eye
- Points 42-47: Right eye
- Points 48-67: Mouth
- Points 68-69: Pupils

Author: Pose Analysis Pipeline
Date: 2024
================================================================================
"""

# ============================================================================
# INDIVIDUAL LANDMARK DEFINITIONS
# ============================================================================

# Eye landmark indices for left and right eyes
# These 6 points form the contour of each eye
EYES = {
    "L": [37, 38, 39, 40, 41, 42],  # Left eye: outer corner to inner corner
    "R": [43, 44, 45, 46, 47, 48],  # Right eye: inner corner to outer corner
}

# Reference anchor points for head stabilization
# These are the outer corners of the eyes, used for normalization
ANCHOR_L = 37  # Left eye outer corner
ANCHOR_R = 46  # Right eye outer corner

# Nose bridge landmarks
# Used for face center calculations and stability reference
NOSE_BRIDGE_INDICES = list(range(28, 37))  # Points 28-36

# Mouth landmarks for lip distance calculations
MOUTH_TOP_CENTER = 62    # Top lip center point
MOUTH_BOTTOM_CENTER = 66  # Bottom lip center point
MOUTH_TOP_ALT = 63       # Alternative top lip point
MOUTH_BOTTOM_ALT = 67    # Alternative bottom lip point

# Pupil center points
LEFT_PUPIL = 69   # Left pupil center
RIGHT_PUPIL = 70  # Right pupil center

# Reference landmarks for Procrustes alignment
# These stable points are used to compute the reference shape
PROCRUSTES_REFERENCE_LANDMARKS = [29, 30, 36, 45]  # Temples and nose points

# ============================================================================
# FEATURE-BASED LANDMARK GROUPINGS
# ============================================================================

# Mapping of features/metrics to the keypoint indices they depend on
# Used for quality control and feature extraction
METRIC_KEYPOINTS = {
    "eyes": EYES["L"] + EYES["R"],                          # All eye contour points
    "head_rotation": [ANCHOR_L, ANCHOR_R],                  # Eye corners for rotation
    "mouth_dist": [MOUTH_TOP_CENTER, MOUTH_BOTTOM_CENTER],  # Lip centers
    "pupils_combined": [LEFT_PUPIL, RIGHT_PUPIL],           # Both pupils
    "center_face": NOSE_BRIDGE_INDICES,                     # Nose bridge region
}

# All landmark indices that are relevant for processing
# This is the union of all landmarks used in any metric
RELEVANT_KEYPOINTS = sorted({
    kp for kps in METRIC_KEYPOINTS.values() for kp in kps
})

# ============================================================================
# COLUMN NAME CONVENTIONS
# ============================================================================

def get_column_name(landmark_idx: int, axis: str) -> str:
    """
    Generate the column name for a landmark coordinate or probability.

    Parameters
    ----------
    landmark_idx : int
        The landmark index (0-70 for OpenPose facial landmarks)
    axis : str
        The axis or type: 'x', 'y', or 'prob' for probability/confidence

    Returns
    -------
    str
        The column name, e.g., 'x37', 'y46', 'prob69'

    Examples
    --------
    >>> get_column_name(37, 'x')
    'x37'
    >>> get_column_name(46, 'prob')
    'prob46'
    """
    return f"{axis}{landmark_idx}"


def get_procrustes_column_name(landmark_idx: int, axis: str) -> str:
    """
    Generate the column name for a Procrustes-aligned landmark coordinate.

    Parameters
    ----------
    landmark_idx : int
        The landmark index (0-70 for OpenPose facial landmarks)
    axis : str
        The axis: 'x' or 'y' (no probability for aligned coordinates)

    Returns
    -------
    str
        The column name with _proc suffix, e.g., 'x37_proc', 'y46_proc'

    Examples
    --------
    >>> get_procrustes_column_name(37, 'x')
    'x37_proc'
    """
    return f"{axis}{landmark_idx}_proc"


def check_required_columns(df, landmark_indices: list, include_prob: bool = False) -> bool:
    """
    Check if a DataFrame has all required columns for specified landmarks.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check
    landmark_indices : list of int
        The landmark indices to check for
    include_prob : bool, optional
        Whether to also check for probability columns (default: False)

    Returns
    -------
    bool
        True if all required columns exist, False otherwise

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(columns=['x37', 'y37', 'x46', 'y46'])
    >>> check_required_columns(df, [37, 46])
    True
    >>> check_required_columns(df, [37, 46, 69])
    False
    """
    required_cols = []

    # Add x and y columns for each landmark
    for idx in landmark_indices:
        required_cols.append(get_column_name(idx, 'x'))
        required_cols.append(get_column_name(idx, 'y'))

    # Optionally add probability columns
    if include_prob:
        for idx in landmark_indices:
            required_cols.append(get_column_name(idx, 'prob'))

    # Check if all required columns exist
    return all(col in df.columns for col in required_cols)


def get_missing_columns(df, landmark_indices: list, include_prob: bool = False) -> list:
    """
    Get list of missing columns for specified landmarks.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check
    landmark_indices : list of int
        The landmark indices to check for
    include_prob : bool, optional
        Whether to also check for probability columns (default: False)

    Returns
    -------
    list of str
        List of column names that are missing from the DataFrame

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(columns=['x37', 'y37'])
    >>> get_missing_columns(df, [37, 46])
    ['x46', 'y46']
    """
    required_cols = []

    # Add x and y columns for each landmark
    for idx in landmark_indices:
        required_cols.append(get_column_name(idx, 'x'))
        required_cols.append(get_column_name(idx, 'y'))

    # Optionally add probability columns
    if include_prob:
        for idx in landmark_indices:
            required_cols.append(get_column_name(idx, 'prob'))

    # Return missing columns
    return [col for col in required_cols if col not in df.columns]


# ============================================================================
# REGIONAL GROUPINGS FOR AVERAGING
# ============================================================================

# Define facial regions for regional feature extraction
# Each region is averaged to create a more stable feature
FACIAL_REGIONS = {
    "center_face": NOSE_BRIDGE_INDICES,      # Nose bridge area
    "left_eye": EYES["L"],                   # Left eye region
    "right_eye": EYES["R"],                  # Right eye region
    "left_pupil": [LEFT_PUPIL],              # Left pupil (single point)
    "right_pupil": [RIGHT_PUPIL],            # Right pupil (single point)
}

# ============================================================================
# QUALITY CONTROL MAPPING
# ============================================================================

# Map QC metrics to the feature columns they affect
# Used to mask out bad data during preprocessing
QC_TO_FEATURE_COLUMNS = {
    "eyes": [
        "blink_dist",
        "left_eye_x", "left_eye_y", "left_eye_prob", "left_eye_magnitude",
        "right_eye_x", "right_eye_y", "right_eye_prob", "right_eye_magnitude",
    ],
    "head_rotation": ["head_rotation_angle"],
    "mouth_dist": ["mouth_dist"],
    "pupils_combined": [
        "avg_pupil_x", "avg_pupil_y", "avg_pupil_magnitude",
        "left_pupil_x", "left_pupil_y", "left_pupil_prob", "left_pupil_magnitude",
        "right_pupil_x", "right_pupil_y", "right_pupil_prob", "right_pupil_magnitude",
    ],
    "center_face": [
        "center_face_x", "center_face_y", "center_face_prob", "center_face_magnitude",
    ],
}