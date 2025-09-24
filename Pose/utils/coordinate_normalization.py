"""
================================================================================
COORDINATE NORMALIZATION MODULE FOR FACIAL POSE DATA
================================================================================

This module provides functions for normalizing facial landmark coordinates to
remove effects of head position, rotation, and scale. Two methods are provided:

1. Original Method: Uses eye corners as reference points for stabilization
2. Procrustes Method: Aligns all landmarks to a reference shape (more robust)

The normalization process ensures that facial features can be compared across
different frames and participants regardless of head pose variations.

Author: Pose Analysis Pipeline
Date: 2024
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

from .landmark_config import (
    ANCHOR_L, ANCHOR_R,
    EYES,
    MOUTH_TOP_ALT, MOUTH_BOTTOM_ALT,
    LEFT_PUPIL, RIGHT_PUPIL,
    FACIAL_REGIONS,
    NOSE_BRIDGE_INDICES,
    PROCRUSTES_REFERENCE_LANDMARKS,
    get_column_name,
    get_procrustes_column_name
)


# ============================================================================
# ORIGINAL NORMALIZATION METHOD (EYE-CORNER BASED)
# ============================================================================

def stabilize_points_original(df: pd.DataFrame, landmark_indices: List[int]) -> Dict[int, Tuple[pd.Series, pd.Series]]:
    """
    Apply head stabilization using eye corners as reference points.

    This normalizes for head position, rotation, and size by:
    1. Centering coordinates at the midpoint between eye corners
    2. Rotating to make the eye line horizontal
    3. Scaling by the distance between eye corners (inter-ocular distance)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw landmark coordinates
    landmark_indices : list of int
        Indices of landmarks to stabilize

    Returns
    -------
    dict
        Dictionary mapping landmark index to tuple of (x, y) stabilized coordinates

    Examples
    --------
    >>> df = pd.DataFrame({'x37': [100], 'y37': [100], 'x46': [200], 'y46': [100]})
    >>> stabilized = stabilize_points_original(df, [37, 46])
    >>> x37, y37 = stabilized[37]
    >>> float(x37.iloc[0])  # Normalized x coordinate
    -0.5
    """
    # Get eye corner coordinates (reference points)
    x_left = df[get_column_name(ANCHOR_L, 'x')]
    y_left = df[get_column_name(ANCHOR_L, 'y')]
    x_right = df[get_column_name(ANCHOR_R, 'x')]
    y_right = df[get_column_name(ANCHOR_R, 'y')]

    # Calculate center point (midpoint between eye corners)
    center_x = (x_left + x_right) / 2.0
    center_y = (y_left + y_right) / 2.0

    # Calculate rotation angle to make eye line horizontal
    dx = x_right - x_left
    dy = y_right - y_left
    theta = np.arctan2(dy, dx)

    # Pre-compute rotation matrix components
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Calculate scaling factor (inter-ocular distance)
    # Use a small epsilon to avoid division by zero
    inter_ocular_distance = np.sqrt(dx * dx + dy * dy)
    inter_ocular_distance = inter_ocular_distance.where(
        lambda s: s > 1e-6, np.nan
    )

    # Apply transformation to all requested landmarks
    stabilized = {}
    for landmark_idx in landmark_indices:
        # Get original coordinates
        x_col = get_column_name(landmark_idx, 'x')
        y_col = get_column_name(landmark_idx, 'y')

        if x_col not in df.columns or y_col not in df.columns:
            # If columns don't exist, return NaN series
            stabilized[landmark_idx] = (
                pd.Series(np.nan, index=df.index),
                pd.Series(np.nan, index=df.index)
            )
            continue

        # Translate to center at origin
        x_centered = df[x_col] - center_x
        y_centered = df[y_col] - center_y

        # Rotate by -theta to align horizontally
        x_rotated = cos_theta * x_centered + sin_theta * y_centered
        y_rotated = -sin_theta * x_centered + cos_theta * y_centered

        # Scale by inter-ocular distance
        x_normalized = x_rotated / inter_ocular_distance
        y_normalized = y_rotated / inter_ocular_distance

        stabilized[landmark_idx] = (x_normalized, y_normalized)

    return stabilized


def apply_original_normalization(df: pd.DataFrame, landmark_indices: List[int],
                                suffix: str = "_norm") -> pd.DataFrame:
    """
    Apply original normalization method and add normalized columns to DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw landmark coordinates
    landmark_indices : list of int
        Indices of landmarks to normalize
    suffix : str, optional
        Suffix to add to normalized column names (default: "_norm")

    Returns
    -------
    pd.DataFrame
        DataFrame with added normalized coordinate columns

    Examples
    --------
    >>> df = pd.DataFrame({'x37': [100], 'y37': [100], 'x46': [200], 'y46': [100]})
    >>> df_norm = apply_original_normalization(df, [37, 46])
    >>> 'x37_norm' in df_norm.columns
    True
    """
    # Make a copy to avoid modifying original
    df_result = df.copy()

    # Get stabilized coordinates
    stabilized = stabilize_points_original(df, landmark_indices)

    # Add stabilized coordinates to DataFrame
    for landmark_idx, (x_norm, y_norm) in stabilized.items():
        x_col = f"x{landmark_idx}{suffix}"
        y_col = f"y{landmark_idx}{suffix}"
        df_result[x_col] = x_norm
        df_result[y_col] = y_norm

    return df_result


# ============================================================================
# PROCRUSTES NORMALIZATION METHOD
# ============================================================================

def procrustes_analysis(reference_shape: np.ndarray, target_shape: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Perform Procrustes analysis to align two sets of landmarks.

    Finds the optimal similarity transformation (translation + rotation + scaling)
    to align target_shape to reference_shape. This method is more robust than
    the eye-corner method because it uses all available landmarks.

    Parameters
    ----------
    reference_shape : np.ndarray
        Reference shape to align to, shape (n_landmarks, 2)
    target_shape : np.ndarray
        Shape to be aligned, shape (n_landmarks, 2)

    Returns
    -------
    tuple
        aligned_shape : np.ndarray - Aligned version of target_shape
        scale : float - Scaling factor applied
        rotation : np.ndarray - 2x2 rotation matrix applied
        translation : np.ndarray - Translation vector applied

    Examples
    --------
    >>> ref = np.array([[0, 0], [1, 0], [0, 1]])
    >>> target = np.array([[1, 1], [2, 1], [1, 2]])
    >>> aligned, scale, rot, trans = procrustes_analysis(ref, target)
    >>> aligned.shape
    (3, 2)
    """
    # Ensure inputs are float arrays
    reference_shape = np.array(reference_shape, dtype=float)
    target_shape = np.array(target_shape, dtype=float)

    # Step 1: Remove translation by centering both shapes at origin
    ref_centroid = reference_shape.mean(axis=0)
    target_centroid = target_shape.mean(axis=0)

    ref_centered = reference_shape - ref_centroid
    target_centered = target_shape - target_centroid

    # Step 2: Calculate optimal scaling factor
    ref_scale = np.sqrt(np.sum(ref_centered ** 2))
    target_scale = np.sqrt(np.sum(target_centered ** 2))

    # Avoid division by zero
    if target_scale < 1e-10:
        return target_shape, 1.0, np.eye(2), np.zeros(2)

    scale = ref_scale / target_scale
    target_scaled = target_centered * scale

    # Step 3: Find optimal rotation using Singular Value Decomposition (SVD)
    # Compute cross-covariance matrix
    H = target_scaled.T @ ref_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure we have a proper rotation (not a reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 4: Apply rotation
    target_rotated = target_scaled @ R.T

    # Step 5: Calculate final translation
    translation = ref_centroid - target_rotated.mean(axis=0)
    aligned_shape = target_rotated + translation

    return aligned_shape, scale, R, translation


def get_landmarks_used_in_features() -> List[int]:
    """
    Get all landmark indices that are actually used in feature extraction.
    """
    landmarks_used = set()
    
    # Eye landmarks for blink detection
    if 'EYES' in globals():
        landmarks_used.update(EYES.get("L", []))
        landmarks_used.update(EYES.get("R", []))
    
    # Anchor points
    if 'ANCHOR_L' in globals():
        landmarks_used.add(ANCHOR_L)
    if 'ANCHOR_R' in globals():
        landmarks_used.add(ANCHOR_R)
    
    # Mouth landmarks
    if 'MOUTH_TOP_ALT' in globals():
        landmarks_used.add(MOUTH_TOP_ALT)
    if 'MOUTH_BOTTOM_ALT' in globals():
        landmarks_used.add(MOUTH_BOTTOM_ALT)
    
    # Pupil landmarks
    if 'LEFT_PUPIL' in globals():
        landmarks_used.add(LEFT_PUPIL)
    if 'RIGHT_PUPIL' in globals():
        landmarks_used.add(RIGHT_PUPIL)
    
    # Regional landmarks
    if 'FACIAL_REGIONS' in globals():
        for region_landmarks in FACIAL_REGIONS.values():
            landmarks_used.update(region_landmarks)
    
    # Nose landmarks (for pupil relative features)
    if 'NOSE_BRIDGE_INDICES' in globals():
        landmarks_used.update(NOSE_BRIDGE_INDICES)
    
    return sorted(list(landmarks_used))


def compute_procrustes_alignment(df: pd.DataFrame,
                                reference_landmarks: Optional[List[int]] = None,
                                landmarks_to_transform: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Apply Procrustes alignment to all landmarks in a DataFrame.

    Process:
    1. Selects stable landmarks as reference points
    2. Calculates the average shape across all valid frames
    3. Aligns each frame to this average shape
    4. Adds new columns with "_proc" suffix for aligned coordinates

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw landmark coordinates
    reference_landmarks : list of int, optional
        Indices of landmarks to use for alignment reference
        If None, uses default stable landmarks (temples and nose)

    Returns
    -------
    pd.DataFrame
        DataFrame with added Procrustes-aligned coordinate columns

    Examples
    --------
    >>> df = pd.read_csv('participant_001.csv')
    >>> df_aligned = compute_procrustes_alignment(df)
    >>> 'x37_proc' in df_aligned.columns
    True
    """
    # Use default reference landmarks if not specified
    if reference_landmarks is None:
        reference_landmarks = PROCRUSTES_REFERENCE_LANDMARKS

    if landmarks_to_transform is None:
        landmarks_to_transform = get_landmarks_used_in_features()

    available_landmarks_to_transform = []
    for landmark in landmarks_to_transform:
        x_col = get_column_name(landmark, 'x')
        y_col = get_column_name(landmark, 'y')
        if x_col in df.columns and y_col in df.columns:
            available_landmarks_to_transform.append(landmark)
    print(f"Transforming {len(available_landmarks_to_transform)} feature-relevant landmarks")

    # Check reference landmark availability
    available_ref_landmarks = []
    for i in reference_landmarks:
        x_col = get_column_name(i, 'x')
        y_col = get_column_name(i, 'y')
        if x_col in df.columns and y_col in df.columns:
            available_ref_landmarks.append(i)

    if len(available_ref_landmarks) < 3:
        print(f"Warning: Need at least 3 reference landmarks, found {len(available_ref_landmarks)}")
        return df

    # Extract landmark coordinates into 3D array: (frames, landmarks, coordinates)
    n_frames = len(df)
    n_ref_landmarks = len(available_ref_landmarks)
    ref_coords = np.zeros((n_frames, n_ref_landmarks, 2))
    valid_frames = np.ones(n_frames, dtype=bool)

    coords = np.zeros((n_frames, n_landmarks, 2))
    valid_frames = np.ones(n_frames, dtype=bool)

    for frame_idx in range(n_frames):
        for lm_idx, landmark in enumerate(available_ref_landmarks):
            x_val = df.loc[frame_idx, get_column_name(landmark, 'x')]
            y_val = df.loc[frame_idx, get_column_name(landmark, 'y')]

            if pd.isna(x_val) or pd.isna(y_val):
                valid_frames[frame_idx] = False
                break

            ref_coords[frame_idx, lm_idx, 0] = x_val
            ref_coords[frame_idx, lm_idx, 1] = y_val

    if valid_frames.sum() == 0:
        print("Warning: No frames with complete reference data")
        return df

    # Calculate reference shape (average of all valid frames)
    reference_shape = coords[valid_frames].mean(axis=0)

    # Apply Procrustes alignment to each frame
    df_aligned = df.copy()

    # Process each frame - but only transform the landmarks we need
    for frame_idx in range(n_frames):
        if not valid_frames[frame_idx]:
            # Set only the landmarks we're transforming to NaN
            for landmark in available_landmarks_to_transform:
                df_aligned.loc[frame_idx, get_procrustes_column_name(landmark, 'x')] = np.nan
                df_aligned.loc[frame_idx, get_procrustes_column_name(landmark, 'y')] = np.nan
            continue

        try:
            # Get transformation from reference landmarks
            current_ref_shape = ref_coords[frame_idx]
            aligned_ref_shape, scale, rotation, translation = procrustes_analysis(
                reference_shape, current_ref_shape
            )

            # Calculate transformation parameters
            orig_ref_centroid = current_ref_shape.mean(axis=0)
            target_ref_centroid = reference_shape.mean(axis=0)

            # Apply transformation to only the landmarks we need for features
            for landmark in available_landmarks_to_transform:
                x_col = get_column_name(landmark, 'x')
                y_col = get_column_name(landmark, 'y')
                
                orig_x = df.loc[frame_idx, x_col]
                orig_y = df.loc[frame_idx, y_col]
                
                if not (pd.isna(orig_x) or pd.isna(orig_y)):
                    # Apply transformation
                    centered_x = orig_x - orig_ref_centroid[0]
                    centered_y = orig_y - orig_ref_centroid[1]
                    
                    scaled_x = centered_x * scale
                    scaled_y = centered_y * scale
                    
                    rotated_x = rotation[0, 0] * scaled_x + rotation[0, 1] * scaled_y
                    rotated_y = rotation[1, 0] * scaled_x + rotation[1, 1] * scaled_y
                    
                    final_x = rotated_x + target_ref_centroid[0]
                    final_y = rotated_y + target_ref_centroid[1]
                    
                    # Store result
                    df_aligned.loc[frame_idx, get_procrustes_column_name(landmark, 'x')] = final_x
                    df_aligned.loc[frame_idx, get_procrustes_column_name(landmark, 'y')] = final_y
                else:
                    df_aligned.loc[frame_idx, get_procrustes_column_name(landmark, 'x')] = np.nan
                    df_aligned.loc[frame_idx, get_procrustes_column_name(landmark, 'y')] = np.nan

        except Exception as e:
            print(f"Warning: Frame {frame_idx} transformation failed: {e}")
            for landmark in available_landmarks_to_transform:
                df_aligned.loc[frame_idx, get_procrustes_column_name(landmark, 'x')] = np.nan
                df_aligned.loc[frame_idx, get_procrustes_column_name(landmark, 'y')] = np.nan

    print(f"Procrustes alignment completed for {len(available_landmarks_to_transform)} feature landmarks")
    return df_aligned


def get_best_coordinate(df: pd.DataFrame, landmark: int, axis: str,
                        prefer_procrustes: bool = True) -> pd.Series:
    """
    Get coordinate series, preferring Procrustes-aligned version if available.

    This helper function intelligently selects between original and Procrustes-aligned
    coordinates based on availability and preference.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing landmark coordinates
    landmark : int
        Landmark index
    axis : str
        Axis ('x' or 'y')
    prefer_procrustes : bool, optional
        Whether to prefer Procrustes coordinates when available (default: True)

    Returns
    -------
    pd.Series
        Coordinate series (may contain NaN values)

    Examples
    --------
    >>> df = pd.DataFrame({'x37': [100], 'x37_proc': [0.5]})
    >>> coords = get_best_coordinate(df, 37, 'x', prefer_procrustes=True)
    >>> float(coords.iloc[0])
    0.5
    """
    proc_col = get_procrustes_column_name(landmark, axis)
    orig_col = get_column_name(landmark, axis)

    if prefer_procrustes and proc_col in df.columns:
        # Use Procrustes coordinates, filling NaN with original
        return df[proc_col].fillna(df.get(orig_col, np.nan))
    elif orig_col in df.columns:
        # Use original coordinates
        return df[orig_col]
    else:
        # Neither exists, return NaN series
        return pd.Series(np.nan, index=df.index)