"""
================================================================================
MAIN PROCESSING PIPELINE FOR FACIAL POSE DATA
================================================================================

This module coordinates the complete preprocessing workflow for facial pose data.
It orchestrates all the individual processing steps from quality control through
feature extraction to final output generation.

The pipeline includes:
1. Quality control analysis
2. Coordinate normalization (Procrustes or original)
3. Feature extraction
4. Quality control masking
5. Temporal filtering
6. Output generation and reporting

Author: Pose Analysis Pipeline
Date: 2024
================================================================================
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import Optional, Dict, List

from .quality_control import run_quality_control_batch
from .coordinate_normalization import compute_procrustes_alignment
from .feature_extraction import extract_all_features, extract_features_flexible
from .masking import load_bad_windows, apply_bad_window_masks
from .temporal_filtering import apply_butterworth_filter
from .landmark_config import QC_TO_FEATURE_COLUMNS


# ============================================================================
# COMPLETE PROCESSING PIPELINE
# ============================================================================

def run_complete_pose_pipeline(
    raw_input_dir: str,
    output_base_dir: str,
    window_size: int = 1800,
    overlap: float = 0.0,
    confidence_threshold: float = 0.3,
    max_interpolation: int = 60,
    coordinate_system: str = "procrustes",
    apply_temporal_filter: bool = True,
    sampling_rate: float = 60.0,
    cutoff_frequency: float = 10.0,
    filter_order: int = 4,
    qc_to_columns_map: Optional[Dict[str, List[str]]] = None,
    feature_procrustes_config: Optional[Dict[str, bool]] = None,
    pupil_relative_to_nose: bool = False
) -> Dict[str, str]:
    """
    Execute the complete facial pose processing pipeline.

    This function coordinates all processing steps from raw landmark data
    to final processed features ready for analysis.

    Parameters
    ----------
    raw_input_dir : str
        Directory containing raw CSV files with landmark data
    output_base_dir : str
        Base directory for all outputs (subdirectories will be created)
    window_size : int, optional
        QC window size in frames (default: 1800 = 30s at 60fps)
    overlap : float, optional
        QC window overlap fraction (default: 0.0)
    confidence_threshold : float, optional
        Minimum landmark confidence threshold (default: 0.3)
    max_interpolation : int, optional
        Maximum consecutive frames to interpolate (default: 60)
    coordinate_system : str, optional
        Coordinate normalization method: "procrustes" or "original" (default: "procrustes")
    apply_temporal_filter : bool, optional
        Whether to apply Butterworth filtering (default: True)
    sampling_rate : float, optional
        Data sampling rate in Hz (default: 60.0)
    cutoff_frequency : float, optional
        Butterworth filter cutoff frequency (default: 10.0)
    filter_order : int, optional
        Butterworth filter order (default: 4)
    qc_to_columns_map : dict, optional
        Custom mapping of QC metrics to feature columns
    feature_procrustes_config : dict, optional
        Per-feature Procrustes settings. Example:
        {'blink_dist': True, 'mouth_dist': False, 'pupils': True}
    pupil_relative_to_nose : bool, optional
        Whether to calculate pupil features relative to nose center (default: False)

    Returns
    -------
    dict
        Dictionary with paths to all output files and directories

    Examples
    --------
    >>> paths = run_complete_pose_pipeline(
    ...     'data/raw_pose',
    ...     'data/processed',
    ...     coordinate_system='procrustes',
    ...     apply_temporal_filter=True
    ... )
    >>> print(f"Features saved to: {paths['feature_dir']}")
    """
    print("="*80)
    print("FACIAL POSE PROCESSING PIPELINE")
    print("="*80)
    print(f"Input directory: {raw_input_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Coordinate system: {coordinate_system}")
    print(f"Temporal filtering: {apply_temporal_filter}")
    print()

    # Create output directory structure
    os.makedirs(output_base_dir, exist_ok=True)
    qc_dir = os.path.join(output_base_dir, "quality_control")
    feature_dir = os.path.join(output_base_dir, "feature_data")
    reports_dir = os.path.join(output_base_dir, "reports")

    os.makedirs(qc_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Use default QC mapping if not provided
    if qc_to_columns_map is None:
        qc_to_columns_map = QC_TO_FEATURE_COLUMNS

    # ========================================================================
    # STEP 1: QUALITY CONTROL ANALYSIS
    # ========================================================================
    print("STEP 1: Running Quality Control Analysis...")
    print("-" * 40)

    keypoint_path, metric_path, detail_path = run_quality_control_batch(
        input_dir=raw_input_dir,
        output_dir=qc_dir,
        window_size=window_size,
        overlap=overlap,
        conf_threshold=confidence_threshold,
        max_interpolation=max_interpolation
    )

    # ========================================================================
    # STEP 2: LOAD BAD WINDOWS FOR MASKING
    # ========================================================================
    print("\nSTEP 2: Loading Quality Control Results...")
    print("-" * 40)

    bad_windows_map, step_frames = load_bad_windows(
        detail_path, window_size, overlap
    )

    # ========================================================================
    # STEP 3: PROCESS EACH FILE
    # ========================================================================
    print("\nSTEP 3: Processing Individual Files...")
    print("-" * 40)

    csv_files = [f for f in os.listdir(raw_input_dir) if f.endswith('.csv')]
    processing_report = []

    for csv_file in tqdm(csv_files, desc=f"Processing files"):
        file_path = os.path.join(raw_input_dir, csv_file)
        base_name = os.path.splitext(csv_file)[0]

        try:
            # Load raw landmark data
            df_raw = pd.read_csv(file_path)
            print(f"  Processing {csv_file}: {len(df_raw)} frames")

            # Apply coordinate normalization
            if coordinate_system == "procrustes":
                print(f"    Applying Procrustes alignment...")
                df_raw = compute_procrustes_alignment(df_raw)

            # Extract all features with flexible Procrustes control
            print(f"    Extracting features...")
            df_features = extract_all_features(
                df_raw,
                use_procrustes=(coordinate_system == "procrustes"),
                confidence_threshold=confidence_threshold,
                feature_procrustes_config=feature_procrustes_config,
                pupil_relative_to_nose=pupil_relative_to_nose
            )

            # Apply quality control masks
            print(f"    Applying QC masks...")
            df_masked, mask_stats, total_masked = apply_bad_window_masks(
                filename=csv_file,
                df_features=df_features,
                bad_windows_map=bad_windows_map,
                qc_to_columns_map=qc_to_columns_map
            )

            # Apply temporal filtering
            if apply_temporal_filter:
                print(f"    Applying temporal filtering...")
                df_filtered = apply_butterworth_filter(
                    df_masked,
                    sampling_rate=sampling_rate,
                    cutoff_frequency=cutoff_frequency,
                    filter_order=filter_order,
                    verbose=False
                )
            else:
                df_filtered = df_masked

            # Save processed features
            output_path = os.path.join(feature_dir, f"{base_name}.csv")
            df_filtered.to_csv(output_path, index=False)

            # Record processing statistics
            for qc_metric, stats in mask_stats.items():
                processing_report.append({
                    "file": csv_file,
                    "qc_metric": qc_metric,
                    "frames_total": stats["frames_total"],
                    "frames_masked": stats["frames_masked"],
                    "pct_masked": stats["pct_masked"],
                    "windows_masked": stats["windows_masked"],
                    "coordinate_system": coordinate_system,
                    "temporal_filter": apply_temporal_filter,
                    "cutoff_frequency": cutoff_frequency if apply_temporal_filter else None,
                    "num_features": len(df_filtered.columns)
                })

            print(f"    Saved to: {output_path}")

        except Exception as e:
            print(f"    ERROR processing {csv_file}: {e}")
            # Record error in report
            processing_report.append({
                "file": csv_file,
                "qc_metric": "ERROR",
                "frames_total": 0,
                "frames_masked": 0,
                "pct_masked": np.nan,
                "windows_masked": 0,
                "coordinate_system": coordinate_system,
                "temporal_filter": apply_temporal_filter,
                "error": str(e)
            })

    # ========================================================================
    # STEP 4: GENERATE REPORTS
    # ========================================================================
    print("\nSTEP 4: Generating Reports...")
    print("-" * 40)

    # Save processing report
    report_path = os.path.join(reports_dir, "processing_report.csv")
    pd.DataFrame(processing_report).to_csv(report_path, index=False)

    # Generate summary statistics
    successful_files = [item for item in processing_report if item.get("qc_metric") != "ERROR"]
    if successful_files:
        summary_stats = {
            "total_files_processed": len(csv_files),
            "successful_files": len(set(item["file"] for item in successful_files)),
            "failed_files": len(csv_files) - len(set(item["file"] for item in successful_files)),
            "total_frames": sum(item["frames_total"] for item in successful_files),
            "total_masked_frames": sum(item["frames_masked"] for item in successful_files),
            "overall_masking_rate": (
                sum(item["frames_masked"] for item in successful_files) /
                sum(item["frames_total"] for item in successful_files) * 100
                if sum(item["frames_total"] for item in successful_files) > 0 else 0
            ),
            "coordinate_system": coordinate_system,
            "temporal_filtering": apply_temporal_filter
        }

        summary_path = os.path.join(reports_dir, "processing_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("FACIAL POSE PROCESSING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")

        print(f"Summary statistics:")
        for key, value in summary_stats.items():
            print(f"  {key}: {value}")

    # ========================================================================
    # PIPELINE COMPLETE
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)

    output_paths = {
        "qc_dir": qc_dir,
        "feature_dir": feature_dir,
        "reports_dir": reports_dir,
        "keypoint_qc": keypoint_path,
        "metric_qc": metric_path,
        "bad_windows": detail_path,
        "processing_report": report_path,
        "summary": summary_path if 'summary_path' in locals() else None
    }

    print("Output locations:")
    for key, path in output_paths.items():
        if path:
            print(f"  {key}: {path}")

    return output_paths


# ============================================================================
# SIMPLIFIED PROCESSING FUNCTIONS
# ============================================================================

def process_single_file(
    input_file: str,
    output_file: str,
    coordinate_system: str = "procrustes",
    confidence_threshold: float = 0.3,
    apply_temporal_filter: bool = True,
    sampling_rate: float = 60.0,
    cutoff_frequency: float = 10.0,
    feature_procrustes_config: Optional[Dict[str, bool]] = None,
    pupil_relative_to_nose: bool = False
) -> bool:
    """
    Process a single CSV file through the pose pipeline.

    Simplified function for processing individual files without QC masking.

    Parameters
    ----------
    input_file : str
        Path to input CSV file
    output_file : str
        Path for output CSV file
    coordinate_system : str, optional
        Coordinate system to use ("procrustes" or "original")
    confidence_threshold : float, optional
        Minimum confidence threshold
    apply_temporal_filter : bool, optional
        Whether to apply temporal filtering
    sampling_rate : float, optional
        Sampling rate for filtering
    cutoff_frequency : float, optional
        Cutoff frequency for filtering

    Returns
    -------
    bool
        True if processing succeeded, False otherwise
    """
    try:
        # Load data
        df_raw = pd.read_csv(input_file)

        # Apply coordinate normalization
        if coordinate_system == "procrustes":
            df_raw = compute_procrustes_alignment(df_raw)

        # Extract features with flexible configuration
        df_features = extract_all_features(
            df_raw,
            use_procrustes=(coordinate_system == "procrustes"),
            confidence_threshold=confidence_threshold,
            feature_procrustes_config=feature_procrustes_config,
            pupil_relative_to_nose=pupil_relative_to_nose
        )

        # Apply temporal filtering
        if apply_temporal_filter:
            df_features = apply_butterworth_filter(
                df_features,
                sampling_rate=sampling_rate,
                cutoff_frequency=cutoff_frequency,
                verbose=False
            )

        # Save result
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_features.to_csv(output_file, index=False)

        return True

    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False


def batch_process_files(
    input_dir: str,
    output_dir: str,
    coordinate_system: str = "procrustes",
    apply_temporal_filter: bool = True
) -> List[str]:
    """
    Process all CSV files in a directory (without QC masking).

    Simplified batch processing for when QC masking is not needed.

    Parameters
    ----------
    input_dir : str
        Directory containing input CSV files
    output_dir : str
        Directory for output files
    coordinate_system : str, optional
        Coordinate system to use
    apply_temporal_filter : bool, optional
        Whether to apply temporal filtering

    Returns
    -------
    list of str
        List of successfully processed files
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    processed_files = []

    for csv_file in tqdm(csv_files, desc="Processing files"):
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, csv_file)

        if process_single_file(
            input_path, output_path,
            coordinate_system=coordinate_system,
            apply_temporal_filter=apply_temporal_filter
        ):
            processed_files.append(csv_file)

    print(f"Successfully processed {len(processed_files)}/{len(csv_files)} files")
    return processed_files