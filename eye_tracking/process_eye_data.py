"""Process eye tracking data files and extract metrics.

This script processes raw EyeLink CSV files, performs normalization and event
detection, and extracts windowed metrics for downstream analysis.

Usage:
    python process_eye_data.py [--overwrite]

Output Structure:
    data/processed/
        normalized/     # Normalized gaze coordinates and pupil data
        metrics/        # Windowed eye tracking metrics
        combined/       # Combined metrics across all files
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eye_tracking.utils.config import CFG
from eye_tracking.utils.eye_gaze_utils import (
    load_file_data,
    normalize_gaze_to_screen,
    pupil_blink_detection,
    fixation_detection,
    saccade_detection,
    extract_eye_metrics
)
from Pose.utils.preprocessing_utils import create_condition_mapping, load_participant_info
from Pose.utils.io_utils import load_participant_info_file


def ensure_output_dirs() -> dict[str, Path]:
    """Create output directory structure.

    Returns:
        Dictionary mapping output types to Path objects
    """
    base_dir = Path(CFG.OUT_BASE)
    dirs = {
        'normalized': base_dir / 'normalized',
        'metrics': base_dir / 'metrics',
        'combined': base_dir / 'combined',
    }

    if CFG.SAVE_EVENTS:
        dirs['events'] = base_dir / 'events'

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def map_session_to_condition(session_str: str, participant_id: str,
                             condition_map: dict) -> Optional[str]:
    """Map session string to condition code.

    Args:
        session_str: Session identifier (e.g., 'session01', 'session02')
        participant_id: Participant ID string
        condition_map: Dictionary mapping (participant, trial) to condition

    Returns:
        Condition code ('L', 'M', 'H') or None if not found
    """
    # Extract trial number from session string
    try:
        # session01 -> 1, session02 -> 2, session03 -> 3
        trial_num = int(session_str.replace('session', ''))
    except (ValueError, AttributeError):
        print(f"  Warning: Could not parse trial number from '{session_str}'")
        return None

    # Look up condition in nested mapping: {participant_id: {trial_num: condition}}
    if participant_id not in condition_map:
        print(f"  Warning: Participant {participant_id} not found in condition mapping")
        return None

    trial_map = condition_map[participant_id]
    if trial_num not in trial_map:
        print(f"  Warning: No condition found for participant {participant_id}, trial {trial_num}")
        return None

    return trial_map[trial_num]


def process_single_file(filename: str, raw_dir: Path, output_dirs: dict,
                       condition_map: dict, overwrite: bool = False) -> Optional[pd.DataFrame]:
    """Process a single eye tracking file.

    Args:
        filename: Name of CSV file to process
        raw_dir: Directory containing raw data
        output_dirs: Dictionary of output directory paths
        condition_map: Mapping of (participant, trial) to condition
        overwrite: Whether to overwrite existing output files

    Returns:
        DataFrame with extracted metrics, or None if processing failed
    """
    print(f"\nProcessing: {filename}")

    # Parse filename: 3105_session01.csv -> participant=3105, session=session01
    parts = filename.replace('.csv', '').split('_')
    if len(parts) < 2:
        print(f"  ERROR: Invalid filename format. Expected: <participantID>_session<number>.csv")
        return None

    participant_id = parts[0]
    session_str = parts[1]

    # Map session to condition
    condition = map_session_to_condition(session_str, participant_id, condition_map)
    if condition is None:
        print(f"  Skipping file due to missing condition mapping")
        return None

    # Check if output already exists
    metrics_file = output_dirs['metrics'] / f"{participant_id}_{condition}_eyegaze_metrics.csv"
    if metrics_file.exists() and not overwrite:
        print(f"  Output already exists: {metrics_file.name}")
        print(f"  Use --overwrite to reprocess")
        return None

    # Load data
    file_data = load_file_data(str(raw_dir), filename)
    if file_data is None:
        print(f"  ERROR: Failed to load data")
        return None

    # Extract DataFrame from returned dictionary
    df = file_data['data']

    print(f"  Loaded {len(df)} samples ({len(df)/CFG.SAMPLE_RATE:.1f} seconds)")
    print(f"  Participant: {participant_id}, Condition: {condition}")

    # Normalize gaze coordinates
    df = normalize_gaze_to_screen(df)

    # Add normalized column names for easier access
    if 'R Gaze X Norm' not in df.columns:
        df['R Gaze X Norm'] = df['R Gaze X']
        df['R Gaze Y Norm'] = df['R Gaze Y']
        df['L Gaze X Norm'] = df['L Gaze X']
        df['L Gaze Y Norm'] = df['L Gaze Y']

    # Detect blinks (returns event lists, not masks)
    blink_starts, blink_ends = pupil_blink_detection(
        df['R Pupil Size'].values,
        df['L Pupil Size'].values,
        df['Time Stamp'].values
    )

    # Create boolean blink mask from events
    blink_mask = np.zeros(len(df), dtype=bool)
    for start, end in zip(blink_starts, blink_ends):
        mask_idx = (df['Time Stamp'] >= start[0]) & (df['Time Stamp'] <= end[1])
        blink_mask[mask_idx] = True
    df['Blink'] = blink_mask

    # Detect fixations (returns event lists, not masks)
    fix_starts, fix_ends = fixation_detection(
        df['R Gaze X Norm'].values,
        df['R Gaze Y Norm'].values,
        df['Time Stamp'].values
    )

    # Create boolean fixation mask from events
    fixation_mask = np.zeros(len(df), dtype=bool)
    for start, end in zip(fix_starts, fix_ends):
        mask_idx = (df['Time Stamp'] >= start[0]) & (df['Time Stamp'] <= end[1])
        fixation_mask[mask_idx] = True
    df['Fixation'] = fixation_mask

    # Detect saccades (returns event lists, not masks)
    sac_starts, sac_ends, _, _ = saccade_detection(
        df['R Gaze X Norm'].values,
        df['R Gaze Y Norm'].values,
        df['Time Stamp'].values
    )

    # Create boolean saccade mask from events
    saccade_mask = np.zeros(len(df), dtype=bool)
    for start, end in zip(sac_starts, sac_ends):
        mask_idx = (df['Time Stamp'] >= start[0]) & (df['Time Stamp'] <= end[1])
        saccade_mask[mask_idx] = True
    df['Saccade'] = saccade_mask

    # Save normalized data if requested
    if CFG.SAVE_NORMALIZED:
        norm_file = output_dirs['normalized'] / f"{participant_id}_{condition}_eyegaze_normalized.csv"
        df.to_csv(norm_file, index=False)
        print(f"  Saved normalized data: {norm_file.name}")

    # Save event data if requested
    if CFG.SAVE_EVENTS:
        events_df = df[['Time Stamp', 'Blink', 'Fixation', 'Saccade']].copy()
        events_file = output_dirs['events'] / f"{participant_id}_{condition}_eyegaze_events.csv"
        events_df.to_csv(events_file, index=False)
        print(f"  Saved events: {events_file.name}")

    # Extract windowed metrics
    metrics_df = extract_eye_metrics(df, participant_id, condition)

    if metrics_df is not None and len(metrics_df) > 0:
        # Save metrics
        metrics_df.to_csv(metrics_file, index=False)
        print(f"  Saved metrics: {metrics_file.name}")
        print(f"  Extracted {len(metrics_df)} windows")
        return metrics_df
    else:
        print(f"  WARNING: No metrics extracted")
        return None


def run_eye_tracking_pipeline(overwrite: bool = False) -> None:
    """Run the complete eye tracking processing pipeline.

    Args:
        overwrite: Whether to overwrite existing output files
    """
    print("=" * 70)
    print("Eye Tracking Processing Pipeline")
    print("=" * 70)

    # Setup paths
    raw_dir = Path(CFG.RAW_DIR)
    if not raw_dir.exists():
        print(f"\nERROR: Raw data directory not found: {raw_dir}")
        print(f"Please update EYELINK_RAW_DIR in .env or config.py")
        sys.exit(1)

    print(f"\nInput directory: {raw_dir}")
    print(f"Output directory: {CFG.OUT_BASE}")
    print(f"Overwrite mode: {overwrite}")

    # Create output directories
    output_dirs = ensure_output_dirs()
    print(f"\nOutput structure:")
    for name, path in output_dirs.items():
        print(f"  {name:12s}: {path}")

    # Load participant info and create condition mapping
    print(f"\nLoading participant info: {CFG.PARTICIPANT_INFO_FILE}")
    try:
        participant_info_path = load_participant_info_file()
        participant_info = pd.read_csv(participant_info_path)

        # Create condition mapping
        # Note: create_condition_mapping expects columns:
        # "Participant ID", "session01", "session02", "session03"
        condition_map = create_condition_mapping(participant_info)
        print(f"  Loaded {len(condition_map)} participant-trial-condition mappings")
    except Exception as e:
        print(f"\nERROR: Failed to load participant info: {e}")
        print(f"Make sure {CFG.PARTICIPANT_INFO_FILE} exists in the project root")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Find all CSV files
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        print(f"\nERROR: No CSV files found in {raw_dir}")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} CSV files to process")

    # Process each file
    all_metrics = []
    successful = 0
    skipped = 0
    failed = 0

    for csv_file in csv_files:
        metrics_df = process_single_file(
            csv_file.name,
            raw_dir,
            output_dirs,
            condition_map,
            overwrite=overwrite
        )

        if metrics_df is not None:
            all_metrics.append(metrics_df)
            successful += 1
        elif Path(output_dirs['metrics'],
                  f"{csv_file.stem.split('_')[0]}_*_eyegaze_metrics.csv").parent.exists():
            skipped += 1
        else:
            failed += 1

    # Combine all metrics
    if all_metrics:
        print("\n" + "=" * 70)
        print("Combining metrics from all files...")
        combined_df = pd.concat(all_metrics, ignore_index=True)

        # Sort by participant, condition, window
        combined_df = combined_df.sort_values(
            ['participant', 'condition', 'window_index']
        ).reset_index(drop=True)

        # Save combined metrics
        combined_file = output_dirs['combined'] / 'eyegaze_metrics_all.csv'
        combined_df.to_csv(combined_file, index=False)
        print(f"Saved combined metrics: {combined_file}")
        print(f"  Total windows: {len(combined_df)}")
        print(f"  Participants: {combined_df['participant'].nunique()}")
        print(f"  Conditions: {sorted(combined_df['condition'].unique())}")

    # Print summary
    print("\n" + "=" * 70)
    print("Processing Summary")
    print("=" * 70)
    print(f"Total files:       {len(csv_files)}")
    print(f"Successful:        {successful}")
    print(f"Skipped (exists):  {skipped}")
    print(f"Failed:            {failed}")
    print("=" * 70)

    if failed > 0:
        print("\nWARNING: Some files failed to process. Check error messages above.")
        sys.exit(1)
    else:
        print("\nProcessing complete!")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Process eye tracking data files and extract metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )

    args = parser.parse_args()

    try:
        run_eye_tracking_pipeline(overwrite=args.overwrite)
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Unexpected error during processing:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
