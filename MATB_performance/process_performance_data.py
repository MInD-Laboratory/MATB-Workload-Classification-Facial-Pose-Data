"""Process MATB performance data files and extract metrics.

This script processes raw MATB output CSV files, extracts windowed performance
metrics for all four sub-tasks (system monitoring, communications, tracking,
resource management), and saves individual and combined metric files.

Usage:
    python process_performance_data.py [--overwrite]

Output Structure:
    data/processed/
        metrics/        # Individual participant performance metrics
        combined/       # Combined metrics across all files
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from MATB_performance.utils.config import CFG
from MATB_performance.utils.performance_utils import (
    sysmon_measures,
    comms_measures,
    track_measures,
    resman_measures
)
from Pose.utils.preprocessing_utils import create_condition_mapping
from Pose.utils.io_utils import load_participant_info_file


def ensure_output_dirs() -> dict[str, Path]:
    """Create output directory structure.

    Returns:
        Dictionary mapping output types to Path objects
    """
    base_dir = Path(CFG.OUT_BASE)
    dirs = {
        'metrics': base_dir / 'metrics',
        'combined': base_dir / 'combined',
    }

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
        trial_num = int(session_str.replace('session', '').replace('.csv', ''))
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
    """Process a single MATB performance file.

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
    metrics_file = output_dirs['metrics'] / f"{participant_id}_{condition}_performance_metrics.csv"
    if metrics_file.exists() and not overwrite:
        print(f"  Output already exists: {metrics_file.name}")
        print(f"  Use --overwrite to reprocess")
        # Return existing data for combining
        try:
            return pd.read_csv(metrics_file)
        except:
            return None

    # Load MATB data
    try:
        df_matb = pd.read_csv(raw_dir / filename)
    except Exception as e:
        print(f"  ERROR: Failed to load data: {e}")
        return None

    print(f"  Loaded {len(df_matb)} rows")
    print(f"  Participant: {participant_id}, Condition: {condition}")

    # Filter to event and performance rows
    df_event_performance = df_matb[
        (df_matb['type'] == 'event') | (df_matb['type'] == 'performance')
    ].copy()

    # Ensure scenario_time is numeric
    df_event_performance['scenario_time'] = df_event_performance['scenario_time'].astype(float)

    # Extract metrics for each sub-task
    try:
        sysmon_failure_rate, sysmon_avg_rt, sysmon_events, sysmon_hits = sysmon_measures(
            df_event_performance,
            window_size=CFG.WINDOW_SECONDS,
            overlap=CFG.WINDOW_OVERLAP,
            total_time=CFG.TOTAL_TIME
        )

        comms_failure_rate, comms_events, comms_own_events, comms_avg_rt = comms_measures(
            df_event_performance,
            window_size=CFG.WINDOW_SECONDS,
            overlap=CFG.WINDOW_OVERLAP,
            total_time=CFG.TOTAL_TIME
        )

        track_failure_rate = track_measures(
            df_event_performance,
            window_size=CFG.WINDOW_SECONDS,
            overlap=CFG.WINDOW_OVERLAP,
            total_time=CFG.TOTAL_TIME
        )

        resman_failure_rate = resman_measures(
            df_event_performance,
            window_size=CFG.WINDOW_SECONDS,
            overlap=CFG.WINDOW_OVERLAP,
            total_time=CFG.TOTAL_TIME
        )
    except Exception as e:
        print(f"  ERROR: Failed to extract metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Build metrics DataFrame
    n_windows = len(sysmon_failure_rate)
    step = int(CFG.WINDOW_SECONDS * (1 - CFG.WINDOW_OVERLAP))
    window_starts = [w * step for w in range(n_windows)]
    window_ends = [start + CFG.WINDOW_SECONDS for start in window_starts]

    metrics_df = pd.DataFrame({
        'participant': participant_id,
        'condition': condition,
        'window_index': list(range(n_windows)),
        'start_time': window_starts,
        'end_time': window_ends,
        'sysmon_failure_rate': sysmon_failure_rate,
        'sysmon_average_reaction_times': sysmon_avg_rt,
        'comms_failure_rate': comms_failure_rate,
        'comms_events': comms_events,
        'comms_own_events': comms_own_events,
        'comms_average_reaction_times': comms_avg_rt,
        'track_failure_rate': track_failure_rate,
        'resman_failure_rate': resman_failure_rate
    })

    # Calculate aggregate metrics
    metrics_df['average_accuracy'] = 100 - (
        metrics_df['sysmon_failure_rate'] +
        metrics_df['comms_failure_rate'] +
        metrics_df['track_failure_rate'] +
        metrics_df['resman_failure_rate']
    ) / 4

    metrics_df['average_reaction_time'] = metrics_df[[
        'sysmon_average_reaction_times',
        'comms_average_reaction_times'
    ]].mean(axis=1) / 1000  # Convert to seconds

    # Save individual metrics file
    if CFG.SAVE_INDIVIDUAL:
        metrics_df.to_csv(metrics_file, index=False)
        print(f"  Saved metrics: {metrics_file.name}")
        print(f"  Extracted {len(metrics_df)} windows")

    return metrics_df


def run_performance_pipeline(overwrite: bool = False) -> None:
    """Run the complete MATB performance processing pipeline.

    Args:
        overwrite: Whether to overwrite existing output files
    """
    print("=" * 70)
    print("MATB Performance Processing Pipeline")
    print("=" * 70)

    # Setup paths
    raw_dir = Path(CFG.RAW_DIR)
    if not raw_dir.exists():
        print(f"\nERROR: Raw data directory not found: {raw_dir}")
        print(f"Please update MATB_RAW_DIR in .env to point to PNAS-MATB/matb_outputs/")
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
        condition_map = create_condition_mapping(participant_info)
        print(f"  Loaded {len(condition_map)} participant-trial-condition mappings")
    except Exception as e:
        print(f"\nERROR: Failed to load participant info: {e}")
        print(f"Make sure {CFG.PARTICIPANT_INFO_FILE} exists in PNAS-MATB folder")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Find all MATB CSV files
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
            if overwrite or not (output_dirs['metrics'] / f"{csv_file.stem.split('_')[0]}_*_performance_metrics.csv").parent.exists():
                successful += 1
            else:
                skipped += 1
        else:
            failed += 1

    # Combine all metrics
    if all_metrics and CFG.SAVE_COMBINED:
        print("\n" + "=" * 70)
        print("Combining metrics from all files...")
        combined_df = pd.concat(all_metrics, ignore_index=True)

        # Sort by participant, condition, window
        combined_df = combined_df.sort_values(
            ['participant', 'condition', 'window_index']
        ).reset_index(drop=True)

        # Save combined metrics
        combined_file = output_dirs['combined'] / 'performance_metrics_all.csv'
        combined_df.to_csv(combined_file, index=False)
        print(f"Saved combined metrics: {combined_file}")
        print(f"  Total windows: {len(combined_df)}")
        print(f"  Participants: {combined_df['participant'].nunique()}")
        print(f"  Conditions: {sorted(combined_df['condition'].unique())}")

        # Save processing summary
        summary = {
            'config': {
                'window_seconds': CFG.WINDOW_SECONDS,
                'window_overlap': CFG.WINDOW_OVERLAP,
                'total_time': CFG.TOTAL_TIME
            },
            'files_processed': len(all_metrics),
            'files_skipped': skipped,
            'files_errored': failed,
            'participants': sorted([str(p) for p in combined_df['participant'].unique()])
        }

        summary_file = output_dirs['combined'] / 'processing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved processing summary: {summary_file}")

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
        description='Process MATB performance data files and extract metrics',
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
        run_performance_pipeline(overwrite=args.overwrite)
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
