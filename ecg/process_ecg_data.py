#!/usr/bin/env python3
"""
ECG Data Processing Pipeline

Processes Zephyr ECG data:
1. Load raw ECG signals and summary data
2. Clean signals and detect R-peaks
3. Extract HRV features (time, frequency, non-linear domains)
4. Save processed signals and features

Usage:
    python process_ecg_data.py [--overwrite]

Arguments:
    --overwrite: Reprocess and overwrite existing processed files

Output structure:
    data/processed/
    ├── signals/          # Cleaned ECG signals, R-peaks, HR per file
    ├── features/         # HRV features per file
    └── combined/         # All features combined into one CSV
"""

import argparse
from pathlib import Path
import pandas as pd
import json
from typing import Dict
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import CFG
from utils.ecg_utils import (
    import_zephyr_ecg_data,
    processing_ecg_signal,
    extract_windowed_hrv_features,
    parse_ecg_filename,
    map_session_to_condition
)

# Import pose utilities for condition mapping
from Pose.utils.io_utils import load_participant_info_file
from Pose.utils.preprocessing_utils import create_condition_mapping


def ensure_output_dirs():
    """Create output directory structure."""
    dirs = [
        Path(CFG.OUT_BASE) / "signals",
        Path(CFG.OUT_BASE) / "features",
        Path(CFG.OUT_BASE) / "combined"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directories ready: {CFG.OUT_BASE}")


def process_single_file(
    filename: str,
    raw_dir: Path,
    output_dirs: dict,
    condition_map: dict,
    overwrite: bool = False
) -> Dict:
    """Process one ECG file.

    Args:
        filename: ECG filename (e.g., '3208_ecg_session01.csv')
        raw_dir: Raw data directory
        output_dirs: Dictionary of output directories
        condition_map: Nested dict {participant_id: {trial_num: condition}}
        overwrite: Whether to overwrite existing files

    Returns:
        Dictionary with processing stats
    """
    # Parse filename
    participant_id, session_num = parse_ecg_filename(filename)
    if participant_id is None:
        return {'status': 'skipped', 'reason': 'invalid_filename', 'filename': filename}

    # Map session to condition using pose utilities
    session_str = f"session{session_num:02d}"
    condition = map_session_to_condition(session_str, participant_id, condition_map)

    if condition is None:
        return {
            'status': 'skipped',
            'reason': 'no_condition_mapping',
            'filename': filename,
            'participant': participant_id
        }

    # Check if already processed
    signal_file = output_dirs['signals'] / f"{participant_id}_{condition}_ecg_signals.csv"
    feature_file = output_dirs['features'] / f"{participant_id}_{condition}_ecg_features.csv"

    if not overwrite and signal_file.exists() and feature_file.exists():
        return {
            'status': 'skipped',
            'reason': 'already_exists',
            'filename': filename,
            'participant': participant_id,
            'condition': condition
        }

    try:
        # Load data (both ECG waveform and summary)
        ecg_data, summary_data = import_zephyr_ecg_data(
            str(raw_dir),
            participant_id=participant_id,
            session_num=session_num
        )

        # Validate ECG data has required columns
        if 'EcgWaveform' not in ecg_data.columns:
            return {
                'status': 'error',
                'error': f'Missing EcgWaveform column in data. Found columns: {list(ecg_data.columns)}',
                'filename': filename,
                'participant': participant_id
            }

        # Process signal: clean, detect R-peaks, calculate HR
        signals, rpeaks = processing_ecg_signal(
            ecg_data['EcgWaveform'].values,
            sampling_rate=CFG.SAMPLE_RATE,
            plot_signal=False
        )

        # Save signals if configured
        if CFG.SAVE_SIGNALS:
            signals['participant'] = participant_id
            signals['condition'] = condition
            signals.to_csv(signal_file, index=False)

        # Extract HRV features using 60-second windows with 50% overlap
        hrv_features = extract_windowed_hrv_features(
            signals,
            rpeaks,
            window_seconds=CFG.WINDOW_SECONDS,
            overlap=CFG.WINDOW_OVERLAP,
            sr=CFG.SAMPLE_RATE
        )

        # Check if feature extraction succeeded
        if hrv_features.empty:
            return {
                'status': 'error',
                'error': 'HRV feature extraction failed (signal may be too short or no R-peaks)',
                'filename': filename,
                'participant': participant_id
            }

        # Add metadata to features (participant, condition, filename)
        hrv_features['participant'] = participant_id
        hrv_features['condition'] = condition
        hrv_features['filename'] = filename

        # Save features if configured
        if CFG.SAVE_FEATURES:
            hrv_features.to_csv(feature_file, index=False)

        return {
            'status': 'success',
            'participant': participant_id,
            'condition': condition,
            'filename': filename,
            'samples': len(signals),
            'windows': len(hrv_features),
            'features': len(hrv_features.columns)
        }

    except FileNotFoundError as e:
        return {
            'status': 'error',
            'error': f'File not found: {str(e)}',
            'participant': participant_id,
            'filename': filename
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'participant': participant_id,
            'filename': filename
        }


def combine_features(output_dirs: dict) -> Path:
    """Combine all feature files into one CSV.

    Args:
        output_dirs: Dictionary of output directories

    Returns:
        Path to combined features file
    """
    features_dir = output_dirs['features']
    all_features = []

    for feature_file in features_dir.glob("*_ecg_features.csv"):
        df = pd.read_csv(feature_file)
        all_features.append(df)

    if len(all_features) > 0:
        combined = pd.concat(all_features, ignore_index=True)
        output_file = output_dirs['combined'] / "ecg_features_all.csv"
        combined.to_csv(output_file, index=False)

        print(f"✓ Combined features saved: {output_file}")
        print(f"  Total records: {len(combined)}")
        print(f"  Participants: {combined['participant'].nunique()}")
        print(f"  Conditions: {sorted(combined['condition'].unique())}")
        return output_file

    return None


def run_ecg_pipeline(overwrite: bool = False):
    """Run complete ECG processing pipeline.

    Args:
        overwrite: Whether to reprocess existing files
    """
    print("="*60)
    print("ECG Data Processing Pipeline")
    print("="*60)

    # Setup output directories
    ensure_output_dirs()
    raw_dir = Path(CFG.RAW_DIR)
    output_dirs = {
        'signals': Path(CFG.OUT_BASE) / "signals",
        'features': Path(CFG.OUT_BASE) / "features",
        'combined': Path(CFG.OUT_BASE) / "combined"
    }

    # Load participant info and create condition mapping
    print("\nLoading participant info and condition mapping...")
    try:
        participant_info_path = load_participant_info_file()
        participant_info = pd.read_csv(participant_info_path)
        condition_map = create_condition_mapping(participant_info)
        print(f"✓ Loaded condition mapping for {len(condition_map)} participants")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure participant_info.csv is accessible.")
        return
    except Exception as e:
        print(f"ERROR loading participant info: {e}")
        return

    # Find ECG files (only the ecg files, not summary files)
    if not raw_dir.exists():
        print(f"ERROR: Raw data directory not found: {raw_dir}")
        print(f"Please check ECG_RAW_DIR in .env or config")
        return

    ecg_files = sorted([f.name for f in raw_dir.glob("*_ecg_session*.csv")])
    if len(ecg_files) == 0:
        print(f"ERROR: No ECG files found in {raw_dir}")
        print("Expected pattern: *_ecg_session*.csv")
        return

    print(f"\n✓ Found {len(ecg_files)} ECG files")

    # Process files
    print("\nProcessing files...")
    results = []
    for filename in tqdm(ecg_files, desc="Processing"):
        result = process_single_file(
            filename, raw_dir, output_dirs, condition_map, overwrite
        )
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)

    successful = [r for r in results if r['status'] == 'success']
    skipped = [r for r in results if r['status'] == 'skipped']
    errors = [r for r in results if r['status'] == 'error']

    print(f"Successful: {len(successful)}")
    print(f"Skipped:    {len(skipped)}")
    print(f"Errors:     {len(errors)}")

    if len(skipped) > 0:
        skip_reasons = {}
        for s in skipped:
            reason = s.get('reason', 'unknown')
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        print("\nSkip reasons:")
        for reason, count in skip_reasons.items():
            print(f"  - {reason}: {count}")

    if len(errors) > 0:
        print("\nErrors (showing first 5):")
        for err in errors[:5]:
            print(f"  - {err.get('filename', 'unknown')}: {err.get('error', 'unknown error')}")

    # Combine features
    if len(successful) > 0:
        print("\nCombining features...")
        combine_features(output_dirs)

        # Save processing summary
        summary = {
            'config': {
                'sample_rate': CFG.SAMPLE_RATE,
                'cleaning_method': CFG.CLEANING_METHOD,
                'peak_method': CFG.PEAK_METHOD,
                'quality_method': CFG.QUALITY_METHOD,
                'interpolation_method': CFG.INTERPOLATION_METHOD,
                'window_seconds': CFG.WINDOW_SECONDS,
                'window_overlap': CFG.WINDOW_OVERLAP
            },
            'files_processed': len(successful),
            'files_skipped': len(skipped),
            'files_errored': len(errors),
            'participants': list(set([r['participant'] for r in successful if 'participant' in r]))
        }

        summary_file = Path(CFG.OUT_BASE) / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Processing summary saved: {summary_file}")
    else:
        print("\nNo files processed successfully.")

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ECG data and extract HRV features"
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing processed files'
    )
    args = parser.parse_args()

    run_ecg_pipeline(overwrite=args.overwrite)
