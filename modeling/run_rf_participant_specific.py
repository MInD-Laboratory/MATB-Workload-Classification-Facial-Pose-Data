#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest Models - Participant-Specific Learning Curves

Trains separate Random Forest models for each participant and evaluates
learning curves showing how performance improves with more training data.

Supports two sampling strategies:
- stratified: Random N windows from EACH condition (balanced)
- temporal_stratified: First N windows (in time) from EACH condition (with buffer to avoid leakage)

Usage:
    python run_rf_participant_specific.py
    python run_rf_participant_specific.py --strategy temporal_stratified
    python run_rf_participant_specific.py --pose-variant original
    python run_rf_participant_specific.py --dry-run

Arguments:
    --strategy: Sampling strategy (stratified or temporal_stratified)
    --pose-variant: Pose normalization (original, procrustes_participant, procrustes_global)
    --overwrite: Overwrite existing results
    --dry-run: Show what would run without executing
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
import numpy as np

from utils import (
    POSE_VARIANTS,
    DEFAULT_MODEL_CONFIG,
    CLASS_CONFIGS,
    get_feature_groups,
    get_all_model_configs,
    get_config_dirname,
    save_run_settings,
    find_matching_run,
    check_model_complete,
    run_participant_specific_model,
    print_summary
)

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

OUTPUT_BASE = Path("model_output") / "participant_specific"

# ============================================================================
# EXPERIMENT DEFINITIONS (31 total)
# ============================================================================

EXPERIMENTS = [
    # -------------------------------
    # Individual Modalities (5)
    # -------------------------------
    {"name": "pose", "feature_groups": ["pose"]},
    {"name": "performance", "feature_groups": ["performance"]},
    {"name": "eye", "feature_groups": ["eye_tracking"]},
    {"name": "gsr", "feature_groups": ["gsr"]},
    {"name": "ecg", "feature_groups": ["ecg"]},

    # -------------------------------
    # Two-way Combinations (10)
    # -------------------------------
    {"name": "pose_perf", "feature_groups": ["pose", "performance"]},
    {"name": "pose_eye", "feature_groups": ["pose", "eye_tracking"]},
    {"name": "pose_gsr", "feature_groups": ["pose", "gsr"]},
    {"name": "pose_ecg", "feature_groups": ["pose", "ecg"]},
    {"name": "perf_eye", "feature_groups": ["performance", "eye_tracking"]},
    {"name": "perf_gsr", "feature_groups": ["performance", "gsr"]},
    {"name": "perf_ecg", "feature_groups": ["performance", "ecg"]},
    {"name": "eye_gsr", "feature_groups": ["eye_tracking", "gsr"]},
    {"name": "eye_ecg", "feature_groups": ["eye_tracking", "ecg"]},
    {"name": "gsr_ecg", "feature_groups": ["gsr", "ecg"]},

    # -------------------------------
    # Three-way Combinations (10)
    # -------------------------------
    {"name": "pose_perf_eye", "feature_groups": ["pose", "performance", "eye_tracking"]},
    {"name": "pose_perf_gsr", "feature_groups": ["pose", "performance", "gsr"]},
    {"name": "pose_perf_ecg", "feature_groups": ["pose", "performance", "ecg"]},
    {"name": "pose_eye_gsr", "feature_groups": ["pose", "eye_tracking", "gsr"]},
    {"name": "pose_eye_ecg", "feature_groups": ["pose", "eye_tracking", "ecg"]},
    {"name": "pose_gsr_ecg", "feature_groups": ["pose", "gsr", "ecg"]},
    {"name": "perf_eye_gsr", "feature_groups": ["performance", "eye_tracking", "gsr"]},
    {"name": "perf_eye_ecg", "feature_groups": ["performance", "eye_tracking", "ecg"]},
    {"name": "perf_gsr_ecg", "feature_groups": ["performance", "gsr", "ecg"]},
    {"name": "eye_gsr_ecg", "feature_groups": ["eye_tracking", "gsr", "ecg"]},

    # -------------------------------
    # Four-way Combinations (5)
    # -------------------------------
    {"name": "pose_perf_eye_gsr", "feature_groups": ["pose", "performance", "eye_tracking", "gsr"]},
    {"name": "pose_perf_eye_ecg", "feature_groups": ["pose", "performance", "eye_tracking", "ecg"]},
    {"name": "pose_perf_gsr_ecg", "feature_groups": ["pose", "performance", "gsr", "ecg"]},
    {"name": "pose_eye_gsr_ecg", "feature_groups": ["pose", "eye_tracking", "gsr", "ecg"]},
    {"name": "perf_eye_gsr_ecg", "feature_groups": ["performance", "eye_tracking", "gsr", "ecg"]},

    # -------------------------------
    # Five-way Combination (1)
    # -------------------------------
    {"name": "all_modalities", "feature_groups": ["pose", "performance", "eye_tracking", "gsr", "ecg"]},
]

# ============================================================================
# LEARNING CURVE CONFIGURATION
# ============================================================================

# Training sizes to test (windows per condition)
# IMPORTANT: Each participant has ~15 windows per condition (~45 total)
#
# These represent N windows from EACH condition (L, M, H):
#   - Training size 10 = 10 from L + 10 from M + 10 from H = 30 total training
#   - Test = remaining 4 per condition = 12 total test (not 5 per condition because we skip a window to ensure no leakage)
#   - Max is 11 to leave adequate test data (3 per condition; not 4 because we skip a window to ensure no leakage)

TRAINING_SIZES = [2, 3, 4, 5, 7, 9, 11]  # Windows per condition

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run participant-specific Random Forest models with learning curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rf_participant_specific.py
  python run_rf_participant_specific.py --pose-variant original
  python run_rf_participant_specific.py --dry-run

Method:
  Uses stratified sampling: takes N windows from EACH condition (L, M, H)
  to ensure balanced class representation at all training sizes.

  Training sizes: [2, 3, 4, 5, 7, 9, 11] windows per condition
        """
    )

    parser.add_argument(
        "--pose-variant",
        choices=["original", "procrustes_participant", "procrustes_global"],
        default="procrustes_global",
        help="Pose feature normalization variant to use (default: procrustes_global)"
    )

    parser.add_argument(
        "--strategy",
        choices=["stratified", "temporal_stratified"],
        default="temporal_stratified",
        help="Sampling strategy: 'stratified' (random N per condition) or 'temporal_stratified' (first N in time per condition with buffer)"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite all existing results and recompute from scratch"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )

    return parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main pipeline execution."""
    args = parse_args()

    print("=" * 80)
    print("RANDOM FOREST - PARTICIPANT-SPECIFIC LEARNING CURVES")
    print("=" * 80)
    print(f"\nPose variant: {args.pose_variant}")
    print(f"Description: {POSE_VARIANTS[args.pose_variant]['description']}")

    # Display current configuration
    class_config = DEFAULT_MODEL_CONFIG.get("class_config", "all")
    class_desc = CLASS_CONFIGS[class_config]["description"]
    print(f"Class configuration: {class_desc}")
    print(f"Feature selection: {DEFAULT_MODEL_CONFIG.get('feature_selection', 'none')}")
    print(f"Pose derivatives: {DEFAULT_MODEL_CONFIG.get('use_pose_derivatives', True)}")
    print(f"Time features: {DEFAULT_MODEL_CONFIG.get('use_time_features', False)}")

    # Display sampling strategy
    if args.strategy == "stratified":
        print(f"Sampling: Stratified (random N windows per condition)")
    elif args.strategy == "temporal_stratified":
        print(f"Sampling: Temporal-Stratified (first N windows per condition, with buffer)")

    print(f"Training sizes: {TRAINING_SIZES} (windows per condition)")

    # Check for existing run with matching configuration
    existing_run = find_matching_run(OUTPUT_BASE, args.pose_variant, DEFAULT_MODEL_CONFIG, "participant_specific")

    if existing_run and not args.overwrite:
        print(f"\n✓ Found existing run with matching configuration:")
        print(f"  {existing_run}")

        if args.dry_run:
            print("\n[DRY RUN] Would use existing directory.")
            return

        print("\nOptions:")
        print("  [u] Use existing directory")
        print("  [n] Create new directory (keep existing)")
        print("  [o] Overwrite existing directory")
        print("  [x] Cancel and exit")

        choice = input("\nChoice [u/n/o/x]: ").strip().lower()

        if choice == 'u':
            output_dir = existing_run
            print(f"Using existing directory: {output_dir}")
        elif choice == 'n':
            # Create new directory with timestamp
            dirname = get_config_dirname(args.pose_variant, DEFAULT_MODEL_CONFIG)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = OUTPUT_BASE / f"{dirname}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created new directory: {output_dir}")
        elif choice == 'o':
            output_dir = existing_run
            args.overwrite = True
            print(f"Will overwrite: {output_dir}")
        else:  # cancel
            print("Exiting.")
            return
    else:
        # Create new output directory
        dirname = get_config_dirname(args.pose_variant, DEFAULT_MODEL_CONFIG)
        output_dir = OUTPUT_BASE / dirname
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")

    # Save configuration settings
    config_to_save = DEFAULT_MODEL_CONFIG.copy()
    config_to_save["training_sizes"] = TRAINING_SIZES
    save_run_settings(output_dir, args.pose_variant, config_to_save, "participant_specific")
    print(f"Overwrite mode: {args.overwrite}")

    # Get feature groups for selected pose variant
    feature_groups = get_feature_groups(args.pose_variant)

    # Add split_strategy and sampling_strategy to each experiment
    for exp in EXPERIMENTS:
        exp["split_strategy"] = "participant_specific"
        exp["sampling_strategy"] = args.strategy

    # Generate all model configs
    all_models = get_all_model_configs(
        EXPERIMENTS,
        feature_groups,
        DEFAULT_MODEL_CONFIG
    )

    print(f"\nConfigured {len(all_models)} experiments")

    # Display experiment list
    print("\nExperiments to run:")
    for i, name in enumerate(all_models.keys(), 1):
        print(f"  {i:2d}. {name}")

    # Handle dry-run mode
    if args.dry_run:
        print("\n[DRY RUN] No models will be executed.")
        print(f"\nWould train {len(all_models)} experiments")
        print(f"Each with {len(TRAINING_SIZES)} training sizes (windows per condition)")
        if args.strategy == "stratified":
            print(f"Using stratified sampling (random windows, balanced across conditions)")
        elif args.strategy == "temporal_stratified":
            print(f"Using temporal-stratified sampling (first N windows per condition, with 1-window buffer)")
        return

    print("\n" + "=" * 80)
    print("NOTE: Participant-specific modeling is computationally intensive")
    print(f"      Will train {len(all_models)} experiments × {len(TRAINING_SIZES)} sizes × ~49 participants")
    print("=" * 80)

    # Filter models that need to run
    models_to_run = []
    for name, config in all_models.items():
        if args.overwrite or not check_model_complete(name, output_dir, config):
            models_to_run.append((name, config))
        else:
            print(f"✓ Skipping (complete): {name}")

    # Execute models
    if models_to_run:
        print(f"\n{'='*80}")
        print(f"RUNNING {len(models_to_run)} PARTICIPANT-SPECIFIC MODELS")
        print("=" * 80)

        for name, config in tqdm(models_to_run, desc="Models"):
            try:
                run_participant_specific_model(
                    name=name,
                    config=config,
                    output_dir=output_dir,
                    training_sizes=TRAINING_SIZES,
                    sampling_strategy=args.strategy,
                    force=args.overwrite,
                )
            except Exception as e:
                print(f"\n✗ Error running {name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\n✓ All models already complete!")

    # Print Summary
    print("\n" + "=" * 80)
    print("✓ PARTICIPANT-SPECIFIC MODELS COMPLETE")
    print("=" * 80)

    print(f"\nResults saved to: {output_dir}")
    print(f"Pose variant: {args.pose_variant}")
    if args.strategy == "stratified":
        print(f"Sampling: Stratified (random windows, balanced across conditions)")
    elif args.strategy == "temporal_stratified":
        print(f"Sampling: Temporal-Stratified (first N windows per condition, with buffer)")
    print("\nNext steps:")
    print("  1. Check individual JSON files for detailed learning curves")
    print("  2. Check *_learning_curve.csv files for easy plotting")
    print("  3. Plot accuracy vs. training size to see calibration needs")
    print("  4. Analyze per-participant variability")


if __name__ == "__main__":
    main()
