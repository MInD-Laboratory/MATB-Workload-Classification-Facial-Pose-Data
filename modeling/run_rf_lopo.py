#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest Models - Leave-One-Participant-Out (LOPO)

Runs Random Forest classifiers using LOPO cross-validation.
Trains on N-1 participants, tests on 1 held-out participant.
Strictest test of generalization to completely unseen individuals.

Usage:
    python run_rf_lopo.py --pose-variant original
    python run_rf_lopo.py --pose-variant procrustes_global --overwrite
    python run_rf_lopo.py --pose-variant procrustes_participant --dry-run

Arguments:
    --pose-variant: Pose normalization (original, procrustes_participant, procrustes_global)
    --overwrite: Overwrite existing results
    --dry-run: Show what would run without executing

Note: LOPO is more computationally intensive than other split strategies
      (runs N times where N = number of participants, typically ~43)
"""

import argparse
from pathlib import Path
from tqdm import tqdm

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
    prompt_user_action,
    run_lopo_model,
    print_summary
)

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

OUTPUT_BASE = Path("model_output") / "lopo"

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
# COMMAND LINE ARGUMENTS
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Random Forest models with LOPO cross-validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rf_lopo.py --pose-variant original
  python run_rf_lopo.py --pose-variant procrustes_global --overwrite
  python run_rf_lopo.py --pose-variant procrustes_participant --dry-run

Note: LOPO is computationally intensive (~43 folds per experiment)
      Estimated runtime: 10-15 hours for all 31 experiments
        """
    )

    parser.add_argument(
        "--pose-variant",
        choices=["original", "procrustes_participant", "procrustes_global"],
        default="procrustes_global",
        help="Pose feature normalization variant to use (default: procrustes_global)"
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
    print("RANDOM FOREST WORKLOAD DETECTION - LEAVE-ONE-PARTICIPANT-OUT")
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
    print(f"\nNote: LOPO runs ~43 folds per experiment (one per participant)")
    print(f"      This is computationally intensive but provides strictest generalization test")

    # Check for existing run with matching configuration
    existing_run = find_matching_run(OUTPUT_BASE, args.pose_variant, DEFAULT_MODEL_CONFIG, "lopo")

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
    save_run_settings(output_dir, args.pose_variant, DEFAULT_MODEL_CONFIG, "lopo")
    print(f"Overwrite mode: {args.overwrite}")

    # Get feature groups for selected pose variant
    feature_groups = get_feature_groups(args.pose_variant)

    # Add split_strategy to each experiment (for logging purposes)
    for exp in EXPERIMENTS:
        exp["split_strategy"] = "lopo"

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
        return

    # Handle existing results and user prompt
    if not args.overwrite:
        existing_models = [
            name for name, config in all_models.items()
            if check_model_complete(name, output_dir, config)
        ]

        if existing_models:
            print(f"\n✓ Found {len(existing_models)} existing results")
            print(f"→ Will run {len(all_models) - len(existing_models)} new experiments")

            action = prompt_user_action()

            if action == "overwrite":
                args.overwrite = True
            elif action == "skip":
                pass
            else:  # cancel or continue (LOPO doesn't support resume)
                if action == "continue":
                    print("Note: LOPO does not support resume mode (runs all participants atomically)")
                print("Exiting.")
                return

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
        print(f"RUNNING {len(models_to_run)} LOPO RF MODELS")
        print("=" * 80)
        print(f"This may take several hours...")

        for name, config in tqdm(models_to_run, desc="Models"):
            try:
                run_lopo_model(
                    name=name,
                    config=config,
                    output_dir=output_dir,
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
    print("✓ RF MODELS COMPLETE - LOPO")
    print("=" * 80)

    print_summary(output_dir)

    print(f"\nResults saved to: {output_dir}")
    print(f"Pose variant: {args.pose_variant}")
    print("\nNext steps:")
    print("  1. Check experiment_log.csv for summary metrics")
    print("  2. Compare LOPO results with random and participant split strategies")
    print("  3. Examine per-participant results in individual JSON files")


if __name__ == "__main__":
    main()
