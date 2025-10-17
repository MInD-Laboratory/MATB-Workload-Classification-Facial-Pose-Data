#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run RF models with different configuration combinations.

This script allows you to systematically test different combinations of config parameters
by automatically updating config.py, clearing cache, and running the models.

Usage:
    # Test all combinations of backward/none feature selection and derivatives on/off
    python run_config_combinations.py --method lopo --vary feature_selection use_pose_derivatives

    # Test normalization on/off with PCA on/off
    python run_config_combinations.py --method random --vary normalize_features use_pca

    # Test all boolean parameters
    python run_config_combinations.py --method lopo --vary use_pose_derivatives normalize_features use_pca tune_hyperparameters

    # Dry run to see what would be executed
    python run_config_combinations.py --method lopo --vary normalize_features --dry-run

Arguments:
    --method: Which modeling strategy (random, participant, lopo, specific)
    --vary: Which config parameters to vary (space-separated list)
    --pose-variant: Pose normalization variant (default: procrustes_global)
    --dry-run: Show what would run without executing
    --n-seeds: Number of seeds per run (default: 10)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from itertools import product
from datetime import datetime
import shutil

# ============================================================================
# CONFIG PARAMETER DEFINITIONS
# ============================================================================

# Boolean parameters that can be toggled
BOOLEAN_PARAMS = {
    'use_pose_derivatives': [True, False],
    'normalize_features': [True, False],
    'use_pca': [True, False],
    'tune_hyperparameters': [True, False],
    'use_time_features': [True, False],
    'include_order': [True, False],
}

# Categorical parameters
CATEGORICAL_PARAMS = {
    'feature_selection': ['backward', 'forward', None],
    'class_config': ['all', 'L_vs_H', 'L_vs_M', 'M_vs_H', 'LM_vs_H', 'L_vs_MH'],
}

ALL_PARAMS = {**BOOLEAN_PARAMS, **CATEGORICAL_PARAMS}

# Method to script mapping
METHOD_TO_SCRIPT = {
    'random': 'run_rf_random_split.py',
    'participant': 'run_rf_participant_split.py',
    'lopo': 'run_rf_lopo.py',
    'specific': 'run_rf_participant_specific.py',
}

# ============================================================================
# CONFIG FILE MANIPULATION
# ============================================================================

def read_config_file():
    """Read current config.py file."""
    config_path = Path('utils/config.py')
    with open(config_path, 'r') as f:
        return f.read()


def update_config_file(param_updates):
    """
    Update DEFAULT_MODEL_CONFIG in config.py with new parameter values.

    Args:
        param_updates (dict): Parameter name -> new value
    """
    config_path = Path('utils/config.py')

    with open(config_path, 'r') as f:
        lines = f.readlines()

    # Find DEFAULT_MODEL_CONFIG section
    in_config_section = False
    updated_lines = []

    for line in lines:
        # Detect start of DEFAULT_MODEL_CONFIG
        if 'DEFAULT_MODEL_CONFIG = {' in line:
            in_config_section = True
            updated_lines.append(line)
            continue

        # Detect end of DEFAULT_MODEL_CONFIG
        if in_config_section and line.strip() == '}':
            in_config_section = False
            updated_lines.append(line)
            continue

        # Update parameters within the config section
        if in_config_section:
            updated = False
            for param_name, new_value in param_updates.items():
                # Check if this line contains the parameter
                if f'"{param_name}":' in line or f"'{param_name}':" in line:
                    # Extract indentation
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent

                    # Format new value
                    if isinstance(new_value, bool):
                        value_str = 'True' if new_value else 'False'
                    elif isinstance(new_value, str):
                        value_str = f'"{new_value}"'
                    elif new_value is None:
                        value_str = 'None'
                    else:
                        value_str = str(new_value)

                    # Preserve comment if present
                    comment = ''
                    if '#' in line:
                        comment = '  ' + line[line.index('#'):]
                    else:
                        comment = '\n'

                    # Create updated line
                    updated_lines.append(f'{indent_str}"{param_name}": {value_str},{comment}')
                    updated = True
                    break

            if not updated:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    # Write updated config
    with open(config_path, 'w') as f:
        f.writelines(updated_lines)


def clear_python_cache():
    """Clear Python bytecode cache to ensure config changes are loaded."""
    cache_dirs = ['utils/__pycache__', '__pycache__']
    for cache_dir in cache_dirs:
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)
            print(f"  Cleared {cache_dir}")


# ============================================================================
# COMBINATION GENERATION
# ============================================================================

def generate_combinations(vary_params, n_seeds=20):
    """
    Generate all combinations of parameter values.

    Args:
        vary_params (list): List of parameter names to vary
        n_seeds (int): Number of seeds to use

    Returns:
        list: List of dicts, each representing one configuration
    """
    # Get all possible values for each parameter
    param_values = []
    param_names = []

    for param in vary_params:
        if param not in ALL_PARAMS:
            raise ValueError(
                f"Unknown parameter: '{param}'. "
                f"Available: {list(ALL_PARAMS.keys())}"
            )
        param_names.append(param)
        param_values.append(ALL_PARAMS[param])

    # Add n_seeds as a constant
    param_names.append('n_seeds')

    # Generate all combinations
    combinations = []
    for combo in product(*param_values):
        config = dict(zip(param_names, combo))
        config['n_seeds'] = n_seeds
        combinations.append(config)

    return combinations


def get_config_description(config):
    """Generate human-readable description of config."""
    parts = []
    for key, value in sorted(config.items()):
        if key == 'n_seeds':
            continue
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif value is None:
            parts.append(f"{key}=none")
        else:
            parts.append(f"{key}={value}")
    return ', '.join(parts)


# ============================================================================
# EXECUTION
# ============================================================================

def run_combination(method, pose_variant, config, dry_run=False):
    """
    Run a single configuration.

    Args:
        method (str): Modeling method (random, lopo, etc.)
        pose_variant (str): Pose variant to use
        config (dict): Configuration parameters
        dry_run (bool): If True, don't actually run

    Returns:
        bool: True if successful, False if failed
    """
    script = METHOD_TO_SCRIPT[method]

    print(f"\n{'='*80}")
    print(f"Configuration: {get_config_description(config)}")
    print(f"{'='*80}")

    if dry_run:
        print(f"[DRY RUN] Would run: python {script} --pose-variant {pose_variant} --overwrite")
        return True

    # Update config.py
    print("1. Updating config.py...")
    update_config_file(config)

    # Clear cache
    print("2. Clearing Python cache...")
    clear_python_cache()

    # Run the script
    print(f"3. Running {script}...")
    cmd = [
        'python',
        script,
        '--pose-variant', pose_variant,
        '--overwrite'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        print(f"\n✓ Configuration completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Configuration failed with error code {e.returncode}")
        return False

    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        return False


# ============================================================================
# LOGGING
# ============================================================================

def log_run(method, pose_variant, combinations, results, log_dir='run_logs'):
    """
    Log the results of a combination run.

    Args:
        method (str): Modeling method
        pose_variant (str): Pose variant
        combinations (list): List of config dicts
        results (list): List of success/failure bools
        log_dir (str): Directory to save logs
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{method}_{timestamp}.json"

    log_data = {
        'timestamp': datetime.now().isoformat(),
        'method': method,
        'pose_variant': pose_variant,
        'total_combinations': len(combinations),
        'successful': sum(results),
        'failed': len(results) - sum(results),
        'combinations': [
            {
                'config': combo,
                'success': success,
                'description': get_config_description(combo)
            }
            for combo, success in zip(combinations, results)
        ]
    }

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"\n✓ Run log saved to: {log_file}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RF models with different configuration combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test derivatives on/off with normalization on/off
  python run_config_combinations.py --method lopo --vary use_pose_derivatives normalize_features

  # Test all feature selection options
  python run_config_combinations.py --method random --vary feature_selection

  # Test PCA and hyperparameter tuning
  python run_config_combinations.py --method random --vary use_pca tune_hyperparameters

  # Dry run to preview
  python run_config_combinations.py --method lopo --vary normalize_features --dry-run

Available parameters to vary:
  Boolean: use_pose_derivatives, normalize_features, use_pca, tune_hyperparameters,
           use_time_features, include_order
  Categorical: feature_selection (backward/forward/none),
               class_config (all/L_vs_H/L_vs_M/M_vs_H/LM_vs_H/L_vs_MH)
        """
    )

    parser.add_argument(
        '--method',
        choices=['random', 'participant', 'lopo', 'specific'],
        required=True,
        help='Modeling strategy to run'
    )

    parser.add_argument(
        '--vary',
        nargs='+',
        required=True,
        help='Config parameters to vary (space-separated)'
    )

    parser.add_argument(
        '--pose-variant',
        choices=['original', 'procrustes_participant', 'procrustes_global'],
        default='procrustes_global',
        help='Pose normalization variant (default: procrustes_global)'
    )

    parser.add_argument(
        '--n-seeds',
        type=int,
        default=20,
        help='Number of random seeds per run (default: 20)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would run without executing'
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    args = parse_args()

    print("=" * 80)
    print("RF MODEL CONFIGURATION COMBINATIONS")
    print("=" * 80)
    print(f"\nMethod: {args.method}")
    print(f"Pose variant: {args.pose_variant}")
    print(f"Parameters to vary: {', '.join(args.vary)}")
    print(f"Seeds per run: {args.n_seeds}")

    # Generate combinations
    try:
        combinations = generate_combinations(args.vary, n_seeds=args.n_seeds)
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return 1

    print(f"\nTotal combinations: {len(combinations)}")
    print("\nCombinations to run:")
    for i, combo in enumerate(combinations, 1):
        print(f"  {i}. {get_config_description(combo)}")

    if args.dry_run:
        print("\n[DRY RUN] No models will be executed.")
        return 0

    # Confirm with user
    print("\n" + "=" * 80)
    print(f"This will run {len(combinations)} complete model runs.")
    print("Each run will train all 31 experiment combinations.")
    print("This may take several hours depending on the method.")
    print("=" * 80)

    response = input("\nContinue? [y/n]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return 0

    # Save original config for restoration
    print("\nBacking up original config.py...")
    original_config = read_config_file()
    backup_path = Path('utils/config.py.backup')
    with open(backup_path, 'w') as f:
        f.write(original_config)
    print(f"✓ Backup saved to {backup_path}")

    # Run all combinations
    results = []
    start_time = datetime.now()

    try:
        for i, combo in enumerate(combinations, 1):
            print(f"\n{'#'*80}")
            print(f"# COMBINATION {i}/{len(combinations)}")
            print(f"{'#'*80}")

            success = run_combination(
                method=args.method,
                pose_variant=args.pose_variant,
                config=combo,
                dry_run=args.dry_run
            )
            results.append(success)

            # Stop if failed and user wants to abort
            if not success:
                response = input("\nCombination failed. Continue with remaining? [y/n]: ").strip().lower()
                if response != 'y':
                    print("Stopping execution.")
                    break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        # Restore original config
        print("\nRestoring original config.py...")
        with open('utils/config.py', 'w') as f:
            f.write(original_config)
        print("✓ Config restored")

        # Clear cache one more time
        clear_python_cache()

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("COMBINATION RUN COMPLETE")
    print("=" * 80)
    print(f"\nTotal combinations: {len(combinations)}")
    print(f"Successful: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    print(f"Duration: {duration}")

    # Log results
    log_run(args.method, args.pose_variant, combinations, results)

    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
