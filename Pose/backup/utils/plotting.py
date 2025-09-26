"""
================================================================================
PLOTTING UTILITIES FOR FACIAL POSE DATA
================================================================================

This module provides functions for visualizing facial pose data at various
stages of processing. Includes functions for quality control visualization,
feature plotting, and summary statistics visualization.

Author: Pose Analysis Pipeline
Date: 2024
================================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Optional, Dict, Tuple
import os
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import textwrap


# Set default plotting style
plt.style.use('default')
sns.set_palette("husl")


# ============================================================================
# QUALITY CONTROL VISUALIZATIONS
# ============================================================================

def plot_qc_summary(qc_summary_csv: str, save_path: Optional[str] = None) -> None:
    """
    Plot summary of quality control results.

    Creates a bar chart showing the percentage of bad windows for each metric.

    Parameters
    ----------
    qc_summary_csv : str
        Path to QC summary CSV file (metric_bad_windows.csv)
    save_path : str, optional
        Path to save the plot. If None, displays the plot

    Examples
    --------
    >>> plot_qc_summary('data/qc/metric_bad_windows.csv', 'figures/qc_summary.png')
    """
    # Load QC data
    df = pd.read_csv(qc_summary_csv)

    # Calculate percentage bad by metric
    metric_summary = df.groupby('metric').agg({
        'bad_windows': 'sum',
        'total_windows': 'sum'
    }).reset_index()
    metric_summary['pct_bad'] = (metric_summary['bad_windows'] /
                                 metric_summary['total_windows']) * 100

    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_summary['metric'], metric_summary['pct_bad'])
    plt.xlabel('Facial Region')
    plt.ylabel('Percentage of Bad Windows (%)')
    plt.title('Quality Control Summary: Bad Windows by Facial Region')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"QC summary plot saved to: {save_path}")
    else:
        plt.show()


def plot_qc_timeline(bad_windows_csv: str, filename: str,
                    save_path: Optional[str] = None) -> None:
    """
    Plot timeline showing bad windows for a specific file.

    Creates a timeline visualization showing when each metric has bad windows.

    Parameters
    ----------
    bad_windows_csv : str
        Path to bad window details CSV (metric_bad_window_indices.csv)
    filename : str
        Name of file to visualize
    save_path : str, optional
        Path to save the plot

    Examples
    --------
    >>> plot_qc_timeline('data/qc/metric_bad_window_indices.csv',
    ...                  'participant_001.csv', 'figures/timeline.png')
    """
    # Load bad window data
    df = pd.read_csv(bad_windows_csv)
    file_data = df[df['file'] == filename].copy()

    if file_data.empty:
        print(f"No bad windows found for file: {filename}")
        return

    # Create timeline plot
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = file_data['metric'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))

    for i, metric in enumerate(metrics):
        metric_data = file_data[file_data['metric'] == metric]

        for _, row in metric_data.iterrows():
            ax.barh(i, row['end_frame_exclusive'] - row['start_frame'],
                   left=row['start_frame'], height=0.6,
                   color=colors[i], alpha=0.7, label=metric if _ == 0 else "")

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Frame Number')
    ax.set_title(f'Bad Window Timeline: {filename}')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline plot saved to: {save_path}")
    else:
        plt.show()


# ============================================================================
# FEATURE VISUALIZATION
# ============================================================================

def plot_feature_timeseries(df: pd.DataFrame, features: List[str],
                           title: str = "Feature Time Series",
                           save_path: Optional[str] = None,
                           df_original: Optional[pd.DataFrame] = None,
                           show_comparison: bool = True) -> None:
    """
    Plot multiple features as time series with optional comparison to original data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing normalized/processed features
    features : list of str
        List of feature column names to plot
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
    df_original : pd.DataFrame, optional
        DataFrame containing original (non-normalized) features for comparison
    show_comparison : bool, optional
        Whether to show comparison with original data (if df_original provided)

    Examples
    --------
    >>> plot_feature_timeseries(df, ['blink_dist', 'mouth_dist'],
    ...                         'Facial Features Over Time')
    >>> plot_feature_timeseries(df_normalized, ['blink_dist'],
    ...                         df_original=df_raw, show_comparison=True)
    """
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features),
                            sharex=True)

    if n_features == 1:
        axes = [axes]

    for i, feature in enumerate(features):
        if feature in df.columns:
            # Plot original data if available (dashed line)
            if df_original is not None and show_comparison and feature in df_original.columns:
                axes[i].plot(df_original.index, df_original[feature],
                           linewidth=1, linestyle='--', alpha=0.5, color='gray',
                           label='Original (non-normalized)')

            # Plot processed/normalized data (solid line)
            axes[i].plot(df.index, df[feature], linewidth=1, color='blue',
                       label='Processed/Normalized')
            axes[i].set_ylabel(feature)
            axes[i].grid(True, alpha=0.3)

            # Add NaN regions as shaded areas
            nan_mask = df[feature].isna()
            if nan_mask.any():
                axes[i].fill_between(df.index, axes[i].get_ylim()[0],
                                   axes[i].get_ylim()[1],
                                   where=nan_mask, alpha=0.3, color='red',
                                   label='Masked data')

            # Add legend if we have comparison
            if df_original is not None and show_comparison and feature in df_original.columns:
                axes[i].legend(loc='upper right', fontsize=8)
        else:
            axes[i].text(0.5, 0.5, f'Feature "{feature}" not found',
                        transform=axes[i].transAxes, ha='center')

    axes[-1].set_xlabel('Frame Number')
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to: {save_path}")
    else:
        plt.show()


def plot_normalization_comparison(df_original: pd.DataFrame,
                                 df_normalized: pd.DataFrame,
                                 features: List[str],
                                 title: str = "Normalization Effects",
                                 save_path: Optional[str] = None) -> None:
    """
    Create a detailed comparison plot showing the effects of normalization.

    Parameters
    ----------
    df_original : pd.DataFrame
        Original non-normalized features
    df_normalized : pd.DataFrame
        Normalized/processed features
    features : list of str
        Features to compare
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot

    Examples
    --------
    >>> plot_normalization_comparison(df_raw, df_proc, ['blink_dist', 'mouth_dist'])
    """
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 2, figsize=(14, 3 * n_features))

    if n_features == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(features):
        if feature in df_original.columns and feature in df_normalized.columns:
            # Time series comparison (left plot)
            axes[i, 0].plot(df_original.index[:500], df_original[feature][:500],
                          linewidth=1, linestyle='--', alpha=0.6, color='gray',
                          label='Original')
            axes[i, 0].plot(df_normalized.index[:500], df_normalized[feature][:500],
                          linewidth=1, color='blue', label='Normalized')
            axes[i, 0].set_ylabel(feature)
            axes[i, 0].set_xlabel('Frame (first 500)')
            axes[i, 0].legend(loc='upper right')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_title(f'{feature} - Time Series')

            # Distribution comparison (right plot)
            axes[i, 1].hist(df_original[feature].dropna(), bins=50, alpha=0.5,
                          color='gray', label='Original', density=True)
            axes[i, 1].hist(df_normalized[feature].dropna(), bins=50, alpha=0.5,
                          color='blue', label='Normalized', density=True)
            axes[i, 1].set_xlabel(feature)
            axes[i, 1].set_ylabel('Density')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_title(f'{feature} - Distribution')

            # Add statistics text
            orig_mean = df_original[feature].mean()
            orig_std = df_original[feature].std()
            norm_mean = df_normalized[feature].mean()
            norm_std = df_normalized[feature].std()

            stats_text = (f'Original: μ={orig_mean:.3f}, σ={orig_std:.3f}\n'
                         f'Normalized: μ={norm_mean:.3f}, σ={norm_std:.3f}')
            axes[i, 1].text(0.02, 0.98, stats_text, transform=axes[i, 1].transAxes,
                          verticalalignment='top', fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()


def plot_feature_distributions(df: pd.DataFrame, features: List[str],
                             group_by: Optional[str] = None,
                             save_path: Optional[str] = None) -> None:
    """
    Plot feature distributions with optional grouping.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features
    features : list of str
        List of feature column names to plot
    group_by : str, optional
        Column name to group by (e.g., 'condition')
    save_path : str, optional
        Path to save the plot

    Examples
    --------
    >>> plot_feature_distributions(df, ['center_face_magnitude_rms'],
    ...                           group_by='condition')
    """
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature in enumerate(features):
        if feature in df.columns:
            if group_by and group_by in df.columns:
                # Grouped distributions
                for group in df[group_by].unique():
                    if pd.notna(group):
                        group_data = df[df[group_by] == group][feature].dropna()
                        axes[i].hist(group_data, alpha=0.7, bins=30, label=str(group))
                axes[i].legend()
            else:
                # Single distribution
                axes[i].hist(df[feature].dropna(), bins=30, alpha=0.7)

            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, f'Feature "{feature}" not found',
                        transform=axes[i].transAxes, ha='center')

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    else:
        plt.show()


def plot_correlation_matrix(correlation_matrix: pd.DataFrame,
                          save_path: Optional[str] = None) -> None:
    """
    Plot correlation matrix heatmap.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Correlation matrix from statistics module
    save_path : str, optional
        Path to save the plot

    Examples
    --------
    >>> corr_matrix = calculate_correlation_matrix(df)
    >>> plot_correlation_matrix(corr_matrix, 'figures/correlation.png')
    """
    plt.figure(figsize=(12, 10))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Create heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, square=True, cbar_kws={"shrink": 0.8})

    plt.title('Feature Correlation Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to: {save_path}")
    else:
        plt.show()


# ============================================================================
# COMPARISON PLOTS
# ============================================================================

def plot_before_after_filtering(df_before: pd.DataFrame, df_after: pd.DataFrame,
                               feature: str, save_path: Optional[str] = None) -> None:
    """
    Plot feature before and after temporal filtering.

    Parameters
    ----------
    df_before : pd.DataFrame
        DataFrame before filtering
    df_after : pd.DataFrame
        DataFrame after filtering
    feature : str
        Feature name to compare
    save_path : str, optional
        Path to save the plot

    Examples
    --------
    >>> plot_before_after_filtering(df_raw, df_filtered, 'center_face_magnitude')
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Before filtering
    ax1.plot(df_before.index, df_before[feature], 'b-', alpha=0.7, linewidth=1)
    ax1.set_ylabel(f'{feature} (Original)')
    ax1.set_title('Before Temporal Filtering')
    ax1.grid(True, alpha=0.3)

    # After filtering
    ax2.plot(df_after.index, df_after[feature], 'r-', linewidth=1)
    ax2.set_ylabel(f'{feature} (Filtered)')
    ax2.set_xlabel('Frame Number')
    ax2.set_title('After Temporal Filtering')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Before/after plot saved to: {save_path}")
    else:
        plt.show()


def plot_condition_comparison(df: pd.DataFrame, features: List[str],
                            condition_column: str,
                            save_path: Optional[str] = None) -> None:
    """
    Plot feature comparison across experimental conditions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and condition labels
    features : list of str
        Features to compare
    condition_column : str
        Column name containing condition labels
    save_path : str, optional
        Path to save the plot

    Examples
    --------
    >>> plot_condition_comparison(df, ['center_face_magnitude_rms'], 'condition')
    """
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(6 * n_features, 6))
    if n_features == 1:
        axes = [axes]

    for i, feature in enumerate(features):
        if feature in df.columns and condition_column in df.columns:
            # Box plot
            df.boxplot(column=feature, by=condition_column, ax=axes[i])
            axes[i].set_title(f'{feature} by {condition_column}')
            axes[i].set_xlabel(condition_column)
            axes[i].set_ylabel(feature)
        else:
            axes[i].text(0.5, 0.5, f'Column not found',
                        transform=axes[i].transAxes, ha='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Condition comparison saved to: {save_path}")
    else:
        plt.show()


# ============================================================================
# SUMMARY REPORT GENERATION
# ============================================================================

def generate_visual_report(feature_data_dir: str, qc_data_dir: str,
                         output_dir: str, sample_files: Optional[List[str]] = None) -> None:
    """
    Generate a comprehensive visual report of the pose analysis pipeline.

    Creates multiple plots summarizing QC results, feature distributions,
    and processing outcomes.

    Parameters
    ----------
    feature_data_dir : str
        Directory containing processed feature files
    qc_data_dir : str
        Directory containing QC results
    output_dir : str
        Directory to save report figures
    sample_files : list of str, optional
        Specific files to include in detailed plots

    Examples
    --------
    >>> generate_visual_report('data/features', 'data/qc', 'reports/figures')
    """
    os.makedirs(output_dir, exist_ok=True)

    # QC Summary
    qc_metric_path = os.path.join(qc_data_dir, "metric_bad_windows.csv")
    if os.path.exists(qc_metric_path):
        plot_qc_summary(qc_metric_path,
                       os.path.join(output_dir, "qc_summary.png"))

    # Feature distributions from sample files
    if sample_files is None:
        feature_files = [f for f in os.listdir(feature_data_dir) if f.endswith('.csv')]
        sample_files = feature_files[:3]  # First 3 files

    for sample_file in sample_files:
        file_path = os.path.join(feature_data_dir, sample_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Key features to visualize
            key_features = ['blink_dist', 'mouth_dist', 'head_rotation_angle',
                          'center_face_magnitude']
            available_features = [f for f in key_features if f in df.columns]

            if available_features:
                # Time series
                plot_feature_timeseries(
                    df, available_features,
                    f"Feature Time Series: {sample_file}",
                    os.path.join(output_dir, f"timeseries_{sample_file.replace('.csv', '.png')}")
                )

                # Distributions
                plot_feature_distributions(
                    df, available_features,
                    save_path=os.path.join(output_dir, f"distributions_{sample_file.replace('.csv', '.png')}")
                )

    print(f"Visual report generated in: {output_dir}")


def plot_pipeline_summary(processing_report_csv: str,
                         save_path: Optional[str] = None) -> None:
    """
    Plot summary of pipeline processing results.

    Parameters
    ----------
    processing_report_csv : str
        Path to processing report CSV
    save_path : str, optional
        Path to save the plot

    Examples
    --------
    >>> plot_pipeline_summary('reports/processing_report.csv')
    """
    df = pd.read_csv(processing_report_csv)

    # Filter out error rows
    df_clean = df[df['qc_metric'] != 'ERROR'].copy()

    if df_clean.empty:
        print("No successful processing data to plot")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Masking rates by metric
    metric_masking = df_clean.groupby('qc_metric')['pct_masked'].mean()
    ax1.bar(metric_masking.index, metric_masking.values)
    ax1.set_ylabel('Average % Masked')
    ax1.set_title('Data Masking by QC Metric')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Processing success rate
    total_files = len(df['file'].unique())
    error_files = len(df[df['qc_metric'] == 'ERROR']['file'].unique())
    success_files = total_files - error_files

    ax2.pie([success_files, error_files], labels=['Success', 'Error'],
           autopct='%1.1f%%', startangle=90)
    ax2.set_title('Processing Success Rate')

    # 3. Frame counts
    frame_stats = df_clean.groupby('file').agg({
        'frames_total': 'first',
        'frames_masked': 'sum'
    }).reset_index()

    ax3.scatter(frame_stats['frames_total'], frame_stats['frames_masked'], alpha=0.6)
    ax3.set_xlabel('Total Frames')
    ax3.set_ylabel('Total Masked Frames')
    ax3.set_title('Masking vs File Size')

    # 4. Features extracted
    if 'num_features' in df_clean.columns:
        feature_counts = df_clean.groupby('file')['num_features'].first()
        ax4.hist(feature_counts, bins=20, alpha=0.7)
        ax4.set_xlabel('Number of Features')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Feature Count Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pipeline summary saved to: {save_path}")
    else:
        plt.show()


# ============================================================================
# STATISTICAL COMPARISON PLOTS
# ============================================================================

def plot_statistical_bars(means: Dict[str, float],
                         sems: Dict[str, float],
                         pvals: Dict[Tuple[str, str], float],
                         ylabel: str,
                         title: str = "",
                         colors: Optional[List[str]] = None,
                         figsize: Tuple[float, float] = (5, 6),
                         save_path: Optional[str] = None) -> None:
    """
    Create bar plot with error bars and significance brackets.

    Parameters
    ----------
    means : dict
        Dictionary mapping condition names to mean values
    sems : dict
        Dictionary mapping condition names to standard errors
    pvals : dict
        Dictionary mapping condition pairs to p-values
    ylabel : str
        Y-axis label
    title : str, optional
        Plot title
    colors : list of str, optional
        Colors for bars (default: blue gradient)
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> means = {'Low': 0.5, 'Moderate': 0.7, 'High': 0.9}
    >>> sems = {'Low': 0.05, 'Moderate': 0.06, 'High': 0.04}
    >>> pvals = {('Low', 'Moderate'): 0.03, ('Low', 'High'): 0.001}
    >>> plot_statistical_bars(means, sems, pvals, 'Feature Value')
    """
    if colors is None:
        colors = ['#c6dbef', '#6baed6', '#2171b5']  # Light to dark blue

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    conditions = list(means.keys())
    mean_values = [means[c] for c in conditions]
    sem_values = [sems[c] for c in conditions]

    # Create bars
    x_pos = np.arange(len(conditions))
    bars = ax.bar(x_pos, mean_values, yerr=sem_values,
                 capsize=5, color=colors[:len(conditions)],
                 edgecolor='black', linewidth=1.5)

    # Calculate y-limits based on data
    lower_whiskers = [m - s for m, s in zip(mean_values, sem_values)]
    upper_whiskers = [m + s for m, s in zip(mean_values, sem_values)]
    y_min = min(lower_whiskers)
    y_max = max(upper_whiskers)
    y_range = y_max - y_min

    # Add significance brackets
    sig_pairs = []
    for (c1, c2), pval in pvals.items():
        if pval < 0.05:
            idx1 = conditions.index(c1) if c1 in conditions else -1
            idx2 = conditions.index(c2) if c2 in conditions else -1
            if idx1 >= 0 and idx2 >= 0:
                sig_pairs.append((idx1, idx2, pval))

    # Sort by span length (shortest first)
    sig_pairs = sorted(sig_pairs, key=lambda x: abs(x[1] - x[0]))

    # Draw brackets
    bracket_height = 0.15 * y_range
    bracket_gap = 0.03 * y_range
    y_start = y_max + 0.05 * y_range

    for i, (idx1, idx2, pval) in enumerate(sig_pairs):
        y = y_start + i * bracket_height
        # Draw bracket
        ax.plot([x_pos[idx1], x_pos[idx1], x_pos[idx2], x_pos[idx2]],
               [y, y + bracket_gap, y + bracket_gap, y],
               'k-', linewidth=1.5)

        # Add significance stars
        if pval < 0.001:
            stars = '***'
        elif pval < 0.01:
            stars = '**'
        else:
            stars = '*'

        ax.text((x_pos[idx1] + x_pos[idx2]) / 2, y + bracket_gap,
               stars, ha='center', va='bottom',
               fontsize=14, fontweight='bold')

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set y-limits with padding
    ax.set_ylim(y_min - 0.1 * y_range,
               y_start + len(sig_pairs) * bracket_height + 0.1 * y_range)

    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Statistical bar plot saved to: {save_path}")
    else:
        plt.show()


def plot_feature_comparison_matrix(stats_results: pd.DataFrame,
                                  features: List[str],
                                  conditions: List[str],
                                  figsize: Tuple[float, float] = (12, 8),
                                  save_path: Optional[str] = None) -> None:
    """
    Create a matrix of bar plots comparing multiple features across conditions.

    Parameters
    ----------
    stats_results : pd.DataFrame
        DataFrame with statistical results from perform_feature_analysis
    features : list of str
        Features to plot
    conditions : list of str
        Condition names
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure
    """
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    colors = ['#c6dbef', '#6baed6', '#2171b5']  # Condition colors

    for idx, feature in enumerate(features):
        ax = axes[idx]
        feature_data = stats_results[stats_results['feature'] == feature]

        if feature_data.empty:
            ax.set_visible(False)
            continue

        # Extract means and SEMs
        means = []
        sems = []
        for condition in conditions:
            mean_col = f'mean_{condition}'
            sem_col = f'sem_{condition}'
            if mean_col in feature_data.columns:
                means.append(feature_data[mean_col].values[0])
                sems.append(feature_data[sem_col].values[0])
            else:
                means.append(0)
                sems.append(0)

        # Create bar plot
        x_pos = np.arange(len(conditions))
        ax.bar(x_pos, means, yerr=sems, capsize=4,
              color=colors[:len(conditions)],
              edgecolor='black', linewidth=1)

        # Add significance indicators if p < 0.05
        if 'p_value' in feature_data.columns:
            p_val = feature_data['p_value'].values[0]
            if p_val < 0.05:
                ax.text(0.5, 0.95, f'p={p_val:.3f}',
                       transform=ax.transAxes,
                       ha='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions, fontsize=10)
        ax.set_ylabel(feature, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Feature Comparisons Across Conditions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature comparison matrix saved to: {save_path}")
    else:
        plt.show()


def plot_condition_summary(summary_df: pd.DataFrame,
                          feature: str,
                          condition_column: str = 'condition',
                          plot_type: str = 'bar',
                          colors: Optional[List[str]] = None,
                          figsize: Tuple[float, float] = (8, 6),
                          save_path: Optional[str] = None) -> None:
    """
    Create summary plot for a feature across conditions.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics from calculate_feature_summary_by_condition
    feature : str
        Feature to plot
    condition_column : str
        Column containing conditions
    plot_type : str
        Type of plot: 'bar', 'box', 'violin'
    colors : list of str, optional
        Colors for conditions
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if colors is None:
        colors = ['#c6dbef', '#6baed6', '#2171b5']

    fig, ax = plt.subplots(figsize=figsize)

    # Filter for the specific feature
    feature_data = summary_df[summary_df['feature'] == feature]

    if feature_data.empty:
        print(f"No data found for feature: {feature}")
        return

    conditions = feature_data[condition_column].unique()
    x_pos = np.arange(len(conditions))

    if plot_type == 'bar':
        means = feature_data.groupby(condition_column)['mean'].first()
        sems = feature_data.groupby(condition_column)['sem'].first()

        ax.bar(x_pos, means, yerr=sems, capsize=5,
              color=colors[:len(conditions)],
              edgecolor='black', linewidth=1.5)
        ax.set_ylabel(f'{feature} (mean ± SEM)', fontsize=12)

    elif plot_type == 'box':
        # For box plots, we need the original data, not summary
        print("Box plot requires original data, not summary statistics")
        return

    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, fontsize=12, fontweight='bold')
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_title(f'{feature} by Condition', fontsize=14, fontweight='bold')

    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Condition summary plot saved to: {save_path}")
    else:
        plt.show()