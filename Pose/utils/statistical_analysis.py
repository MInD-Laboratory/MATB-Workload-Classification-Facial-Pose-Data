"""
================================================================================
STATISTICAL ANALYSIS MODULE FOR FACIAL POSE DATA
================================================================================

This module provides functions for statistical analysis of facial pose features.
It includes normalization methods, descriptive statistics, and analysis
functions commonly used in pose analysis research.

Author: Pose Analysis Pipeline
Date: 2024
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, kruskal, mannwhitneyu
import warnings


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def apply_z_score_normalization(df: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               by_participant: bool = False,
                               participant_column: str = 'participant') -> pd.DataFrame:
    """
    Apply z-score normalization to convert values to standard deviations from mean.

    Formula: z = (x - mean) / std

    Parameters
    ----------
    df : pd.DataFrame
        Input data to normalize
    columns : list of str, optional
        Specific columns to normalize. If None, normalizes all numeric columns
    by_participant : bool, optional
        Whether to normalize within each participant separately (default: False)
    participant_column : str, optional
        Column name containing participant IDs (default: 'participant')

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized values

    Examples
    --------
    >>> df_norm = apply_z_score_normalization(df, ['center_face_magnitude_rms'])
    >>> # Participant-specific normalization
    >>> df_norm = apply_z_score_normalization(df, by_participant=True)
    """
    df_normalized = df.copy()

    # Determine columns to normalize
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    # Apply normalization
    for col in columns:
        if col not in df.columns:
            continue

        if by_participant and participant_column in df.columns:
            # Normalize within each participant
            for participant in df[participant_column].unique():
                if pd.isna(participant):
                    continue

                mask = df[participant_column] == participant
                participant_data = df.loc[mask, col]

                if len(participant_data) > 1 and participant_data.std() > 0:
                    df_normalized.loc[mask, col] = (
                        participant_data - participant_data.mean()
                    ) / participant_data.std()
        else:
            # Global normalization
            if df[col].std() > 0:
                df_normalized[col] = (df[col] - df[col].mean()) / df[col].std()

    return df_normalized


def apply_percentile_normalization(df: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 percentile_range: Tuple[float, float] = (5, 95)) -> pd.DataFrame:
    """
    Apply percentile-based normalization to handle outliers.

    Scales values to [0, 1] range based on specified percentiles.

    Parameters
    ----------
    df : pd.DataFrame
        Input data to normalize
    columns : list of str, optional
        Specific columns to normalize. If None, normalizes all numeric columns
    percentile_range : tuple of float, optional
        (min_percentile, max_percentile) for scaling (default: (5, 95))

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized values

    Examples
    --------
    >>> df_norm = apply_percentile_normalization(df, percentile_range=(10, 90))
    """
    df_normalized = df.copy()

    # Determine columns to normalize
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    for col in columns:
        if col not in df.columns:
            continue

        # Calculate percentiles
        p_min = np.percentile(df[col].dropna(), percentile_range[0])
        p_max = np.percentile(df[col].dropna(), percentile_range[1])

        # Scale to [0, 1] range
        if p_max > p_min:
            df_normalized[col] = (df[col] - p_min) / (p_max - p_min)
            # Clip values outside percentile range
            df_normalized[col] = df_normalized[col].clip(0, 1)

    return df_normalized


def apply_robust_normalization(df: pd.DataFrame,
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Apply robust normalization using median and median absolute deviation (MAD).

    More robust to outliers than z-score normalization.
    Formula: z = (x - median) / MAD

    Parameters
    ----------
    df : pd.DataFrame
        Input data to normalize
    columns : list of str, optional
        Specific columns to normalize. If None, normalizes all numeric columns

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized values
    """
    df_normalized = df.copy()

    # Determine columns to normalize
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    for col in columns:
        if col not in df.columns:
            continue

        median_val = df[col].median()
        mad_val = stats.median_abs_deviation(df[col].dropna(), nan_policy='omit')

        if mad_val > 0:
            df_normalized[col] = (df[col] - median_val) / mad_val

    return df_normalized


# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def calculate_summary_statistics(df: pd.DataFrame,
                               group_by: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate comprehensive summary statistics for all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    group_by : str, optional
        Column name to group by (e.g., 'condition', 'participant')

    Returns
    -------
    pd.DataFrame
        Summary statistics table

    Examples
    --------
    >>> summary = calculate_summary_statistics(df, group_by='condition')
    """
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    if group_by and group_by in df.columns:
        # Grouped statistics
        summary = df.groupby(group_by)[numeric_cols].agg([
            'count', 'mean', 'std', 'min', 'max', 'median',
            lambda x: x.quantile(0.25),  # Q1
            lambda x: x.quantile(0.75),  # Q3
        ])
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
    else:
        # Overall statistics
        summary = df[numeric_cols].agg([
            'count', 'mean', 'std', 'min', 'max', 'median',
            lambda x: x.quantile(0.25),  # Q1
            lambda x: x.quantile(0.75),  # Q3
        ]).T
        summary.columns = ['count', 'mean', 'std', 'min', 'max', 'median', 'Q1', 'Q3']

    return summary


def calculate_rms_values(df: pd.DataFrame,
                        time_window: Optional[int] = None,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate Root Mean Square (RMS) values for time series data.

    RMS is useful for measuring the magnitude of oscillations or movements.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data
    time_window : int, optional
        Window size for rolling RMS calculation. If None, calculates global RMS
    columns : list of str, optional
        Specific columns to process. If None, processes all numeric columns

    Returns
    -------
    pd.DataFrame
        DataFrame with RMS values

    Examples
    --------
    >>> rms_df = calculate_rms_values(df, time_window=300)  # 5-second windows at 60fps
    """
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    rms_df = pd.DataFrame()

    for col in columns:
        if col not in df.columns:
            continue

        if time_window is None:
            # Global RMS
            rms_df[f'{col}_rms'] = [np.sqrt(np.mean(df[col].dropna() ** 2))] * len(df)
        else:
            # Rolling RMS
            rms_df[f'{col}_rms'] = df[col].rolling(
                window=time_window,
                center=True,
                min_periods=1
            ).apply(lambda x: np.sqrt(np.mean(x ** 2)))

    return rms_df


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def calculate_correlation_matrix(df: pd.DataFrame,
                               method: str = 'pearson',
                               min_periods: int = 30) -> pd.DataFrame:
    """
    Calculate correlation matrix between all numeric variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    method : str, optional
        Correlation method ('pearson', 'spearman', 'kendall'), default: 'pearson'
    min_periods : int, optional
        Minimum number of observations required per pair, default: 30

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    return df[numeric_cols].corr(method=method, min_periods=min_periods)


def find_highly_correlated_features(correlation_matrix: pd.DataFrame,
                                  threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """
    Find pairs of features with high correlation.

    Useful for identifying redundant features or multicollinearity issues.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Correlation matrix from calculate_correlation_matrix()
    threshold : float, optional
        Correlation threshold for "high" correlation (default: 0.8)

    Returns
    -------
    list of tuple
        List of (feature1, feature2, correlation) tuples
    """
    high_corr_pairs = []

    # Get upper triangle of correlation matrix (avoid duplicates)
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Find high correlations
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            corr_val = upper_triangle.loc[idx, col]
            if not pd.isna(corr_val) and abs(corr_val) >= threshold:
                high_corr_pairs.append((idx, col, corr_val))

    # Sort by absolute correlation value (descending)
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return high_corr_pairs


# ============================================================================
# TIME SERIES ANALYSIS
# ============================================================================

def calculate_temporal_features(df: pd.DataFrame,
                              columns: Optional[List[str]] = None,
                              window_size: int = 300) -> pd.DataFrame:
    """
    Calculate temporal features for time series analysis.

    Computes rolling statistics that capture temporal dynamics.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data
    columns : list of str, optional
        Columns to analyze. If None, uses all numeric columns
    window_size : int, optional
        Window size for rolling calculations (default: 300 frames = 5 seconds at 60fps)

    Returns
    -------
    pd.DataFrame
        DataFrame with temporal features
    """
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    temporal_df = pd.DataFrame()

    for col in columns:
        if col not in df.columns:
            continue

        # Rolling mean (smoothed signal)
        temporal_df[f'{col}_rolling_mean'] = df[col].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()

        # Rolling standard deviation (variability)
        temporal_df[f'{col}_rolling_std'] = df[col].rolling(
            window=window_size, center=True, min_periods=1
        ).std()

        # Rolling range (max - min)
        temporal_df[f'{col}_rolling_range'] = (
            df[col].rolling(window=window_size, center=True, min_periods=1).max() -
            df[col].rolling(window=window_size, center=True, min_periods=1).min()
        )

    return temporal_df


# ============================================================================
# FEATURE SELECTION HELPERS
# ============================================================================

def calculate_feature_importance_variance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate feature importance based on variance.

    Features with very low variance may not be informative.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature data

    Returns
    -------
    pd.DataFrame
        DataFrame with variance statistics
    """
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    variance_stats = []
    for col in numeric_cols:
        var_val = df[col].var()
        std_val = df[col].std()
        cv_val = std_val / abs(df[col].mean()) if df[col].mean() != 0 else np.inf

        variance_stats.append({
            'feature': col,
            'variance': var_val,
            'std': std_val,
            'coefficient_of_variation': cv_val,
            'non_null_count': df[col].notna().sum()
        })

    return pd.DataFrame(variance_stats).sort_values('variance', ascending=False)


def detect_outliers_iqr(df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       multiplier: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using Interquartile Range (IQR) method.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    columns : list of str, optional
        Columns to check for outliers. If None, checks all numeric columns
    multiplier : float, optional
        IQR multiplier for outlier detection (default: 1.5)

    Returns
    -------
    pd.DataFrame
        DataFrame with outlier indicators
    """
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    outlier_df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outlier_df[f'{col}_is_outlier'] = (
            (df[col] < lower_bound) | (df[col] > upper_bound)
        )

    return outlier_df


# ============================================================================
# GROUP COMPARISONS AND STATISTICAL TESTING
# ============================================================================

def compare_groups_statistical(df: pd.DataFrame,
                              feature: str,
                              group_column: str = 'condition',
                              test_type: str = 'auto') -> Dict:
    """
    Perform statistical comparisons between groups for a given feature.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing features and group labels
    feature : str
        Feature column to compare
    group_column : str, optional
        Column containing group labels (default: 'condition')
    test_type : str, optional
        Statistical test to use: 'auto', 'parametric', 'nonparametric'

    Returns
    -------
    dict
        Dictionary containing test results, p-values, and effect sizes
    """
    # Get unique groups and filter out NaN values
    groups = df[group_column].dropna().unique()
    n_groups = len(groups)

    results = {
        'feature': feature,
        'groups': groups.tolist(),
        'n_groups': n_groups
    }

    # Calculate descriptive statistics for each group
    group_stats = {}
    group_data = {}
    for group in groups:
        group_df = df[df[group_column] == group][feature].dropna()
        group_data[group] = group_df
        group_stats[group] = {
            'mean': group_df.mean(),
            'std': group_df.std(),
            'sem': group_df.sem(),
            'median': group_df.median(),
            'n': len(group_df),
            'ci_95': (group_df.mean() - 1.96 * group_df.sem(),
                     group_df.mean() + 1.96 * group_df.sem())
        }

    results['group_statistics'] = group_stats

    # Determine which test to use
    if test_type == 'auto':
        # Check normality for each group
        normality_passed = True
        for group, data in group_data.items():
            if len(data) >= 3:
                _, p_norm = stats.shapiro(data)
                if p_norm < 0.05:
                    normality_passed = False
                    break

        test_type = 'parametric' if normality_passed else 'nonparametric'

    results['test_type'] = test_type

    # Perform appropriate omnibus test
    if n_groups > 2:
        if test_type == 'parametric':
            # One-way ANOVA
            f_stat, p_value = f_oneway(*group_data.values())
            results['omnibus_test'] = 'One-way ANOVA'
            results['test_statistic'] = f_stat
            results['omnibus_p_value'] = p_value

            # Effect size (eta-squared)
            ss_between = sum(len(data) * (data.mean() - df[feature].mean())**2
                           for data in group_data.values())
            ss_total = sum((data - df[feature].mean())**2
                         for data in group_data.values() for data in data)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            results['effect_size'] = eta_squared
            results['effect_size_label'] = 'eta_squared'
        else:
            # Kruskal-Wallis H test
            h_stat, p_value = kruskal(*group_data.values())
            results['omnibus_test'] = 'Kruskal-Wallis H'
            results['test_statistic'] = h_stat
            results['omnibus_p_value'] = p_value

    elif n_groups == 2:
        group_list = list(group_data.values())
        if test_type == 'parametric':
            # Independent t-test
            t_stat, p_value = ttest_ind(group_list[0], group_list[1])
            results['test'] = 'Independent t-test'
            results['test_statistic'] = t_stat
            results['p_value'] = p_value

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group_list[0])-1) * group_list[0].std()**2 +
                                 (len(group_list[1])-1) * group_list[1].std()**2) /
                                (len(group_list[0]) + len(group_list[1]) - 2))
            cohens_d = (group_list[0].mean() - group_list[1].mean()) / pooled_std
            results['effect_size'] = cohens_d
            results['effect_size_label'] = 'cohens_d'
        else:
            # Mann-Whitney U test
            u_stat, p_value = mannwhitneyu(group_list[0], group_list[1])
            results['test'] = 'Mann-Whitney U'
            results['test_statistic'] = u_stat
            results['p_value'] = p_value

    # Perform pairwise comparisons if more than 2 groups
    if n_groups > 2 and results.get('omnibus_p_value', 1) < 0.05:
        pairwise_results = {}
        group_names = list(groups)

        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                g1, g2 = group_names[i], group_names[j]
                data1 = group_data[g1]
                data2 = group_data[g2]

                if test_type == 'parametric':
                    t_stat, p_val = ttest_ind(data1, data2)
                else:
                    u_stat, p_val = mannwhitneyu(data1, data2)

                # Apply Bonferroni correction
                p_val_corrected = p_val * (n_groups * (n_groups - 1) / 2)
                p_val_corrected = min(p_val_corrected, 1.0)

                pairwise_results[f'{g1}_vs_{g2}'] = {
                    'p_value': p_val,
                    'p_value_corrected': p_val_corrected,
                    'significant': p_val_corrected < 0.05
                }

        results['pairwise_comparisons'] = pairwise_results

    return results


def perform_feature_analysis(df: pd.DataFrame,
                            features: List[str],
                            group_column: str = 'condition',
                            participant_column: Optional[str] = 'participant') -> pd.DataFrame:
    """
    Perform comprehensive statistical analysis on multiple features.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing features and grouping variables
    features : list of str
        List of feature columns to analyze
    group_column : str, optional
        Column containing group/condition labels
    participant_column : str, optional
        Column containing participant IDs for repeated measures

    Returns
    -------
    pd.DataFrame
        Summary table with statistical results for all features
    """
    results_list = []

    for feature in features:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in dataframe")
            continue

        # Perform statistical comparison
        stats_results = compare_groups_statistical(df, feature, group_column)

        # Create summary row
        summary = {
            'feature': feature,
            'test_used': stats_results.get('omnibus_test', stats_results.get('test', 'N/A')),
            'p_value': stats_results.get('omnibus_p_value', stats_results.get('p_value', np.nan)),
            'effect_size': stats_results.get('effect_size', np.nan),
            'effect_size_type': stats_results.get('effect_size_label', 'N/A')
        }

        # Add group means
        for group, group_stats in stats_results['group_statistics'].items():
            summary[f'mean_{group}'] = group_stats['mean']
            summary[f'sem_{group}'] = group_stats['sem']
            summary[f'n_{group}'] = group_stats['n']

        # Add pairwise comparison results if available
        if 'pairwise_comparisons' in stats_results:
            for comparison, comp_results in stats_results['pairwise_comparisons'].items():
                summary[f'p_{comparison}'] = comp_results['p_value_corrected']

        results_list.append(summary)

    return pd.DataFrame(results_list)


def calculate_feature_summary_by_condition(df: pd.DataFrame,
                                          features: List[str],
                                          condition_column: str = 'condition',
                                          participant_column: Optional[str] = 'participant',
                                          time_column: Optional[str] = 'minute') -> Dict[str, pd.DataFrame]:
    """
    Calculate comprehensive summary statistics by experimental condition.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with features and conditions
    features : list of str
        Features to summarize
    condition_column : str
        Column containing condition labels
    participant_column : str, optional
        Column for participant IDs
    time_column : str, optional
        Column for time/block information

    Returns
    -------
    dict
        Dictionary with summary dataframes for different aggregation levels
    """
    summaries = {}

    # Overall summary by condition
    condition_summary = []
    for condition in df[condition_column].dropna().unique():
        cond_df = df[df[condition_column] == condition]

        for feature in features:
            if feature not in cond_df.columns:
                continue

            feature_data = cond_df[feature].dropna()

            condition_summary.append({
                'condition': condition,
                'feature': feature,
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'sem': feature_data.sem(),
                'median': feature_data.median(),
                'q25': feature_data.quantile(0.25),
                'q75': feature_data.quantile(0.75),
                'min': feature_data.min(),
                'max': feature_data.max(),
                'n': len(feature_data),
                'ci_lower': feature_data.mean() - 1.96 * feature_data.sem(),
                'ci_upper': feature_data.mean() + 1.96 * feature_data.sem()
            })

    summaries['by_condition'] = pd.DataFrame(condition_summary)

    # Summary by participant and condition if participant column exists
    if participant_column and participant_column in df.columns:
        participant_summary = []

        for (participant, condition), group_df in df.groupby([participant_column, condition_column]):
            for feature in features:
                if feature not in group_df.columns:
                    continue

                feature_data = group_df[feature].dropna()
                if len(feature_data) > 0:
                    participant_summary.append({
                        'participant': participant,
                        'condition': condition,
                        'feature': feature,
                        'mean': feature_data.mean(),
                        'std': feature_data.std(),
                        'median': feature_data.median(),
                        'n': len(feature_data)
                    })

        summaries['by_participant_condition'] = pd.DataFrame(participant_summary)

    # Summary over time if time column exists
    if time_column and time_column in df.columns:
        time_summary = []

        for (time, condition), group_df in df.groupby([time_column, condition_column]):
            for feature in features:
                if feature not in group_df.columns:
                    continue

                feature_data = group_df[feature].dropna()
                if len(feature_data) > 0:
                    time_summary.append({
                        'time': time,
                        'condition': condition,
                        'feature': feature,
                        'mean': feature_data.mean(),
                        'std': feature_data.std(),
                        'n': len(feature_data)
                    })

        summaries['by_time_condition'] = pd.DataFrame(time_summary)

    return summaries