# utils/stats_utils.py
"""
Statistical utility functions for analyzing experimental data across different conditions.
Provides functions for multiple comparison correction and group comparisons with automatic
test selection based on data normality.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

# Fixed order for experimental conditions: Low, Medium, High workload
COND_ORDER = ("L", "M", "H")

def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    The Holm-Bonferroni method is a step-down procedure that controls the family-wise
    error rate when conducting multiple hypothesis tests. It's less conservative than
    the standard Bonferroni correction while still maintaining strong control over Type I errors.

    Args:
        pvals: Dictionary mapping comparison labels to their raw p-values

    Returns:
        Dictionary with the same keys as input, but with corrected p-values

    Note:
        - P-values are sorted in ascending order
        - Each p-value is multiplied by (m - i + 1) where m is total comparisons and i is rank
        - Corrected p-values are capped at 1.0
    """
    # Sort p-values in ascending order for step-down procedure
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)  # Total number of comparisons
    corrected = {}

    # Apply Holm-Bonferroni correction formula to each p-value
    for i, (lbl, p) in enumerate(items, start=1):
        # Multiply by (m - i + 1) and cap at 1.0
        corrected[lbl] = min(p * (m - i + 1), 1.0)

    # Return corrected p-values in original key order
    return {k: corrected[k] for k in pvals.keys()}

def compare_groups_statistical(df: pd.DataFrame, metric: str, test_type: str = "auto"):
    """
    Perform comprehensive statistical comparison across experimental conditions.

    Conducts omnibus test across all groups (L/M/H conditions), followed by pairwise
    comparisons with Holm-Bonferroni correction for multiple comparisons.

    Args:
        df: DataFrame containing 'condition' column and the metric column to analyze
        metric: Name of the column containing the dependent variable to analyze
        test_type: Type of test to use - "auto" (default), "parametric", or "nonparametric"
                  "auto" selects based on Shapiro-Wilk normality test

    Returns:
        Dictionary containing:
        - metric: Name of the analyzed metric
        - test_type: Type of test used (parametric/nonparametric)
        - omnibus: Results of omnibus test (name, statistic, p-value)
        - descriptives: DataFrame with mean, std, sem, median for each condition
        - pairwise: DataFrame with pairwise comparisons if >2 groups

    Potential Issues:
        - Shapiro-Wilk test may have low power with small sample sizes (<20)
        - Exception handling in normality test might mask real errors
    """
    # Extract relevant data and remove missing values
    work = df[["condition", metric]].dropna()

    # Create list of value arrays for each condition that exists in the data
    groups = [work[work["condition"] == c][metric].astype(float).values
              for c in COND_ORDER if (work["condition"] == c).any()]

    # Track which condition names are actually present in the data
    group_names = [c for c in COND_ORDER if (work["condition"] == c).any()]
    k = len(groups)  # Number of groups present

    # Calculate descriptive statistics for each condition
    desc = []
    for c in group_names:
        vals = work.loc[work["condition"] == c, metric].astype(float)
        desc.append({
            "condition": c,
            "n": int(vals.count()),  # Sample size
            "mean": float(vals.mean()),  # Arithmetic mean
            "std": float(vals.std(ddof=1)),  # Sample standard deviation (n-1 denominator)
            "sem": float(vals.sem()),  # Standard error of the mean
            "median": float(vals.median())  # Median (robust to outliers)
        })

    # Create DataFrame and ensure conditions appear in standard order
    desc_df = pd.DataFrame(desc).set_index("condition").reindex(COND_ORDER)

    # Automatically choose test type based on normality if set to "auto"
    if test_type == "auto":
        normal = True  # Assume normal until proven otherwise

        # Test normality for each group using Shapiro-Wilk test
        for vals in groups:
            # Need at least 3 values for Shapiro-Wilk test
            if len(vals) >= 3:
                try:
                    _, p = stats.shapiro(vals)
                except Exception as e:
                    # If Shapiro-Wilk fails, default to nonparametric (more conservative)
                    # This can happen with certain data patterns or numerical issues
                    import warnings
                    warnings.warn(f"Shapiro-Wilk test failed, defaulting to nonparametric: {e}")
                    normal = False
                    break

                # If any group is non-normal (p < 0.05), use nonparametric tests
                if p < 0.05:
                    normal = False
                    break

        test_type = "parametric" if normal else "nonparametric"

    # Perform omnibus test across all groups
    omni_name = None
    omni_stat = float("nan")
    omni_p = float("nan")

    # Only perform tests if we have at least 2 groups
    if k >= 2:
        if test_type == "parametric":
            # Use ANOVA for 3+ groups, t-test for exactly 2 groups
            omni_name = "One-way ANOVA" if k > 2 else "Independent t-test"
            if k > 2:
                # F-test for comparing means across multiple groups
                omni_stat, omni_p = stats.f_oneway(*groups)
            else:
                # Welch's t-test (unequal variances) for two groups
                omni_stat, omni_p = stats.ttest_ind(*groups, equal_var=False)
        else:
            # Use Kruskal-Wallis for 3+ groups, Mann-Whitney U for 2 groups
            omni_name = "Kruskal–Wallis" if k > 2 else "Mann–Whitney U"
            if k > 2:
                # Non-parametric alternative to one-way ANOVA
                omni_stat, omni_p = stats.kruskal(*groups)
            else:
                # Non-parametric alternative to t-test
                omni_stat, omni_p = stats.mannwhitneyu(*groups, alternative="two-sided")

    # Perform pairwise comparisons with Holm-Bonferroni correction
    pairs = {}

    # Only do pairwise comparisons if we have more than 2 groups
    if k > 2:
        raw_p = {}  # Store raw p-values before correction

        # Compare all pairs of groups
        for i in range(k):
            for j in range(i+1, k):
                a, b = groups[i], groups[j]
                label = f"{group_names[i]} vs {group_names[j]}"

                # Use appropriate test based on test_type
                if test_type == "parametric":
                    # Welch's t-test for unequal variances
                    _, p = stats.ttest_ind(a, b, equal_var=False)
                else:
                    # Mann-Whitney U test for non-parametric comparison
                    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")

                raw_p[label] = float(p)

        # Apply Holm-Bonferroni correction to control family-wise error rate
        corr = holm_bonferroni(raw_p) if raw_p else {}

        # Store both raw and corrected p-values with significance flag
        for label in raw_p:
            pairs[label] = {
                "p_raw": raw_p[label],  # Original p-value
                "p_holm": corr[label],  # Corrected p-value
                "significant_0.05": corr[label] < 0.05  # Whether significant at α=0.05
            }

    # Return comprehensive results dictionary
    return {
        "metric": metric,
        "test_type": test_type,
        "omnibus": {"name": omni_name, "stat": float(omni_stat), "p": float(omni_p)},
        "descriptives": desc_df,
        "pairwise": pd.DataFrame.from_dict(pairs, orient="index") if pairs else pd.DataFrame()
    }
