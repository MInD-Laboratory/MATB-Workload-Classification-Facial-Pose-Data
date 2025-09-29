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
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

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


# ---------- R mixed models ----------------------------------------
# Ensure emmeans installed on the R side (best-effort)
try:
    ro.r('if (!require("emmeans"))  install.packages("emmeans",  repos="https://cloud.r-project.org")')
except Exception:
    # If R or rpy2 not present, let caller handle the exception
    pass

def run_lmer_rpy2(df, dv, feature_label):
    """
    Fit a mixed effects model with lmerTest::lmer and do pairwise emmeans.
    Returns:
        pairwise_p: dict {('L','M'), ('L','H'), ('M','H') -> p-value}
        means:      dict {'L','M','H' -> emmean}
        cis:        dict {'L','M','H' -> (lower, upper)}
    """
    # ---- prep pandas data ----
    cols = ["participant", "condition", "session_order_numeric", "window_index", dv]
    dat = df[cols].dropna().copy()
    dat = dat.rename(columns={dv: "dv"})
    # keep your intended order visible in pandas; R will get an ordered factor
    dat["condition"] = pd.Categorical(dat["condition"], categories=["L", "M", "H"], ordered=True)

    # ---- import R packages ----
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import pandas2ri
    
    lmerTest = importr("lmerTest")
    emmeans   = importr("emmeans")
    base      = importr("base")

    # ---- pandas -> R (scoped; DO NOT use pandas2ri.activate) ----
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_dat = ro.conversion.py2rpy(dat)

    ro.globalenv["dat"] = r_dat
    ro.r("""
        # ensure types on the R side
        dat$participant <- factor(dat$participant)
        dat$condition   <- factor(dat$condition, levels = c("L","M","H"), ordered = TRUE)
        dat$window_index <- as.numeric(dat$window_index)
        dat$session_order_numeric <- as.numeric(dat$session_order_numeric)
    """)

    # ---- fit model ----
    # Using window_index instead of minute
    formula = f'dv ~ condition + session_order_numeric + window_index + (window_index|participant)'
    ro.r(f"fit <- lmerTest::lmer({formula}, data = dat)")

    # optional: print summary for visibility
    print(f"\n=== {feature_label} (R lmerTest) ===")
    print(ro.r("summary(fit)"))

    # ---- emmeans + pairwise ----
    ro.r("emm <- emmeans::emmeans(fit, ~ condition)")

    # to pandas
    emm_df  = ro.r("as.data.frame(emm)")
    
    # Try to get confidence intervals - handle different possible return formats
    try:
        ci_df = ro.r("as.data.frame(confint(emm, level = 0.95))")
    except Exception as e:
        print(f"[WARN] Could not compute confint, using SE from emmeans: {e}")
        ci_df = emm_df.copy()  # fallback to using SE from emmeans output
    
    pwc_df  = ro.r('as.data.frame(pairs(emm, adjust = "tukey"))')

    with localconverter(ro.default_converter + pandas2ri.converter):
        emm_pd = ro.conversion.rpy2py(emm_df)
        ci_pd  = ro.conversion.rpy2py(ci_df)
        pwc_pd = ro.conversion.rpy2py(pwc_df)

    # ---- build outputs ----
    # means keyed by 'L','M','H'
    means = {str(r["condition"]): float(r["emmean"]) for _, r in emm_pd.iterrows()}

    # CI column names can vary - try multiple approaches
    cis = {}
    try:
        # Method 1: Look for lower/upper columns in ci_pd
        lower_col = next((c for c in ci_pd.columns if "lower" in c.lower()), None)
        upper_col = next((c for c in ci_pd.columns if "upper" in c.lower()), None)
        
        if lower_col and upper_col:
            cis = {str(r["condition"]): (float(r[lower_col]), float(r[upper_col])) 
                   for _, r in ci_pd.iterrows()}
        else:
            raise ValueError("No lower/upper columns found")
    except (StopIteration, ValueError, KeyError) as e:
        # Method 2: Fall back to computing CI from SE in emmeans output
        print(f"[WARN] Computing CIs from SE instead of confint: {e}")
        se_col = next((c for c in emm_pd.columns if c.lower() in ["se", "std.error", "stderr"]), None)
        if se_col:
            for _, r in emm_pd.iterrows():
                cond = str(r["condition"])
                mean = float(r["emmean"])
                se = float(r[se_col])
                # 95% CI using 1.96 * SE
                cis[cond] = (mean - 1.96 * se, mean + 1.96 * se)
        else:
            print("[ERROR] Could not find SE column either, using NaN for CIs")
            for cond in means.keys():
                cis[cond] = (float('nan'), float('nan'))

    # pairwise p-values keyed by tuple ('L','M'), etc.
    pcol = "p.value" if "p.value" in pwc_pd.columns else next((c for c in pwc_pd.columns if c.lower().startswith("p")), None)
    pairwise_p = {}
    if pcol:
        for _, r in pwc_pd.iterrows():
            contrast_str = str(r["contrast"])
            # Handle different contrast formats: "L - M" or "L-M"
            parts = [s.strip() for s in contrast_str.replace(" - ", "-").split("-")]
            if len(parts) == 2:
                g1, g2 = parts
                pairwise_p[(g1, g2)] = float(r[pcol])
    else:
        print("[WARN] Could not find p-value column in pairwise comparisons")

    return pairwise_p, means, cis


def barplot_ax(ax, means, sems, pvals,
               ylabel, metric_name,
               colors=None,
               bar_width=0.80,
               ylim_padding=(0.4, 0.1)):
    """
    Draws a bar plot with error bars and significance brackets.
    """
    if colors is None:
        colors = ['#c6dbef', '#6baed6', '#2171b5']

    import textwrap
    import numpy as _np

    x = _np.arange(len(means))
    ax.bar(x, means, yerr=sems, capsize=4,
           color=colors, width=bar_width,
           edgecolor="black")

    lowers = [m - (s if not _np.isnan(s) else 0) for m, s in zip(means, sems)]
    uppers = [m + (s if not _np.isnan(s) else 0) for m, s in zip(means, sems)]
    y_min = min(lowers)
    y_max = max(uppers)
    y_span = y_max - y_min if y_max > y_min else 1.0

    pairs = [(0,1,pvals[0]), (0,2,pvals[1]), (1,2,pvals[2])]
    sig_pairs = [(i, j, p) for (i, j, p) in pairs if (p is not None and not _np.isnan(p) and p < 0.05)]
    sig_pairs = sorted(sig_pairs, key=lambda t: (t[1]-t[0]))

    h_step = 0.2 * y_span
    line_h = 0.03 * y_span
    y0 = y_max + 0.04 * y_span

    for idx, (i, j, p) in enumerate(sig_pairs):
        y = y0 + idx * h_step
        ax.plot([x[i], x[i], x[j], x[j]],
                [y, y+line_h, y+line_h, y],
                lw=1.5, color='black', clip_on=False)
        stars = '***' if p < .001 else '**' if p < .01 else '*'
        ax.text((x[i]+x[j])/2, y+0.25*line_h, stars,
                ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='black', clip_on=False)

    ax.set_xlim(-0.5, len(means)-0.5)
    ax.set_xticks([])

    wrapped_ylabel = "\n".join(textwrap.wrap(ylabel, width=25))
    ax.set_ylabel(wrapped_ylabel, weight='bold', fontsize=12)

    ax.set_ylim(y_min - ylim_padding[0]*y_span,
                y_max + ylim_padding[1]*y_span + len(sig_pairs)*h_step)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.spines[['top','right']].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.tick_params(axis='y', width=1.3, labelsize=11)
    for lab in ax.get_yticklabels():
        lab.set_fontweight('bold')
