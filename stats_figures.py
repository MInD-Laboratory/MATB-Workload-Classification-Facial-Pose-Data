import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

def print_means(df, dv, group='condition'):
    """
    Print and return the means of a dependent variable (dv) grouped by a specified column (default: 'condition').
    df: DataFrame
    dv: dependent variable/column name
    group: column to group by (default 'condition')
    Returns: Series of means
    """
    means = df.groupby(group, observed=False)[dv].mean()
    print(f"Means for {dv}:")
    for cond, val in means.items():
        print(f"  {cond}: {val:.3f}")
    return means

def run_rpy2_lmer(df, dv, feature_label):
    """
    Fit a mixed effects model using R's lmerTest::lmer and get pairwise comparisons using emmeans, via rpy2.
    df: DataFrame with all required columns
    dv: dependent variable/column name
    feature_label: label for printing/model output
    Returns:
        pairwise_p: dict of p-values for ('L','M'), ('L','H'), ('M','H')
        means: dict of means for each condition ('L','M','H')
        cis: dict of (lower, upper) 95% CI for each condition ('L','M','H')
    """
    df = df.copy()
    print_means(df, dv)
    pairwise_p = {}
    means = {}
    cis = {}
    # --- Handle participant column robustly ---
    participant_col = "participant_id" if "participant_id" in df.columns else "participant"
    dat = df[[participant_col, "condition", "session_order_numeric", "window_index", dv]].dropna().copy()
    dat = dat.rename(columns={participant_col: "participant_id"})
    # --- Factorise condition ---
    dat["condition"] = pd.Categorical(dat["condition"], categories=["L", "M", "H"], ordered=True)
    # --- Rename dv to a safe column name for R ---
    dat = dat.rename(columns={dv: "dv"})
    # --- Push to R and ensure participant_id is a factor ---
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_dat = robjects.conversion.py2rpy(dat)
    robjects.globalenv["dat"] = r_dat
    robjects.r("dat$participant_id <- factor(dat$participant_id)")
    # --- Load R packages and set up formula ---
    robjects.r("library(lmerTest)")
    robjects.r("library(emmeans)")
    formula = 'dv ~ condition + session_order_numeric + window_index + (1|participant_id)'
    robjects.r(f"library(lmerTest)")
    robjects.r(f"library(emmeans)")
    robjects.r(f"model <- lmer({formula}, data=dat)")
    print(f"\n=== {feature_label} (R lmerTest) ===")
    summary = robjects.r("summary(model)")
    print(summary)
    r_emm = robjects.r("emmeans(model, 'condition')")
    print("\nEstimated marginal means for 'condition':")
    print(r_emm)
    r_pwc = robjects.r("pairs(emmeans(model, 'condition'))")
    print("\nPairwise comparisons for 'condition':")
    print(r_pwc)
    pwc_df = robjects.r('as.data.frame(pairs(emmeans(model, "condition")))')
    with localconverter(robjects.default_converter + pandas2ri.converter):
        pwc_pd = robjects.conversion.rpy2py(pwc_df)
    for idx, row in pwc_pd.iterrows():
        contrast = row['contrast']
        pval = row['p.value']
        g1, g2 = [s.strip() for s in contrast.split('-')]
        pairwise_p[(g1, g2)] = pval
    emm_df = robjects.r('as.data.frame(emmeans(model, "condition"))')
    with localconverter(robjects.default_converter + pandas2ri.converter):
        emm_pd = robjects.conversion.rpy2py(emm_df)
    cond_map_r = {'1': 'L', '1.0': 'L', 'L': 'L',
                '2': 'M', '2.0': 'M', 'M': 'M',
                '3': 'H', '3.0': 'H', 'H': 'H'}
    for idx, row in emm_pd.iterrows():
        cond_raw = str(row['condition'])
        key = cond_map_r.get(cond_raw, cond_raw)
        means[key] = row['emmean']
        cis[key] = (row['lower.CL'], row['upper.CL'])
    return pairwise_p, means, cis

def barplot_ax(ax, means, sems, pvals,
               ylabel, metric_name,
               colors=None,
               bar_width=0.80,
               ylim_padding=(0.4, 0.1)):
    """
    Draw a bar plot with error bars and significance brackets for three conditions.
    ax: matplotlib axis
    means: list of means for each condition
    sems: list of standard errors for each condition
    pvals: list of p-values for pairwise comparisons (L-M, L-H, M-H)
    ylabel: y-axis label
    metric_name: name of the metric (for title/labeling)
    colors: list of bar colors (optional)
    bar_width: width of bars
    ylim_padding: tuple for y-axis padding
    """
    import textwrap
    if colors is None:
        colors = ['#c6dbef', '#6baed6', '#2171b5']
    x = np.arange(len(means))
    ax.bar(x, means, yerr=sems, capsize=4,
           color=colors, width=bar_width,
           edgecolor="black")
    lowers = [m - (s if not np.isnan(s) else 0) for m, s in zip(means, sems)]
    uppers = [m + (s if not np.isnan(s) else 0) for m, s in zip(means, sems)]
    y_min = min(lowers)
    y_max = max(uppers)
    y_span = y_max - y_min if y_max > y_min else 1.0
    pairs = [(0,1,pvals[0]), (0,2,pvals[1]), (1,2,pvals[2])]
    sig_pairs = [(i, j, p) for (i, j, p) in pairs if p < 0.05]
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
                fontsize=13, fontweight='bold',
                color='black', clip_on=False)
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
