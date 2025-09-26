# utils/nb_utils.py
from __future__ import annotations
from pathlib import Path
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List

# ---------- disk status ----------
def outputs_exist(base: str | Path) -> dict:
    base = Path(base)
    per_frame = {
        "procrustes_global": list((base / "features" / "per_frame" / "procrustes_global").glob("*.csv")),
        "procrustes_participant": list((base / "features" / "per_frame" / "procrustes_participant").glob("*.csv")),
        "original": list((base / "features" / "per_frame" / "original").glob("*.csv")),
    }
    linear = {
        "procrustes_global": (base / "linear_metrics" / "procrustes_global_linear.csv").exists(),
        "procrustes_participant": (base / "linear_metrics" / "procrustes_participant_linear.csv").exists(),
        "original": (base / "linear_metrics" / "original_linear.csv").exists(),
    }
    any_per_frame = any(len(v) > 0 for v in per_frame.values())
    any_linear = any(linear.values())
    return {"per_frame": per_frame, "linear": linear,
            "any_per_frame": any_per_frame, "any_linear": any_linear}

# ---------- file picking ----------
def pick_norm_file(out_base: str | Path, sample_norm: str | Path | None = None) -> Path:
    if sample_norm:
        return Path(sample_norm)
    norm_dir = Path(out_base) / "norm_screen"
    files = sorted(norm_dir.glob("*_norm.csv"))
    if not files:
        raise FileNotFoundError(f"No normalized CSVs in {norm_dir}. Run the pipeline first.")
    return files[0]

# ---------- column access ----------
def find_col(df: pd.DataFrame, axis: str, i: int) -> str | None:
    c1, c2 = f"{axis}{i}", f"{axis.upper()}{i}"
    return c1 if c1 in df.columns else (c2 if c2 in df.columns else None)

def series_num(df: pd.DataFrame, axis: str, i: int, n: int) -> pd.Series:
    c = find_col(df, axis, i)
    return pd.to_numeric(df[c], errors="coerce") if c else pd.Series([np.nan]*n)

# ---------- slicing ----------
def slice_first_seconds(df: pd.DataFrame, fps: int, seconds: int) -> pd.DataFrame:
    n = len(df)
    end = min(n, fps * seconds)
    return df.iloc[:end].reset_index(drop=True)

# ---------- metrics & plotting ----------
META_COLS = {"source","participant","condition","window_index","t_start_frame","t_end_frame"}

def ensure_condition_order(df: pd.DataFrame, cond_order=("L","M","H")) -> pd.DataFrame:
    if "condition" in df.columns:
        df["condition"] = pd.Categorical(df["condition"], categories=list(cond_order), ordered=True)
    return df

def candidate_metric_cols(df: pd.DataFrame) -> List[str]:
    num_cols = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    priority, others = [], []
    for c in num_cols:
        lc = c.lower()
        if lc.endswith("_mean_abs_vel") or lc.endswith("_mean_abs_acc") or lc.endswith("_rms"):
            priority.append(c)
        else:
            others.append(c)
    return priority + others

def default_metric(cols: List[str]) -> str | None:
    if not cols: return None
    lowers = [c.lower() for c in cols]
    prefs = [
        "blink_aperture_rms", "mouth_aperture_rms", "center_face_magnitude_rms",
        "blink_aperture_mean_abs_vel", "mouth_aperture_mean_abs_vel",
    ]
    for exact in prefs:
        if exact in lowers:
            return cols[lowers.index(exact)]
    for i, lc in enumerate(lowers):
        if lc.endswith("_rms"):
            return cols[i]
    return cols[0]

def sem(series) -> float:
    s = pd.Series(series).astype(float)
    return s.std(ddof=1) / np.sqrt(max(s.count(), 1))

def bar_by_condition(df: pd.DataFrame, metric: str, cond_order=("L","M","H"),
                     colors=("#1f77b4","#f1c40f","#8B0000"), title_suffix: str = ""):
    df = ensure_condition_order(df, cond_order)
    grouped = df.groupby("condition")[metric].agg(["mean", sem]).reindex(cond_order)
    idx = np.arange(len(cond_order))
    means = grouped["mean"].to_numpy(dtype=float)
    errs  = grouped["sem"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(idx, means, yerr=errs, capsize=4, width=0.75, color=list(colors), edgecolor="black", alpha=0.9)
    ax.set_xticks(idx); ax.set_xticklabels(cond_order)
    ax.set_xlabel("Condition"); ax.set_ylabel("Mean ± SEM")
    ttl = f"{metric} by Condition" + (f" — {title_suffix}" if title_suffix else "")
    ax.set_title(ttl); ax.set_xlim(-0.5, len(cond_order)-0.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.grid(axis="y", alpha=0.25)
    for x, m in zip(idx, means):
        if np.isfinite(m):
            ax.text(x, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig, ax

# ---------- stats ----------
def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    corrected = {}
    for i, (lbl, p) in enumerate(items, start=1):
        corrected[lbl] = min(p * (m - i + 1), 1.0)
    return {k: corrected[k] for k in pvals.keys()}
