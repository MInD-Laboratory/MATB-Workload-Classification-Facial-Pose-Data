#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import textwrap
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure utils on path
import sys as _sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parent
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

# Config, flags
from utils import (
    CFG, Config,
    RUN_FILTER, RUN_MASK, RUN_INTERP_FILTER, RUN_NORM, RUN_TEMPLATES,
    RUN_FEATURES_PROCRUSTES_GLOBAL, RUN_FEATURES_PROCRUSTES_PARTICIPANT, RUN_FEATURES_ORIGINAL,
    RUN_LINEAR, SCALE_BY_INTEROCULAR,
    SAVE_REDUCED, SAVE_MASKED, SAVE_INTERP_FILTERED, SAVE_NORM,
    SAVE_PER_FRAME_PROCRUSTES_GLOBAL, SAVE_PER_FRAME_PROCRUSTES_PARTICIPANT, SAVE_PER_FRAME_ORIGINAL,
    OVERWRITE, OVERWRITE_TEMPLATES, SCIPY_AVAILABLE
)

# IO + helpers
from utils.io_utils import (
    ensure_dirs, load_raw_files, parse_participant_condition,
    detect_conf_prefix_case_insensitive, relevant_indices, filter_df_to_relevant,
    confidence_mask, write_per_frame_metrics, save_json_summary
)
from utils.signal_utils import interpolate_run_limited, butterworth_segment_filter
from utils.normalize_utils import normalize_to_screen, interocular_series
from utils.features_utils import (
    procrustes_features_for_file, original_features_for_file, compute_linear_from_perframe_dir
)
from utils.window_utils import window_features

# ---------- small helpers ----------
def _read_if_exists(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None

def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _print_skip(msg: str) -> None:
    print(f"[skip] {msg}")

def _print_ok(msg: str) -> None:
    print(f"[OK] {msg}")

def _err(msg: str) -> None:
    print(msg)
    sys.exit(1)

def _any_pre_steps_enabled() -> bool:
    return any([RUN_FILTER, RUN_MASK, RUN_INTERP_FILTER, RUN_NORM])

# ---------- pipeline ----------
def run_pipeline():
    ensure_dirs()

    # ---------- Linear-only gate ----------
    def _linear_only_mode() -> bool:
        return RUN_LINEAR and not any([
            RUN_FILTER, RUN_MASK, RUN_INTERP_FILTER, RUN_NORM, RUN_TEMPLATES,
            RUN_FEATURES_PROCRUSTES_GLOBAL, RUN_FEATURES_PROCRUSTES_PARTICIPANT, RUN_FEATURES_ORIGINAL
        ])

    if _linear_only_mode():
        lm_dir = Path(CFG.OUT_BASE) / "linear_metrics"
        lm_dir.mkdir(parents=True, exist_ok=True)

        def _do_linear(out_name: str):
            src = ("procrustes_global" if "procrustes_global" in out_name
                   else "procrustes_participant" if "procrustes_participant" in out_name
                   else "original")
            per_frame_dir = Path(CFG.OUT_BASE) / "features" / "per_frame" / src
            if not (per_frame_dir.exists() and any(per_frame_dir.glob("*.csv"))):
                _print_skip(f"Missing per-frame dir for '{src}': {per_frame_dir}")
                return {}
            out_path = lm_dir / out_name
            drops = compute_linear_from_perframe_dir(
                per_frame_dir, out_path, CFG.FPS, CFG.WINDOW_SECONDS, CFG.WINDOW_OVERLAP, SCALE_BY_INTEROCULAR
            )
            _print_ok(f"Wrote {out_path}")
            return drops

        routes = []
        for src_flag, route in [(True, "procrustes_global"), (True, "procrustes_participant"), (True, "original")]:
            per_frame_dir = Path(CFG.OUT_BASE) / "features" / "per_frame" / route
            if per_frame_dir.exists() and any(per_frame_dir.glob("*.csv")):
                routes.append(route)

        if not routes:
            summary = {
                "config": asdict(CFG),
                "flags": {k: globals()[k] for k in [
                    "RUN_FILTER","RUN_MASK","RUN_INTERP_FILTER","RUN_NORM","RUN_TEMPLATES",
                    "RUN_FEATURES_PROCRUSTES_GLOBAL","RUN_FEATURES_PROCRUSTES_PARTICIPANT",
                    "RUN_FEATURES_ORIGINAL","RUN_LINEAR",
                    "SAVE_REDUCED","SAVE_MASKED","SAVE_INTERP_FILTERED","SAVE_NORM",
                    "OVERWRITE","OVERWRITE_TEMPLATES","SCALE_BY_INTEROCULAR"
                ]},
                "masking_overall": {},
                "window_drops": {"procrustes_global": {}, "procrustes_participant": {}, "original": {}, "linear_metrics": {}}
            }
            save_json_summary(Path(CFG.OUT_BASE) / "pipeline_summary.json", summary)
            _err("No per-frame CSVs found. Enable a RUN_FEATURES_* step once, or provide per_frame dirs.")

        linear_drop_totals = {}
        if "procrustes_global" in routes:
            linear_drop_totals["procrustes_global"] = _do_linear("procrustes_global_linear.csv")
        if "procrustes_participant" in routes:
            linear_drop_totals["procrustes_participant"] = _do_linear("procrustes_participant_linear.csv")
        if "original" in routes:
            linear_drop_totals["original"] = _do_linear("original_linear.csv")

        summary = {
            "config": asdict(CFG),
            "flags": {k: globals()[k] for k in [
                "RUN_FILTER","RUN_MASK","RUN_INTERP_FILTER","RUN_NORM","RUN_TEMPLATES",
                "RUN_FEATURES_PROCRUSTES_GLOBAL","RUN_FEATURES_PROCRUSTES_PARTICIPANT",
                "RUN_FEATURES_ORIGINAL","RUN_LINEAR",
                "SAVE_REDUCED","SAVE_MASKED","SAVE_INTERP_FILTERED","SAVE_NORM",
                "OVERWRITE","OVERWRITE_TEMPLATES","SCALE_BY_INTEROCULAR"
            ]},
            "masking_overall": {},
            "window_drops": {"procrustes_global": {}, "procrustes_participant": {}, "original": {}, "linear_metrics": linear_drop_totals}
        }
        save_json_summary(Path(CFG.OUT_BASE) / "pipeline_summary.json", summary)
        return

    # ---------- Full mode (with cache-aware steps) ----------
    files = load_raw_files()
    perfile_data: dict[str, dict[str, pd.DataFrame]] = {}
    perfile_meta: dict[str, dict[str, str]] = {}
    perfile_mask_stats: dict[str, dict] = {}

    # FAST PATH: if no pre steps enabled and all normalized files exist, load them and skip Steps 1–5
    norm_cache_ok = False
    if not _any_pre_steps_enabled():
        norm_cache_ok = all((Path(CFG.OUT_BASE) / "norm_screen" / (fp.stem + "_norm.csv")).exists() for fp in files)
        if norm_cache_ok:
            print("\n=== Using cached normalized data; skipping Steps 1–5 ===")
            for fp in files:
                norm_p = Path(CFG.OUT_BASE) / "norm_screen" / (fp.stem + "_norm.csv")
                df_norm = pd.read_csv(norm_p)
                pid, cond = parse_participant_condition(fp.name)
                # detect conf prefix on normalized file (confidence columns are kept)
                conf_prefix = detect_conf_prefix_case_insensitive(list(df_norm.columns))
                perfile_meta[fp.name] = {"participant": pid, "condition": cond, "conf_prefix": conf_prefix}
                perfile_data[fp.name] = {"norm": df_norm}

    # If we couldn't fast-path, run (or load) Steps 1–5 with cache toggles
    if not norm_cache_ok:
        print("\n=== Steps 1–3: Load → Filter → Mask ===")
        for fp in tqdm(files, desc="Load/Filter/Mask", unit="file"):
            df_raw = pd.read_csv(fp)
            conf_prefix = detect_conf_prefix_case_insensitive(list(df_raw.columns))
            indices = relevant_indices()

            # Step 2: FILTER
            reduced_p = Path(CFG.OUT_BASE) / "reduced" / (fp.stem + "_reduced.csv")
            if RUN_FILTER:
                if reduced_p.exists() and not OVERWRITE:
                    df_reduced = pd.read_csv(reduced_p)
                    _print_skip(f"Reduced exists, OVERWRITE=False → {reduced_p}")
                else:
                    df_reduced = filter_df_to_relevant(df_raw, conf_prefix, indices)
                    if SAVE_REDUCED:
                        _save_df(df_reduced, reduced_p)
            else:
                df_reduced = _read_if_exists(reduced_p)
                if df_reduced is None:
                    _err(f"RUN_FILTER=False but no cached reduced at {reduced_p}. Enable RUN_FILTER once to generate it.")

            # Step 3: MASK
            masked_p = Path(CFG.OUT_BASE) / "masked" / (fp.stem + "_masked.csv")
            if RUN_MASK:
                if masked_p.exists() and not OVERWRITE:
                    df_masked = pd.read_csv(masked_p)
                    _print_skip(f"Masked exists, OVERWRITE=False → {masked_p}")
                    # We still populate stats neutrally
                    perfile_mask_stats[fp.name] = {"frames": len(df_masked), "n_landmarks_considered": 0,
                                                   "total_coord_values": 0, "total_coords_masked": 0, "pct_coords_masked": 0.0}
                else:
                    df_masked, stats = confidence_mask(df_reduced, conf_prefix, indices, CFG.CONF_THRESH)
                    perfile_mask_stats[fp.name] = stats["overall"]
                    if SAVE_MASKED:
                        _save_df(df_masked, masked_p)
            else:
                cached = _read_if_exists(masked_p)
                df_masked = cached if cached is not None else df_reduced.copy()
                # neutral stats
                perfile_mask_stats[fp.name] = {"frames": len(df_masked), "n_landmarks_considered": 0,
                                               "total_coord_values": 0, "total_coords_masked": 0, "pct_coords_masked": 0.0}

            perfile_data[fp.name] = {"reduced": df_reduced, "masked": df_masked}
            pid, cond = parse_participant_condition(fp.name)
            perfile_meta[fp.name] = {"participant": pid, "condition": cond, "conf_prefix": conf_prefix}

        # Step 4: INTERPOLATE + FILTER
        print("\n=== Step 4: Interpolate + Filter ===")
        for fp in tqdm(files, desc="Interp/Filter", unit="file"):
            name = fp.name
            interp_p = Path(CFG.OUT_BASE) / "interp_filtered" / (fp.stem + "_interp_filt.csv")
            if RUN_INTERP_FILTER:
                if not SCIPY_AVAILABLE:
                    _err("SciPy required for RUN_INTERP_FILTER=True.")
                if interp_p.exists() and not OVERWRITE:
                    dfm = pd.read_csv(interp_p)
                    _print_skip(f"Interp/filtered exists, OVERWRITE=False → {interp_p}")
                else:
                    dfm = perfile_data[name]["masked"].copy()
                    for col in dfm.columns:
                        lc = col.lower()
                        if lc.startswith("x") or lc.startswith("y"):
                            dfm[col] = interpolate_run_limited(dfm[col], CFG.MAX_INTERP_RUN)
                            dfm[col] = butterworth_segment_filter(dfm[col], CFG.FILTER_ORDER, CFG.CUTOFF_HZ, CFG.FPS)
                    if SAVE_INTERP_FILTERED:
                        _save_df(dfm, interp_p)
                perfile_data[name]["interp_filt"] = dfm
            else:
                cached = _read_if_exists(interp_p)
                perfile_data[name]["interp_filt"] = cached if cached is not None else perfile_data[name]["masked"].copy()

        # Step 5: NORMALIZE
        print("\n=== Step 5: Normalize to screen ===")
        for fp in tqdm(files, desc="Normalize", unit="file"):
            name = fp.name
            norm_p = Path(CFG.OUT_BASE) / "norm_screen" / (fp.stem + "_norm.csv")
            if RUN_NORM:
                if norm_p.exists() and not OVERWRITE:
                    df_norm = pd.read_csv(norm_p)
                    _print_skip(f"Norm exists, OVERWRITE=False → {norm_p}")
                else:
                    dfc = perfile_data[name]["interp_filt"]
                    df_norm = normalize_to_screen(dfc, CFG.IMG_WIDTH, CFG.IMG_HEIGHT)
                    if SAVE_NORM:
                        _save_df(df_norm, norm_p)
                perfile_data[name]["norm"] = df_norm
            else:
                cached = _read_if_exists(norm_p)
                if cached is None:
                    _err(f"RUN_NORM=False but no cached normalized at {norm_p}. Enable RUN_NORM once to generate it.")
                perfile_data[name]["norm"] = cached

    # Step 6: Templates
    print("\n=== Step 6: Templates (global + per-participant) ===")
    templ_dir = Path(CFG.OUT_BASE) / "templates"
    global_templ_path = templ_dir / "global_template.csv"

    part_to_files: dict[str, list[str]] = {}
    for fp in files:
        pid = perfile_meta[fp.name]["participant"]
        part_to_files.setdefault(pid, []).append(fp.name)

    def _compute_template(names: list[str]) -> pd.DataFrame:
        if not names:
            return pd.DataFrame()
        cols = perfile_data[names[0]]["norm"].columns
        accum = [perfile_data[n]["norm"][cols].astype(float) for n in names]
        big = pd.concat(accum, axis=0, ignore_index=True)
        x_cols = [c for c in cols if c.lower().startswith("x")]
        y_cols = [c for c in cols if c.lower().startswith("y")]
        templ = pd.DataFrame(index=[0], columns=x_cols + y_cols, dtype=float)
        templ[x_cols] = big[x_cols].mean(axis=0, skipna=True).values
        templ[y_cols] = big[y_cols].mean(axis=0, skipna=True).values
        return templ

    if RUN_TEMPLATES:
        if global_templ_path.exists() and not OVERWRITE_TEMPLATES:
            global_template = pd.read_csv(global_templ_path)
            _print_skip(f"Global template exists, OVERWRITE_TEMPLATES=False → {global_templ_path}")
        else:
            all_names = [fp.name for fp in files]
            global_template = _compute_template(all_names)
            _save_df(global_template, global_templ_path)

        participant_templates: dict[str, pd.DataFrame] = {}
        for pid, names in tqdm(part_to_files.items(), desc="Participant templates", unit="participant"):
            part_path = templ_dir / f"participant_{pid}_template.csv"
            if part_path.exists() and not OVERWRITE_TEMPLATES:
                participant_templates[pid] = pd.read_csv(part_path)
                _print_skip(f"Participant template exists, OVERWRITE_TEMPLATES=False → {part_path}")
            else:
                templ = _compute_template(names)
                _save_df(templ, part_path)
                participant_templates[pid] = templ
    else:
        # Must load cached, Procrustes requires templates
        if not global_templ_path.exists():
            _err("RUN_TEMPLATES=False but no cached global template. Enable RUN_TEMPLATES once to generate.")
        global_template = pd.read_csv(global_templ_path)
        participant_templates = {}
        for pid in part_to_files:
            part_path = templ_dir / f"participant_{pid}_template.csv"
            if not part_path.exists():
                _err(f"RUN_TEMPLATES=False but missing participant template: {part_path}")
            participant_templates[pid] = pd.read_csv(part_path)

    # Step 7: Features
    print("\n=== Step 7: Features (windowed) ===")
    win = CFG.WINDOW_SECONDS * CFG.FPS
    hop = int(win * (1.0 - CFG.WINDOW_OVERLAP)) or max(1, win // 2)
    rel_idxs = relevant_indices()
    feat_dir = Path(CFG.OUT_BASE) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    procrustes_global_rows = []
    procrustes_part_rows = []
    original_rows = []
    procrustes_global_drops_agg = {}
    procrustes_part_drops_agg = {}
    original_drops_agg = {}

    if RUN_FEATURES_PROCRUSTES_GLOBAL or RUN_FEATURES_PROCRUSTES_PARTICIPANT:
        print("Computing Procrustes features...")
        for fp in tqdm(files, desc="Procrustes features", unit="file"):
            name = fp.name
            pid = perfile_meta[name]["participant"]
            cond = perfile_meta[name]["condition"]
            df_norm = perfile_data[name]["norm"]

            if RUN_FEATURES_PROCRUSTES_GLOBAL:
                feats = procrustes_features_for_file(df_norm, global_template, rel_idxs)
                io = interocular_series(df_norm, perfile_meta[name]["conf_prefix"]).values
                n_frames = len(io)
                if SAVE_PER_FRAME_PROCRUSTES_GLOBAL:
                    write_per_frame_metrics(feat_dir, "procrustes_global", pid, cond, feats, io, n_frames)
                dfw, drops = window_features(feats, io, CFG.FPS, win, hop)
                dfw.insert(0, "condition", cond); dfw.insert(0, "participant", pid); dfw.insert(0, "source", "procrustes_global")
                procrustes_global_rows.append(dfw)
                for k, v in drops.items():
                    procrustes_global_drops_agg[k] = procrustes_global_drops_agg.get(k, 0) + v

            if RUN_FEATURES_PROCRUSTES_PARTICIPANT:
                templ = participant_templates[pid]
                feats = procrustes_features_for_file(df_norm, templ, rel_idxs)
                io = interocular_series(df_norm, perfile_meta[name]["conf_prefix"]).values
                n_frames = len(io)
                if SAVE_PER_FRAME_PROCRUSTES_PARTICIPANT:
                    write_per_frame_metrics(feat_dir, "procrustes_participant", pid, cond, feats, io, n_frames)
                dfw, drops = window_features(feats, io, CFG.FPS, win, hop)
                dfw.insert(0, "condition", cond); dfw.insert(0, "participant", pid); dfw.insert(0, "source", "procrustes_participant")
                procrustes_part_rows.append(dfw)
                for k, v in drops.items():
                    procrustes_part_drops_agg[k] = procrustes_part_drops_agg.get(k, 0) + v

    if RUN_FEATURES_ORIGINAL:
        print("Computing Original (no Procrustes) features...")
        for fp in tqdm(files, desc="Original features", unit="file"):
            name = fp.name
            pid = perfile_meta[name]["participant"]
            cond = perfile_meta[name]["condition"]
            df_norm = perfile_data[name]["norm"]

            feats = original_features_for_file(df_norm)
            io = interocular_series(df_norm, perfile_meta[name]["conf_prefix"]).values
            n_frames = len(io)
            if SAVE_PER_FRAME_ORIGINAL:
                write_per_frame_metrics(feat_dir, "original", pid, cond, feats, io, n_frames)
            dfw, drops = window_features(feats, io, CFG.FPS, win, hop)
            dfw.insert(0, "condition", cond); dfw.insert(0, "participant", pid); dfw.insert(0, "source", "original")
            original_rows.append(dfw)
            for k, v in drops.items():
                original_drops_agg[k] = original_drops_agg.get(k, 0) + v

    # Save Step 7 outputs (combined)
    if RUN_FEATURES_PROCRUSTES_GLOBAL and procrustes_global_rows:
        _save_df(pd.concat(procrustes_global_rows, ignore_index=True), feat_dir / "procrustes_global_features.csv")
    if RUN_FEATURES_PROCRUSTES_PARTICIPANT and procrustes_part_rows:
        _save_df(pd.concat(procrustes_part_rows, ignore_index=True), feat_dir / "procrustes_participant_features.csv")
    if RUN_FEATURES_ORIGINAL and original_rows:
        _save_df(pd.concat(original_rows, ignore_index=True), feat_dir / "original_features.csv")

    # Step 8: Linear metrics
    print("\n=== Step 8: Interocular scaling + linear metrics ===")
    lm_dir = Path(CFG.OUT_BASE) / "linear_metrics"
    lm_dir.mkdir(parents=True, exist_ok=True)

    linear_drop_totals = {}
    if RUN_LINEAR:
        if RUN_FEATURES_PROCRUSTES_GLOBAL:
            linear_drop_totals["procrustes_global"] = compute_linear_from_perframe_dir(
                feat_dir / "per_frame" / "procrustes_global",
                lm_dir / "procrustes_global_linear.csv",
                CFG.FPS, CFG.WINDOW_SECONDS, CFG.WINDOW_OVERLAP, SCALE_BY_INTEROCULAR
            )
        if RUN_FEATURES_PROCRUSTES_PARTICIPANT:
            linear_drop_totals["procrustes_participant"] = compute_linear_from_perframe_dir(
                feat_dir / "per_frame" / "procrustes_participant",
                lm_dir / "procrustes_participant_linear.csv",
                CFG.FPS, CFG.WINDOW_SECONDS, CFG.WINDOW_OVERLAP, SCALE_BY_INTEROCULAR
            )
        if RUN_FEATURES_ORIGINAL:
            linear_drop_totals["original"] = compute_linear_from_perframe_dir(
                feat_dir / "per_frame" / "original",
                lm_dir / "original_linear.csv",
                CFG.FPS, CFG.WINDOW_SECONDS, CFG.WINDOW_OVERLAP, SCALE_BY_INTEROCULAR
            )

    # Summary
    summary = {
        "config": asdict(CFG),
        "flags": {k: globals()[k] for k in [
            "RUN_FILTER","RUN_MASK","RUN_INTERP_FILTER","RUN_NORM","RUN_TEMPLATES",
            "RUN_FEATURES_PROCRUSTES_GLOBAL","RUN_FEATURES_PROCRUSTES_PARTICIPANT",
            "RUN_FEATURES_ORIGINAL","RUN_LINEAR",
            "SAVE_REDUCED","SAVE_MASKED","SAVE_INTERP_FILTERED","SAVE_NORM",
            "OVERWRITE","OVERWRITE_TEMPLATES","SCALE_BY_INTEROCULAR"
        ]},
        "masking_overall": perfile_mask_stats,
        "window_drops": {
            "procrustes_global": procrustes_global_drops_agg if RUN_FEATURES_PROCRUSTES_GLOBAL else {},
            "procrustes_participant": procrustes_part_drops_agg if RUN_FEATURES_PROCRUSTES_PARTICIPANT else {},
            "original": original_drops_agg if RUN_FEATURES_ORIGINAL else {},
            "linear_metrics": linear_drop_totals if RUN_LINEAR else {}
        }
    }
    save_json_summary(Path(CFG.OUT_BASE) / "pipeline_summary.json", summary)
    print("\nSummary written to:", Path(CFG.OUT_BASE) / "pipeline_summary.json")
    print("Done.")

if __name__ == "__main__":
    print("POSE PIPELINE — standalone mode")
    print("Config:")
    for k, v in asdict(CFG).items():
        print(f"  {k}: {v}")
    print("\nFlags:")
    print(textwrap.indent(
        "\n".join([f"{k}: {globals()[k]}" for k in [
            "RUN_FILTER","RUN_MASK","RUN_INTERP_FILTER","RUN_NORM",
            "RUN_TEMPLATES","RUN_FEATURES_PROCRUSTES_GLOBAL","RUN_FEATURES_PROCRUSTES_PARTICIPANT",
            "RUN_FEATURES_ORIGINAL","RUN_LINEAR",
            "SAVE_REDUCED","SAVE_MASKED","SAVE_INTERP_FILTERED","SAVE_NORM",
            "OVERWRITE","OVERWRITE_TEMPLATES","SCALE_BY_INTEROCULAR"
        ]]),
        "  "
    ))
    if not SCIPY_AVAILABLE and RUN_INTERP_FILTER:
        print("\nERROR: scipy is required for RUN_INTERP_FILTER. Install scipy or set RUN_INTERP_FILTER=False.")
        sys.exit(1)
    run_pipeline()
