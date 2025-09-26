from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from pathlib import Path

from .config import CFG, SCALE_BY_INTEROCULAR
from .io_utils import detect_conf_prefix_case_insensitive, relevant_indices, find_real_colname
from .normalize_utils import interocular_series
from .geometry_utils import procrustes_frame_to_template, angle_between_points
from .window_utils import windows_indices, is_distance_like_metric, linear_metrics

# --------- Small helpers used by features -----------------------------------
def blink_aperture_from_points(eye_top: np.ndarray, eye_bot: np.ndarray) -> float:
    top_mean = eye_top.mean(axis=0)
    bot_mean = eye_bot.mean(axis=0)
    return float(abs(top_mean[1] - bot_mean[1]))

def mouth_aperture(p63: np.ndarray, p67: np.ndarray) -> float:
    return float(np.linalg.norm(p67 - p63))

# --------- Per-file Procrustes features -------------------------------------
def procrustes_features_for_file(df_norm: pd.DataFrame,
                                 template_df: pd.DataFrame,
                                 rel_idxs: List[int]) -> Dict[str, np.ndarray]:
    cols = list(df_norm.columns)
    xs, ys = [], []
    conf_prefix = detect_conf_prefix_case_insensitive(cols)
    for i in rel_idxs:
        xs.append(find_real_colname("x", i, cols))
        ys.append(find_real_colname("y", i, cols))

    templ_xy = np.column_stack([template_df[xs].values[0], template_df[ys].values[0]])
    n = len(df_norm)
    head_rot = np.full(n, np.nan, float)
    head_tx  = np.full(n, np.nan, float)
    head_ty  = np.full(n, np.nan, float)
    head_s   = std_s = np.full(n, np.nan, float)
    head_motion_mag = np.full(n, np.nan, float)
    blink_ap = np.full(n, np.nan, float)
    mouth_ap = np.full(n, np.nan, float)
    pupil_av = np.full(n, np.nan, float)

    def idx_of(lmk: int) -> int:
        return rel_idxs.index(lmk) if lmk in rel_idxs else -1

    L_top_idxs = [idx_of(38), idx_of(39)]
    L_bot_idxs = [idx_of(41), idx_of(42)]
    R_top_idxs = [idx_of(44), idx_of(45)]
    R_bot_idxs = [idx_of(47), idx_of(48)]
    left_eye_ring  = [idx_of(i) for i in [37,38,39,40,41,42] if idx_of(i) >= 0]
    right_eye_ring = [idx_of(i) for i in [43,44,45,46,47,48] if idx_of(i) >= 0]
    mouth_pair = (idx_of(63), idx_of(67))
    eye_pair = (idx_of(37), idx_of(46))

    for t in range(n):
        fx, fy = [], []
        for xc, yc in zip(xs, ys):
            fx.append(df_norm.iloc[t, df_norm.columns.get_loc(xc)] if xc else np.nan)
            fy.append(df_norm.iloc[t, df_norm.columns.get_loc(yc)] if yc else np.nan)
        frame_xy = np.column_stack([np.asarray(fx, float), np.asarray(fy, float)])
        available = np.isfinite(frame_xy).all(axis=1) & np.isfinite(templ_xy).all(axis=1)

        ok, s, tx, ty, R, Xtrans = procrustes_frame_to_template(frame_xy, templ_xy, available)
        if not ok:
            continue

        head_s[t] = s
        head_tx[t] = tx
        head_ty[t] = ty
        head_motion_mag[t] = math.sqrt(tx*tx + ty*ty + (s - 1.0)**2)

        i37, i46 = eye_pair
        if i37 >= 0 and i46 >= 0 and np.isfinite(Xtrans[i37]).all() and np.isfinite(Xtrans[i46]).all():
            head_rot[t] = angle_between_points(Xtrans[i37], Xtrans[i46])

        def safe_points(idxs: List[int]) -> Optional[np.ndarray]:
            pts = [Xtrans[i] for i in idxs if i >= 0 and np.isfinite(Xtrans[i]).all()]
            return np.vstack(pts) if len(pts) == 2 else None

        Ltop = safe_points(L_top_idxs); Lbot = safe_points(L_bot_idxs)
        Rtop = safe_points(R_top_idxs); Rbot = safe_points(R_bot_idxs)
        vals = []
        if Ltop is not None and Lbot is not None:
            vals.append(blink_aperture_from_points(Ltop, Lbot))
        if Rtop is not None and Rbot is not None:
            vals.append(blink_aperture_from_points(Rtop, Rbot))
        if vals:
            blink_ap[t] = float(np.mean(vals))

        m63, m67 = mouth_pair
        if m63 >= 0 and m67 >= 0 and np.isfinite(Xtrans[m63]).all() and np.isfinite(Xtrans[m67]).all():
            mouth_ap[t] = mouth_aperture(Xtrans[m63], Xtrans[m67])

        def eye_center(idxs: List[int]) -> Optional[np.ndarray]:
            pts = [Xtrans[i] for i in idxs if i >= 0 and np.isfinite(Xtrans[i]).all()]
            if len(pts) >= 3:
                return np.vstack(pts).mean(axis=0)
            return None

        cL = eye_center(left_eye_ring)
        cR = eye_center(right_eye_ring)

        i69 = rel_idxs.index(69) if 69 in rel_idxs else -1
        i70 = rel_idxs.index(70) if 70 in rel_idxs else -1

        vals = []
        if i69 >= 0 and cL is not None and np.isfinite(Xtrans[i69]).all():
            vals.append(float(np.linalg.norm(Xtrans[i69] - cL)))
        if i70 >= 0 and cR is not None and np.isfinite(Xtrans[i70]).all():
            vals.append(float(np.linalg.norm(Xtrans[i70] - cR)))
        if vals:
            pupil_av[t] = float(np.mean(vals))

    return {
        "head_rotation_rad": head_rot,
        "head_tx": head_tx,
        "head_ty": head_ty,
        "head_scale": head_s,
        "head_motion_mag": head_motion_mag,
        "blink_aperture": blink_ap,
        "mouth_aperture": mouth_ap,
        "pupil_metric": pupil_av
    }

# --------- Per-file "original" features -------------------------------------
def original_features_for_file(df_norm: pd.DataFrame) -> Dict[str, np.ndarray]:
    cols = list(df_norm.columns)

    def col(i: int, axis: str) -> pd.Series:
        c = find_real_colname(axis, i, cols)
        return df_norm[c].astype(float) if c else pd.Series([np.nan]*len(df_norm))

    n = len(df_norm)
    x37, y37 = col(37, "x"), col(37, "y")
    x46, y46 = col(46, "x"), col(46, "y")

    head_rot = np.full(n, np.nan, float)
    vdx = (x46 - x37).values
    vdy = (y46 - y37).values
    valid = np.isfinite(vdx) & np.isfinite(vdy)
    head_rot[valid] = np.arctan2(vdy[valid], vdx[valid])

    def avg2(s1, s2): return (s1 + s2) / 2.0
    Ltop = avg2(col(38,"y"), col(39,"y"))
    Lbot = avg2(col(41,"y"), col(42,"y"))
    Rtop = avg2(col(44,"y"), col(45,"y"))
    Rbot = avg2(col(47,"y"), col(48,"y"))
    blink = np.full(n, np.nan, float)
    L_ok = Ltop.notna() & Lbot.notna()
    R_ok = Rtop.notna() & Rbot.notna()
    if L_ok.any():
        blink[L_ok] = np.abs(Ltop[L_ok].values - Lbot[L_ok].values)
    if R_ok.any():
        tmp = np.abs(Rtop[R_ok].values - Rbot[R_ok].values)
        both = L_ok & R_ok
        blink[both] = (blink[both] + tmp[both]) / 2.0
        onlyR = R_ok & (~L_ok)
        blink[onlyR] = tmp[onlyR]

    x63, y63 = col(63,"x"), col(63,"y")
    x67, y67 = col(67,"x"), col(67,"y")
    mouth = np.sqrt((x67 - x63)**2 + (y67 - y63)**2).values

    def eye_center_xy(ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        xs = [col(i,"x").values for i in ids if find_real_colname("x", i, cols)]
        ys = [col(i,"y").values for i in ids if find_real_colname("y", i, cols)]
        if not xs or not ys:
            return np.full(n, np.nan), np.full(n, np.nan)
        x_mat = np.vstack(xs); y_mat = np.vstack(ys)
        return np.nanmean(x_mat, axis=0), np.nanmean(y_mat, axis=0)

    cLx, cLy = eye_center_xy([37,38,39,40,41,42])
    cRx, cRy = eye_center_xy([43,44,45,46,47,48])

    pLx = col(69, "x").values if find_real_colname("x", 69, cols) else np.full(n, np.nan)
    pLy = col(69, "y").values if find_real_colname("y", 69, cols) else np.full(n, np.nan)
    pRx = col(70, "x").values if find_real_colname("x", 70, cols) else np.full(n, np.nan)
    pRy = col(70, "y").values if find_real_colname("y", 70, cols) else np.full(n, np.nan)

    dL = np.sqrt((pLx - cLx)**2 + (pLy - cLy)**2)
    dR = np.sqrt((pRx - cRx)**2 + (pRy - cRy)**2)
    pupil = np.where(np.isfinite(dL) & np.isfinite(dR),
                     (dL + dR) / 2.0,
                     np.where(np.isfinite(dL), dL,
                              np.where(np.isfinite(dR), dR, np.nan)))

    nose_x = [col(i, "x").values for i in CFG.CENTER_FACE]
    nose_y = [col(i, "y").values for i in CFG.CENTER_FACE]
    nose_x = np.vstack(nose_x) if len(nose_x) else np.empty((0,n))
    nose_y = np.vstack(nose_y) if len(nose_y) else np.empty((0,n))
    cfm = np.full(n, np.nan, float)
    if nose_x.size and nose_y.size:
        mean_x = np.nanmean(nose_x, axis=1, keepdims=True)
        mean_y = np.nanmean(nose_y, axis=1, keepdims=True)
        dx = nose_x - mean_x
        dy = nose_y - mean_y
        dists = np.sqrt(dx**2 + dy**2)
        cfm = np.sqrt(np.nanmean(dists**2, axis=0))

    return {
        "head_rotation_rad": head_rot,
        "blink_aperture": blink,
        "mouth_aperture": mouth,
        "pupil_metric": pupil,
        "center_face_magnitude": cfm
    }

# --------- Linear-from-perframe helper --------------------------------------
def compute_linear_from_perframe_dir(per_frame_dir: Path,
                                     out_csv: Path,
                                     fps: int,
                                     window_seconds: int,
                                     window_overlap: float,
                                     scale_by_interocular: bool = True) -> Dict[str, int]:
    rows = []
    drops_agg: Dict[str, int] = {}
    files = sorted(per_frame_dir.glob("*.csv"))
    for pf in files:
        df = pd.read_csv(pf)
        pid = str(df["participant"].iloc[0]) if "participant" in df.columns and len(df) else "NA"
        cond = str(df["condition"].iloc[0]) if "condition" in df.columns and len(df) else "NA"
        metric_cols = [c for c in df.columns if c not in ("participant","condition","frame","interocular")]
        io = df["interocular"].to_numpy(float) if "interocular" in df.columns else np.full(len(df), np.nan)

        scaled = {}
        for k in metric_cols:
            arr = df[k].to_numpy(float)
            if scale_by_interocular and is_distance_like_metric(k) and np.isfinite(io).any():
                scaled[k] = arr / io
            else:
                scaled[k] = arr

        win = window_seconds * fps
        hop = int(win * (1.0 - window_overlap))
        hop = max(1, hop)
        n = len(df)
        for (s, e, widx) in windows_indices(n, win, hop):
            base = {"source": per_frame_dir.name, "participant": pid, "condition": cond,
                    "window_index": widx, "t_start_frame": s, "t_end_frame": e}
            for k, arr in scaled.items():
                seg = arr[s:e]
                if np.any(~np.isfinite(seg)) or len(seg) < 3:
                    drops_agg[k] = drops_agg.get(k, 0) + 1
                    base[f"{k}_mean_abs_vel"] = np.nan
                    base[f"{k}_mean_abs_acc"] = np.nan
                    base[f"{k}_rms"] = np.nan
                else:
                    v, a, r = linear_metrics(seg.astype(float), fps)
                    base[f"{k}_mean_abs_vel"] = v
                    base[f"{k}_mean_abs_acc"] = a
                    base[f"{k}_rms"] = r
            rows.append(base)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return drops_agg
