"""
Failure cases, step 1: rank candidate images from the out-of-fold predictions.

Six real held-out errors are shown in three groups of two:
  A. adjacent-age ambiguity      : G4<->G5, G6<->G7
  B. landmark / pose / occlusion : strong pose or landmark misalignment; occlusion or difficult crop
  C. appearance / image quality  : low image quality; appearance-age mismatch

The script does not pick the final cases. It
  1. checks the predictions and summarises the error distribution -> results/failure_cases/oof_error_stats.json
  2. measures every image (cached)                                  -> results/failure_cases/oof_image_factors.csv
     (blur, brightness, face detection, head pose, affine residual, occlusion proxy)
  3. filters and ranks candidates for each slot                     -> results/failure_cases/candidates/<slot>.csv/.png
After visual inspection of the contact sheets, record the chosen sample_idx in
selected_cases.json (see selected_cases.example.json) and run make_failure_figure.

Five slots use errors of one group and one slot an error of two groups, which
reflects the typical OOF error distribution (mostly adjacent-group errors).

Usage (from the repository root):
    python -m analysis.failure_cases.select_failure_cases --pred results/faceord/oof_predictions.xlsx
    python -m analysis.failure_cases.select_failure_cases --workers 8 --top 16 --recompute
"""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from faceord.config import Config
from faceord.landmarks import load_cache
from .fc_common import (DEFAULT_DATA, DEFAULT_PRED, FACTORS_CSV, K, OUT_DIR,
                       REPO_ROOT, image_factors, load_oof, sha256)

ADJACENT_PAIRS = [(4, 5), (6, 7)]     # G4<->G5 and G6<->G7


def error_stats(df: pd.DataFrame) -> dict:
    """Accuracy, error-distance distribution, top confusions and the confusion matrix."""
    err  = df[df["age_wrong"]]
    dist = err["age_dist"].abs()
    cm   = (pd.crosstab(df["age_true"], df["age_pred"])
              .reindex(index=range(K), columns=range(K), fill_value=0))
    off  = sorted(((t, p, int(cm.loc[t, p])) for t in range(K)
                   for p in range(K) if t != p), key=lambda x: -x[2])
    return {
        "n_samples"            : len(df),
        "n_age_errors"         : len(err),
        "age_accuracy"         : round(float(1 - df["age_wrong"].mean()), 4),
        "error_distance_counts": {int(k): int(v)
                                  for k, v in dist.value_counts().sort_index().items()},
        "error_distance_pct"   : {int(k): round(100 * v, 2) for k, v in
                                  dist.value_counts(normalize=True).sort_index().items()},
        "top_confusions"       : [{"true": t, "pred": p, "count": c}
                                  for t, p, c in off[:10]],
        "confusion_matrix"     : cm.to_numpy().tolist(),
        "n_gender_errors"            : int(df["gender_wrong"].sum()),
        "n_gender_errors_age_correct": int((df["gender_wrong"] & ~df["age_wrong"]).sum()),
    }


def compute_factors(df: pd.DataFrame, workers: int, recompute: bool, data_root: Path,
                    landmark_cache: str, backend: str, dlib_predictor: str) -> pd.DataFrame:
    """Image factors of every sample; reuses oof_image_factors.csv unless `recompute`."""
    if FACTORS_CSV.exists() and not recompute:
        f = pd.read_csv(FACTORS_CSV, index_col="sample_idx")
        if ("landmark_backend" in f and (f["landmark_backend"] == backend).all()
                and len(f) == len(df) and (f["filename"].to_numpy() == df["filename"].to_numpy()).all()):
            print(f"  [cache] {FACTORS_CSV.name}")
            return f
    points, sources = load_cache(landmark_cache, gamma=Config.GAMMA, backend=backend, with_source=True)
    keys = [Path(path).relative_to(data_root).as_posix() for path in df["path"]]
    cached = [(points[k], sources[k]) if k in points else None for k in keys]
    print(f"  Measuring {len(df)} images ({workers} workers, "
          f"{sum(c is not None for c in cached)} cached landmark sets)...")
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        n = len(df)
        for i, r in enumerate(ex.map(image_factors, df["path"].tolist(), cached,
                                     [backend] * n, [dlib_predictor] * n, chunksize=64)):
            rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f"    {i + 1}/{len(df)}")
    f = pd.DataFrame(rows, index=df.index)
    f.insert(0, "filename", df["filename"].to_numpy())
    f["landmark_backend"] = backend
    f.to_csv(FACTORS_CSV, index_label="sample_idx")
    return f


def add_percentiles(f: pd.DataFrame) -> pd.DataFrame:
    """Percentiles over all samples, so thresholds do not depend on the measurement scale."""
    f = f.copy()
    f["skin_min"] = f[["skin_r0", "skin_r1", "skin_r3"]].min(axis=1)   # eyes and mouth
    for c in ["blur_lapvar", "brightness", "contrast", "align_res", "skin_min"]:
        f[f"{c}_pct"] = f[c].rank(pct=True)
    return f


def factor_error_rates(df: pd.DataFrame, f: pd.DataFrame) -> list:
    """Age error rate under each image condition (descriptive, not causal)."""
    yaw = f["yaw"].abs()
    subsets = {
        "all samples"                           : pd.Series(True, index=f.index),
        "no face detected (full-image box)"     : ~f["detected"],
        "face detected"                         : f["detected"],
        "|yaw| >= 30 deg"                       : f["detected"] & (yaw >= 30),
        "|yaw| < 30 deg"                        : f["detected"] & (yaw < 30),
        "blur (Laplacian var, bottom 10%)"      : f["blur_lapvar_pct"] <= 0.10,
        "dark (brightness, bottom 5%)"          : f["brightness_pct"] <= 0.05,
    }
    out = []
    for name, m in subsets.items():
        sub = df[m]
        err = sub[sub["age_wrong"]]
        out.append({
            "subset"               : name,
            "n"                    : int(m.sum()),
            "age_error_rate"       : round(float(sub["age_wrong"].mean()), 4),
            "errors_within_1_group": round(float((err["age_dist"].abs() == 1).mean()), 4)
                                     if len(err) else None,
        })
    return out


def build_slots(X: pd.DataFrame, cm: np.ndarray) -> list:
    """
    Candidate slots: key, failure factor (panel label), selection rule, ranked candidates.
    'clean' = sharp, normally lit, face detected, near-frontal and no suspected
    occlusion, so the error cannot be attributed to image quality.
    """
    yaw = X["yaw"].abs()
    quality_ok = ((X["blur_lapvar_pct"] >= 0.30)
                  & X["brightness_pct"].between(0.10, 0.90)
                  & (X["contrast_pct"] >= 0.10))
    clean = (quality_ok & X["detected"] & (yaw < 20) & (X["roll"].abs() < 15)
             & (X["align_res_pct"] <= 0.80) & (X["skin_min_pct"] >= 0.20))
    d1 = X["age_wrong"] & (X["age_dist"].abs() == 1)
    d2 = X["age_wrong"] & (X["age_dist"].abs() == 2)

    slots = []

    # A. Adjacent-age ambiguity: dominant confusion direction of each pair
    for a, b in ADJACENT_PAIRS:
        t, p = (a, b) if cm[a, b] >= cm[b, a] else (b, a)
        cell = X[(X["age_true"] == t) & (X["age_pred"] == p)]
        med  = cell["confidence"].median()
        c    = cell[clean.loc[cell.index]].copy()
        c["_key"] = (c["confidence"] - med).abs()
        slots.append(dict(
            key=f"A_adjacent_G{t}_to_G{p}", factor="adjacent-age ambiguity",
            rule=(f"true G{t} → pred G{p} (dominant direction: {cm[t, p]} vs "
                  f"{cm[p, t]}); clean image; confidence closest to cell median "
                  f"{med:.2f}"),
            cands=c.sort_values("_key")))

    # B1. Strong pose / landmark misalignment: largest 5-point affine residual
    c = X[d1 & quality_ok & X["detected"]]
    slots.append(dict(
        key="B1_pose_landmark", factor="strong pose / landmark misalignment",
        rule="|error| = 1; image quality OK; face detected; ranked by 5-point affine residual",
        cands=c.sort_values("align_res", ascending=False)))

    # B2. Occlusion / difficult crop: no detected face first, then low skin consistency
    c = X[d1 & quality_ok & (~X["detected"] | (X["skin_min_pct"] <= 0.05))].copy()
    c["_key"] = c["detected"].astype(int)
    slots.append(dict(
        key="B2_occlusion_crop", factor="occlusion / difficult crop",
        rule=("|error| = 1; image quality OK; no face detected OR lowest 5% "
              "eye/mouth skin consistency; MUST be confirmed visually"),
        cands=c.sort_values(["_key", "skin_min"])))

    # C1. Low image quality: blur, dark, overexposed or low contrast
    worst = pd.concat([X["blur_lapvar_pct"], X["brightness_pct"],
                       1 - X["brightness_pct"], X["contrast_pct"]], axis=1).min(axis=1)
    low_q = ((X["blur_lapvar_pct"] <= 0.10) | (X["brightness_pct"] <= 0.05)
             | (X["brightness_pct"] >= 0.95) | (X["contrast_pct"] <= 0.05))
    c = X[d1 & low_q].copy()
    c["_key"] = worst.loc[c.index]
    slots.append(dict(
        key="C1_low_quality", factor="low image quality (blur / poor lighting)",
        rule="|error| = 1; bottom 10% sharpness or extreme brightness/contrast",
        cands=c.sort_values("_key")))

    # C2. Appearance-age mismatch: the single +/-2 slot, clean image, confident model;
    #     candidates for both directions (looks older / looks younger).
    for key, mask, desc in [("older", X["age_dist"] > 0, "looks older (pred > true)"),
                            ("younger", X["age_dist"] < 0, "looks younger (pred < true)")]:
        c = X[d2 & clean & mask]
        slots.append(dict(
            key=f"C2_appearance_{key}", factor="appearance–age mismatch",
            rule=f"|error| = 2 (the single ±2 slot); {desc}; clean image; ranked by confidence",
            cands=c.sort_values("confidence", ascending=False)))

    # Optional: gender wrong but age correct, can replace one slot
    c = X[X["gender_wrong"] & ~X["age_wrong"] & clean]
    slots.append(dict(
        key="opt_gender_error", factor="gender error (age correct)",
        rule="optional replacement slot; age correct, gender wrong; clean image",
        cands=c.sort_values("gender_conf", ascending=False)))
    return slots


def contact_sheet(c: pd.DataFrame, path: Path, title: str, top: int):
    """Grid of the top candidates with their predictions and image factors."""
    c = c.head(top)
    if c.empty:
        return
    cols = 4
    rows = math.ceil(len(c) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.85),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (idx, r) in zip(axes.flat, c.iterrows()):
        ax.imshow(cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB))
        yaw = "n/a" if np.isnan(r["yaw"]) else f"{r['yaw']:+.0f}°"
        ax.set_title(
            f"#{idx}  fold {r['fold']}  age {r['age']}\n"
            f"G{r['age_true']}→G{r['age_pred']}  p={r['confidence']:.2f}  "
            f"g:{r['gender_true']}→{r['gender_pred']}\n"
            f"yaw {yaw}  lap {r['blur_lapvar']:.0f}  "
            f"face det. {'Y' if r['detected'] else 'N'}", fontsize=6.5)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


_CSV_COLS = ["fold", "filename", "age", "race", "age_true", "age_pred",
             "age_dist", "confidence", "gender_true", "gender_pred",
             "gender_conf", "detected", "yaw", "pitch", "roll", "align_res",
             "blur_lapvar", "brightness", "contrast", "skin_min"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--landmark-cache", default=Config.LANDMARK_CACHE)
    ap.add_argument("--landmark-backend", default=Config.LANDMARK_BACKEND, choices=["2d_fan", "dlib"])
    ap.add_argument("--dlib-predictor", default=Config.DLIB_PREDICTOR)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--top", type=int, default=16,
                    help="candidates per contact sheet")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore the cached oof_image_factors.csv")
    args = ap.parse_args()

    cand_dir = OUT_DIR / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] OOF predictions: {args.pred}")
    df    = load_oof(args.pred, args.data_root)
    stats = error_stats(df)
    print(f"  N={stats['n_samples']}  age acc={stats['age_accuracy']:.4f}  "
          f"errors={stats['n_age_errors']}")
    print(f"  |pred−true| among errors (%): {stats['error_distance_pct']}")

    print("[2/3] Image factors")
    f = add_percentiles(compute_factors(df, args.workers, args.recompute, args.data_root,
                                        args.landmark_cache, args.landmark_backend,
                                        args.dlib_predictor))
    stats["factor_error_rates"] = factor_error_rates(df, f)
    for r in stats["factor_error_rates"]:
        print(f"  {r['subset']:42s} n={r['n']:6d}  age err={r['age_error_rate']:.3f}")

    pred = args.pred.resolve()
    stats["source"] = {"predictions": str(pred.relative_to(REPO_ROOT)
                                          if pred.is_relative_to(REPO_ROOT) else pred),
                       "sha256": sha256(pred)}
    with open(OUT_DIR / "oof_error_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print("[3/3] Candidates per slot")
    X = df.join(f.drop(columns="filename"))
    for s in build_slots(X, np.array(stats["confusion_matrix"])):
        cands = s["cands"]
        cands[_CSV_COLS].head(50).to_csv(cand_dir / f"{s['key']}.csv",
                                         index_label="sample_idx")
        contact_sheet(cands, cand_dir / f"{s['key']}.png",
                      f"{s['key']} — {s['factor']}\n{s['rule']}", args.top)
        print(f"  {s['key']:28s} {len(cands):5d} candidates -> candidates/{s['key']}.png")

    print("\nInspect the contact sheets, record the chosen cases in "
          "analysis/failure_cases/selected_cases.json and run make_failure_figure.")


if __name__ == "__main__":
    main()
