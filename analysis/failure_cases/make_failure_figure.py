"""
Failure cases, step 2: 2x3 figure of the selected cases.

Reads selected_cases.json (sample_idx and failure factor of each visually
confirmed case). Every number on the panels (true and predicted group,
confidence, age) is taken from the out-of-fold predictions; a selected sample
that was predicted correctly is rejected.

Outputs (results/failure_cases/):
  fig_failure_cases.pdf / .png   2x3 figure (vector PDF, 300 dpi PNG)
  fig_failure_cases.tex          figure* environment with caption
  selected_cases_manifest.csv    full record of every case

Usage (from the repository root):
    python -m analysis.failure_cases.make_failure_figure --pred results/faceord/oof_predictions.xlsx
"""

import argparse
import json
import textwrap
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from faceord.config import Config
from faceord.landmarks import load_cache
from .fc_common import (DEFAULT_DATA, DEFAULT_PRED, FACTORS_CSV, OUT_DIR,
                       PROB_COLS, RACES, SELECTION, group_name, load_oof,
                       pipeline_landmarks, sha256)

plt.rcParams.update({
    "font.family"     : "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size"       : 8,
    "pdf.fonttype"    : 42,     # embed TrueType so PDF text stays selectable
})

GENDERS      = ["male", "female"]
LM_COLOR     = "#00E5FF"       # landmarks used by the model

# Landmark model name and caption text for images without a detected face
_DETECTOR_TEXT = {
    "2d_fan": ("2D-FAN", "the face detector finds no face and 2D-FAN is applied to the whole image"),
    "dlib"  : ("dlib", "dlib detects no face and the shape predictor is applied to the whole image"),
}
WRAP_CHARS   = 42


def check_selection(cases: list, df: pd.DataFrame):
    """Reject duplicates and correctly predicted samples; report the error distances."""
    idx = [c["sample_idx"] for c in cases]
    if len(set(idx)) != len(idx):
        raise ValueError("selected_cases.json contains duplicate sample_idx values")
    for c in cases:
        r = df.loc[c["sample_idx"]]
        if not (r["age_wrong"] or r["gender_wrong"]):
            raise ValueError(f"#{c['sample_idx']} is correct for both age and gender, "
                             "so it is not a failure case")

    dist = [int(abs(df.loc[i, "age_dist"])) for i in idx if df.loc[i, "age_wrong"]]
    print(f"  |pred - true| of the age errors: {sorted(dist)}")
    if dist and not any(d >= 2 for d in dist):
        print("  [warning] all errors are adjacent-group errors")
    if any(d >= 3 for d in dist):
        print("  [warning] an error of three or more groups is rare in the OOF predictions")


def panel_lines(r: pd.Series, factor: str) -> list:
    """Panel text: true label, prediction, confidence and the wrapped failure factor."""
    if r["age_wrong"]:
        true = rf"$\mathbf{{True}}$: {group_name(r['age_true'])}, age {r['age']}"
        pred = rf"$\mathbf{{Pred}}$: {group_name(r['age_pred'])}"
        conf = r["confidence"]
    else:                                   # gender error only
        true = (rf"$\mathbf{{True}}$: {GENDERS[r['gender_true']]}, "
                f"{group_name(r['age_true'])}")
        pred = rf"$\mathbf{{Pred}}$: {GENDERS[r['gender_pred']]} (age correct)"
        conf = r["gender_conf"]
    wrapped = textwrap.wrap(f"Failure factor: {factor}", WRAP_CHARS)
    wrapped[0] = wrapped[0].replace("Failure factor:", r"$\mathbf{Failure\ factor}$:", 1)
    return [true, pred, rf"$\mathbf{{Confidence}}$: {conf:.2f}", *wrapped]


def draw_overlay(ax, img_bgr: np.ndarray, cached=None, backend: str = Config.LANDMARK_BACKEND,
                 dlib_predictor: str = Config.DLIB_PREDICTOR) -> bool:
    """
    Draw the 68 landmarks used by FaceOrd on the original image.

    The aligned image is not shown: for rotated faces the 5-point affine with
    BORDER_REPLICATE stretches the borders, which could be mistaken for an
    artefact. Returns whether the face detector found the face.
    """
    h, w = img_bgr.shape[:2]
    lm    = pipeline_landmarks(img_bgr, cached, backend, dlib_predictor)
    scale = np.array([w / 224.0, h / 224.0])
    pts = lm["pts"] * scale
    ax.scatter(pts[:, 0], pts[:, 1], s=3.5, c=LM_COLOR, edgecolors="black",
               linewidths=0.25, zorder=3)
    ax.set_xlim(-0.5, w - 0.5)              # keep out-of-frame landmarks from widening the axes
    ax.set_ylim(h - 0.5, -0.5)
    return lm["detected"]


def make_figure(cases: list, df: pd.DataFrame, out_stem: Path, landmarks: dict,
                backend: str = Config.LANDMARK_BACKEND, dlib_predictor: str = Config.DLIB_PREDICTOR) -> dict:
    """Draw the figure; returns {panel letter: face detected} for panels with an overlay."""
    overlays = {}
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 6.6))
    for k, (ax, c) in enumerate(zip(axes.flat, cases)):
        r = df.loc[c["sample_idx"]]
        img = cv2.imread(r["path"])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        if c.get("landmark_overlay"):
            overlays[chr(97 + k)] = draw_overlay(ax, img, landmarks.get(c["sample_idx"]),
                                                 backend, dlib_predictor)
        ax.text(0.03, 0.97, f"({chr(97 + k)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
        for j, line in enumerate(panel_lines(r, c["failure_factor"])):
            ax.text(0.0, -0.035 - j * 0.085, line, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.17,
                        wspace=0.08, hspace=0.52)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return overlays


def overlay_sentence(overlays: dict, backend: str = Config.LANDMARK_BACKEND) -> str:
    """Caption sentence describing the landmark overlays of the panels."""
    name, failure = _DETECTOR_TEXT[backend]
    def _fmt(letters):
        return f"({', '.join(letters)})"
    missing = [p for p, det in overlays.items() if not det]
    parts = []
    if overlays:
        parts.append(f"In {_fmt(list(overlays))}, dots mark the 68 {name} landmarks used "
                     "for region decomposition.")
    if missing:
        parts.append(f"In {_fmt(missing)}, {failure}.")
    return "\n  ".join(parts)


def latex_snippet(cases: list, stats: dict, overlays: dict, backend: str = Config.LANDMARK_BACKEND) -> str:
    """LaTeX figure* environment whose caption is generated from the statistics and panels."""
    pct  = {int(k): v for k, v in stats["error_distance_pct"].items()}
    n_err = stats["n_age_errors"]
    within2 = pct.get(1, 0) + pct.get(2, 0)
    # Merge consecutive panels with the same factor: "(a, b) adjacent-age ambiguity"
    groups = []
    for k, c in enumerate(cases):
        if groups and groups[-1][1] == c["failure_factor"]:
            groups[-1][0].append(chr(97 + k))
        else:
            groups.append(([chr(97 + k)], c["failure_factor"]))
    panels = "; ".join(f"({', '.join(ls)}) {f}" for ls, f in groups)
    panels = panels.replace("–", "--")
    return rf"""% Generated by analysis.failure_cases.make_failure_figure.
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{fig_failure_cases.pdf}}
  \caption{{Representative failure cases of FaceOrd on held-out UTKFace test
  samples. The examples illustrate errors associated with adjacent age-group
  ambiguity, challenging pose or landmark localization, occlusion, and reduced
  image quality: {panels}.
  Most age errors remain close to neighboring ordinal categories, consistent with the confusion-matrix pattern reported in the
  Supplementary Material: of the {n_err:,} misclassified out-of-fold samples,
  {pct.get(1, 0):.1f}\% fall into an adjacent age group and {within2:.1f}\%
  within two groups. Confidence is the predicted-group probability from the
  fold model for which the image was held out.
  {overlay_sentence(overlays, backend)}}}
  \label{{fig:failure_cases}}
\end{{figure*}}
"""


def write_manifest(cases: list, df: pd.DataFrame, factors: pd.DataFrame, path: Path):
    """CSV with labels, predictions, probabilities and image factors of every selected case."""
    rows = []
    for k, c in enumerate(cases):
        r = df.loc[c["sample_idx"]]
        f = factors.loc[c["sample_idx"]]
        rows.append({
            "panel": f"({chr(97 + k)})", "slot": c["slot"],
            "sample_idx": c["sample_idx"], "fold": r["fold"],
            "filename": r["filename"], "age": r["age"],
            "gender": GENDERS[r["gender_true"]], "race": RACES[r["race"]],
            "age_true": group_name(r["age_true"]),
            "age_pred": group_name(r["age_pred"]),
            "age_dist": r["age_dist"], "confidence": round(r["confidence"], 4),
            "gender_pred": GENDERS[r["gender_pred"]],
            "gender_conf": round(r["gender_conf"], 4),
            "failure_factor": c["failure_factor"],
            "face_detected": f["detected"], "yaw": f["yaw"],
            "blur_lapvar": f["blur_lapvar"], "brightness": f["brightness"],
            **{p: round(r[p], 4) for p in PROB_COLS},
            "visual_note": c.get("visual_note", ""),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--selection", type=Path, default=SELECTION)
    ap.add_argument("--landmark-cache", default=Config.LANDMARK_CACHE)
    ap.add_argument("--landmark-backend", default=Config.LANDMARK_BACKEND, choices=["2d_fan", "dlib"])
    ap.add_argument("--dlib-predictor", default=Config.DLIB_PREDICTOR)
    args = ap.parse_args()

    selection = json.loads(args.selection.read_text())
    source_sha = sha256(args.pred)
    if selection.get("source_sha256") != source_sha:
        raise ValueError("source_sha256 in the selection does not match the prediction file")
    cases = selection["cases"]
    if len(cases) != 6:
        raise ValueError(f"The 2x3 layout needs exactly 6 cases, got {len(cases)}")

    df = load_oof(args.pred, args.data_root)
    check_selection(cases, df)
    stats   = json.loads((OUT_DIR / "oof_error_stats.json").read_text())
    if stats.get("source", {}).get("sha256") != source_sha:
        raise ValueError("oof_error_stats.json belongs to a different prediction file; rerun select_failure_cases")
    factors = pd.read_csv(FACTORS_CSV, index_col="sample_idx")

    points, sources = load_cache(args.landmark_cache, gamma=Config.GAMMA,
                                 backend=args.landmark_backend, with_source=True)
    landmarks = {}
    for c in cases:
        key = Path(df.loc[c["sample_idx"], "path"]).relative_to(args.data_root).as_posix()
        if key in points:
            landmarks[c["sample_idx"]] = (points[key], sources[key])

    out = OUT_DIR / "fig_failure_cases"
    overlays = make_figure(cases, df, out, landmarks, args.landmark_backend, args.dlib_predictor)
    out.with_suffix(".tex").write_text(latex_snippet(cases, stats, overlays, args.landmark_backend))
    write_manifest(cases, df, factors, OUT_DIR / "selected_cases_manifest.csv")
    print(f"  → {out.with_suffix('.pdf')}\n  → {out.with_suffix('.png')}\n"
          f"  → {out.with_suffix('.tex')}\n"
          f"  → {OUT_DIR / 'selected_cases_manifest.csv'}")


if __name__ == "__main__":
    main()
