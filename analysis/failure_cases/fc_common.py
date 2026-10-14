"""
Shared utilities of the failure-case analysis (select_failure_cases.py and
make_failure_figure.py).

The input is the out-of-fold prediction file written by `faceord.kfold`, in
which every image is predicted by the fold model that held it out. Landmark
overlays use the functions of `faceord.preprocessing` (resize 224 -> CLAHE +
gamma -> 68 landmarks (2D-FAN or dlib) -> 5-point affine), so they show what the
model sees.
"""

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from faceord import preprocessing as fp          # noqa: E402
from faceord.config import Config                 # noqa: E402
from faceord.dataset import UTKFaceDataset        # noqa: E402
from faceord.landmarks import estimate_landmarks  # noqa: E402


SCRIPT_DIR   = Path(__file__).resolve().parent
OUT_DIR      = REPO_ROOT / "results" / "failure_cases"
DEFAULT_PRED = REPO_ROOT / "results" / "faceord" / "oof_predictions.xlsx"
DEFAULT_DATA = REPO_ROOT / "data" / "UTKFace"
SELECTION    = SCRIPT_DIR / "selected_cases.json"
FACTORS_CSV  = OUT_DIR / "oof_image_factors.csv"

K = Config.K
PROB_COLS = [f"age_prob_{k}" for k in range(K)]

# Group labels G0..G7 (same indices as in the code)
GROUP_RANGES = ["0–2", "3–6", "7–13", "14–21", "22–33", "34–44", "45–53", "≥54"]
RACES        = ["White", "Black", "Asian", "Indian", "Others"]


def group_name(k: int) -> str:
    """Group index -> display string, e.g. 5 -> 'G5 (34–44)'."""
    return f"G{k} ({GROUP_RANGES[k]})"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_oof(pred_path: Path = DEFAULT_PRED,
             data_root: Path = DEFAULT_DATA) -> pd.DataFrame:
    """
    Read the out-of-fold predictions and attach image information (path, exact
    age, race) and per-sample error/confidence columns.

    Checks (raise on failure):
      - age_pred == argmax(age_prob_*) for every sample;
      - sample_idx is a permutation of 0..N-1 in UTKFaceDataset order;
      - age group and gender parsed from the file names match age_true / gender_true.
    """
    from faceord.eval_from_predictions import read_predictions
    df = read_predictions(pred_path)

    samples = UTKFaceDataset(str(data_root), train=False).samples
    N = len(samples)
    if len(df) != N or sorted(df["sample_idx"]) != list(range(N)):
        raise ValueError(f"sample_idx does not match the {N} images in {data_root}")

    df = df.set_index("sample_idx", drop=False).sort_index()
    if "filename" in df:
        expected_names = [Path(samples[i][0]).name for i in df.index]
        if df["filename"].tolist() != expected_names:
            raise ValueError("Prediction filenames differ from the dataset ordering")
    df["path"]     = [samples[i][0] for i in df.index]
    df["filename"] = [Path(p).name for p in df["path"]]
    parts          = df["filename"].str.split("_")
    df["age"]      = parts.str[0].astype(int)
    df["race"]     = parts.str[2].astype(int)

    group_ok  = np.array([samples[i][1] for i in df.index]) == df["age_true"]
    gender_ok = np.array([samples[i][2] for i in df.index]) == df["gender_true"]
    if not (group_ok.all() and gender_ok.all()):
        raise ValueError("Labels parsed from file names differ from age_true/gender_true")

    P = df[PROB_COLS].to_numpy()
    df["confidence"]   = P[np.arange(N), df["age_pred"].to_numpy()]
    df["age_dist"]     = df["age_pred"] - df["age_true"]
    df["age_wrong"]    = df["age_dist"] != 0
    df["gender_wrong"] = df["gender_pred"] != df["gender_true"]
    df["gender_conf"]  = np.where(df["gender_pred"] == 1,
                                  df["gender_prob"], 1.0 - df["gender_prob"])
    return df


# Generic 6-point 3D face model for solvePnP
# (nose tip 30, chin 8, outer eye corners 36/45, mouth corners 48/54)
_MODEL_3D = np.array([(0, 0, 0), (0, -330, -65),
                      (-225, 170, -135), (225, 170, -135),
                      (-150, -150, -125), (150, -150, -125)], dtype=np.float64)
_PNP_IDX  = [30, 8, 36, 45, 48, 54]


def pipeline_landmarks(img_bgr: np.ndarray, cached=None, backend: str = Config.LANDMARK_BACKEND,
                       dlib_predictor: str = Config.DLIB_PREDICTOR) -> dict:
    """
    Steps 1-4 of FaceOrdPreprocessor on an original image.

    `cached` is an optional (points, source) pair from the landmark cache;
    otherwise landmarks are estimated with `backend` ("2d_fan" or "dlib").
    Returns:
      img224      : 224x224 image after CLAHE + gamma (input of the landmark model)
      pts         : (68, 2) landmarks on img224
      detected    : True when the face detector found the face; False when the
                    landmark model ran on the whole image as the face box
      aligned     : image after the 5-point affine alignment (model input)
      pts_aligned : (68, 2) landmarks on the aligned image
    """
    img = fp.clahe_gamma(cv2.resize(img_bgr, (224, 224)), Config.GAMMA)
    if cached is not None:
        pts, source = cached
    else:
        pts, source = estimate_landmarks(img, backend, dlib_predictor=dlib_predictor)
    aligned, pts_aligned = fp.affine_align(img, pts)
    return {"img224": img, "pts": pts, "detected": source == "detector",
            "aligned": aligned, "pts_aligned": pts_aligned}


def head_pose(pts: np.ndarray, size: int = 224) -> tuple:
    """(yaw, pitch, roll) in degrees from the 68 landmarks via solvePnP."""
    cam = np.array([[size, 0, size / 2], [0, size, size / 2], [0, 0, 1]],
                   dtype=np.float64)
    _, rvec, _ = cv2.solvePnP(_MODEL_3D, pts[_PNP_IDX].astype(np.float64),
                              cam, np.zeros(4))
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, _ = cv2.RQDecomp3x3(R)[0]
    # The model y axis points up and the image y axis down -> pitch is around +/-180
    pitch = (pitch + 360.0) % 360.0 - 180.0
    eye_l, eye_r = pts[36:42].mean(0), pts[42:48].mean(0)
    roll = float(np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])))
    return float(yaw), float(pitch), roll


def align_residual(pts: np.ndarray, size: int = 224) -> float:
    """
    RMS residual of the 5-point least-squares affine fit used by `affine_align`,
    normalised by the template eye distance (0.30 * size). Large values mean the
    landmarks deviate from a frontal configuration (strong pose or mislocalisation).
    """
    src = np.array([pts[36:42].mean(0), pts[42:48].mean(0),
                    pts[30], pts[48], pts[54]], dtype=np.float64)
    dst = np.array([[0.35, 0.37], [0.65, 0.37], [0.50, 0.55],
                    [0.37, 0.70], [0.63, 0.70]]) * size
    src_h = np.hstack([src, np.ones((5, 1))])
    M, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
    rms = np.sqrt(((src_h @ M - dst) ** 2).sum(1).mean())
    return float(rms / (0.30 * size))


def region_box(pts_aligned: np.ndarray, idxs: list, pad: float = 0.3,
               size: int = 224) -> tuple:
    """Bounding box (x0, y0, x1, y1) of a region, computed as in `extract_regions`."""
    p = pts_aligned[idxs]
    (x0, y0), (x1, y1) = p.min(0), p.max(0)
    bw, bh = x1 - x0, y1 - y0
    return (max(0, int(x0 - pad * bw)), max(0, int(y0 - pad * bh)),
            min(size, int(x1 + pad * bw)), min(size, int(y1 + pad * bh)))


def region_skin_consistency(aligned: np.ndarray, pts_aligned: np.ndarray,
                            delta_e: float = 25.0) -> np.ndarray:
    """
    Occlusion proxy: for each semantic region (same boxes as `extract_regions`),
    the share of pixels whose Lab colour is close to a skin reference (median of
    the nose-bridge area, landmarks 28-30). Sunglasses, hair or other occluders
    lower the share in the eye or mouth regions. Used only to rank candidates;
    occlusion must be confirmed visually.
    """
    lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
    size = lab.shape[0]

    def _box(idxs, pad):
        return region_box(pts_aligned, idxs, pad, size)

    x0, y0, x1, y1 = _box([28, 29, 30, 31, 35], 0.0)
    ref_patch = lab[y0:y1 + 1, x0:x1 + 1].reshape(-1, 3)
    if len(ref_patch) == 0:
        return np.full(len(fp.REGION_LANDMARK_GROUPS), np.nan)
    ref = np.median(ref_patch, axis=0)

    out = []
    for m, idxs in fp.REGION_LANDMARK_GROUPS.items():
        x0, y0, x1, y1 = _box(idxs, 0.3)
        patch = lab[y0:y1, x0:x1].reshape(-1, 3)
        if len(patch) == 0:
            out.append(np.nan)
            continue
        out.append(float((np.linalg.norm(patch - ref, axis=1) < delta_e).mean()))
    return np.array(out)


def image_factors(path: str, cached=None, backend: str = Config.LANDMARK_BACKEND,
                  dlib_predictor: str = Config.DLIB_PREDICTOR) -> dict:
    """Diagnostic measurements of one image (run in a process pool)."""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lm = pipeline_landmarks(img, cached, backend, dlib_predictor)

    row = {
        "blur_lapvar": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness" : float(gray.mean()),
        "contrast"   : float(gray.std()),
        "detected"   : bool(lm["detected"]),
    }
    row["yaw"], row["pitch"], row["roll"] = head_pose(lm["pts"])
    row["align_res"] = align_residual(lm["pts"])
    skin = region_skin_consistency(lm["aligned"], lm["pts_aligned"])
    for m, v in enumerate(skin):
        row[f"skin_r{m}"] = v
    return row
