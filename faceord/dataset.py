"""
UTKFace dataset.

File names follow [age]_[gender]_[race]_[datetime].jpg with gender 0 = male,
1 = female. Exact ages map to K=8 ordered groups:
  0: 0-2    1: 3-6    2: 7-13   3: 14-21
  4: 22-33  5: 34-44  6: 45-53  7: 54+

Samples are sorted by file name; this order defines `sample_idx` in every
split manifest and prediction file.
"""

import os
import re
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .age_mapping import AGE_BINS
from .config import Config
from .landmarks import load_cache
from .preprocessing import FaceOrdPreprocessor


# Display labels of the age groups
GROUP_LABELS = [
    "0-2", "3-6", "7-13", "14-21",
    "22-33", "34-44", "45-53", "54+",
]


def age_to_group(age: int) -> int:
    """Exact age in years -> group index 0..K-1."""
    for i in range(len(AGE_BINS) - 1, -1, -1):
        if age >= AGE_BINS[i]:
            return i
    return 0


_FNAME_RE = re.compile(r'^(\d+)_(\d+)_(\d+)_.*\.(jpg|png|jpeg)$', re.IGNORECASE)


def _parse_utk_filename(fname: str):
    """UTKFace file name -> (age, gender), or None if it does not follow the format."""
    m = _FNAME_RE.match(os.path.basename(fname))
    if m is None:
        return None
    age    = int(m.group(1))
    gender = int(m.group(2))
    if gender not in (0, 1):
        return None
    return age, gender


class UTKFaceDataset(Dataset):
    """
    UTKFace for FaceOrd and the comparison models.

    Each item is a dict with
      'global_img' : (3, 224, 224)  aligned, normalised face
      'regions'    : (M, 3, R, R)   semantic region patches
      'age_group'  : long           age group 0..K-1
      'gender'     : long           0 = male, 1 = female

    Args:
      root  : directory with the UTKFace images
      train : apply training augmentation
      cfg   : Config; LANDMARK_CACHE is used when the file exists, otherwise
              landmarks are estimated on the fly (much slower)
    """

    def __init__(self, root: str, train: bool = True, cfg=None):
        if cfg is None:
            cfg = Config()

        self.root  = Path(root)
        self.train = train
        self.cfg   = cfg
        self.preprocessor = FaceOrdPreprocessor.from_config(cfg, train=train)

        # (path, age group, gender) of every valid image
        self.samples = []
        for fpath in sorted(self.root.glob("*")):
            if not fpath.is_file():
                continue
            parsed = _parse_utk_filename(fpath.name)
            if parsed is None:
                continue
            age, gender = parsed
            if age < 0 or age > 116:
                continue
            self.samples.append((str(fpath), age_to_group(age), gender))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid UTKFace images found in '{root}'.\n"
                f"Expected file names such as 25_0_2_20170116174525125.jpg"
            )

        self.landmarks = load_cache(cfg.LANDMARK_CACHE, gamma=cfg.GAMMA, backend=cfg.LANDMARK_BACKEND)
        if not self.landmarks:
            warnings.warn(f"Landmark cache '{cfg.LANDMARK_CACHE}' not found; landmarks are estimated "
                          "on the fly. Build it once with `python -m faceord.landmarks`.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, age_group, gender = self.samples[idx]

        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            raise RuntimeError(f"Cannot decode image: {fpath}")
        landmarks = self.landmarks.get(Path(fpath).relative_to(self.root).as_posix())
        # Per-sample generator keeps landmark perturbations reproducible for a given fold seed.
        global_t, regions_t = self.preprocessor(img_bgr, landmarks,
                                                rng=np.random.default_rng(self.cfg.TRAIN_SEED + idx))

        return {
            "global_img" : global_t,                              # (3, H, W)
            "regions"    : regions_t,                             # (M, 3, R, R)
            "age_group"  : torch.tensor(age_group, dtype=torch.long),
            "gender"     : torch.tensor(gender,    dtype=torch.long),
        }
