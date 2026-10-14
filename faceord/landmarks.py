"""68-point facial landmarks.

Two backends are available (Config.LANDMARK_BACKEND):
  2d_fan : pretrained 2D-FAN of face-alignment 1.4.1 with the SFD face detector
           (default); created lazily, kept in evaluation mode, never optimised
  dlib   : dlib HOG face detector + 68-point shape predictor
           (shape_predictor_68_face_landmarks.dat, Config.DLIB_PREDICTOR)

Landmarks are estimated once on the resized 224x224 CLAHE/gamma image (the
same input as in the preprocessing pipeline) and stored in a cache that every
experiment reuses:

    python -m faceord.landmarks --data data/UTKFace --out data/landmarks_utkface.npz
    python -m faceord.landmarks --data /data/adience --recursive \
        --out /data/adience/landmarks.npz --device cuda
    python -m faceord.landmarks --backend dlib --out data/landmarks_utkface_dlib.npz

Cache keys are image paths relative to the data root (POSIX separators).
"""
import argparse
from collections import Counter
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import random

import numpy as np

FACE_ALIGNMENT_VERSION = '1.4.1'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Index of the mirrored landmark for each of the 68 iBUG points (horizontal flip).
FLIP_INDEX = np.array(
    list(range(16, -1, -1))                       # jaw 0-16
    + list(range(26, 16, -1))                     # eyebrows 17-26
    + [27, 28, 29, 30]                            # nose bridge
    + [35, 34, 33, 32, 31]                        # lower nose
    + [45, 44, 43, 42, 47, 46]                    # eye 36-41
    + [39, 38, 37, 36, 41, 40]                    # eye 42-47
    + [54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55]   # outer lip
    + [64, 63, 62, 61, 60, 67, 66, 65])           # inner lip


@lru_cache(maxsize=4)
def get_fan(device='cpu', detector='sfd'):
    """Return a frozen 2D-FAN instance, created once per (device, detector)."""
    import face_alignment
    import torch
    if version('face-alignment') != FACE_ALIGNMENT_VERSION:
        raise RuntimeError(f'FaceOrd requires face-alignment=={FACE_ALIGNMENT_VERSION}')
    # Building the network must not consume the training RNG streams.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            fan = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                              device=device, flip_input=False, face_detector=detector)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    fan.face_alignment_net.eval().requires_grad_(False)
    # face-alignment enables cuDNN benchmarking on CUDA; restore deterministic settings.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return fan


def fan_landmarks(img_bgr, device='cpu', detector='sfd'):
    """Landmarks of the largest detected face as ``(points (68, 2), source)``.

    The supported datasets contain face crops, so when the face detector finds
    nothing, 2D-FAN is applied with the whole image as the face box and
    ``source`` is ``"full_box"`` instead of ``"detector"``.
    """
    fan = get_fan(device, detector)
    rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
    points, _, boxes = fan.get_landmarks_from_image(rgb, return_bboxes=True)
    source = 'detector'
    if points is None or not len(points):
        h, w = rgb.shape[:2]
        boxes = [np.array([0, 0, w, h], dtype=np.float32)]
        points = fan.get_landmarks_from_image(rgb, detected_faces=boxes)
        source = 'full_box'
    if points is None or not len(points):
        raise RuntimeError('2D-FAN returned no landmarks')
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    pts = np.asarray(points[int(np.argmax(areas))], dtype=np.float32)[:, :2]
    if pts.shape != (68, 2) or not np.isfinite(pts).all():
        raise RuntimeError('Invalid 2D-FAN landmarks')
    return pts, source


@lru_cache(maxsize=2)
def get_dlib(predictor_path):
    """Return the dlib HOG face detector and 68-point shape predictor, loaded once."""
    import dlib
    if not Path(predictor_path).is_file():
        raise FileNotFoundError(f'dlib shape predictor not found at {predictor_path}; download '
                                'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
    return dlib.get_frontal_face_detector(), dlib.shape_predictor(str(predictor_path))


def dlib_landmarks(img_bgr, predictor_path):
    """Landmarks of the largest dlib face as ``(points (68, 2), source)``.

    Without a detected face the shape predictor uses the whole image as the face
    box and ``source`` is ``"full_box"``.
    """
    import cv2
    import dlib
    detector, predictor = get_dlib(str(predictor_path))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces):
        box, source = max(faces, key=lambda d: d.width() * d.height()), 'detector'
    else:
        h, w = gray.shape
        box, source = dlib.rectangle(0, 0, w - 1, h - 1), 'full_box'
    shape = predictor(gray, box)
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)
    return pts, source


def estimate_landmarks(img_bgr, backend='2d_fan', device='cpu', detector='sfd', dlib_predictor=None):
    """``(points, source)`` from the selected backend ("2d_fan" or "dlib")."""
    if backend == '2d_fan':
        return fan_landmarks(img_bgr, device, detector)
    if backend == 'dlib':
        return dlib_landmarks(img_bgr, dlib_predictor)
    raise ValueError(f'Unknown landmark backend: {backend!r}')


def backend_version(backend):
    """Installed version of the landmark library."""
    if backend == 'dlib':
        import dlib
        return str(dlib.__version__)
    return version('face-alignment')


def flip_landmarks(points, width):
    """Landmarks of the horizontally flipped image, with left/right indices swapped."""
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] = (width - 1) - pts[:, 0]
    return pts[FLIP_INDEX]


def transform_landmarks(points, matrix):
    """Apply a 2x3 affine matrix (as used by cv2.warpAffine) to (N, 2) points."""
    m = np.asarray(matrix, dtype=np.float32)
    return np.asarray(points, dtype=np.float32) @ m[:, :2].T + m[:, 2]


def perturb_landmarks(points, jitter=0.0, dropout=0.0, scale='face_diagonal', rng=None):
    """Perturb aligned landmarks for the robustness study.

    jitter : independent Gaussian noise per coordinate with std = jitter * D, where
             D is the landmark bounding-box diagonal ("face_diagonal") or the
             distance between the eye centres ("interocular").
    dropout: floor(68 * dropout) points are removed (set to NaN). Region boxes are
             built from the remaining points; an empty region becomes a black patch.
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.shape != (68, 2) or not np.isfinite(pts).all():
        raise ValueError('Expected 68 finite landmark coordinates')
    if jitter < 0 or not 0 <= dropout < 1:
        raise ValueError('Invalid landmark perturbation')
    rng = rng if rng is not None else np.random.default_rng()
    if scale == 'face_diagonal':
        distance = np.linalg.norm(pts.max(0) - pts.min(0))
    elif scale == 'interocular':
        distance = np.linalg.norm(pts[36:42].mean(0) - pts[42:48].mean(0))
    else:
        raise ValueError('Unknown landmark scale')
    pts += rng.normal(0, jitter * distance, pts.shape).astype(np.float32)
    dropped = rng.choice(68, size=int(np.floor(dropout * 68)), replace=False)
    pts[dropped] = np.nan
    return pts


def load_cache(path, gamma=None, backend=None, with_source=False):
    """Read a cache written by :func:`build_cache`.

    Returns ``{relative path: (68, 2) float32}`` (and ``{relative path: source}``
    when ``with_source``). A missing file gives empty dictionaries. When `gamma`
    or `backend` is given, the cache must have been built with the same setting.
    """
    if not path or not Path(path).is_file():
        return ({}, {}) if with_source else {}
    with np.load(path, allow_pickle=False) as data:
        if backend is not None and str(data['backend']) != backend:
            raise ValueError(f'{path} was built with the {data["backend"]} backend, expected {backend}')
        if gamma is not None and not np.isclose(float(data['gamma']), gamma):
            raise ValueError(f'{path} was computed with gamma={float(data["gamma"])}, expected {gamma}')
        keys = data['keys'].tolist()
        points = dict(zip(keys, data['landmarks'].astype(np.float32)))
        sources = dict(zip(keys, data['source'].tolist()))
    return (points, sources) if with_source else points


def build_cache(data_root, out_path, backend='2d_fan', device='cpu', detector='sfd',
                dlib_predictor=None, gamma=1.2, recursive=False):
    """Estimate landmarks for every image under ``data_root`` and save them to ``out_path``."""
    import cv2
    from .preprocessing import clahe_gamma
    root = Path(data_root)
    files = sorted(p for p in root.glob('**/*' if recursive else '*')
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise RuntimeError(f'No images found in {root}')
    keys, points, sources, unreadable = [], [], [], []
    for number, path in enumerate(files, 1):
        image = cv2.imread(str(path))
        if image is None:
            unreadable.append(path.relative_to(root).as_posix())
            continue
        pts, source = estimate_landmarks(clahe_gamma(cv2.resize(image, (224, 224)), gamma),
                                         backend, device, detector, dlib_predictor)
        keys.append(path.relative_to(root).as_posix())
        points.append(pts)
        sources.append(source)
        if number % 1000 == 0:
            print(f'  {number}/{len(files)}', flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, keys=np.array(keys), landmarks=np.stack(points).astype(np.float32),
                        source=np.array(sources), backend=np.array(backend),
                        library_version=np.array(backend_version(backend)),
                        face_detector=np.array(detector if backend == '2d_fan' else 'dlib_hog'),
                        gamma=np.array(gamma))
    counts = dict(Counter(sources))
    print(f'{len(keys)} images -> {out_path}; landmark source: {counts}')
    if unreadable:
        print(f'Skipped {len(unreadable)} unreadable images, e.g. {unreadable[:3]}')
    return counts


def main():
    from .config import Config
    p = argparse.ArgumentParser(description='Estimate and cache 68 landmarks per image')
    p.add_argument('--config', default='configs/faceord.json')
    p.add_argument('--data', help='Image root (default: DATA_ROOT of the configuration)')
    p.add_argument('--out', help='Output .npz (default: LANDMARK_CACHE of the configuration)')
    p.add_argument('--backend', choices=['2d_fan', 'dlib'], help='Default: LANDMARK_BACKEND of the configuration')
    p.add_argument('--dlib-predictor', help='Default: DLIB_PREDICTOR of the configuration')
    p.add_argument('--device', help='Device for 2D-FAN, e.g. cuda or cpu')
    p.add_argument('--recursive', action='store_true', help='Include subdirectories (MORPH-II, Adience)')
    args = p.parse_args()
    cfg = Config.from_json(args.config)
    build_cache(args.data or cfg.DATA_ROOT, args.out or cfg.LANDMARK_CACHE,
                args.backend or cfg.LANDMARK_BACKEND, args.device or cfg.LANDMARK_DEVICE,
                cfg.LANDMARK_FACE_DETECTOR, args.dlib_predictor or cfg.DLIB_PREDICTOR,
                cfg.GAMMA, args.recursive)


if __name__ == '__main__':
    main()
