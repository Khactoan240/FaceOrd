"""Landmark robustness: evaluate each outer-fold checkpoint on its own test fold under
landmark jitter or dropout.

    python -m faceord.landmark_robustness --checkpoints results/faceord/fold_*/best.pt \
        --splits results/shared_splits.json --data data/UTKFace

Perturbations act on the aligned landmarks before region construction; the global
image is unchanged. Jitter adds Gaussian noise with std = level * D per coordinate
(D = landmark bounding-box diagonal, or interocular distance with --scale
interocular); dropout removes floor(68 * level) points.
"""
import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
from .checkpoint import load_checkpoint
from .dataset import UTKFaceDataset
from .engine import collect_predictions, prediction_metrics, make_loader, resolve_device, write_json, aggregate_fold_metrics
from .splits import load_or_create_folds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoints', required=True, nargs='+', type=Path)
    p.add_argument('--splits', required=True, type=Path)
    p.add_argument('--data', required=True)
    p.add_argument('--output', default='results/landmark_robustness', type=Path)
    p.add_argument('--jitters', nargs='+', type=float, default=[0.0, .01, .03, .05])
    p.add_argument('--dropouts', nargs='+', type=float, default=[0.0, .1, .2, .3])
    p.add_argument('--scale', choices=['face_diagonal', 'interocular'], default='face_diagonal')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--landmark-cache', help='Override LANDMARK_CACHE of the checkpoints')
    args = p.parse_args()
    if not args.splits.is_file():
        p.error('An existing outer-split manifest is required')
    if (args.output / 'metrics.json').exists():
        raise FileExistsError('Choose a new robustness output directory')
    conditions = [(0., 0.)] + [(v, 0.) for v in args.jitters if v] + [(0., v) for v in args.dropouts if v]
    conditions = list(dict.fromkeys(conditions))
    measurements = {f'jitter_{j:g}_dropout_{d:g}': [] for j, d in conditions}
    seen, provenance, shared_split, identity = set(), [], None, None
    device = resolve_device(args.device)
    for path in args.checkpoints:
        model, cfg, payload = load_checkpoint(path, device)
        metadata = payload['metadata']
        if metadata.get('purpose') != 'outer_cv' or payload['model_name'] != 'faceord':
            raise ValueError('Use FaceOrd outer-fold checkpoints from faceord.kfold')
        fold = metadata['fold']
        if fold in seen:
            raise ValueError('Duplicate outer fold checkpoint')
        seen.add(fold)
        # All checkpoints must share model and preprocessing settings; seeds, paths and
        # per-fold tuned loss coefficients may differ.
        comparable = {key: value for key, value in cfg.to_dict().items() if key not in
                      {'TRAIN_SEED', 'SAVE_DIR', 'DATA_ROOT', 'LANDMARK_CACHE', 'DLIB_PREDICTOR', 'NUM_WORKERS', 'DEVICE',
                       'LABEL_DIST_TAU', 'LAMBDA_MONO', 'LAMBDA_DIST', 'FIXED_AGE_WEIGHT', 'FIXED_GENDER_WEIGHT'}}
        comparable['model_name'], comparable['variant'] = payload['model_name'], payload['variant']
        if identity is not None and comparable != identity:
            raise ValueError('Robustness checkpoints use different model/preprocessing configurations')
        identity = comparable
        cfg = replace(cfg, DATA_ROOT=args.data, NUM_WORKERS=args.workers, DEVICE=args.device,
                      LANDMARK_JITTER=0.0, LANDMARK_DROPOUT=0.0, LANDMARK_SCALE=args.scale,
                      LANDMARK_CACHE=args.landmark_cache or cfg.LANDMARK_CACHE)
        cfg.validate()
        dataset = UTKFaceDataset(args.data, train=False, cfg=cfg)
        folds, split = load_or_create_folds(args.splits, dataset.samples, args.data, cfg)
        if split != metadata['split'] or (shared_split is not None and split != shared_split):
            raise ValueError('Checkpoint, dataset and split manifest do not match')
        shared_split = split
        indices = next(item['test'] for item in folds if item['fold'] == fold)
        if indices != metadata['test_indices']:
            raise ValueError('Held-out indices differ from the checkpoint')
        for jitter, dropout in conditions:
            condition = replace(cfg, LANDMARK_JITTER=jitter, LANDMARK_DROPOUT=dropout)
            condition.validate()
            dataset = UTKFaceDataset(args.data, train=False, cfg=condition)
            pred = collect_predictions(model, make_loader(dataset, indices, condition), device,
                                       include_cumulatives=True)
            key = f'jitter_{jitter:g}_dropout_{dropout:g}'
            measurements[key].append({'fold': fold, **prediction_metrics(pred)})
        provenance.append({'fold': fold, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        del model
    if seen != set(range(1, shared_split['k'] + 1)):
        raise ValueError('Supply the complete outer-fold checkpoint set')
    write_json(args.output / 'metrics.json', {'evaluation': 'outer_test', 'split': shared_split,
               'checkpoint_provenance': provenance, 'scale': args.scale,
               'jitter_distribution': 'independent zero-mean Gaussian per coordinate, std=jitter*D',
               'D': 'aligned landmark bounding-box diagonal' if args.scale == 'face_diagonal' else 'aligned interocular distance',
               'dropout': 'floor(68*p) points sampled without replacement; skip dropped points in regional boxes; empty region is black',
               'perturbation_stage': 'after affine alignment, before region construction; clean global image retained',
               'conditions': {key: {'folds': rows, 'summary': aggregate_fold_metrics(rows)} for key, rows in measurements.items()}})


if __name__ == '__main__':
    main()
