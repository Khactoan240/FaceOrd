"""Sensitivity to lambda_mono or lambda_dist with three inner folds (30 epochs) per outer fold.

    python -m faceord.loss_sweep --parameter LAMBDA_MONO --values 0 0.05 0.1 0.2 \
        --outer-splits results/shared_splits.json --output results/lambda_mono

The value with the highest mean inner-test age AUC is stored per outer fold in
selected_settings.json (input of `faceord.kfold --hyperparameter-selection`).
"""
import argparse
from dataclasses import replace
from pathlib import Path
from .config import Config
from .dataset import UTKFaceDataset
from .engine import run_cv, write_json
from .logger import LogSession
from .splits import load_or_create_folds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/faceord.json')
    p.add_argument('--outer-splits', default='results/shared_splits.json')
    p.add_argument('--parameter', required=True, choices=['LAMBDA_MONO', 'LAMBDA_DIST'])
    p.add_argument('--values', required=True, nargs='+', type=float, help='Grid of coefficients')
    p.add_argument('--data')
    p.add_argument('--landmark-cache')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    if min(args.values) < 0 or len(set(args.values)) != len(args.values):
        p.error('Use distinct, nonnegative coefficients')
    cfg = Config.from_json(args.config)
    cfg.DEVICE, cfg.NUM_WORKERS = args.device, args.workers
    if args.data:
        cfg.DATA_ROOT = args.data
    if args.landmark_cache:
        cfg.LANDMARK_CACHE = args.landmark_cache
    dataset = UTKFaceDataset(cfg.DATA_ROOT, train=False, cfg=cfg)
    folds, meta = load_or_create_folds(args.outer_splits, dataset.samples, cfg.DATA_ROOT, cfg)
    selected, measurements = {}, {}
    with LogSession(f'loss_sweep_{args.parameter.lower()}'):
        for outer in folds:
            pool = sorted(outer['train'] + outer['val'])
            scores = {}
            for value in args.values:
                directory = args.output / f'outer_{outer["fold"]}' / f'value_{value:g}'
                inner = replace(cfg, KFOLD_K=3, EPOCHS=30, SAVE_DIR=str(directory), **{args.parameter: value})
                result = run_cv(inner, pool_indices=pool,
                                split_path=args.output / f'outer_{outer["fold"]}' / 'inner_splits.json')
                scores[value] = result['summary']['age']['auc']['mean']
                measurements[f'{outer["fold"]}/{value}'] = result
            selected[str(outer['fold'])] = {args.parameter: max(scores, key=scores.get)}
        write_json(args.output / 'selected_settings.json', {
            'outer_split': meta, 'selected_settings': selected, 'selection_metric': 'inner_test_age_auc',
            'epochs': 30, 'inner_folds': 3, 'grid': args.values, 'parameter': args.parameter})
        write_json(args.output / 'sweep_metrics.json', measurements)


if __name__ == '__main__':
    main()
