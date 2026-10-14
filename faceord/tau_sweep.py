"""Sensitivity to tau with three inner folds (30 epochs) inside each outer training portion.

    python -m faceord.tau_sweep --outer-splits results/shared_splits.json --output results/tau_sweep

For every outer fold the tau with the highest mean inner-test age AUC is written to
selected_tau_by_outer_fold.json (input of `faceord.kfold --tau-selection`).
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
    p.add_argument('--data')
    p.add_argument('--output', default='results/tau_sweep')
    p.add_argument('--outer-splits', default='results/shared_splits.json')
    p.add_argument('--outer-fold', type=int, help='Run one outer fold; default runs all')
    p.add_argument('--taus', nargs='+', type=float, default=[0.5, 1, 1.5, 2, 3])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--workers', type=int)
    p.add_argument('--landmark-cache')
    args = p.parse_args()
    cfg = Config.from_json(args.config)
    if args.data:
        cfg.DATA_ROOT = args.data
    if args.landmark_cache:
        cfg.LANDMARK_CACHE = args.landmark_cache
    dataset = UTKFaceDataset(cfg.DATA_ROOT, train=False, cfg=cfg)
    outer_folds, meta = load_or_create_folds(args.outer_splits, dataset.samples, cfg.DATA_ROOT, cfg)
    if any(t <= 0 for t in args.taus):
        p.error('Every tau must be positive')
    if args.outer_fold is not None and args.outer_fold not in range(1, cfg.KFOLD_K + 1):
        p.error('outer-fold is outside the configured fold range')
    chosen, all_results = {}, {}
    with LogSession('tau_sweep'):
        for outer in outer_folds:
            fold = outer['fold']
            if args.outer_fold is not None and fold != args.outer_fold:
                continue
            # Inner folds are drawn from the outer training portion (train + validation) only.
            pool = sorted(outer['train'] + outer['val'])
            scores = {}
            for tau in args.taus:
                destination = Path(args.output) / f'outer_{fold}' / f'tau_{tau:g}'
                inner_cfg = replace(cfg, KFOLD_K=3, EPOCHS=args.epochs, LABEL_DIST_TAU=tau,
                                    SAVE_DIR=str(destination), DEVICE=args.device,
                                    NUM_WORKERS=cfg.NUM_WORKERS if args.workers is None else args.workers)
                result = run_cv(inner_cfg, pool_indices=pool,
                                split_path=Path(args.output) / f'outer_{fold}' / 'inner_splits.json')
                scores[str(tau)] = result['summary']['age']['auc']['mean']
                all_results[f'outer_{fold}/tau_{tau:g}'] = result
            chosen[str(fold)] = float(max(scores, key=scores.get))
        write_json(Path(args.output) / 'selected_tau_by_outer_fold.json',
                   {'outer_split': meta, 'selected_tau': chosen,
                    'selection_metric': 'inner_test_age_auc', 'inner_folds': 3, 'epochs': args.epochs})
        write_json(Path(args.output) / 'tau_sweep_summary.json', all_results)


if __name__ == '__main__':
    main()
