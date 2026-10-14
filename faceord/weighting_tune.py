"""Fixed task weights tuned on the validation subset of each outer fold.

    python -m faceord.weighting_tune --age-weights 0.5 1 2 \
        --outer-splits results/shared_splits.json --output results/weighting_tune

For every outer fold, FaceOrd is trained with fixed weights for each candidate
age weight; the weight with the highest validation age AUC is written to
selected_settings.json (input of `faceord.kfold --hyperparameter-selection`).
"""
import argparse
from dataclasses import replace
from pathlib import Path
from .config import Config
from .dataset import UTKFaceDataset
from .engine import train_one, seed_everything, class_weights, write_json
from .experiment import make_model, make_criterion
from .logger import LogSession
from .splits import load_or_create_folds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/faceord.json')
    p.add_argument('--outer-splits', default='results/shared_splits.json')
    p.add_argument('--age-weights', required=True, nargs='+', type=float)
    p.add_argument('--gender-weight', default=1.0, type=float)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--data')
    p.add_argument('--landmark-cache')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    if min(args.age_weights) <= 0 or args.gender_weight <= 0 or len(set(args.age_weights)) != len(args.age_weights):
        p.error('Use distinct positive weights')
    cfg = Config.from_json(args.config)
    cfg.DEVICE, cfg.NUM_WORKERS = args.device, args.workers
    if args.data:
        cfg.DATA_ROOT = args.data
    if args.landmark_cache:
        cfg.LANDMARK_CACHE = args.landmark_cache
    eval_ds = UTKFaceDataset(cfg.DATA_ROOT, train=False, cfg=cfg)
    train_ds = UTKFaceDataset(cfg.DATA_ROOT, train=True, cfg=cfg)
    folds, meta = load_or_create_folds(args.outer_splits, eval_ds.samples, cfg.DATA_ROOT, cfg)
    selected, trials = {}, []
    with LogSession('weighting_tune'):
        for split in folds:
            best_score, best_weight = -float('inf'), None
            for weight in args.age_weights:
                trial = replace(cfg, EPOCHS=30, WEIGHTING='fixed', FIXED_AGE_WEIGHT=weight,
                                FIXED_GENDER_WEIGHT=args.gender_weight,
                                TRAIN_SEED=cfg.FOLD_SEEDS[split['fold'] - 1])
                # The datasets read TRAIN_SEED for their per-sample generators.
                train_ds.cfg.TRAIN_SEED = trial.TRAIN_SEED
                eval_ds.cfg.TRAIN_SEED = trial.TRAIN_SEED
                seed_everything(trial.TRAIN_SEED)
                cw = class_weights([eval_ds.samples[i][1] for i in split['train']], 8, cfg.CLASS_WEIGHTS)
                model = make_model(trial)
                selected_model, validation = train_one(
                    model, make_criterion(trial, 'faceord', 'f_full', cw),
                    train_ds, eval_ds, split['train'], split['val'], trial, 'faceord', 'f_full',
                    args.output / f'fold_{split["fold"]}' / f'age_weight_{weight:g}',
                    {'purpose': 'validation_weight_tuning', 'fold': split['fold'], 'split': meta})
                score = validation['age']['auc']
                trials.append({'fold': split['fold'], 'age_weight': weight, 'validation': validation})
                if score is not None and score > best_score:
                    best_score, best_weight = score, weight
                del model, selected_model
            if best_weight is None:
                raise ValueError('Validation age AUC is undefined for all candidates')
            selected[str(split['fold'])] = {'WEIGHTING': 'fixed', 'FIXED_AGE_WEIGHT': best_weight,
                                            'FIXED_GENDER_WEIGHT': args.gender_weight}
        write_json(args.output / 'selected_settings.json', {
            'outer_split': meta, 'selected_settings': selected,
            'selection_metric': 'validation_age_auc', 'epochs': 30, 'trials': trials})


if __name__ == '__main__':
    main()
