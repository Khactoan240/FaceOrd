"""Five-fold outer cross-validation of FaceOrd or a comparison model.

    python -m faceord.kfold --config configs/faceord.json --splits results/shared_splits.json
    python -m faceord.kfold --model resnet50 --config configs/resnet50.json --splits results/shared_splits.json
    python -m faceord.kfold --final --train-seed 3407 --output results/final_source_seed_3407
"""
import argparse
from .config import Config
from .engine import run_cv, run_final
from .experiment import BASELINES, VARIANTS
from .logger import LogSession


def parser():
    """Command-line options shared by kfold and ablation."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/faceord.json')
    p.add_argument('--model', choices=['faceord', *BASELINES], default='faceord')
    p.add_argument('--variant', choices=list(VARIANTS), default='f_full')
    p.add_argument('--data')
    p.add_argument('--output')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu', 'mps'])
    p.add_argument('--workers', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--k', type=int)
    p.add_argument('--tau', type=float)
    p.add_argument('--landmark-cache', help='Landmark cache built by faceord.landmarks')
    p.add_argument('--splits', help='Shared outer train/val/test split manifest')
    p.add_argument('--tau-selection', help='selected_tau_by_outer_fold.json from faceord.tau_sweep')
    p.add_argument('--hyperparameter-selection', help='selected_settings.json from loss_sweep/weighting_tune')
    p.add_argument('--train-seed', type=int, help='Seed of a final source-domain run')
    p.add_argument('--no-clean', action='store_true', help='Disable DataCleaner')
    p.add_argument('--final', action='store_true',
                   help='Train one source-domain checkpoint on all UTKFace images for cross-dataset transfer')
    return p


def config_from_args(args):
    """Load the JSON configuration and apply command-line overrides."""
    cfg = Config.from_json(args.config)
    for argument, field in [('data', 'DATA_ROOT'), ('output', 'SAVE_DIR'), ('device', 'DEVICE'),
                            ('workers', 'NUM_WORKERS'), ('epochs', 'EPOCHS'), ('k', 'KFOLD_K'),
                            ('tau', 'LABEL_DIST_TAU'), ('train_seed', 'TRAIN_SEED'),
                            ('landmark_cache', 'LANDMARK_CACHE')]:
        value = getattr(args, argument, None)
        if value is not None:
            setattr(cfg, field, value)
    if args.no_clean:
        cfg.CLEAN = False
    cfg.validate()
    return cfg


def main():
    args = parser().parse_args()
    cfg = config_from_args(args)
    with LogSession(f"{'final' if args.final else 'kfold'}_{args.model}_{args.variant}"):
        if args.final:
            run_final(cfg, args.model, args.variant)
        else:
            run_cv(cfg, args.model, args.variant, args.splits, tau_selection=args.tau_selection,
                   hyperparameter_selection=args.hyperparameter_selection)


if __name__ == '__main__':
    main()
