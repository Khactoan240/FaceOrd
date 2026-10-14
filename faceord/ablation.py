"""Cumulative ablation (a)-(f) on the shared outer folds.

    python -m faceord.ablation --splits results/shared_splits.json --output results/ablation

All variants use ConvNeXt-S, 30 epochs and the same DataCleaner, class weighting
and seeds, so only the added component changes between consecutive variants.
"""
from dataclasses import replace
from pathlib import Path
from .engine import run_cv, write_json
from .experiment import VARIANTS
from .kfold import parser, config_from_args
from .logger import LogSession


def main():
    p = parser()
    p.description = __doc__
    p.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    args = p.parse_args()
    cfg = config_from_args(args)
    root = Path(args.output or 'results/ablation')
    splits = args.splits or root / 'splits.json'
    results = {}
    with LogSession('ablation'):
        for variant in args.variants:
            variant_cfg = replace(cfg, SAVE_DIR=str(root / variant))
            results[variant] = run_cv(variant_cfg, variant=variant, split_path=splits)
        write_json(root / 'ablation_summary.json', results)


if __name__ == '__main__':
    main()
