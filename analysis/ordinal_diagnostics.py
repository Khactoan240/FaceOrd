"""Monotonicity violations of the raw cumulative age outputs in an out-of-fold prediction file.

    python -m analysis.ordinal_diagnostics --predictions results/faceord/oof_predictions.csv \
        --splits results/shared_splits.json

Requires the raw columns age_c_0..age_c_6 (exported by faceord.kfold for ordinal models).
"""
import argparse
import hashlib
import json
from pathlib import Path
from faceord.eval_from_predictions import read_predictions, verify_manifest
from faceord.metrics import cumulative_diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--splits', required=True, type=Path)
    p.add_argument('--output', default=Path('results/ordinal_diagnostics.json'), type=Path)
    p.add_argument('--tolerance', default=0.0, type=float)
    args = p.parse_args()
    df = read_predictions(args.predictions)
    split = verify_manifest(df, args.splits)
    columns = [f'age_c_{k}' for k in range(7)]
    if not set(columns) <= set(df):
        raise ValueError('Columns age_c_0..age_c_6 are required; they cannot be recovered from age_prob_*')
    result = {'source_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(), 'split': split,
              'pooled': cumulative_diagnostics(df[columns].to_numpy(), args.tolerance),
              'folds': [{'fold': int(fold), **cumulative_diagnostics(group[columns].to_numpy(), args.tolerance)}
                        for fold, group in df.groupby('fold')]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
