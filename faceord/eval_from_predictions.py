"""Metrics from saved out-of-fold predictions (CSV or XLSX), without loading a model.

    python -m faceord.eval_from_predictions --predictions results/faceord/oof_predictions.csv \
        --splits results/shared_splits.json --output results/faceord/oof_metrics.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import compute_metrics, validate_probabilities, cumulative_diagnostics

PROB_COLS = [f'age_prob_{i}' for i in range(8)]


def read_predictions(path):
    """Read and validate a prediction file (labels, probabilities and argmax consistency)."""
    path = Path(path)
    df = pd.read_excel(path, sheet_name='predictions') if path.suffix.lower() == '.xlsx' else pd.read_csv(path)
    required = {'fold', 'sample_idx', 'age_true', 'age_pred', 'gender_true', 'gender_pred',
                'gender_prob', *PROB_COLS}
    missing = required - set(df)
    if missing:
        raise ValueError(f'Missing prediction columns: {sorted(missing)}')
    for column in ['fold', 'sample_idx', 'age_true', 'age_pred', 'gender_true', 'gender_pred']:
        values = df[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f'{column} must contain finite integers')
        df[column] = values.astype(int)
    if df.empty or df['sample_idx'].duplicated().any() or (df['sample_idx'] < 0).any():
        raise ValueError('OOF sample indices must be unique and nonnegative')
    if not df['age_true'].between(0, 7).all() or not df['gender_true'].isin([0, 1]).all():
        raise ValueError('Invalid age or gender labels')
    if (df['fold'] < 1).any():
        raise ValueError('Fold IDs must be positive')
    ap = validate_probabilities(df[PROB_COLS].to_numpy())
    gp = validate_probabilities(df['gender_prob'].to_numpy())
    if not np.array_equal(df['age_pred'].to_numpy(), ap.argmax(axis=1)):
        raise ValueError('age_pred differs from argmax(age_prob); probabilities may have been edited')
    if not np.array_equal(df['gender_pred'].to_numpy(), (gp >= 0.5).astype(int)):
        raise ValueError('gender_pred differs from the 0.5 probability threshold')
    return df


def evaluate_dataframe(df):
    """Age/gender metrics of a prediction table, with ordinal diagnostics when age_c_* exist."""
    result = {'age': compute_metrics(df['age_true'], df[PROB_COLS].to_numpy()),
              'gender': compute_metrics(df['gender_true'], df['gender_prob'].to_numpy(), binary=True)}
    columns = [f'age_c_{i}' for i in range(7)]
    if any(col in df for col in columns):
        if not all(col in df for col in columns):
            raise ValueError('All seven raw cumulative columns are required')
        result['cumulative_diagnostics'] = cumulative_diagnostics(df[columns].to_numpy())
    return result


def summarize(folds):
    """Mean and sample standard deviation (ddof=1) across folds."""
    result = {'age': {}, 'gender': {}}
    for task in result:
        for metric in ['accuracy', 'f1', 'auc', 'map', 'brier', 'ece', 'mae_class', 'adj_acc']:
            values = [f[task].get(metric) for f in folds]
            values = [v for v in values if v is not None]
            if values:
                result[task][metric] = {'mean': float(np.mean(values)),
                                      'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    return result


def verify_manifest(df, manifest_path):
    """Check that every fold predicts exactly its outer test indices; returns split metadata."""
    from .splits import validate_split
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['k'] < 2 or len(manifest['folds']) != manifest['k']:
        raise ValueError('Outer evaluation requires a complete CV split manifest')
    for fold in manifest['folds']:
        validate_split(fold['train'], fold['val'], fold['test'], manifest['n_samples'])
    if sorted(df['sample_idx'].tolist()) != list(range(manifest['n_samples'])):
        raise ValueError('OOF file must cover the split-manifest dataset exactly once')
    if set(df['fold']) != {fold['fold'] for fold in manifest['folds']}:
        raise ValueError('OOF fold IDs differ from the split manifest')
    for split in manifest['folds']:
        predicted = set(df.loc[df['fold'] == split['fold'], 'sample_idx'])
        if predicted != set(split['test']):
            raise ValueError(f'Prediction indices differ from outer test fold {split["fold"]}')
    meta = {key: manifest[key] for key in ['dataset_sha256', 'n_samples', 'k', 'seed', 'val_fraction']}
    meta['split_sha256'] = hashlib.sha256(
        json.dumps(manifest['folds'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--splits', type=Path, help='Split manifest used to verify fold membership')
    p.add_argument('--output', type=Path, default=Path('results/evaluation/metrics.json'))
    args = p.parse_args()
    df = read_predictions(args.predictions)
    split_meta = verify_manifest(df, args.splits) if args.splits else None
    folds = [{'fold': int(fold), **evaluate_dataframe(group)} for fold, group in df.groupby('fold')]
    result = {'source_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              'n_samples': len(df), 'evaluation': 'outer_test' if split_meta else 'oof_without_split_manifest',
              'split': split_meta, 'folds': folds, 'summary': summarize(folds),
              'pooled': evaluate_dataframe(df)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))
    print(f'Saved evaluation to {args.output}')


if __name__ == '__main__':
    main()
