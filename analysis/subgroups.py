"""Subgroup and per-class analysis of an out-of-fold prediction file.

    python -m analysis.subgroups --predictions results/faceord/oof_predictions.csv \
        --splits results/shared_splits.json

Metrics by gender, race (UTKFace label) and true age group; per-class precision,
recall, F1 and support; raw and row-normalised confusion matrices.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from faceord.eval_from_predictions import read_predictions, verify_manifest, evaluate_dataframe


def class_report(y, predicted, n_classes):
    """Per-class precision/recall/F1/support and confusion matrices (counts and row-normalised)."""
    labels = list(range(n_classes))
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=labels, zero_division=0)
    counts = confusion_matrix(y, predicted, labels=labels)
    denominators = counts.sum(1, keepdims=True)
    normalized = np.divide(counts, denominators, out=np.zeros_like(counts, dtype=float), where=denominators != 0)
    return {'classes': [{'class': k, 'precision': float(precision[k]), 'recall': float(recall[k]),
                         'f1': float(f1[k]), 'support': int(support[k])} for k in labels],
            'confusion_counts': counts.tolist(), 'confusion_row_normalized': normalized.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--splits', required=True, type=Path)
    p.add_argument('--output', default=Path('results/subgroups.json'), type=Path)
    args = p.parse_args()
    df = read_predictions(args.predictions)
    split = verify_manifest(df, args.splits)
    if 'race_label' not in df:
        if 'filename' not in df:
            raise ValueError('Race subgroup analysis requires race_label or original UTKFace filenames')
        import re
        def race_from_name(value):
            match = re.match(r'^\d+_[01]_([0-4])_', Path(str(value)).name)
            return int(match.group(1)) if match else None
        df['race_label'] = df['filename'].map(race_from_name)
    result = {'source_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(), 'split': split,
              'pooled': evaluate_dataframe(df), 'age_classes': class_report(df.age_true, df.age_pred, 8),
              'gender_classes': class_report(df.gender_true, df.gender_pred, 2), 'subgroups': {}}
    for column in ['gender_true', 'race_label', 'age_true']:
        result['subgroups'][column] = [{'group': str(value), 'n_samples': len(group),
                                       **evaluate_dataframe(group)} for value, group in df.groupby(column, dropna=False)]
    result['interpretation'] = 'AUC is null where a subgroup contains a single class (e.g. age AUC within one true age group).'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
