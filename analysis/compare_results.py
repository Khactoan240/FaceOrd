"""Paired comparison of FaceOrd against other methods on the same outer folds.

    python -m analysis.compare_results --reference results/faceord/metrics.json \
        --others results/resnet50/metrics.json results/facexformer/metrics.json

For age and gender AUC: paired t-test over folds, mean difference with a 95%
Student-t confidence interval, Cohen's d_z, and Holm-adjusted p-values computed
separately within the age and gender families.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel, t as student_t


def holm_adjust(p_values):
    """Holm step-down adjusted p-values (same order as the input)."""
    values = np.asarray(p_values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError('Holm correction requires finite p-values in [0, 1]')
    order = np.argsort(values, kind='stable')
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted.tolist()


def paired_statistics(reference, other, task):
    """Paired statistics of the per-fold AUC of `task` for two metrics.json results on identical splits."""
    if reference.get('evaluation') != 'outer_test' or other.get('evaluation') != 'outer_test':
        raise ValueError('Statistical comparison requires verified outer-test results')
    if not reference.get('split', {}).get('split_sha256') or reference['split'] != other.get('split'):
        raise ValueError('Paired tests require identical dataset, fold seed and split settings')
    ref = {f['fold']: f[task]['auc'] for f in reference['folds']}
    alt = {f['fold']: f[task]['auc'] for f in other['folds']}
    if set(ref) != set(alt) or len(ref) < 2 or len(ref) != reference['split']['k']:
        raise ValueError('Both models must cover the same complete outer fold set')
    keys = sorted(ref)
    if any(ref[k] is None or alt[k] is None for k in keys):
        raise ValueError('AUC is undefined for at least one fold')
    a, b = np.array([ref[k] for k in keys]), np.array([alt[k] for k in keys])
    diff = a - b
    sd = float(np.std(diff, ddof=1))
    if sd == 0:
        # A finite estimate is undefined for constant nonzero paired differences.
        t, p, dz = (0.0, 1.0, 0.0) if np.all(diff == 0) else (None, None, None)
    else:
        test = ttest_rel(a, b)
        t, p, dz = float(test.statistic), float(test.pvalue), float(diff.mean() / sd)
    margin = float(student_t.ppf(.975, len(keys) - 1) * sd / np.sqrt(len(keys)))
    return {'task': task, 'n_folds': len(keys), 'delta_auc_pp': float(diff.mean() * 100),
            'paired_sd_pp': sd * 100, 't': t, 'p': p, 'cohens_dz': dz,
            'ci95_delta_auc_pp': [(float(diff.mean()) - margin) * 100, (float(diff.mean()) + margin) * 100],
            'raw_significance': 'undefined' if p is None else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--others', required=True, nargs='+', type=Path)
    p.add_argument('--output', type=Path, default=Path('results/statistics.json'))
    args = p.parse_args()
    ref = json.loads(args.reference.read_text())
    rows = []
    for path in args.others:
        other = json.loads(path.read_text())
        for task in ['age', 'gender']:
            rows.append({'comparison': path.parent.name, **paired_statistics(ref, other, task)})
    for task in ['age', 'gender']:
        family = [row for row in rows if row['task'] == task]
        # Undefined tests conservatively count in the family as p=1.
        adjusted = holm_adjust([row['p'] if row['p'] is not None else 1.0 for row in family])
        for row, value in zip(family, adjusted):
            row['holm_p'] = value if row['p'] is not None else None
            row['significance'] = 'undefined' if row['p'] is None else '**' if value < .01 else '*' if value < .05 else 'ns'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'tests': rows, 'multiple_comparison_adjustment': 'Holm separately within age and gender',
                                      'ci': 'paired mean difference, Student t, 95%, unadjusted'}, indent=2, allow_nan=False))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
