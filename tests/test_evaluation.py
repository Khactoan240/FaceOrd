"""Tests of metrics, splits, prediction files, statistics, label mapping and landmark geometry."""
import json
import numpy as np
import pandas as pd
import pytest
from faceord.config import Config
from faceord.metrics import compute_metrics, expected_calibration_error
from faceord.splits import create_folds, validate_split, load_or_create_folds
from faceord.eval_from_predictions import read_predictions, verify_manifest
from analysis.compare_results import paired_statistics, holm_adjust
from faceord.metrics import cumulative_diagnostics
from faceord.age_mapping import adience_age_group, exact_age_group, fairface_age_group
from faceord.landmarks import FLIP_INDEX, flip_landmarks, transform_landmarks


def test_perfect_probabilities_and_confidence_one():
    age = np.tile(np.arange(8), 2)
    p = np.eye(8)[age]
    result = compute_metrics(age, p)
    assert result['auc'] == 1 and result['f1'] == 1 and result['brier'] == 0
    assert result['ece'] == 0
    assert expected_calibration_error(np.array([1, 0]), np.array([0.0, 1.0])) == 1


def test_missing_age_groups_are_reported():
    result = compute_metrics(np.array([3, 3, 4, 4]), np.eye(8)[[3, 3, 4, 4]])
    assert result['auc_classes'] == ['3', '4']
    assert result['auc'] == 1
    assert compute_metrics([1, 1], [0.7, 0.8], binary=True)['auc'] is None


def test_invalid_probability_mass_is_rejected():
    with pytest.raises(ValueError, match='sum to one'):
        compute_metrics([0], [[0.8] * 8])
    with pytest.raises(ValueError, match='finite'):
        compute_metrics([0], [[float('nan')] * 8])


def test_outer_splits_are_disjoint_and_cover_once():
    labels = np.tile(np.arange(8), 100)
    folds = create_folds(labels)
    assert sorted(i for fold in folds for i in fold['test']) == list(range(800))
    assert folds == create_folds(labels)
    for fold in folds:
        validate_split(fold['train'], fold['val'], fold['test'], 800)
        for subset in ['train', 'val', 'test']:
            assert set(labels[fold[subset]]) == set(range(8))
    with pytest.raises(ValueError, match='disjoint'):
        validate_split([0, 1], [1], [2], 3)


def test_split_manifest_rejects_changed_dataset(tmp_path):
    cfg = Config()
    samples = [(str(tmp_path / f'{i}.jpg'), i % 8, i % 2) for i in range(800)]
    path = tmp_path / 'splits.json'
    folds, _ = load_or_create_folds(path, samples, tmp_path, cfg)
    assert folds == load_or_create_folds(path, samples, tmp_path, cfg)[0]
    samples[0] = (str(tmp_path / 'different.jpg'), 0, 0)
    with pytest.raises(ValueError, match='differs'):
        load_or_create_folds(path, samples, tmp_path, cfg)


def small_oof():
    labels = np.arange(8)
    data = {'fold': np.ones(8, int), 'sample_idx': labels, 'age_true': labels,
            'age_pred': labels, 'gender_true': labels % 2, 'gender_pred': labels % 2,
            'gender_prob': labels % 2}
    data.update({f'age_prob_{k}': np.eye(8)[:, k] for k in range(8)})
    return pd.DataFrame(data)


def test_oof_rejects_edited_probabilities_and_duplicate_indices(tmp_path):
    path = tmp_path / 'predictions.csv'
    df = small_oof()
    df.to_csv(path, index=False)
    assert len(read_predictions(path)) == 8
    df.loc[0, ['age_prob_0', 'age_prob_1']] = [0, 1]
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='argmax'):
        read_predictions(path)
    df = small_oof()
    df.loc[1, 'sample_idx'] = 0
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='unique'):
        read_predictions(path)


def test_oof_fold_membership(tmp_path):
    df = small_oof()
    df.loc[4:, 'fold'] = 2
    metadata = {'n_samples': 8, 'dataset_sha256': 'fixture', 'k': 2, 'seed': 42,
                'val_fraction': 0.1, 'folds': [
                    {'fold': 1, 'train': [4, 5, 6], 'val': [7], 'test': [0, 1, 2, 3]},
                    {'fold': 2, 'train': [0, 1, 2], 'val': [3], 'test': [4, 5, 6, 7]}]}
    path = tmp_path / 'splits.json'
    path.write_text(json.dumps(metadata))
    assert verify_manifest(df, path)['dataset_sha256'] == 'fixture'
    metadata['folds'][0]['train'] = [0, 4, 5, 6]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='disjoint'):
        verify_manifest(df, path)


def test_paired_statistics_require_verified_same_folds():
    split = {'dataset_sha256': 'fixture', 'split_sha256': 'same', 'k': 5, 'seed': 42, 'val_fraction': 0.1}
    ref = {'evaluation': 'outer_test', 'split': split,
           'folds': [{'fold': i + 1, 'age': {'auc': v}} for i, v in enumerate([.90, .91, .92, .91, .90])]}
    other = {'evaluation': 'outer_test', 'split': dict(split),
             'folds': [{'fold': i + 1, 'age': {'auc': v}} for i, v in enumerate([.85, .84, .86, .87, .85])]}
    result = paired_statistics(ref, other, 'age')
    assert result['delta_auc_pp'] > 0 and result['p'] < .05
    lo, hi = result['ci95_delta_auc_pp']
    assert lo < result['delta_auc_pp'] < hi
    other['split']['seed'] = 99
    with pytest.raises(ValueError, match='identical'):
        paired_statistics(ref, other, 'age')


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / 'bad_config.json'
    path.write_text('{"UNKNOWN_KEY": 0.18}')
    with pytest.raises(ValueError, match='Unknown'):
        Config.from_json(path)


def test_holm_order_and_paired_interval():
    assert np.allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])


def test_adience_native_mapping_and_contiguous_exact_ages():
    categories = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    assert [adience_age_group(value) for value in categories] == list(range(8))
    assert [exact_age_group(value) for value in [0, 3, 7, 14, 22, 34, 45, 54]] == list(range(8))
    with pytest.raises(ValueError):
        adience_age_group('(3, 6)')


def test_violation_diagnostics_use_raw_cumulative_pairs():
    c = np.array([[.9, .8, .7, .6, .5, .4, .3], [.9, .8, .9, .6, .5, .4, .3]])
    report = cumulative_diagnostics(c)
    assert report['pairwise_violation_rate'] == 1 / 12
    assert report['sample_violation_rate'] == .5
    assert np.isclose(report['mean_violation_magnitude_all_pairs'], .1 / 12)
    assert np.isclose(report['mean_violation_magnitude_violated_pairs'], .1)


def test_landmark_flip_mirrors_points_and_swaps_sides():
    pts = np.random.default_rng(0).uniform(0, 223, (68, 2)).astype(np.float32)
    assert sorted(FLIP_INDEX.tolist()) == list(range(68))
    assert np.array_equal(FLIP_INDEX[FLIP_INDEX], np.arange(68))
    flipped = flip_landmarks(pts, 224)
    assert np.allclose(flip_landmarks(flipped, 224), pts)
    assert np.allclose(flipped[36], [223 - pts[45, 0], pts[45, 1]])   # outer eye corners
    assert np.allclose(flipped[30], [223 - pts[30, 0], pts[30, 1]])   # nose tip stays


def test_affine_landmark_transform():
    pts = np.array([[10., 20.], [30., 40.]], dtype=np.float32)
    shift = np.array([[1., 0., 5.], [0., 1., -2.]])
    assert np.allclose(transform_landmarks(pts, shift), pts + [5, -2])


def test_fairface_ranges_map_to_age_groups():
    assert [fairface_age_group(v) for v in ['0-2', '3-9', '10-19', '20-29', '30-39']] == [0, 1, 3, 4, 5]
    assert fairface_age_group('more than 70') == 7
    with pytest.raises(ValueError):
        fairface_age_group('25-35')
