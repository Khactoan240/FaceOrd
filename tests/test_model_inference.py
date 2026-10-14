"""Tests of the ordinal head, temperature scaling, DataCleaner resampling and the ablation variants."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
from faceord.models import OrdinalAgeHead
from faceord.losses import _ordinal_targets
from faceord.data_cleaner import rebalance_indices
from faceord.config import Config
from faceord.experiment import make_model


def test_ordinal_endpoints_and_probability_simplex():
    targets = _ordinal_targets(torch.tensor([0, 7]), 8)
    assert targets[0].sum() == 0 and targets[1].sum() == 7
    head = OrdinalAgeHead(7, 8)
    with torch.no_grad():
        head.classifiers.weight.copy_(torch.eye(7))
        head.classifiers.bias.zero_()
        _, _, probs = head(torch.tensor([[-3., 3., -2., 2., -1., 1., 0.]]))
    assert torch.all(probs > 0)
    assert torch.allclose(probs.sum(1), torch.ones(1))


def test_resampling_preserves_size_and_near_miss_rule():
    indices = list(range(10, 16))
    age = np.array([0, 0, 1, 2, 3, 4])
    gender = np.zeros(6, int)
    ap = np.eye(8)[[0, 0, 2, 3, 7, 7]]
    gp = np.zeros(6)
    result = rebalance_indices(indices, age, gender, ap, gp, fraction=1, seed=42)
    assert len(result) == 6 and set(result) <= set(indices)
    assert result.count(12) == 2 and result.count(13) == 2
    assert result.count(14) == 1 and result.count(15) == 1
    perfect = np.eye(8)[age]
    assert rebalance_indices(indices, age, gender, perfect, gp, fraction=1) == indices
    assert rebalance_indices(indices, age, gender, ap, gp, fraction=0) == indices


def test_tiny_cumulative_variants_have_expected_modules():
    cfg = Config(BACKBONE='scratch', USE_PRETRAINED_BACKBONE=False, D=16,
                 GLOBAL_CHANNELS=[4, 8], REGION_CHANNELS=[4, 8], TRANSFORMER_DIM_FF=32)
    global_only = make_model(cfg, variant='a_global_flat')
    local = make_model(cfg, variant='d_local_regions')
    full = make_model(cfg, variant='f_full')
    assert not hasattr(global_only, 'regional_encoder')
    assert hasattr(local, 'regional_encoder') and not hasattr(local, 'transformer')
    assert not hasattr(local, 'fusion')
    assert hasattr(full, 'transformer') and hasattr(full, 'fusion')
    del global_only, local, full


def test_cleaner_counts_easy_pool_and_allows_small_near_pool():
    indices = list(range(12))
    age = np.zeros(12, int)
    ap = np.eye(8)[[0] * 10 + [1, 7]]
    gender = np.zeros(12, int)
    result, info = rebalance_indices(indices, age, gender, ap, np.zeros(12),
                                      fraction=.2, return_info=True)
    assert info['n_easy'] == 10 and info['n_near_miss'] == 1
    assert info['actual_replacements'] == 2 and result.count(10) == 3
    assert len(result) == 12 and result.count(11) == 1


def test_temperature_scaling_of_ordinal_head():
    from faceord.calibration import apply_temperature, calibrated_ece, fit_temperature
    torch.manual_seed(0)
    logits = torch.randn(64, 7) * 3
    probs = OrdinalAgeHead.masses_from_cumulative(torch.sigmoid(logits))
    assert torch.allclose(apply_temperature(logits.numpy(), 1.0, ordinal=True), probs)
    labels = torch.randint(0, 8, (64,)).numpy()
    temperature = fit_temperature(labels, logits.numpy(), ordinal=True)
    assert 0.05 <= temperature <= 10
    assert 0 <= calibrated_ece(labels, logits.numpy(), temperature, ordinal=True) <= 1
