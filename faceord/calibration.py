"""Temperature scaling of the age head, fitted on the validation subset.

The age logits are divided by a temperature T: softmax(z / T) for a flat head,
and sigmoid(a_k / T) followed by the ordinal mass reconstruction for the ordinal
head. T minimises the age negative log-likelihood on the validation subset of an
outer fold and is used only for the calibrated ECE (`ece_calibrated`); all other
metrics use the uncalibrated probabilities.
"""
import torch
import torch.nn.functional as F

from .metrics import expected_calibration_error
from .models import OrdinalAgeHead


def apply_temperature(age_logits, temperature, ordinal):
    """Age-group probabilities (torch, (N, K)) after dividing the logits by `temperature`."""
    logits = torch.as_tensor(age_logits, dtype=torch.float32) / temperature
    if ordinal:
        return OrdinalAgeHead.masses_from_cumulative(torch.sigmoid(logits))
    return F.softmax(logits, dim=1)


def fit_temperature(age_true, age_logits, ordinal):
    """Temperature in [0.05, 10] minimising the age NLL (LBFGS on log T)."""
    labels = torch.as_tensor(age_true, dtype=torch.long)
    rows = torch.arange(len(labels))
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        probs = apply_temperature(age_logits, log_t.exp(), ordinal)
        nll = -torch.log(probs[rows, labels]).mean()
        nll.backward()
        return nll

    with torch.enable_grad():
        optimizer.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 10.0))


def calibrated_ece(age_true, age_logits, temperature, ordinal):
    """10-bin ECE of the temperature-scaled age probabilities."""
    with torch.no_grad():
        probs = apply_temperature(age_logits, temperature, ordinal).numpy()
    return expected_calibration_error(age_true, probs)
