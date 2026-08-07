import pytest
import torch
from neurodxfm.downstream import (
    DirichletPrediction,
    EarlyStopping,
    LinearProbe,
    NonlinearProbe,
    TemperatureScaler,
    estimated_months_to_onset,
    evidential_decision,
    retention_ratio,
    t1_only_input,
)


def test_linear_probe_shape() -> None:
    probe = LinearProbe(32, 2, 0.1)
    assert probe(torch.randn(4, 32)).shape == (4, 2)


def test_nonlinear_probe_shape() -> None:
    probe = NonlinearProbe(32, 3, 0.1)
    assert probe(torch.randn(4, 32)).shape == (4, 3)


def test_temperature_scaler_identity() -> None:
    scaler = TemperatureScaler()
    logits = torch.randn(4, 2)
    assert torch.equal(scaler(logits), logits)


def test_dirichlet_probability() -> None:
    prediction = DirichletPrediction(torch.tensor([[2.0, 2.0]]))
    assert torch.equal(prediction.probability, torch.tensor([[0.5, 0.5]]))


def test_dirichlet_uncertainty_decreases_with_evidence() -> None:
    low = DirichletPrediction(torch.tensor([[2.0, 2.0]])).uncertainty
    high = DirichletPrediction(torch.tensor([[20.0, 20.0]])).uncertainty
    assert high < low


def test_early_stopping() -> None:
    stopping = EarlyStopping(2)
    assert not stopping.update(0.5)
    assert not stopping.update(0.4)
    assert stopping.update(0.3)


def test_early_stopping_resets() -> None:
    stopping = EarlyStopping(2)
    stopping.update(0.5)
    stopping.update(0.4)
    assert not stopping.update(0.6)


def test_evidential_decision_abstains() -> None:
    evidence = torch.tensor([[1.0, 1.0], [20.0, 2.0]])
    labels, _ = evidential_decision(evidence, 0.5)
    assert labels[0] == -1
    assert labels[1] == 0


def test_t1_only_input() -> None:
    volumes = torch.ones(2, 5, 3, 3, 3)
    result = t1_only_input(volumes)
    assert result[:, 0].sum() > 0
    assert result[:, 1:].sum() == 0


def test_retention_ratio() -> None:
    assert retention_ratio(0.941, 0.957) == pytest.approx(0.983281, rel=1e-4)


def test_retention_ratio_rejects_zero() -> None:
    with pytest.raises(ValueError):
        retention_ratio(0.9, 0.0)


def test_estimated_months_monotonic() -> None:
    risk = torch.tensor([0.1, 0.5, 0.9])
    months = estimated_months_to_onset(risk, -10.0, 0.0)
    assert months[0] > months[1] > months[2]
