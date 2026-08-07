import numpy as np
import torch
from neurodxfm.metrics import (
    accuracy,
    auc,
    average_precision,
    balanced_accuracy,
    binary_report,
    brier_score,
    concordance_index,
    delong_difference,
    expected_calibration_error,
    f1_score,
    holm_bonferroni,
    intraclass_correlation,
    negative_predictive_value,
    precision,
    prevalence_adjusted_ppv,
    sensitivity,
    spearman,
    specificity,
    threshold_at_specificity,
)

TARGET = torch.tensor([0, 0, 1, 1])
PERFECT = torch.tensor([0.0, 0.1, 0.9, 1.0])


def test_auc_perfect() -> None:
    assert auc(TARGET, PERFECT) == 1.0


def test_auc_reversed() -> None:
    assert auc(TARGET, 1.0 - PERFECT) == 0.0


def test_average_precision_perfect() -> None:
    assert average_precision(TARGET, PERFECT) == 1.0


def test_sensitivity() -> None:
    assert sensitivity(TARGET, PERFECT) == 1.0


def test_specificity() -> None:
    assert specificity(TARGET, PERFECT) == 1.0


def test_precision() -> None:
    assert precision(TARGET, PERFECT) == 1.0


def test_negative_predictive_value() -> None:
    assert negative_predictive_value(TARGET, PERFECT) == 1.0


def test_accuracy() -> None:
    assert accuracy(TARGET, PERFECT) == 1.0


def test_balanced_accuracy() -> None:
    assert balanced_accuracy(TARGET, PERFECT) == 1.0


def test_f1() -> None:
    assert f1_score(TARGET, PERFECT) == 1.0


def test_brier() -> None:
    assert brier_score(TARGET, TARGET.float()) == 0.0


def test_calibration_error() -> None:
    assert expected_calibration_error(TARGET, TARGET.float()) == 0.0


def test_threshold_at_specificity() -> None:
    threshold = threshold_at_specificity(TARGET, PERFECT, 1.0)
    assert threshold > 0.1


def test_prevalence_adjusted_ppv() -> None:
    assert prevalence_adjusted_ppv(1.0, 1.0, 0.05) == 1.0


def test_spearman() -> None:
    values = torch.arange(8).float()
    assert spearman(values, values) == 1.0


def test_concordance_index() -> None:
    time = torch.tensor([1.0, 2.0, 3.0])
    risk = torch.tensor([3.0, 2.0, 1.0])
    event = torch.ones(3)
    assert concordance_index(time, risk, event) == 1.0


def test_intraclass_correlation() -> None:
    values = np.tile(np.arange(10)[:, None], (1, 3)).astype(np.float64)
    assert intraclass_correlation(values) == 1.0


def test_delong_identical() -> None:
    difference, p_value = delong_difference(TARGET, PERFECT, PERFECT)
    assert difference == 0.0
    assert p_value == 1.0


def test_holm_bonferroni() -> None:
    adjusted = holm_bonferroni([0.01, 0.02, 0.5])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_binary_report() -> None:
    report = binary_report(TARGET, PERFECT)
    assert report.auc == 1.0
    assert report.sensitivity == 1.0
    assert report.specificity == 1.0
