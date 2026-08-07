import pytest
import torch
from neurodxfm.preprocessing import (
    bias_correct,
    center_of_mass,
    finite_volume,
    foreground_mask,
    gaussian_kernel,
    identity_transform,
    percentile_clip,
    quality_report,
    reject_counterfactual,
    resample,
    structural_similarity,
    validate_spacing,
    zscore_foreground,
)


def test_finite_volume() -> None:
    volume = torch.tensor([float("nan"), float("inf"), 1.0])
    assert torch.equal(finite_volume(volume), torch.tensor([0.0, 0.0, 1.0]))


def test_percentile_clip() -> None:
    volume = torch.arange(100).float()
    result = percentile_clip(volume, 10.0, 90.0)
    assert result.min() > 0
    assert result.max() < 99


def test_percentile_clip_rejects_nonfinite() -> None:
    with pytest.raises(ValueError):
        percentile_clip(torch.full((4,), float("nan")))


def test_foreground_mask() -> None:
    volume = torch.zeros(8, 8, 8)
    volume[2:6, 2:6, 2:6] = 1.0
    mask = foreground_mask(volume)
    assert mask.sum() == 64


def test_zscore_foreground() -> None:
    volume = torch.zeros(8, 8, 8)
    volume[2:6, 2:6, 2:6] = torch.arange(64).reshape(4, 4, 4).float() + 1.0
    result = zscore_foreground(volume)
    selected = result[result != 0]
    assert abs(float(selected.mean())) < 0.1


def test_bias_correct_constant() -> None:
    volume = torch.ones(8, 8, 8)
    mask = torch.ones_like(volume, dtype=torch.bool)
    result = bias_correct(volume, mask)
    assert torch.allclose(result, volume, atol=1e-3)


def test_center_of_mass() -> None:
    mask = torch.zeros(5, 5, 5, dtype=torch.bool)
    mask[2, 3, 4] = True
    assert torch.equal(center_of_mass(mask), torch.tensor([2.0, 3.0, 4.0]))


def test_identity_resampling() -> None:
    volume = torch.rand(8, 8, 8)
    transform = identity_transform((8, 8, 8))
    result = resample(volume, transform).squeeze(0)
    assert torch.allclose(result, volume, atol=1e-5)


def test_gaussian_kernel_normalized() -> None:
    assert gaussian_kernel(1.5).sum().item() == pytest.approx(1.0)


def test_structural_similarity_identity() -> None:
    volume = torch.rand(1, 8, 8, 8)
    assert structural_similarity(volume, volume).item() == pytest.approx(1.0, abs=1e-5)


def test_counterfactual_rejection() -> None:
    first = torch.zeros(1, 8, 8, 8)
    second = torch.ones_like(first)
    assert reject_counterfactual(first, second)


def test_quality_report_fields() -> None:
    volume = torch.zeros(16, 16, 16)
    volume[4:12, 4:12, 4:12] = torch.rand(8, 8, 8) + 5.0
    report = quality_report(volume)
    assert report.finite_fraction == 1.0
    assert 0.0 < report.foreground_fraction < 1.0


def test_validate_spacing() -> None:
    assert validate_spacing((1.0, 1.0, 1.0))
    assert not validate_spacing((1.0, 1.0, 1.2))
