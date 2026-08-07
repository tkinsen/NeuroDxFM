from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class SpatialTransform:
    matrix: Tensor
    output_shape: tuple[int, int, int]


@dataclass(frozen=True)
class QualityReport:
    finite_fraction: float
    foreground_fraction: float
    signal_mean: float
    signal_std: float
    signal_to_noise: float
    contrast_to_noise: float
    entropy: float
    passed: bool


def finite_volume(volume: Tensor) -> Tensor:
    return torch.nan_to_num(volume.float(), nan=0.0, posinf=0.0, neginf=0.0)


def percentile_clip(volume: Tensor, lower: float = 0.5, upper: float = 99.5) -> Tensor:
    finite = volume[torch.isfinite(volume)]
    if finite.numel() == 0:
        raise ValueError("volume contains no finite values")
    low = torch.quantile(finite, lower / 100.0)
    high = torch.quantile(finite, upper / 100.0)
    return volume.clamp(low, high)


def foreground_mask(volume: Tensor, quantile: float = 0.15) -> Tensor:
    positive = volume[volume > 0]
    if positive.numel() == 0:
        return torch.zeros_like(volume, dtype=torch.bool)
    threshold = torch.quantile(positive, quantile)
    return (volume >= threshold) & (volume > 0)


def zscore_foreground(volume: Tensor, mask: Tensor | None = None) -> Tensor:
    selected = foreground_mask(volume) if mask is None else mask
    values = volume[selected]
    if values.numel() == 0:
        return torch.zeros_like(volume)
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(1e-6)
    result = (volume - mean) / std
    return torch.where(selected, result, torch.zeros_like(result))


def bias_field_basis(shape: tuple[int, int, int], order: int, device: torch.device) -> Tensor:
    axes = [torch.linspace(-1.0, 1.0, length, device=device) for length in shape]
    coordinates = torch.meshgrid(*axes, indexing="ij")
    terms: list[Tensor] = []
    for first in range(order + 1):
        for second in range(order + 1 - first):
            for third in range(order + 1 - first - second):
                terms.append(coordinates[0].pow(first) * coordinates[1].pow(second) * coordinates[2].pow(third))
    return torch.stack(terms)


def estimate_bias_field(volume: Tensor, mask: Tensor, order: int = 3, regularization: float = 1e-3) -> Tensor:
    basis = bias_field_basis(tuple(volume.shape), order, volume.device)
    selected_basis = basis[:, mask].transpose(0, 1)
    selected_values = volume[mask].clamp_min(1e-4).log()
    gram = selected_basis.transpose(0, 1) @ selected_basis
    identity = torch.eye(gram.shape[0], device=volume.device, dtype=volume.dtype)
    coefficients = torch.linalg.solve(gram + regularization * identity, selected_basis.transpose(0, 1) @ selected_values)
    return torch.exp(torch.einsum("c,cdhw->dhw", coefficients, basis))


def bias_correct(volume: Tensor, mask: Tensor | None = None) -> Tensor:
    selected = foreground_mask(volume) if mask is None else mask
    if selected.sum() < 100:
        return volume
    field = estimate_bias_field(volume, selected)
    corrected = volume / field.clamp_min(1e-4)
    scale = volume[selected].median() / corrected[selected].median().clamp_min(1e-4)
    return corrected * scale


def center_of_mass(mask: Tensor) -> Tensor:
    coordinates = torch.nonzero(mask, as_tuple=False).float()
    if len(coordinates) == 0:
        return torch.tensor([(size - 1) / 2.0 for size in mask.shape], device=mask.device)
    return coordinates.mean(dim=0)


def affine_grid(matrix: Tensor, output_shape: tuple[int, int, int], batch: int = 1) -> Tensor:
    theta = matrix[:3].unsqueeze(0).expand(batch, -1, -1)
    return F.affine_grid(theta, (batch, 1, *output_shape), align_corners=True)


def resample(volume: Tensor, transform: SpatialTransform, mode: str = "bilinear") -> Tensor:
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif volume.ndim == 4:
        volume = volume.unsqueeze(0)
    if volume.ndim != 5:
        raise ValueError("volume must have three or four dimensions")
    grid = affine_grid(transform.matrix.to(volume), transform.output_shape, volume.shape[0])
    result = F.grid_sample(volume, grid, mode=mode, padding_mode="zeros", align_corners=True)
    return result.squeeze(0)


def identity_transform(shape: tuple[int, int, int], device: torch.device | None = None) -> SpatialTransform:
    matrix = torch.eye(4, device=device)
    return SpatialTransform(matrix, shape)


def compose_transforms(first: SpatialTransform, second: SpatialTransform) -> SpatialTransform:
    return SpatialTransform(second.matrix @ first.matrix, second.output_shape)


def rigid_matrix(angles: Tensor, translation: Tensor, scale: Tensor | None = None) -> Tensor:
    ax, ay, az = angles
    cx, sx = torch.cos(ax), torch.sin(ax)
    cy, sy = torch.cos(ay), torch.sin(ay)
    cz, sz = torch.cos(az), torch.sin(az)
    one = torch.ones((), device=angles.device)
    zero = torch.zeros((), device=angles.device)
    rx = torch.stack((one, zero, zero, zero, cx, -sx, zero, sx, cx)).reshape(3, 3)
    ry = torch.stack((cy, zero, sy, zero, one, zero, -sy, zero, cy)).reshape(3, 3)
    rz = torch.stack((cz, -sz, zero, sz, cz, zero, zero, zero, one)).reshape(3, 3)
    rotation = rz @ ry @ rx
    if scale is not None:
        rotation = rotation @ torch.diag(scale)
    matrix = torch.eye(4, device=angles.device)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def random_rigid(
    shape: tuple[int, int, int],
    max_degrees: float,
    scale_range: tuple[float, float],
    translation_range: float,
    generator: torch.Generator,
    device: torch.device,
) -> SpatialTransform:
    angles = (torch.rand(3, generator=generator, device=device) * 2.0 - 1.0) * np.deg2rad(max_degrees)
    translation = (torch.rand(3, generator=generator, device=device) * 2.0 - 1.0) * translation_range
    scale = torch.empty(3, device=device).uniform_(scale_range[0], scale_range[1], generator=generator)
    return SpatialTransform(rigid_matrix(angles, translation, scale), shape)


def gaussian_kernel(sigma: float, truncate: float = 3.0, device: torch.device | None = None) -> Tensor:
    radius = max(int(truncate * sigma + 0.5), 1)
    coordinate = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coordinate / sigma).square())
    return kernel / kernel.sum()


def separable_blur(volume: Tensor, sigma: float) -> Tensor:
    kernel = gaussian_kernel(sigma, device=volume.device).to(volume.dtype)
    radius = len(kernel) // 2
    result = volume.reshape(-1, 1, *volume.shape[-3:])
    kernels = (
        kernel.view(1, 1, -1, 1, 1),
        kernel.view(1, 1, 1, -1, 1),
        kernel.view(1, 1, 1, 1, -1),
    )
    pads = ((radius, 0, 0), (0, radius, 0), (0, 0, radius))
    for axis_kernel, axis_pad in zip(kernels, pads, strict=True):
        result = F.conv3d(result, axis_kernel, padding=axis_pad)
    return result.reshape(volume.shape)


def structural_similarity(first: Tensor, second: Tensor, window_sigma: float = 1.5) -> Tensor:
    first = first.float()
    second = second.float()
    mean_first = separable_blur(first, window_sigma)
    mean_second = separable_blur(second, window_sigma)
    variance_first = separable_blur(first.square(), window_sigma) - mean_first.square()
    variance_second = separable_blur(second.square(), window_sigma) - mean_second.square()
    covariance = separable_blur(first * second, window_sigma) - mean_first * mean_second
    dynamic = torch.maximum(first.max(), second.max()) - torch.minimum(first.min(), second.min())
    c1 = (0.01 * dynamic).square()
    c2 = (0.03 * dynamic).square()
    numerator = (2.0 * mean_first * mean_second + c1) * (2.0 * covariance + c2)
    denominator = (mean_first.square() + mean_second.square() + c1) * (variance_first + variance_second + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean()


def shannon_entropy(volume: Tensor, bins: int = 256) -> float:
    finite = volume[torch.isfinite(volume)]
    histogram = torch.histc(finite, bins=bins, min=float(finite.min()), max=float(finite.max()))
    probability = histogram / histogram.sum().clamp_min(1.0)
    probability = probability[probability > 0]
    return float(-(probability * probability.log2()).sum())


def quality_report(volume: Tensor) -> QualityReport:
    finite = torch.isfinite(volume)
    clean = finite_volume(volume)
    mask = foreground_mask(clean)
    signal = clean[mask]
    background = clean[~mask]
    signal_mean = float(signal.mean()) if signal.numel() else 0.0
    signal_std = float(signal.std(unbiased=False)) if signal.numel() else 0.0
    noise_std = float(background.std(unbiased=False)) if background.numel() else 0.0
    signal_to_noise = signal_mean / max(noise_std, 1e-6)
    contrast = abs(signal_mean - float(background.mean())) if background.numel() else signal_mean
    contrast_to_noise = contrast / max(noise_std, 1e-6)
    finite_fraction = float(finite.float().mean())
    foreground_fraction = float(mask.float().mean())
    entropy = shannon_entropy(clean)
    passed = finite_fraction > 0.999 and 0.05 < foreground_fraction < 0.8 and signal_to_noise > 2.0
    return QualityReport(
        finite_fraction,
        foreground_fraction,
        signal_mean,
        signal_std,
        signal_to_noise,
        contrast_to_noise,
        entropy,
        passed,
    )


def reject_counterfactual(original: Tensor, candidate: Tensor, minimum_ssim: float = 0.85) -> bool:
    return float(structural_similarity(original, candidate)) < minimum_ssim


def batch_quality(volumes: Iterable[Tensor]) -> list[QualityReport]:
    return [quality_report(volume) for volume in volumes]


def save_tensor(volume: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(volume.detach().cpu(), path)


def voxel_spacing_from_affine(affine: NDArray[np.float64]) -> tuple[float, float, float]:
    spacing = np.sqrt(np.sum(np.asarray(affine[:3, :3], dtype=np.float64) ** 2, axis=0))
    return float(spacing[0]), float(spacing[1]), float(spacing[2])


def validate_spacing(spacing: Sequence[float], target: float = 1.0, tolerance: float = 0.05) -> bool:
    return all(abs(float(value) - target) <= tolerance for value in spacing)
