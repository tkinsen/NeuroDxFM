from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    effect: float
    interval: Interval | None


def normal_cdf(value: float) -> float:
    return float(0.5 * torch.erfc(torch.tensor(-value / np.sqrt(2.0))).item())


def normal_survival(value: float) -> float:
    return 1.0 - normal_cdf(value)


def percentile_interval(values: Sequence[float], confidence: float = 0.95) -> Interval:
    array = np.asarray(values, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    return Interval(float(array.mean()), float(np.quantile(array, alpha)), float(np.quantile(array, 1.0 - alpha)), confidence)


def stratified_bootstrap_indices(labels: Array, generator: np.random.Generator) -> Array:
    result: list[NDArray[np.int64]] = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        result.append(generator.choice(indices, len(indices), replace=True))
    return np.concatenate(result).astype(np.float64)


def stratified_bootstrap(
    labels: Sequence[float],
    values: Sequence[float],
    metric: Callable[[Array, Array], float],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 3407,
) -> Interval:
    target = np.asarray(labels, dtype=np.float64)
    score = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        indices = stratified_bootstrap_indices(target, generator).astype(np.int64)
        estimates.append(metric(target[indices], score[indices]))
    alpha = (1.0 - confidence) / 2.0
    return Interval(metric(target, score), float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha)), confidence)


def paired_bootstrap_difference(
    labels: Sequence[float],
    first: Sequence[float],
    second: Sequence[float],
    metric: Callable[[Array, Array], float],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 3407,
) -> TestResult:
    target = np.asarray(labels, dtype=np.float64)
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    observed = metric(target, left) - metric(target, right)
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        indices = stratified_bootstrap_indices(target, generator).astype(np.int64)
        estimates.append(metric(target[indices], left[indices]) - metric(target[indices], right[indices]))
    interval = percentile_interval(estimates, confidence)
    p_value = 2.0 * min(np.mean(np.asarray(estimates) <= 0.0), np.mean(np.asarray(estimates) >= 0.0))
    standard_error = np.std(estimates, ddof=1)
    statistic = observed / max(standard_error, 1e-12)
    return TestResult(float(statistic), float(min(p_value, 1.0)), float(observed), interval)


def mcnemar(first_correct: Sequence[bool], second_correct: Sequence[bool], continuity: bool = True) -> TestResult:
    first = np.asarray(first_correct, dtype=bool)
    second = np.asarray(second_correct, dtype=bool)
    b = int(np.sum(first & ~second))
    c = int(np.sum(~first & second))
    correction = 1.0 if continuity else 0.0
    statistic = (max(abs(b - c) - correction, 0.0) ** 2) / max(b + c, 1)
    p_value = float(torch.erfc(torch.sqrt(torch.tensor(statistic / 2.0))).item())
    effect = (b - c) / max(len(first), 1)
    return TestResult(statistic, p_value, effect, None)


def cohen_d(first: Sequence[float], second: Sequence[float], paired: bool = False) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if paired:
        differences = left - right
        return float(differences.mean() / max(differences.std(ddof=1), 1e-12))
    pooled = np.sqrt(((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1)) / max(len(left) + len(right) - 2, 1))
    return float((left.mean() - right.mean()) / max(pooled, 1e-12))


def cliffs_delta(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    greater = np.sum(left[:, None] > right[None, :])
    lower = np.sum(left[:, None] < right[None, :])
    return float((greater - lower) / (len(left) * len(right)))


def permutation_test(
    first: Sequence[float],
    second: Sequence[float],
    statistic: Callable[[Array, Array], float],
    iterations: int = 10000,
    seed: int = 3407,
) -> TestResult:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    observed = statistic(left, right)
    combined = np.concatenate((left, right))
    generator = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        shuffled = generator.permutation(combined)
        value = statistic(shuffled[: len(left)], shuffled[len(left) :])
        exceed += abs(value) >= abs(observed)
    p_value = (exceed + 1) / (iterations + 1)
    return TestResult(observed, p_value, observed, None)


def benjamini_hochberg(values: Sequence[float]) -> list[float]:
    p_values = np.asarray(values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 1.0
    count = len(p_values)
    for reversed_rank, index in enumerate(order[::-1], start=1):
        rank = count - reversed_rank + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = running
    return adjusted.tolist()


def bonferroni(values: Sequence[float]) -> list[float]:
    count = len(values)
    return [min(float(value) * count, 1.0) for value in values]


def holm(values: Sequence[float]) -> list[float]:
    p_values = np.asarray(values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for position, index in enumerate(order):
        running = max(running, (count - position) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Interval:
    if total <= 0:
        raise ValueError("total must be positive")
    alpha = 1.0 - confidence
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else 1.6448536269514722
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return Interval(proportion, centre - margin, centre + margin, confidence)


def fisher_transform_interval(correlation: float, count: int, confidence: float = 0.95) -> Interval:
    clipped = np.clip(correlation, -0.999999, 0.999999)
    transformed = np.arctanh(clipped)
    standard_error = 1.0 / np.sqrt(max(count - 3, 1))
    z = 1.959963984540054
    lower = np.tanh(transformed - z * standard_error)
    upper = np.tanh(transformed + z * standard_error)
    return Interval(correlation, float(lower), float(upper), confidence)


def variance_components(values: Tensor, seed_axis: int = 0, site_axis: int = 1) -> dict[str, float]:
    array = values.detach().double()
    grand = array.mean()
    seed_means = array.mean(dim=site_axis)
    site_means = array.mean(dim=seed_axis)
    seed_variance = float(((seed_means - grand) ** 2).mean())
    site_variance = float(((site_means - grand) ** 2).mean())
    fitted = seed_means.unsqueeze(site_axis) + site_means.unsqueeze(seed_axis) - grand
    residual_variance = float(((array - fitted) ** 2).mean())
    total = seed_variance + site_variance + residual_variance
    return {
        "seed": seed_variance,
        "site": site_variance,
        "residual": residual_variance,
        "seed_fraction": seed_variance / max(total, 1e-12),
        "site_fraction": site_variance / max(total, 1e-12),
        "residual_fraction": residual_variance / max(total, 1e-12),
    }
