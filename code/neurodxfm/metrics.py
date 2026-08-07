from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

FloatArray = NDArray[np.float64]


def as_numpy(value: Tensor | Sequence[float]) -> FloatArray:
    if isinstance(value, Tensor):
        return np.asarray(value.detach().cpu().double().numpy(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def binary_confusion(target: FloatArray, score: FloatArray, threshold: float) -> tuple[int, int, int, int]:
    predicted = score >= threshold
    positive = target == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    return tp, fp, tn, fn


def sensitivity(target: Tensor | Sequence[float], score: Tensor | Sequence[float], threshold: float = 0.5) -> float:
    tp, _, _, fn = binary_confusion(as_numpy(target), as_numpy(score), threshold)
    return tp / max(tp + fn, 1)


def specificity(target: Tensor | Sequence[float], score: Tensor | Sequence[float], threshold: float = 0.5) -> float:
    _, fp, tn, _ = binary_confusion(as_numpy(target), as_numpy(score), threshold)
    return tn / max(tn + fp, 1)


def precision(target: Tensor | Sequence[float], score: Tensor | Sequence[float], threshold: float = 0.5) -> float:
    tp, fp, _, _ = binary_confusion(as_numpy(target), as_numpy(score), threshold)
    return tp / max(tp + fp, 1)


def negative_predictive_value(target: Tensor | Sequence[float], score: Tensor | Sequence[float], threshold: float = 0.5) -> float:
    _, _, tn, fn = binary_confusion(as_numpy(target), as_numpy(score), threshold)
    return tn / max(tn + fn, 1)


def accuracy(target: Tensor | Sequence[float], score: Tensor | Sequence[float], threshold: float = 0.5) -> float:
    tp, fp, tn, fn = binary_confusion(as_numpy(target), as_numpy(score), threshold)
    return (tp + tn) / max(tp + fp + tn + fn, 1)


def balanced_accuracy(target: Tensor | Sequence[float], score: Tensor | Sequence[float], threshold: float = 0.5) -> float:
    return 0.5 * (sensitivity(target, score, threshold) + specificity(target, score, threshold))


def f1_score(target: Tensor | Sequence[float], score: Tensor | Sequence[float], threshold: float = 0.5) -> float:
    p = precision(target, score, threshold)
    r = sensitivity(target, score, threshold)
    return 2.0 * p * r / max(p + r, 1e-12)


def roc_curve(target: Tensor | Sequence[float], score: Tensor | Sequence[float]) -> tuple[FloatArray, FloatArray, FloatArray]:
    y = as_numpy(target).astype(np.int64)
    s = as_numpy(score)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    distinct = np.where(np.diff(s))[0]
    indices = np.r_[distinct, y.size - 1]
    true = np.cumsum(y)[indices]
    false = 1 + indices - true
    true = np.r_[0, true]
    false = np.r_[0, false]
    tpr = true / max(true[-1], 1)
    fpr = false / max(false[-1], 1)
    thresholds = np.r_[np.inf, s[indices]]
    return fpr.astype(np.float64), tpr.astype(np.float64), thresholds.astype(np.float64)


def auc(target: Tensor | Sequence[float], score: Tensor | Sequence[float]) -> float:
    fpr, tpr, _ = roc_curve(target, score)
    return float(np.trapezoid(tpr, fpr))


def average_precision(target: Tensor | Sequence[float], score: Tensor | Sequence[float]) -> float:
    y = as_numpy(target).astype(np.int64)
    s = as_numpy(score)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    cumulative = np.cumsum(y)
    ranks = np.arange(1, len(y) + 1)
    return float(np.sum(cumulative / ranks * y) / max(np.sum(y), 1))


def threshold_at_specificity(target: Tensor | Sequence[float], score: Tensor | Sequence[float], target_specificity: float = 0.8) -> float:
    fpr, _, thresholds = roc_curve(target, score)
    valid = np.where(fpr <= 1.0 - target_specificity)[0]
    return float(thresholds[valid[-1]]) if len(valid) else float("inf")


def brier_score(target: Tensor | Sequence[float], probability: Tensor | Sequence[float]) -> float:
    y = as_numpy(target)
    p = as_numpy(probability)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(target: Tensor | Sequence[float], probability: Tensor | Sequence[float], bins: int = 10) -> float:
    y = as_numpy(target)
    p = as_numpy(probability)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (p >= lower) & (p < upper if index < bins - 1 else p <= upper)
        count = int(selected.sum())
        if count:
            result += count / total * abs(float(y[selected].mean()) - float(p[selected].mean()))
    return result


def maximum_calibration_error(target: Tensor | Sequence[float], probability: Tensor | Sequence[float], bins: int = 10) -> float:
    y = as_numpy(target)
    p = as_numpy(probability)
    edges = np.linspace(0.0, 1.0, bins + 1)
    errors: list[float] = []
    for index in range(bins):
        selected = (p >= edges[index]) & (p < edges[index + 1] if index < bins - 1 else p <= edges[index + 1])
        if selected.any():
            errors.append(abs(float(y[selected].mean()) - float(p[selected].mean())))
    return max(errors, default=0.0)


def prevalence_adjusted_ppv(sens: float, spec: float, prevalence: float) -> float:
    numerator = sens * prevalence
    return numerator / max(numerator + (1.0 - spec) * (1.0 - prevalence), 1e-12)


def spearman(x: Tensor | Sequence[float], y: Tensor | Sequence[float]) -> float:
    left = as_numpy(x)
    right = as_numpy(y)
    left_rank = np.argsort(np.argsort(left)).astype(np.float64)
    right_rank = np.argsort(np.argsort(right)).astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def concordance_index(time: Tensor | Sequence[float], risk: Tensor | Sequence[float], event: Tensor | Sequence[float]) -> float:
    t = as_numpy(time)
    r = as_numpy(risk)
    e = as_numpy(event)
    comparable = 0
    concordant = 0.0
    for left in range(len(t)):
        for right in range(left + 1, len(t)):
            if t[left] == t[right]:
                continue
            earlier, later = (left, right) if t[left] < t[right] else (right, left)
            if e[earlier] != 1:
                continue
            comparable += 1
            if r[earlier] > r[later]:
                concordant += 1.0
            elif r[earlier] == r[later]:
                concordant += 0.5
    return concordant / max(comparable, 1)


def intraclass_correlation(matrix: FloatArray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    subjects, raters = values.shape
    grand = values.mean()
    subject_means = values.mean(axis=1)
    residual = values - subject_means[:, None]
    between = raters * np.sum((subject_means - grand) ** 2) / max(subjects - 1, 1)
    within = np.sum(residual**2) / max(subjects * (raters - 1), 1)
    return float((between - within) / max(between + (raters - 1) * within, 1e-12))


def bootstrap_interval(
    target: Tensor | Sequence[float],
    score: Tensor | Sequence[float],
    metric: Callable[[FloatArray, FloatArray], float],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 3407,
) -> tuple[float, float, float]:
    y = as_numpy(target)
    s = as_numpy(score)
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        indices = generator.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        estimates.append(metric(y[indices], s[indices]))
    alpha = (1.0 - confidence) / 2.0
    return metric(y, s), float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def delong_variance(target: FloatArray, score: FloatArray) -> tuple[float, float]:
    positive = score[target == 1]
    negative = score[target == 0]
    comparisons = (positive[:, None] > negative[None, :]).astype(np.float64)
    comparisons += 0.5 * (positive[:, None] == negative[None, :])
    auc_value = float(comparisons.mean())
    positive_component = comparisons.mean(axis=1)
    negative_component = comparisons.mean(axis=0)
    variance = np.var(positive_component, ddof=1) / len(positive) + np.var(negative_component, ddof=1) / len(negative)
    return auc_value, float(variance)


def delong_difference(
    target: Tensor | Sequence[float], first: Tensor | Sequence[float], second: Tensor | Sequence[float]
) -> tuple[float, float]:
    y = as_numpy(target)
    a = as_numpy(first)
    b = as_numpy(second)
    positive = y == 1
    negative = y == 0

    def components(score: FloatArray) -> tuple[FloatArray, FloatArray]:
        matrix = (score[positive, None] > score[None, negative]).astype(np.float64)
        matrix += 0.5 * (score[positive, None] == score[None, negative])
        return matrix.mean(axis=1), matrix.mean(axis=0)

    ap, an = components(a)
    bp, bn = components(b)
    difference = auc(y, a) - auc(y, b)
    variance = np.var(ap - bp, ddof=1) / len(ap) + np.var(an - bn, ddof=1) / len(an)
    z = abs(difference) / max(np.sqrt(variance), 1e-12)
    p = float(torch.erfc(torch.tensor(z / np.sqrt(2.0))).item())
    return difference, p


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


@dataclass(frozen=True)
class BinaryReport:
    auc: float
    average_precision: float
    sensitivity: float
    specificity: float
    precision: float
    npv: float
    f1: float
    balanced_accuracy: float
    brier: float
    ece: float
    threshold: float


def binary_report(target: Tensor | Sequence[float], score: Tensor | Sequence[float], operating_specificity: float = 0.8) -> BinaryReport:
    threshold = threshold_at_specificity(target, score, operating_specificity)
    return BinaryReport(
        auc(target, score),
        average_precision(target, score),
        sensitivity(target, score, threshold),
        specificity(target, score, threshold),
        precision(target, score, threshold),
        negative_predictive_value(target, score, threshold),
        f1_score(target, score, threshold),
        balanced_accuracy(target, score, threshold),
        brier_score(target, score),
        expected_calibration_error(target, score),
        threshold,
    )
