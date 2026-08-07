from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .metrics import (
    BinaryReport,
    auc,
    binary_report,
    bootstrap_interval,
    holm_bonferroni,
    prevalence_adjusted_ppv,
    sensitivity,
    specificity,
)


@dataclass(frozen=True)
class CohortPrediction:
    cohort: str
    subject_ids: tuple[str, ...]
    target: Tensor
    score: Tensor
    site: tuple[str, ...]
    sex: tuple[str, ...]
    race: tuple[str, ...]
    age: Tensor
    vendor: tuple[str, ...]
    field_strength: tuple[str, ...]

    def validate(self) -> None:
        length = len(self.subject_ids)
        values = (
            len(self.target),
            len(self.score),
            len(self.site),
            len(self.sex),
            len(self.race),
            len(self.age),
            len(self.vendor),
            len(self.field_strength),
        )
        if any(value != length for value in values):
            raise ValueError("prediction fields have unequal lengths")
        if len(set(self.subject_ids)) != length:
            raise ValueError("subject identifiers are not unique")


@dataclass(frozen=True)
class CohortSummary:
    cohort: str
    count: int
    report: BinaryReport
    auc_lower: float
    auc_upper: float
    ppv_1_percent: float
    ppv_5_percent: float
    ppv_10_percent: float
    ppv_20_percent: float


@dataclass(frozen=True)
class SubgroupSummary:
    dimension: str
    group: str
    count: int
    auc: float
    sensitivity: float
    specificity: float


@dataclass(frozen=True)
class SiteConcordance:
    summaries: tuple[CohortSummary, ...]
    maximum_auc_drop: float
    minimum_auc: float
    maximum_auc: float
    mcid_pass: bool


def summarize_cohort(prediction: CohortPrediction, iterations: int = 2000) -> CohortSummary:
    prediction.validate()
    report = binary_report(prediction.target, prediction.score, 0.8)
    _, lower, upper = bootstrap_interval(
        prediction.target,
        prediction.score,
        lambda target, score: auc(target, score),
        iterations,
    )
    sens = report.sensitivity
    spec = report.specificity
    return CohortSummary(
        prediction.cohort,
        len(prediction.target),
        report,
        lower,
        upper,
        prevalence_adjusted_ppv(sens, spec, 0.01),
        prevalence_adjusted_ppv(sens, spec, 0.05),
        prevalence_adjusted_ppv(sens, spec, 0.10),
        prevalence_adjusted_ppv(sens, spec, 0.20),
    )


def cross_site_concordance(predictions: Sequence[CohortPrediction], iterations: int = 2000) -> SiteConcordance:
    summaries = tuple(summarize_cohort(prediction, iterations) for prediction in predictions)
    values = [summary.report.auc for summary in summaries]
    maximum = max(values)
    minimum = min(values)
    drop = maximum - minimum
    return SiteConcordance(summaries, drop, minimum, maximum, drop <= 0.05)


def _group_indices(values: Sequence[str]) -> Mapping[str, list[int]]:
    result: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        result.setdefault(value, []).append(index)
    return result


def subgroup_reports(prediction: CohortPrediction, minimum_size: int = 20) -> list[SubgroupSummary]:
    dimensions = {
        "sex": prediction.sex,
        "race": prediction.race,
        "site": prediction.site,
        "vendor": prediction.vendor,
        "field_strength": prediction.field_strength,
    }
    reports: list[SubgroupSummary] = []
    for dimension, values in dimensions.items():
        for group, indices in _group_indices(values).items():
            if len(indices) < minimum_size:
                continue
            target = prediction.target[indices]
            score = prediction.score[indices]
            if target.unique().numel() < 2:
                continue
            reports.append(
                SubgroupSummary(
                    dimension,
                    group,
                    len(indices),
                    auc(target, score),
                    sensitivity(target, score),
                    specificity(target, score),
                )
            )
    age = prediction.age
    boundaries = ((0.0, 60.0), (60.0, 70.0), (70.0, 80.0), (80.0, float("inf")))
    for lower, upper in boundaries:
        selected = torch.where((age >= lower) & (age < upper))[0]
        if len(selected) < minimum_size or prediction.target[selected].unique().numel() < 2:
            continue
        target = prediction.target[selected]
        score = prediction.score[selected]
        reports.append(
            SubgroupSummary(
                "age",
                f"{lower:g}-{upper:g}",
                len(selected),
                auc(target, score),
                sensitivity(target, score),
                specificity(target, score),
            )
        )
    return reports


def permutation_gap_test(
    target: Tensor,
    score: Tensor,
    groups: Sequence[str],
    first: str,
    second: str,
    iterations: int = 10000,
    seed: int = 3407,
) -> tuple[float, float]:
    group_array = np.asarray(groups)
    selected = np.where((group_array == first) | (group_array == second))[0]
    first_mask = group_array[selected] == first
    y = target[selected]
    s = score[selected]
    observed = auc(y[first_mask], s[first_mask]) - auc(y[~first_mask], s[~first_mask])
    generator = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        permuted = generator.permutation(first_mask)
        difference = auc(y[permuted], s[permuted]) - auc(y[~permuted], s[~permuted])
        exceed += abs(difference) >= abs(observed)
    return observed, (exceed + 1) / (iterations + 1)


def mcid_crossing(values: Sequence[float], baselines: Sequence[float], threshold: float, direction: str = "higher") -> float:
    if len(values) != len(baselines):
        raise ValueError("value and baseline arrays must match")
    differences = np.asarray(values) - np.asarray(baselines)
    crossed = differences >= threshold if direction == "higher" else differences <= -threshold
    return float(crossed.mean())


def site_classifier_accuracy(logits: Tensor, site: Tensor) -> float:
    return float((logits.argmax(dim=-1) == site).float().mean())


def false_positive_rate(target: Tensor, score: Tensor, threshold: float = 0.5) -> float:
    return 1.0 - specificity(target, score, threshold)


def age_propensity_weights(age: Tensor, diagnosis: Tensor, bins: int = 20) -> Tensor:
    boundaries = torch.linspace(float(age.min()), float(age.max()) + 1e-6, bins + 1, device=age.device)
    assignments = torch.bucketize(age, boundaries[1:-1])
    weights = torch.ones_like(age, dtype=torch.float32)
    for index in range(bins):
        selected = assignments == index
        if not selected.any():
            continue
        prevalence = diagnosis[selected].float().mean().clamp(0.05, 0.95)
        weights[selected & (diagnosis == 1)] = 0.5 / prevalence
        weights[selected & (diagnosis == 0)] = 0.5 / (1.0 - prevalence)
    return weights / weights.mean()


def export_summary(path: Path, concordance: SiteConcordance, subgroups: Sequence[SubgroupSummary]) -> None:
    import json

    payload = {
        "concordance": asdict(concordance),
        "subgroups": [asdict(item) for item in subgroups],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def corrected_p_values(tests: Mapping[str, float]) -> dict[str, float]:
    names = list(tests)
    adjusted = holm_bonferroni([tests[name] for name in names])
    return dict(zip(names, adjusted, strict=True))


def evaluate_repeated_seeds(seed_scores: Mapping[int, Tensor], target: Tensor) -> dict[str, float]:
    values = np.asarray([auc(target, score) for score in seed_scores.values()])
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "seeds": float(len(values)),
    }
