import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExpectedResult:
    identifier: str
    task: str
    cohort: str
    metric: str
    expected: float
    tolerance: float
    seeds: int
    sample_size: int

    def accepts(self, observed: float) -> bool:
        return abs(observed - self.expected) <= self.tolerance


EXPECTED_RESULTS = (
    ExpectedResult("table1_adni_mci", "MCI to AD conversion", "ADNI", "auc", 0.887, 0.015, 20, 296),
    ExpectedResult("table1_oasis_mci", "MCI to AD conversion", "OASIS-3", "auc", 0.871, 0.020, 20, 245),
    ExpectedResult("table1_nacc_mci", "MCI to AD conversion", "NACC", "auc", 0.862, 0.025, 20, 4200),
    ExpectedResult("table2_adni_ad", "AD versus CN", "ADNI", "auc", 0.957, 0.012, 20, 417),
    ExpectedResult("table3_adni_t1", "AD versus CN", "ADNI", "auc", 0.941, 0.012, 20, 417),
    ExpectedResult("table3_oasis_t1", "AD versus CN", "OASIS-3", "auc", 0.924, 0.018, 20, 1378),
    ExpectedResult("table3_aibl_t1", "AD versus CN", "AIBL", "auc", 0.908, 0.022, 20, 768),
    ExpectedResult("table3_nacc_t1", "AD versus CN", "NACC", "auc", 0.931, 0.015, 20, 4200),
    ExpectedResult("preclinical_sensitivity", "Amyloid positive CN", "ADNI and AIBL", "sensitivity", 0.728, 0.030, 20, 400),
    ExpectedResult("ppmi_prodromal", "Prodromal PD", "PPMI", "auc", 0.812, 0.025, 20, 400),
    ExpectedResult("nifd_genetic", "Genetic FTD", "NIFD", "auc", 0.793, 0.030, 20, 150),
    ExpectedResult("dian_lead_time", "Mutation carrier onset", "DIAN", "months", 27.4, 4.0, 20, 214),
    ExpectedResult("dian_spearman", "Amyloid concordance", "DIAN", "spearman", 0.714, 0.055, 20, 214),
    ExpectedResult("site_discriminator", "Protocol invariance", "Four sites", "accuracy", 0.283, 0.035, 20, 6763),
    ExpectedResult("hcp_false_positive", "Healthy reference", "HCP-Aging", "fpr", 0.041, 0.018, 20, 725),
    ExpectedResult("openneuro_ood", "OOD neurodegeneration", "OpenNeuro", "auc", 0.897, 0.035, 20, 312),
    ExpectedResult("calibration_ad", "AD versus CN", "ADNI", "ece", 0.018, 0.012, 20, 417),
    ExpectedResult("calibration_mci", "MCI to AD conversion", "ADNI", "ece", 0.023, 0.015, 20, 296),
    ExpectedResult("calibration_amyloid", "Amyloid positive CN", "ADNI", "ece", 0.031, 0.018, 20, 400),
)


@dataclass(frozen=True)
class Comparison:
    identifier: str
    observed: float
    expected: float
    tolerance: float
    difference: float
    passed: bool


def compare_results(observed: Mapping[str, float]) -> list[Comparison]:
    comparisons: list[Comparison] = []
    for expected in EXPECTED_RESULTS:
        if expected.identifier not in observed:
            continue
        value = observed[expected.identifier]
        difference = value - expected.expected
        comparisons.append(
            Comparison(
                expected.identifier,
                value,
                expected.expected,
                expected.tolerance,
                difference,
                expected.accepts(value),
            )
        )
    return comparisons


def require_complete(observed: Mapping[str, float]) -> tuple[str, ...]:
    expected = {item.identifier for item in EXPECTED_RESULTS}
    return tuple(sorted(expected - set(observed)))


def aggregate_seed_files(paths: Sequence[Path]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("seed result must contain a mapping")
        for key, value in payload.items():
            values.setdefault(key, []).append(float(value))
    return {key: float(np.mean(items)) for key, items in values.items()}


def export_comparisons(path: Path, comparisons: Iterable[Comparison]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("identifier", "observed", "expected", "tolerance", "difference", "passed"))
        writer.writeheader()
        writer.writerows(asdict(comparison) for comparison in comparisons)


def result_matrix(results: Mapping[str, Mapping[int, float]]) -> tuple[list[str], list[int], np.ndarray]:
    identifiers = sorted(results)
    seeds = sorted({seed for values in results.values() for seed in values})
    matrix = np.full((len(identifiers), len(seeds)), np.nan, dtype=np.float64)
    for row, identifier in enumerate(identifiers):
        for column, seed in enumerate(seeds):
            if seed in results[identifier]:
                matrix[row, column] = results[identifier][seed]
    return identifiers, seeds, matrix


def seed_stability(results: Mapping[str, Mapping[int, float]]) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for identifier, values in results.items():
        array = np.asarray(list(values.values()), dtype=np.float64)
        summaries[identifier] = {
            "mean": float(array.mean()),
            "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "range": float(array.max() - array.min()),
            "count": float(len(array)),
        }
    return summaries
