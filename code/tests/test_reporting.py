import pytest
from neurodxfm.reporting import EXPECTED_RESULTS, compare_results, require_complete, result_matrix, seed_stability


def test_expected_results_have_unique_identifiers() -> None:
    identifiers = [item.identifier for item in EXPECTED_RESULTS]
    assert len(identifiers) == len(set(identifiers))


def test_compare_results_passes_exact_values() -> None:
    item = EXPECTED_RESULTS[0]
    comparisons = compare_results({item.identifier: item.expected})
    assert comparisons[0].passed


def test_require_complete() -> None:
    missing = require_complete({})
    assert len(missing) == len(EXPECTED_RESULTS)


def test_result_matrix() -> None:
    identifiers, seeds, matrix = result_matrix({"a": {1: 0.8, 2: 0.9}, "b": {1: 0.7}})
    assert identifiers == ["a", "b"]
    assert seeds == [1, 2]
    assert matrix.shape == (2, 2)


def test_seed_stability() -> None:
    summary = seed_stability({"a": {1: 0.8, 2: 0.9}})
    assert summary["a"]["mean"] == pytest.approx(0.85)
    assert summary["a"]["count"] == 2.0
