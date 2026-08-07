from pathlib import Path

import numpy as np
import pytest
import torch
from neurodxfm.data import (
    CounterfactualTransform,
    ScanRecord,
    crop_or_pad,
    longitudinal_pairs,
    manifest_digest,
    modality_dropout,
    normalize_white_matter,
    subject_partition,
    validate_subject_partition,
)


def record(subject: str, months: float, diagnosis: int = 0, apoe4: int = 0) -> ScanRecord:
    return ScanRecord(
        subject,
        f"v{months}",
        "test",
        Path("volume.npy"),
        None,
        None,
        None,
        None,
        diagnosis,
        0,
        months,
        tuple(0.0 for _ in range(68)),
        -1,
        apoe4,
        "site",
        "vendor",
        "3T",
        "t1",
    )


def test_crop_larger_volume() -> None:
    volume = torch.arange(10 * 12 * 14).reshape(10, 12, 14)
    result = crop_or_pad(volume, (8, 8, 8))
    assert result.shape == (8, 8, 8)


def test_pad_smaller_volume() -> None:
    volume = torch.ones(3, 4, 5)
    result = crop_or_pad(volume, (8, 8, 8))
    assert result.shape == (8, 8, 8)
    assert result.sum() == volume.sum()


def test_crop_or_pad_rejects_matrix() -> None:
    with pytest.raises(ValueError):
        crop_or_pad(torch.ones(3, 3), (3, 3, 3))


def test_normalize_white_matter() -> None:
    volume = torch.linspace(0.0, 100.0, 1000).reshape(10, 10, 10)
    result = normalize_white_matter(volume)
    assert torch.quantile(result[result > 0], 0.85).item() == pytest.approx(110.0, rel=0.02)


def test_modality_dropout_preserves_t1() -> None:
    volume = torch.ones(5, 4, 4, 4)
    generator = torch.Generator().manual_seed(1)
    result, missing = modality_dropout(volume, 1.0, generator)
    assert not missing[0]
    assert result[0].sum() > 0
    assert result[1:].sum() == 0


def test_counterfactual_shape() -> None:
    transform = CounterfactualTransform()
    volume = torch.rand(5, 4, 4, 4)
    result = transform(volume, torch.Generator().manual_seed(2))
    assert result.shape == volume.shape


def test_subject_partition_keeps_visits_together() -> None:
    records = [record("a", 0), record("a", 12), record("b", 0), record("c", 0)]
    partitions = subject_partition(records, 2, 3)
    validate_subject_partition(records, partitions)
    ownership = {}
    for fold, indices in enumerate(partitions):
        for index in indices:
            ownership.setdefault(records[index].subject_id, fold)
            assert ownership[records[index].subject_id] == fold


def test_partition_detects_leakage() -> None:
    records = [record("a", 0), record("a", 12)]
    with pytest.raises(ValueError):
        validate_subject_partition(records, [[0], [1]])


def test_longitudinal_pairs() -> None:
    records = [record("a", 0), record("a", 12), record("a", 132), record("b", 0)]
    pairs = longitudinal_pairs(records)
    assert pairs == [(0, 1, 12), (1, 2, 120)]


def test_manifest_digest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("subject_id\na\n", encoding="utf-8")
    first = manifest_digest(path)
    path.write_text("subject_id\nb\n", encoding="utf-8")
    second = manifest_digest(path)
    assert first != second


def test_numpy_volume_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "volume.npy"
    np.save(path, np.ones((3, 3, 3), dtype=np.float32))
    loaded = np.load(path)
    assert loaded.shape == (3, 3, 3)
