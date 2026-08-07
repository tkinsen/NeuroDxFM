import csv
import hashlib
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from .settings import DataSettings


@dataclass(frozen=True)
class ScanRecord:
    subject_id: str
    visit_id: str
    cohort: str
    t1: Path
    flair: Path | None
    fdg_pet: Path | None
    amyloid_pet: Path | None
    csf: Path | None
    diagnosis: int
    protocol: int
    months: float
    anatomy: tuple[float, ...]
    amyloid: int
    apoe4: int
    site: str
    vendor: str
    field_strength: str
    sequence: str


@dataclass(frozen=True)
class VolumeBatch:
    volumes: Tensor
    counterfactual: Tensor
    reconstruction_target: Tensor
    missing_modalities: Tensor
    protocol: Tensor
    months: Tensor
    anatomy: Tensor
    anatomy_valid: Tensor
    amyloid: Tensor
    amyloid_valid: Tensor
    diagnosis: Tensor
    subject_ids: tuple[str, ...]

    def to(self, device: torch.device | str) -> "VolumeBatch":
        return VolumeBatch(
            self.volumes.to(device),
            self.counterfactual.to(device),
            self.reconstruction_target.to(device),
            self.missing_modalities.to(device),
            self.protocol.to(device),
            self.months.to(device),
            self.anatomy.to(device),
            self.anatomy_valid.to(device),
            self.amyloid.to(device),
            self.amyloid_valid.to(device),
            self.diagnosis.to(device),
            self.subject_ids,
        )


def _optional_path(value: str) -> Path | None:
    stripped = value.strip()
    return Path(stripped) if stripped else None


def _anatomy(value: str) -> tuple[float, ...]:
    if not value.strip():
        return tuple(0.0 for _ in range(68))
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != 68:
        raise ValueError("anatomy must contain 68 values")
    return tuple(float(item) for item in parsed)


def read_manifest(path: Path) -> list[ScanRecord]:
    records: list[ScanRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                ScanRecord(
                    subject_id=row["subject_id"],
                    visit_id=row["visit_id"],
                    cohort=row["cohort"],
                    t1=Path(row["t1"]),
                    flair=_optional_path(row.get("flair", "")),
                    fdg_pet=_optional_path(row.get("fdg_pet", "")),
                    amyloid_pet=_optional_path(row.get("amyloid_pet", "")),
                    csf=_optional_path(row.get("csf", "")),
                    diagnosis=int(row["diagnosis"]),
                    protocol=int(row["protocol"]),
                    months=float(row.get("months", "0")),
                    anatomy=_anatomy(row.get("anatomy", "")),
                    amyloid=int(row.get("amyloid", "-1")),
                    apoe4=int(row.get("apoe4", "-1")),
                    site=row.get("site", "unknown"),
                    vendor=row.get("vendor", "unknown"),
                    field_strength=row.get("field_strength", "unknown"),
                    sequence=row.get("sequence", "unknown"),
                )
            )
    return records


def manifest_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_partition(records: Sequence[ScanRecord], folds: int, seed: int) -> list[list[int]]:
    grouped: dict[tuple[int, int], dict[str, list[int]]] = {}
    for index, record in enumerate(records):
        key = record.diagnosis, record.apoe4
        grouped.setdefault(key, {}).setdefault(record.subject_id, []).append(index)
    result: list[list[int]] = [[] for _ in range(folds)]
    generator = random.Random(seed)
    for subjects in grouped.values():
        identities = list(subjects)
        generator.shuffle(identities)
        for offset, identity in enumerate(identities):
            result[offset % folds].extend(subjects[identity])
    return result


def validate_subject_partition(records: Sequence[ScanRecord], partitions: Sequence[Sequence[int]]) -> None:
    ownership: dict[str, int] = {}
    for fold, indices in enumerate(partitions):
        for index in indices:
            subject = records[index].subject_id
            if subject in ownership and ownership[subject] != fold:
                raise ValueError("subject occurs in multiple partitions")
            ownership[subject] = fold


def load_volume(path: Path) -> Tensor:
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".npy"):
        array = np.load(path, allow_pickle=False)
        return torch.from_numpy(np.asarray(array, dtype=np.float32))
    if suffixes.endswith(".pt"):
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, Tensor):
            raise TypeError("volume file must contain a tensor")
        return value.float()
    raise ValueError(f"unsupported volume format: {suffixes}")


def crop_or_pad(volume: Tensor, shape: tuple[int, int, int]) -> Tensor:
    if volume.ndim != 3:
        raise ValueError("volume must have three dimensions")
    slices: list[slice] = []
    for current, target in zip(volume.shape, shape, strict=True):
        start = max((current - target) // 2, 0)
        slices.append(slice(start, start + min(current, target)))
    cropped = volume[tuple(slices)]
    pads: list[int] = []
    for current, target in reversed(list(zip(cropped.shape, shape, strict=True))):
        difference = target - current
        lower = difference // 2
        pads.extend((lower, difference - lower))
    return F.pad(cropped, pads)


def normalize_white_matter(volume: Tensor, peak: float = 110.0) -> Tensor:
    finite = volume[torch.isfinite(volume)]
    if finite.numel() == 0:
        raise ValueError("volume has no finite voxels")
    positive = finite[finite > 0]
    reference = torch.quantile(positive, 0.85) if positive.numel() else finite.abs().mean().clamp_min(1e-6)
    normalized = volume / reference.clamp_min(1e-6) * peak
    return torch.nan_to_num(normalized, nan=0.0, posinf=peak * 4.0, neginf=0.0)


def random_intensity(volume: Tensor, settings: DataSettings, generator: torch.Generator) -> Tensor:
    brightness = torch.empty((), device=volume.device).uniform_(settings.brightness_low, settings.brightness_high, generator=generator)
    gamma = torch.empty((), device=volume.device).uniform_(settings.gamma_low, settings.gamma_high, generator=generator)
    noise = torch.randn(volume.shape, device=volume.device, dtype=volume.dtype, generator=generator) * settings.noise_std
    minimum = volume.amin()
    maximum = volume.amax()
    scaled = (volume - minimum) / (maximum - minimum).clamp_min(1e-6)
    transformed = scaled.clamp_min(0.0).pow(gamma) * brightness + noise
    return transformed * (maximum - minimum) + minimum


def modality_dropout(volumes: Tensor, probability: float, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    channels = volumes.shape[0]
    missing = torch.rand(channels, generator=generator, device=volumes.device) < probability
    missing[0] = False
    retained = volumes.clone()
    retained[missing] = 0.0
    return retained, missing


class CounterfactualTransform:
    def __init__(self, noise_std: float = 0.02, gamma_range: tuple[float, float] = (0.7, 1.5)) -> None:
        self.noise_std = noise_std
        self.gamma_range = gamma_range

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        gamma = torch.empty((), device=volume.device).uniform_(self.gamma_range[0], self.gamma_range[1], generator=generator)
        noise = torch.randn(volume.shape, generator=generator, device=volume.device, dtype=volume.dtype) * self.noise_std
        lower = volume.amin(dim=(-3, -2, -1), keepdim=True)
        upper = volume.amax(dim=(-3, -2, -1), keepdim=True)
        scaled = (volume - lower) / (upper - lower).clamp_min(1e-6)
        return scaled.clamp(0.0, 1.0).pow(gamma) * (upper - lower) + lower + noise


class NeuroVolumeDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self, records: Sequence[ScanRecord], settings: DataSettings | None = None, training: bool = True, seed: int = 3407
    ) -> None:
        self.records = list(records)
        self.settings = settings or DataSettings()
        self.training = training
        self.seed = seed
        self.counterfactual = CounterfactualTransform(self.settings.noise_std, (self.settings.gamma_low, self.settings.gamma_high))

    def __len__(self) -> int:
        return len(self.records)

    def _modalities(self, record: ScanRecord) -> tuple[Tensor, Tensor]:
        paths = (record.t1, record.flair, record.fdg_pet, record.amyloid_pet, record.csf)
        volumes: list[Tensor] = []
        available: list[bool] = []
        for path in paths:
            if path is None:
                volumes.append(torch.zeros(self.settings.input_shape))
                available.append(False)
            else:
                volume = crop_or_pad(load_volume(path), self.settings.input_shape)
                volumes.append(normalize_white_matter(volume, self.settings.white_matter_peak))
                available.append(True)
        return torch.stack(volumes), torch.tensor(available)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        generator = torch.Generator().manual_seed(self.seed + index)
        target, available = self._modalities(record)
        volumes = target.clone()
        missing = ~available
        if self.training:
            volumes, random_missing = modality_dropout(volumes, self.settings.modality_dropout, generator)
            missing = missing | random_missing
            volumes = random_intensity(volumes, self.settings, generator)
        counterfactual = self.counterfactual(volumes, generator)
        anatomy = torch.tensor(record.anatomy, dtype=torch.float32)
        return {
            "volumes": volumes,
            "counterfactual": counterfactual,
            "target": target,
            "missing": missing,
            "protocol": torch.tensor(record.protocol),
            "months": torch.tensor(record.months),
            "anatomy": anatomy,
            "anatomy_valid": torch.tensor(any(anatomy != 0.0)),
            "amyloid": torch.tensor(max(record.amyloid, 0)),
            "amyloid_valid": torch.tensor(record.amyloid >= 0),
            "diagnosis": torch.tensor(record.diagnosis),
            "subject_id": record.subject_id,
        }


def collate_volume_batch(items: Sequence[Mapping[str, Tensor | str]]) -> VolumeBatch:
    def stack(name: str) -> Tensor:
        values = [item[name] for item in items]
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError(f"{name} must contain tensors")
        return torch.stack(values)

    return VolumeBatch(
        stack("volumes"),
        stack("counterfactual"),
        stack("target"),
        stack("missing"),
        stack("protocol"),
        stack("months"),
        stack("anatomy"),
        stack("anatomy_valid"),
        stack("amyloid"),
        stack("amyloid_valid"),
        stack("diagnosis"),
        tuple(str(item["subject_id"]) for item in items),
    )


class CohortBalancedSampler(Sampler[int]):
    def __init__(self, records: Sequence[ScanRecord], samples: int, seed: int = 3407) -> None:
        self.samples = samples
        self.seed = seed
        self.epoch = 0
        self.groups: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            self.groups.setdefault(record.cohort, []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples

    def __iter__(self) -> Iterator[int]:
        generator = random.Random(self.seed + self.epoch)
        cohorts = sorted(self.groups)
        for position in range(self.samples):
            cohort = cohorts[position % len(cohorts)]
            yield generator.choice(self.groups[cohort])


def longitudinal_pairs(records: Sequence[ScanRecord], low: float = 6.0, high: float = 120.0) -> list[tuple[int, int, float]]:
    subjects: dict[str, list[tuple[int, ScanRecord]]] = {}
    for index, record in enumerate(records):
        subjects.setdefault(record.subject_id, []).append((index, record))
    pairs: list[tuple[int, int, float]] = []
    for visits in subjects.values():
        ordered = sorted(visits, key=lambda item: item[1].months)
        for first in range(len(ordered)):
            for second in range(first + 1, len(ordered)):
                gap = ordered[second][1].months - ordered[first][1].months
                if low <= gap <= high:
                    pairs.append((ordered[first][0], ordered[second][0], gap))
    return pairs
