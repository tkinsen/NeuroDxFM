import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F


class Transform(Protocol):
    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor: ...


@dataclass(frozen=True)
class Compose:
    transforms: Sequence[Transform]

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        result = volume
        for transform in self.transforms:
            result = transform(result, generator)
        return result


@dataclass(frozen=True)
class RandomApply:
    transform: Transform
    probability: float

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        if float(torch.rand((), generator=generator)) < self.probability:
            return self.transform(volume, generator)
        return volume


@dataclass(frozen=True)
class GaussianNoise:
    standard_deviation: float

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        return volume + torch.randn(volume.shape, generator=generator, device=volume.device, dtype=volume.dtype) * self.standard_deviation


@dataclass(frozen=True)
class MultiplicativeBrightness:
    lower: float
    upper: float

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        factor = torch.empty((), device=volume.device).uniform_(self.lower, self.upper, generator=generator)
        return volume * factor


@dataclass(frozen=True)
class GammaContrast:
    lower: float
    upper: float

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        gamma = torch.empty((), device=volume.device).uniform_(self.lower, self.upper, generator=generator)
        minimum = volume.amin(dim=(-3, -2, -1), keepdim=True)
        maximum = volume.amax(dim=(-3, -2, -1), keepdim=True)
        normalized = (volume - minimum) / (maximum - minimum).clamp_min(1e-6)
        return normalized.clamp(0.0, 1.0).pow(gamma) * (maximum - minimum) + minimum


@dataclass(frozen=True)
class RandomFlip:
    axes: tuple[int, ...]
    probability: float

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        selected = [axis for axis in self.axes if float(torch.rand((), generator=generator)) < self.probability]
        return torch.flip(volume, selected) if selected else volume


@dataclass(frozen=True)
class RandomCutout:
    size: tuple[int, int, int]
    probability: float

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        if float(torch.rand((), generator=generator)) >= self.probability:
            return volume
        shape = volume.shape[-3:]
        starts = [
            int(torch.randint(0, max(current - cut + 1, 1), (), generator=generator)) for current, cut in zip(shape, self.size, strict=True)
        ]
        result = volume.clone()
        result[..., starts[0] : starts[0] + self.size[0], starts[1] : starts[1] + self.size[1], starts[2] : starts[2] + self.size[2]] = 0.0
        return result


@dataclass(frozen=True)
class RandomLowResolution:
    lower: float = 0.5
    upper: float = 1.0

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        factor = float(torch.empty(()).uniform_(self.lower, self.upper, generator=generator))
        original = volume.shape[-3:]
        reduced = tuple(max(int(length * factor), 2) for length in original)
        batched = volume.reshape(-1, 1, *original)
        low = F.interpolate(batched, size=reduced, mode="trilinear", align_corners=False)
        restored = F.interpolate(low, size=original, mode="trilinear", align_corners=False)
        return restored.reshape(volume.shape)


@dataclass(frozen=True)
class RandomKSpaceSpike:
    magnitude: float = 3.0
    probability: float = 0.2

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        if float(torch.rand((), generator=generator)) >= self.probability:
            return volume
        spectrum = torch.fft.fftn(volume, dim=(-3, -2, -1))
        indices = [int(torch.randint(0, length, (), generator=generator)) for length in volume.shape[-3:]]
        spectrum[..., indices[0], indices[1], indices[2]] *= self.magnitude
        return torch.fft.ifftn(spectrum, dim=(-3, -2, -1)).real


@dataclass(frozen=True)
class RandomGhosting:
    axis: int = -2
    repetitions: int = 4
    intensity: float = 0.2
    probability: float = 0.2

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        if float(torch.rand((), generator=generator)) >= self.probability:
            return volume
        result = volume.clone()
        length = volume.shape[self.axis]
        for repetition in range(1, self.repetitions + 1):
            shift = max(length * repetition // self.repetitions, 1)
            result = result + torch.roll(volume, shift, dims=self.axis) * self.intensity / repetition
        return result / (1.0 + self.intensity * sum(1.0 / value for value in range(1, self.repetitions + 1)))


@dataclass(frozen=True)
class RandomBiasField:
    coefficient: float = 0.3
    order: int = 3

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        shape = volume.shape[-3:]
        axes = [torch.linspace(-1.0, 1.0, length, device=volume.device) for length in shape]
        coordinates = torch.meshgrid(*axes, indexing="ij")
        field = torch.zeros(shape, device=volume.device, dtype=volume.dtype)
        for first in range(self.order + 1):
            for second in range(self.order + 1 - first):
                for third in range(self.order + 1 - first - second):
                    if first + second + third == 0:
                        continue
                    value = torch.empty((), device=volume.device).uniform_(-self.coefficient, self.coefficient, generator=generator)
                    field = field + value * coordinates[0].pow(first) * coordinates[1].pow(second) * coordinates[2].pow(third)
        return volume * field.exp()


@dataclass(frozen=True)
class RandomMotion:
    max_shift: int = 5
    segments: int = 4

    def __call__(self, volume: Tensor, generator: torch.Generator) -> Tensor:
        axis = int(torch.randint(0, 3, (), generator=generator))
        length = volume.shape[-3 + axis]
        boundaries = torch.linspace(0, length, self.segments + 1).long()
        result = volume.clone()
        for segment in range(self.segments):
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (), generator=generator))
            slices = [slice(None)] * volume.ndim
            slices[-3 + axis] = slice(int(boundaries[segment]), int(boundaries[segment + 1]))
            result[tuple(slices)] = torch.roll(volume[tuple(slices)], shift, dims=-1)
        return result


def pretraining_augmentations() -> Compose:
    return Compose(
        (
            RandomApply(GaussianNoise(0.02), 0.5),
            RandomApply(MultiplicativeBrightness(0.9, 1.1), 0.5),
            RandomApply(GammaContrast(0.7, 1.5), 0.5),
            RandomFlip((-3,), 0.5),
            RandomLowResolution(0.7, 1.0),
            RandomKSpaceSpike(2.0, 0.1),
            RandomGhosting(-2, 4, 0.1, 0.1),
            RandomApply(RandomBiasField(0.2, 3), 0.3),
        )
    )


def trajectory_mask(tokens: int, ratio: float, generator: torch.Generator, device: torch.device) -> Tensor:
    count = max(math.ceil(tokens * ratio), 1)
    indices = torch.randperm(tokens, generator=generator, device=device)[:count]
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    mask[indices] = True
    return mask
