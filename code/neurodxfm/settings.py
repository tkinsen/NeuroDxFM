from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ModelSettings:
    input_channels: int = 5
    input_shape: tuple[int, int, int] = (160, 192, 160)
    patch_size: tuple[int, int, int] = (4, 4, 4)
    window_size: tuple[int, int, int] = (7, 7, 7)
    depths: tuple[int, int, int, int] = (2, 2, 18, 2)
    heads: tuple[int, int, int, int] = (3, 6, 12, 24)
    embedding_dim: int = 96
    representation_dim: int = 768
    projection_dim: int = 256
    anatomical_targets: int = 68
    protocol_classes: int = 48
    dropout: float = 0.1
    attention_dropout: float = 0.0
    stochastic_depth: float = 0.2
    modality_dropout: float = 0.5
    mask_ratio: float = 0.25
    ema_momentum: float = 0.996
    temperature: float = 0.07


@dataclass(frozen=True)
class LossSettings:
    protocol: float = 1.0
    trajectory: float = 1.0
    modality: float = 0.5
    anatomy: float = 0.3
    amyloid: float = 0.1
    grl: float = 1.0
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class OptimizerSettings:
    name: Literal["adamw"] = "adamw"
    learning_rate: float = 3e-4
    downstream_learning_rate: float = 1e-5
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    gradient_clip: float = 1.0


@dataclass(frozen=True)
class ScheduleSettings:
    epochs: int = 200
    phase_boundary: int = 100
    warmup_epochs: int = 10
    scheduler: Literal["cosine"] = "cosine"
    steps_per_epoch: int = 6250
    longitudinal_ratio: int = 3


@dataclass(frozen=True)
class RuntimeSettings:
    seed: int = 3407
    precision: Literal["fp16"] = "fp16"
    world_size: int = 8
    batch_size_per_gpu: int = 8
    gradient_accumulation: int = 4
    workers_per_gpu: int = 12
    device: str = "cuda"
    deterministic: bool = True
    checkpoint_interval: int = 5
    output_directory: Path = Path("runs/main")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size_per_gpu * self.world_size * self.gradient_accumulation


@dataclass(frozen=True)
class DataSettings:
    manifest: Path = Path("data/manifest.csv")
    cache_directory: Path = Path("data/cache")
    input_shape: tuple[int, int, int] = (160, 192, 160)
    modality_dropout: float = 0.5
    modalities: tuple[str, ...] = ("t1", "flair", "fdg_pet", "amyloid_pet", "csf")
    white_matter_peak: float = 110.0
    rotation_degrees: float = 15.0
    scale_low: float = 0.9
    scale_high: float = 1.1
    translation_mm: float = 5.0
    noise_std: float = 0.02
    brightness_low: float = 0.9
    brightness_high: float = 1.1
    gamma_low: float = 0.7
    gamma_high: float = 1.5
    counterfactual_ssim: float = 0.85
    time_gap_low_months: int = 6
    time_gap_high_months: int = 120


@dataclass(frozen=True)
class ExperimentSettings:
    model: ModelSettings = field(default_factory=ModelSettings)
    loss: LossSettings = field(default_factory=LossSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    schedule: ScheduleSettings = field(default_factory=ScheduleSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    data: DataSettings = field(default_factory=DataSettings)
