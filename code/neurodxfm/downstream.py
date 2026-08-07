from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TaskKind(StrEnum):
    AD_CLASSIFICATION = "ad_classification"
    MCI_CONVERSION = "mci_conversion"
    AMYLOID_CN = "amyloid_cn"
    PRODROMAL_PD = "prodromal_pd"
    GENETIC_FTD = "genetic_ftd"
    DIAN_LEAD_TIME = "dian_lead_time"


@dataclass(frozen=True)
class TaskSettings:
    kind: TaskKind
    output_dim: int
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int
    dropout: float
    freeze_encoder: bool
    class_counts: tuple[int, ...]


TASKS = {
    TaskKind.AD_CLASSIFICATION: TaskSettings(TaskKind.AD_CLASSIFICATION, 2, 1e-3, 1e-4, 50, 5, 0.1, True, (213, 204)),
    TaskKind.MCI_CONVERSION: TaskSettings(TaskKind.MCI_CONVERSION, 2, 1e-3, 1e-4, 50, 5, 0.1, True, (132, 164)),
    TaskKind.AMYLOID_CN: TaskSettings(TaskKind.AMYLOID_CN, 2, 1e-3, 1e-4, 50, 5, 0.1, True, (220, 180)),
    TaskKind.PRODROMAL_PD: TaskSettings(TaskKind.PRODROMAL_PD, 2, 1e-3, 1e-4, 50, 5, 0.1, True, (310, 400)),
    TaskKind.GENETIC_FTD: TaskSettings(TaskKind.GENETIC_FTD, 2, 1e-3, 1e-4, 50, 5, 0.1, True, (103, 150)),
    TaskKind.DIAN_LEAD_TIME: TaskSettings(TaskKind.DIAN_LEAD_TIME, 1, 1e-3, 1e-4, 50, 5, 0.1, True, (214,)),
}


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, representation: Tensor) -> Tensor:
        return self.classifier(self.dropout(self.normalization(representation)))


class NonlinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        hidden = input_dim // 2
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, representation: Tensor) -> Tensor:
        return self.network(representation)


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def fit(self, logits: Tensor, labels: Tensor, iterations: int = 100) -> float:
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.01, max_iter=iterations)

        def closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self(logits), labels.long())
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach())


class DirichletPrediction:
    def __init__(self, evidence: Tensor) -> None:
        self.alpha = evidence

    @property
    def probability(self) -> Tensor:
        return self.alpha / self.alpha.sum(dim=-1, keepdim=True)

    @property
    def uncertainty(self) -> Tensor:
        return self.alpha.shape[-1] / self.alpha.sum(dim=-1)

    @property
    def entropy(self) -> Tensor:
        probability = self.probability.clamp_min(1e-8)
        return -(probability * probability.log()).sum(dim=-1)


class EarlyStopping:
    def __init__(self, patience: int, minimum_delta: float = 0.0) -> None:
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.best = float("-inf")
        self.stale = 0

    def update(self, value: float) -> bool:
        if value > self.best + self.minimum_delta:
            self.best = value
            self.stale = 0
            return False
        self.stale += 1
        return self.stale >= self.patience


class ProbeTrainer:
    def __init__(self, probe: nn.Module, settings: TaskSettings, device: torch.device) -> None:
        self.probe = probe.to(device)
        self.settings = settings
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.probe.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        counts = torch.tensor(settings.class_counts, dtype=torch.float32, device=device)
        if len(counts) > 1:
            weights = counts.sum() / counts.clamp_min(1.0)
            self.weights = weights / weights.mean()
        else:
            self.weights = torch.ones(1, device=device)

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        if self.settings.output_dim == 1:
            return F.smooth_l1_loss(logits.flatten(), target.float())
        return F.cross_entropy(logits, target.long(), weight=self.weights)

    def train_epoch(self, batches: Iterable[tuple[Tensor, Tensor]]) -> float:
        self.probe.train()
        total = 0.0
        count = 0
        for representation, target in batches:
            representation = representation.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss(self.probe(representation), target)
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach())
            count += 1
        return total / max(count, 1)


def evidential_decision(evidence: Tensor, uncertainty_threshold: float) -> tuple[Tensor, Tensor]:
    prediction = DirichletPrediction(evidence)
    probability, label = prediction.probability.max(dim=-1)
    abstain = prediction.uncertainty > uncertainty_threshold
    label = torch.where(abstain, torch.full_like(label, -1), label)
    return label, probability


def t1_only_input(volumes: Tensor) -> Tensor:
    result = torch.zeros_like(volumes)
    result[:, 0] = volumes[:, 0]
    return result


def retention_ratio(t1_metric: float, multimodal_metric: float) -> float:
    if multimodal_metric <= 0.0:
        raise ValueError("multimodal metric must be positive")
    return t1_metric / multimodal_metric


def estimated_months_to_onset(risk: Tensor, calibration_slope: float, calibration_intercept: float) -> Tensor:
    return calibration_intercept + calibration_slope * torch.logit(risk.clamp(1e-5, 1.0 - 1e-5))
