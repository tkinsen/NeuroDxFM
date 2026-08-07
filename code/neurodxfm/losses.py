from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import NeuroDxOutput
from .settings import LossSettings


def info_nce(anchor: Tensor, positive: Tensor, temperature: float = 0.07) -> Tensor:
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.transpose(0, 1) / temperature
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)) * 0.5


def masked_future_loss(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    target = target.detach()
    if mask is None:
        return F.mse_loss(prediction, target)
    selected_prediction = prediction[mask]
    selected_target = target[mask]
    if selected_prediction.numel() == 0:
        return prediction.sum() * 0.0
    return F.mse_loss(selected_prediction, selected_target)


def modality_reconstruction_loss(prediction: Tensor, target: Tensor, missing: Tensor) -> Tensor:
    while missing.ndim < prediction.ndim:
        missing = missing.unsqueeze(-1)
    missing = missing.expand_as(prediction)
    difference = F.smooth_l1_loss(prediction, target, reduction="none")
    denominator = missing.sum().clamp_min(1)
    return (difference * missing).sum() / denominator


def anatomical_loss(prediction: Tensor, target: Tensor, valid: Tensor | None = None) -> Tensor:
    target = torch.log1p(target.clamp_min(0.0))
    prediction = torch.log1p(prediction.clamp_min(0.0))
    difference = F.smooth_l1_loss(prediction, target, reduction="none")
    if valid is None:
        return difference.mean()
    weights = valid.to(difference.dtype)
    return (difference * weights).sum() / weights.sum().clamp_min(1.0)


def binary_anchor_loss(logits: Tensor, labels: Tensor, valid: Tensor | None = None) -> Tensor:
    loss = F.cross_entropy(logits, labels.long(), reduction="none")
    if valid is None:
        return loss.mean()
    weights = valid.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def dirichlet_kl(alpha: Tensor, classes: int) -> Tensor:
    prior = torch.ones_like(alpha)
    alpha_sum = alpha.sum(dim=-1, keepdim=True)
    prior_sum = prior.sum(dim=-1, keepdim=True)
    first = torch.lgamma(alpha_sum) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
    second = -torch.lgamma(prior_sum) + torch.lgamma(prior).sum(dim=-1, keepdim=True)
    third = ((alpha - prior) * (torch.digamma(alpha) - torch.digamma(alpha_sum))).sum(dim=-1, keepdim=True)
    return (first + second + third).mean() / classes


def evidential_classification_loss(alpha: Tensor, target: Tensor, annealing: float = 1.0) -> Tensor:
    classes = alpha.shape[-1]
    encoded = F.one_hot(target.long(), num_classes=classes).to(alpha.dtype)
    strength = alpha.sum(dim=-1, keepdim=True)
    probability = alpha / strength
    error = (encoded - probability).square().sum(dim=-1)
    variance = (alpha * (strength - alpha) / (strength.square() * (strength + 1.0))).sum(dim=-1)
    adjusted = (alpha - 1.0) * (1.0 - encoded) + 1.0
    return (error + variance).mean() + annealing * dirichlet_kl(adjusted, classes)


@dataclass(frozen=True)
class LossBatch:
    protocol: Tensor
    future: Tensor | None
    reconstruction_target: Tensor
    missing_modalities: Tensor
    anatomy: Tensor
    anatomy_valid: Tensor | None
    amyloid: Tensor
    amyloid_valid: Tensor | None
    diagnosis: Tensor | None = None


@dataclass(frozen=True)
class LossValues:
    total: Tensor
    protocol: Tensor
    trajectory: Tensor
    modality: Tensor
    anatomy: Tensor
    amyloid: Tensor
    evidential: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "protocol": float(self.protocol.detach()),
            "trajectory": float(self.trajectory.detach()),
            "modality": float(self.modality.detach()),
            "anatomy": float(self.anatomy.detach()),
            "amyloid": float(self.amyloid.detach()),
            "evidential": float(self.evidential.detach()),
        }


class FiveHeadObjective(nn.Module):
    def __init__(self, settings: LossSettings | None = None, temperature: float = 0.07) -> None:
        super().__init__()
        self.settings = settings or LossSettings()
        self.temperature = temperature

    def forward(
        self,
        original: NeuroDxOutput,
        counterfactual: NeuroDxOutput,
        batch: LossBatch,
        epoch: int,
    ) -> LossValues:
        protocol = info_nce(original.projection, counterfactual.projection, self.temperature)
        protocol = protocol + F.cross_entropy(original.protocol_logits, batch.protocol.long())
        zero = original.representation.sum() * 0.0
        trajectory = zero
        if epoch > 100 and original.trajectory is not None and batch.future is not None:
            trajectory = masked_future_loss(original.trajectory, batch.future)
        modality = modality_reconstruction_loss(original.reconstruction, batch.reconstruction_target, batch.missing_modalities)
        anatomy = anatomical_loss(original.anatomy, batch.anatomy, batch.anatomy_valid)
        amyloid = binary_anchor_loss(original.amyloid_logits, batch.amyloid, batch.amyloid_valid)
        evidential = zero
        if batch.diagnosis is not None:
            evidential = evidential_classification_loss(original.evidence, batch.diagnosis, min(epoch / 10.0, 1.0))
        total = (
            self.settings.protocol * protocol
            + self.settings.trajectory * trajectory
            + self.settings.modality * modality
            + self.settings.anatomy * anatomy
            + self.settings.amyloid * amyloid
            + evidential
        )
        return LossValues(total, protocol, trajectory, modality, anatomy, amyloid, evidential)


class ClassBalancedCrossEntropy(nn.Module):
    def __init__(self, counts: Tensor, beta: float = 0.9999) -> None:
        super().__init__()
        effective = 1.0 - torch.pow(torch.tensor(beta), counts.float())
        weights = (1.0 - beta) / effective.clamp_min(1e-12)
        self.register_buffer("weights", weights / weights.sum() * counts.numel())

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits, target.long(), weight=self.weights)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        ce = F.cross_entropy(logits, target.long(), reduction="none")
        probability = torch.exp(-ce)
        loss = (1.0 - probability).pow(self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha
        return loss.mean()


class CoxPartialLikelihood(nn.Module):
    def forward(self, risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
        order = torch.argsort(time, descending=True)
        ordered_risk = risk.flatten()[order]
        ordered_event = event.float().flatten()[order]
        log_cumulative = torch.logcumsumexp(ordered_risk, dim=0)
        likelihood = ordered_risk - log_cumulative
        return -(likelihood * ordered_event).sum() / ordered_event.sum().clamp_min(1.0)


class ConcordanceCorrelationLoss(nn.Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction_mean = prediction.mean()
        target_mean = target.mean()
        covariance = ((prediction - prediction_mean) * (target - target_mean)).mean()
        prediction_variance = prediction.var(unbiased=False)
        target_variance = target.var(unbiased=False)
        coefficient = 2.0 * covariance / (prediction_variance + target_variance + (prediction_mean - target_mean).square() + 1e-8)
        return 1.0 - coefficient
