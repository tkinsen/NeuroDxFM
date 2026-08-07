from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FusionOutput:
    representation: Tensor
    attention: Tensor
    available: Tensor


class ModalityTokenBank(nn.Module):
    def __init__(self, modalities: int, dimension: int) -> None:
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(modalities, dimension))
        self.missing = nn.Parameter(torch.empty(modalities, dimension))
        nn.init.trunc_normal_(self.tokens, std=0.02)
        nn.init.trunc_normal_(self.missing, std=0.02)

    def forward(self, batch: int, available: Tensor) -> Tensor:
        tokens = self.tokens.unsqueeze(0).expand(batch, -1, -1)
        missing = self.missing.unsqueeze(0).expand(batch, -1, -1)
        return torch.where(available.unsqueeze(-1), tokens, missing)


class CrossModalAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension * 4), nn.GELU(), nn.Linear(dimension * 4, dimension)
        )

    def forward(self, query: Tensor, context: Tensor, available: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.normalization(query)
        attended, weights = self.attention(normalized, context, context, key_padding_mask=~available, need_weights=True)
        result = query + attended
        return result + self.output(result), weights


class CrossAttentionFusion(nn.Module):
    def __init__(self, modalities: int, dimension: int, heads: int = 8, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.modalities = modalities
        self.tokens = ModalityTokenBank(modalities, dimension)
        self.layers = nn.ModuleList([CrossModalAttention(dimension, heads, dropout) for _ in range(layers)])
        self.pool = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))

    def forward(self, representations: Tensor, available: Tensor) -> FusionOutput:
        if representations.ndim != 3:
            raise ValueError("representations must have batch, modality, feature dimensions")
        if representations.shape[:2] != available.shape:
            raise ValueError("availability shape must match representation prefix")
        if not available[:, 0].all():
            raise ValueError("T1 must be present")
        x = representations + self.tokens(representations.shape[0], available)
        for layer in self.layers:
            x, _ = layer(x, x, available)
        logits = self.pool(x).squeeze(-1).masked_fill(~available, float("-inf"))
        pooling = logits.softmax(dim=-1)
        fused = torch.einsum("bm,bmd->bd", pooling, x)
        return FusionOutput(fused, pooling, available)


class LateFusion(nn.Module):
    def __init__(self, modalities: int, dimension: int, classes: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(dimension, classes) for _ in range(modalities)])

    def forward(self, representations: Tensor, available: Tensor) -> Tensor:
        logits = torch.stack([head(representations[:, index]) for index, head in enumerate(self.heads)], dim=1)
        weights = available.to(logits.dtype)
        return (logits * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)


class GatedFusion(nn.Module):
    def __init__(self, modalities: int, dimension: int) -> None:
        super().__init__()
        self.gates = nn.Sequential(nn.Linear(modalities * dimension, dimension), nn.GELU(), nn.Linear(dimension, modalities))

    def forward(self, representations: Tensor, available: Tensor) -> FusionOutput:
        flattened = representations.flatten(start_dim=1)
        logits = self.gates(flattened).masked_fill(~available, float("-inf"))
        weights = logits.softmax(dim=-1)
        fused = torch.einsum("bm,bmd->bd", weights, representations)
        return FusionOutput(fused, weights, available)


class ModalityDropout(nn.Module):
    def __init__(self, probability: float, preserve_index: int = 0) -> None:
        super().__init__()
        self.probability = probability
        self.preserve_index = preserve_index

    def forward(self, representations: Tensor, available: Tensor) -> tuple[Tensor, Tensor]:
        if not self.training:
            return representations, available
        dropped = torch.rand_like(available.float()) < self.probability
        dropped[:, self.preserve_index] = False
        retained = available & ~dropped
        return representations * retained.unsqueeze(-1), retained


def fusion_consistency(full: Tensor, partial: Tensor) -> Tensor:
    return 1.0 - F.cosine_similarity(full, partial, dim=-1).mean()


def modality_contribution(model: nn.Module, representations: Tensor, available: Tensor) -> Tensor:
    baseline = model(representations, available)
    baseline_value = baseline.representation if isinstance(baseline, FusionOutput) else baseline
    contributions: list[Tensor] = []
    for index in range(available.shape[1]):
        ablated = available.clone()
        if index != 0:
            ablated[:, index] = False
        output = model(representations, ablated)
        value = output.representation if isinstance(output, FusionOutput) else output
        contributions.append((baseline_value - value).norm(dim=-1))
    return torch.stack(contributions, dim=-1)


def enumerate_modality_sets(names: Sequence[str], required: str = "t1") -> tuple[tuple[str, ...], ...]:
    if required not in names:
        raise ValueError("required modality is absent")
    optional = [name for name in names if name != required]
    combinations: list[tuple[str, ...]] = []
    for mask in range(1 << len(optional)):
        selected = [required]
        selected.extend(name for index, name in enumerate(optional) if mask & (1 << index))
        combinations.append(tuple(selected))
    return tuple(combinations)
