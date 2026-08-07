from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BackboneKind(StrEnum):
    SWIN_UNETR = "swin_unetr"
    RESNET_3D = "resnet_3d"
    VISION_MAMBA_3D = "vision_mamba_3d"
    DINOV2_3D = "dinov2_3d"


class ConvNormActivation(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, kernel: int, stride: int = 1) -> None:
        padding = kernel // 2
        super().__init__(
            nn.Conv3d(input_channels, output_channels, kernel, stride, padding, bias=False),
            nn.GroupNorm(min(32, output_channels), output_channels),
            nn.GELU(),
        )


class ResidualBlock3D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.first = ConvNormActivation(input_channels, output_channels, 3, stride)
        self.second = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, 3, 1, 1, bias=False), nn.GroupNorm(min(32, output_channels), output_channels)
        )
        self.shortcut = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Conv3d(input_channels, output_channels, 1, stride, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.second(self.first(x)) + self.shortcut(x))


class ResNet3D(nn.Module):
    def __init__(self, input_channels: int = 5, representation_dim: int = 768, blocks: tuple[int, ...] = (2, 2, 2, 2)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, 64, 7, 2, 3, bias=False), nn.GroupNorm(32, 64), nn.GELU(), nn.MaxPool3d(3, 2, 1)
        )
        channels = (64, 128, 256, 512)
        stages: list[nn.Module] = []
        previous = 64
        for index, (current, count) in enumerate(zip(channels, blocks, strict=True)):
            units: list[nn.Module] = [ResidualBlock3D(previous, current, 1 if index == 0 else 2)]
            units.extend(ResidualBlock3D(current, current) for _ in range(count - 1))
            stages.append(nn.Sequential(*units))
            previous = current
        self.stages = nn.ModuleList(stages)
        self.projection = nn.Linear(channels[-1], representation_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.projection(x.mean(dim=(-3, -2, -1)))


class PatchTokenizer3D(nn.Module):
    def __init__(self, input_channels: int, dimension: int, patch: int) -> None:
        super().__init__()
        self.projection = nn.Conv3d(input_channels, dimension, patch, patch)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        features = self.projection(x)
        shape = tuple(features.shape[-3:])
        return features.flatten(2).transpose(1, 2), shape


class RotaryPosition3D(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        frequencies = 1.0 / (10000.0 ** (torch.arange(0, dimension, 2).float() / dimension))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        positions = torch.arange(x.shape[1], device=x.device).float()
        angles = torch.outer(positions, self.frequencies)
        sine = angles.sin().unsqueeze(0)
        cosine = angles.cos().unsqueeze(0)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(nn.Linear(dimension, dimension * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension * 4, dimension))

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class DinoVisionTransformer3D(nn.Module):
    def __init__(
        self, input_channels: int = 5, dimension: int = 768, patch: int = 8, depth: int = 12, heads: int = 12, registers: int = 4
    ) -> None:
        super().__init__()
        self.tokenizer = PatchTokenizer3D(input_channels, dimension, patch)
        self.class_token = nn.Parameter(torch.empty(1, 1, dimension))
        self.register_tokens = nn.Parameter(torch.empty(1, registers, dimension))
        self.position = RotaryPosition3D(dimension)
        self.blocks = nn.ModuleList([SelfAttentionBlock(dimension, heads, 0.0) for _ in range(depth)])
        self.norm = nn.LayerNorm(dimension)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        tokens, _ = self.tokenizer(x)
        batch = tokens.shape[0]
        tokens = torch.cat((self.class_token.expand(batch, -1, -1), self.register_tokens.expand(batch, -1, -1), tokens), dim=1)
        tokens = self.position(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens[:, 0])


class SelectiveStateSpace(nn.Module):
    def __init__(self, dimension: int, state_dimension: int = 16, expansion: int = 2) -> None:
        super().__init__()
        inner = dimension * expansion
        self.dimension = dimension
        self.inner = inner
        self.input_projection = nn.Linear(dimension, inner * 2)
        self.convolution = nn.Conv1d(inner, inner, 4, padding=3, groups=inner)
        self.delta = nn.Linear(inner, inner)
        self.b = nn.Linear(inner, state_dimension)
        self.c = nn.Linear(inner, state_dimension)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state_dimension + 1).float()).repeat(inner, 1))
        self.skip = nn.Parameter(torch.ones(inner))
        self.output_projection = nn.Linear(inner, dimension)

    def forward(self, x: Tensor) -> Tensor:
        projected, gate = self.input_projection(x).chunk(2, dim=-1)
        projected = self.convolution(projected.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        projected = F.silu(projected)
        delta = F.softplus(self.delta(projected))
        b = self.b(projected)
        c = self.c(projected)
        a = -torch.exp(self.a_log)
        state = torch.zeros(x.shape[0], self.inner, a.shape[1], device=x.device, dtype=x.dtype)
        outputs: list[Tensor] = []
        for index in range(x.shape[1]):
            transition = torch.exp(delta[:, index].unsqueeze(-1) * a.unsqueeze(0))
            input_term = delta[:, index].unsqueeze(-1) * b[:, index].unsqueeze(1) * projected[:, index].unsqueeze(-1)
            state = transition * state + input_term
            output = torch.einsum("bns,bs->bn", state, c[:, index]) + self.skip * projected[:, index]
            outputs.append(output)
        sequence = torch.stack(outputs, dim=1) * F.silu(gate)
        return self.output_projection(sequence)


class MambaBlock(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.ssm = SelectiveStateSpace(dimension)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ssm(self.norm(x))


class VisionMamba3D(nn.Module):
    def __init__(self, input_channels: int = 5, dimension: int = 768, patch: int = 8, depth: int = 12) -> None:
        super().__init__()
        self.tokenizer = PatchTokenizer3D(input_channels, dimension, patch)
        self.blocks = nn.ModuleList([MambaBlock(dimension) for _ in range(depth)])
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x: Tensor) -> Tensor:
        tokens, _ = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens).mean(dim=1)


@dataclass(frozen=True)
class BackboneSpecification:
    kind: BackboneKind
    input_channels: int = 5
    representation_dim: int = 768


def build_ablation_backbone(specification: BackboneSpecification) -> nn.Module:
    if specification.kind == BackboneKind.RESNET_3D:
        return ResNet3D(specification.input_channels, specification.representation_dim)
    if specification.kind == BackboneKind.VISION_MAMBA_3D:
        return VisionMamba3D(specification.input_channels, specification.representation_dim)
    if specification.kind == BackboneKind.DINOV2_3D:
        heads = max(value for value in range(1, 13) if specification.representation_dim % value == 0)
        return DinoVisionTransformer3D(specification.input_channels, specification.representation_dim, heads=heads)
    raise ValueError("Swin-UNETR is constructed by the primary model module")
