from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .settings import ModelSettings


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    return value, value, value


def window_partition(x: Tensor, window: tuple[int, int, int]) -> Tensor:
    batch, depth, height, width, channels = x.shape
    wd, wh, ww = window
    x = x.view(batch, depth // wd, wd, height // wh, wh, width // ww, ww, channels)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wd * wh * ww, channels)


def window_reverse(
    windows: Tensor,
    window: tuple[int, int, int],
    batch: int,
    depth: int,
    height: int,
    width: int,
) -> Tensor:
    wd, wh, ww = window
    channels = windows.shape[-1]
    x = windows.view(batch, depth // wd, height // wh, width // ww, wd, wh, ww, channels)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(batch, depth, height, width, channels)


class DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random.floor() / keep


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, output: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, output)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(self.dropout(self.activation(self.fc1(x)))))


class RelativePositionBias3D(nn.Module):
    def __init__(self, window: tuple[int, int, int], heads: int) -> None:
        super().__init__()
        self.window = window
        wd, wh, ww = window
        size = (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1)
        self.table = nn.Parameter(torch.zeros(size, heads))
        coordinates = torch.stack(torch.meshgrid(torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij"))
        flattened = torch.flatten(coordinates, 1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += wd - 1
        relative[:, :, 1] += wh - 1
        relative[:, :, 2] += ww - 1
        relative[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative[:, :, 1] *= 2 * ww - 1
        self.register_buffer("index", relative.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self) -> Tensor:
        tokens = self.index.shape[0]
        return self.table[self.index.reshape(-1)].view(tokens, tokens, -1).permute(2, 0, 1)


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, window: tuple[int, int, int], heads: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection_dropout = nn.Dropout(dropout)
        self.relative = RelativePositionBias3D(window, heads)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        attention = attention + self.relative().unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            attention = attention.view(batch_windows // windows, windows, self.heads, tokens, tokens)
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(-1, self.heads, tokens, tokens)
        attention = self.attention_dropout(attention.softmax(dim=-1))
        output = (attention @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.projection_dropout(self.projection(output))


class SwinBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        window: tuple[int, int, int],
        shifted: bool,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.window = window
        self.shift = tuple(v // 2 if shifted else 0 for v in window)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention3D(dim, window, heads, dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4, dim, dropout)

    def _pad(self, x: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        _, depth, height, width, _ = x.shape
        pd = (self.window[0] - depth % self.window[0]) % self.window[0]
        ph = (self.window[1] - height % self.window[1]) % self.window[1]
        pw = (self.window[2] - width % self.window[2]) % self.window[2]
        padded = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))
        return padded, (depth, height, width)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        x, original = self._pad(x)
        batch, depth, height, width, _ = x.shape
        if any(self.shift):
            x = torch.roll(x, shifts=tuple(-v for v in self.shift), dims=(1, 2, 3))
        windows = window_partition(x, self.window)
        windows = self.attention(windows)
        x = window_reverse(windows, self.window, batch, depth, height, width)
        if any(self.shift):
            x = torch.roll(x, shifts=self.shift, dims=(1, 2, 3))
        od, oh, ow = original
        x = x[:, :od, :oh, :ow, :]
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbedding3D(nn.Module):
    def __init__(self, channels: int, dim: int, patch: tuple[int, int, int]) -> None:
        super().__init__()
        self.projection = nn.Conv3d(channels, dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x).permute(0, 2, 3, 4, 1).contiguous()
        return self.norm(x)


class PatchMerging3D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim * 8)
        self.reduction = nn.Linear(dim * 8, dim * 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        _, depth, height, width, _ = x.shape
        x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2, 0, depth % 2))
        pieces = [x[:, i::2, j::2, k::2, :] for i in range(2) for j in range(2) for k in range(2)]
        return self.reduction(self.norm(torch.cat(pieces, dim=-1)))


class EncoderStage(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, window: tuple[int, int, int], rates: list[float]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock3D(dim, heads, window, index % 2 == 1, 0.0, rates[index]) for index in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class SwinEncoder3D(nn.Module):
    def __init__(self, settings: ModelSettings) -> None:
        super().__init__()
        self.embedding = PatchEmbedding3D(settings.input_channels, settings.embedding_dim, settings.patch_size)
        total = sum(settings.depths)
        rates = torch.linspace(0.0, settings.stochastic_depth, total).tolist()
        stages: list[nn.Module] = []
        mergers: list[nn.Module] = []
        cursor = 0
        for index, depth in enumerate(settings.depths):
            dim = settings.embedding_dim * (2**index)
            stages.append(EncoderStage(dim, depth, settings.heads[index], settings.window_size, rates[cursor : cursor + depth]))
            cursor += depth
            if index < len(settings.depths) - 1:
                mergers.append(PatchMerging3D(dim))
        self.stages = nn.ModuleList(stages)
        self.mergers = nn.ModuleList(mergers)
        self.norm = nn.LayerNorm(settings.representation_dim)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.mergers):
                x = self.mergers[index](x)
        return self.norm(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x).mean(dim=(1, 2, 3))


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        return -float(ctx.coefficient) * gradient, None


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, months: Tensor) -> Tensor:
        half = self.dim // 2
        scale = torch.exp(
            torch.arange(half, device=months.device) * (-torch.log(torch.tensor(10000.0, device=months.device)) / max(half - 1, 1))
        )
        angles = months.float().unsqueeze(-1) * scale.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ProtocolHead(nn.Module):
    def __init__(self, dim: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, classes))

    def forward(self, x: Tensor, coefficient: float) -> Tensor:
        return self.network(GradientReverse.apply(x, coefficient))


class TrajectoryHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.time = TimeEmbedding(dim)
        self.predictor = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, representation: Tensor, months: Tensor) -> Tensor:
        return self.predictor(torch.cat((representation, self.time(months)), dim=-1))


class ReconstructionHead(nn.Module):
    def __init__(self, dim: int, channels: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(nn.ConvTranspose3d(dim, dim // 2, 2, 2), nn.GELU(), nn.Conv3d(dim // 2, channels, 1))

    def forward(self, features: Tensor) -> Tensor:
        return self.decoder(features.permute(0, 4, 1, 2, 3))


class EvidentialHead(nn.Module):
    def __init__(self, dim: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim // 2, classes))

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(self.network(x)) + 1.0


class NeuroDxOutput(NamedTuple):
    representation: Tensor
    projection: Tensor
    protocol_logits: Tensor
    trajectory: Tensor | None
    reconstruction: Tensor
    anatomy: Tensor
    amyloid_logits: Tensor
    evidence: Tensor


class NeuroDxFM(nn.Module):
    def __init__(self, settings: ModelSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or ModelSettings()
        dim = self.settings.representation_dim
        self.encoder = SwinEncoder3D(self.settings)
        self.projector = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.settings.projection_dim))
        self.protocol = ProtocolHead(dim, self.settings.protocol_classes)
        self.trajectory = TrajectoryHead(dim)
        self.reconstruction = ReconstructionHead(dim, self.settings.input_channels)
        self.anatomy = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, self.settings.anatomical_targets))
        self.amyloid = nn.Linear(dim, 2)
        self.evidence = EvidentialHead(dim, 2)

    def forward(self, volumes: Tensor, months: Tensor | None = None, grl: float = 1.0) -> NeuroDxOutput:
        features = self.encoder.forward_features(volumes)
        representation = features.mean(dim=(1, 2, 3))
        trajectory = self.trajectory(representation, months) if months is not None else None
        return NeuroDxOutput(
            representation,
            F.normalize(self.projector(representation), dim=-1),
            self.protocol(representation, grl),
            trajectory,
            self.reconstruction(features),
            self.anatomy(representation),
            self.amyloid(representation),
            self.evidence(representation),
        )

    @torch.no_grad()
    def encode(self, volumes: Tensor) -> Tensor:
        return self.encoder(volumes)
