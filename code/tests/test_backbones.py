import pytest
import torch
from neurodxfm.backbones import (
    BackboneKind,
    BackboneSpecification,
    ConvNormActivation,
    DinoVisionTransformer3D,
    MambaBlock,
    PatchTokenizer3D,
    ResidualBlock3D,
    ResNet3D,
    RotaryPosition3D,
    SelectiveStateSpace,
    SelfAttentionBlock,
    VisionMamba3D,
    build_ablation_backbone,
)
from torch import nn


def test_conv_norm_activation_shape() -> None:
    module = ConvNormActivation(2, 8, 3)
    output = module(torch.randn(1, 2, 8, 8, 8))
    assert output.shape == (1, 8, 8, 8, 8)


def test_conv_norm_activation_stride() -> None:
    module = ConvNormActivation(2, 8, 3, 2)
    output = module(torch.randn(1, 2, 8, 8, 8))
    assert output.shape == (1, 8, 4, 4, 4)


def test_residual_block_identity_shape() -> None:
    module = ResidualBlock3D(8, 8)
    output = module(torch.randn(1, 8, 8, 8, 8))
    assert output.shape == (1, 8, 8, 8, 8)


def test_residual_block_downsample_shape() -> None:
    module = ResidualBlock3D(8, 16, 2)
    output = module(torch.randn(1, 8, 8, 8, 8))
    assert output.shape == (1, 16, 4, 4, 4)


def test_resnet_shape() -> None:
    module = ResNet3D(1, 32, (1, 1, 1, 1))
    output = module(torch.randn(1, 1, 32, 32, 32))
    assert output.shape == (1, 32)


def test_patch_tokenizer_shape() -> None:
    module = PatchTokenizer3D(1, 16, 4)
    tokens, shape = module(torch.randn(2, 1, 16, 20, 24))
    assert tokens.shape == (2, 120, 16)
    assert shape == (4, 5, 6)


def test_rotary_position_shape() -> None:
    module = RotaryPosition3D(16)
    values = torch.randn(2, 10, 16)
    assert module(values).shape == values.shape


def test_rotary_position_changes_values() -> None:
    module = RotaryPosition3D(16)
    values = torch.randn(2, 10, 16)
    assert not torch.equal(module(values), values)


def test_attention_block_shape() -> None:
    module = SelfAttentionBlock(16, 4, 0.0)
    values = torch.randn(2, 10, 16)
    assert module(values).shape == values.shape


def test_dino_shape() -> None:
    module = DinoVisionTransformer3D(1, 32, 4, 2, 4, 2)
    output = module(torch.randn(1, 1, 16, 16, 16))
    assert output.shape == (1, 32)


def test_selective_state_space_shape() -> None:
    module = SelectiveStateSpace(16, 4, 1)
    values = torch.randn(2, 8, 16)
    assert module(values).shape == values.shape


def test_selective_state_space_gradient() -> None:
    module = SelectiveStateSpace(16, 4, 1)
    values = torch.randn(2, 8, 16, requires_grad=True)
    module(values).sum().backward()
    assert values.grad is not None


def test_mamba_block_shape() -> None:
    module = MambaBlock(16)
    values = torch.randn(2, 8, 16)
    assert module(values).shape == values.shape


def test_vision_mamba_shape() -> None:
    module = VisionMamba3D(1, 16, 4, 2)
    output = module(torch.randn(1, 1, 16, 16, 16))
    assert output.shape == (1, 16)


@pytest.mark.parametrize(
    "kind,expected",
    [
        (BackboneKind.RESNET_3D, ResNet3D),
        (BackboneKind.VISION_MAMBA_3D, VisionMamba3D),
        (BackboneKind.DINOV2_3D, DinoVisionTransformer3D),
    ],
)
def test_build_ablation_backbone(kind: BackboneKind, expected: type[nn.Module]) -> None:
    module = build_ablation_backbone(BackboneSpecification(kind, 1, 32))
    assert isinstance(module, expected)


def test_build_rejects_primary_backbone() -> None:
    with pytest.raises(ValueError):
        build_ablation_backbone(BackboneSpecification(BackboneKind.SWIN_UNETR, 1, 32))


def test_resnet_parameter_gradient() -> None:
    module = ResNet3D(1, 16, (1, 1, 1, 1))
    output = module(torch.randn(1, 1, 32, 32, 32))
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_dino_parameter_gradient() -> None:
    module = DinoVisionTransformer3D(1, 16, 4, 1, 4, 2)
    output = module(torch.randn(1, 1, 8, 8, 8))
    output.sum().backward()
    assert module.class_token.grad is not None


def test_mamba_parameter_gradient() -> None:
    module = VisionMamba3D(1, 16, 4, 1)
    output = module(torch.randn(1, 1, 8, 8, 8))
    output.sum().backward()
    assert any(parameter.grad is not None for parameter in module.parameters())
