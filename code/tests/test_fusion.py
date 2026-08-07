import pytest
import torch
from neurodxfm.fusion import (
    CrossAttentionFusion,
    GatedFusion,
    LateFusion,
    ModalityDropout,
    enumerate_modality_sets,
    fusion_consistency,
    modality_contribution,
)


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    representations = torch.randn(3, 5, 32)
    available = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, False, False],
            [True, False, False, False, False],
        ]
    )
    return representations, available


def test_cross_attention_shape() -> None:
    representations, available = inputs()
    module = CrossAttentionFusion(5, 32, heads=4)
    output = module(representations, available)
    assert output.representation.shape == (3, 32)
    assert output.attention.shape == (3, 5)


def test_cross_attention_weights_sum_to_one() -> None:
    representations, available = inputs()
    module = CrossAttentionFusion(5, 32, heads=4)
    output = module(representations, available)
    assert torch.allclose(output.attention.sum(dim=-1), torch.ones(3))


def test_cross_attention_masks_missing_modalities() -> None:
    representations, available = inputs()
    module = CrossAttentionFusion(5, 32, heads=4)
    output = module(representations, available)
    assert output.attention[2, 1:].sum() == 0.0


def test_cross_attention_requires_t1() -> None:
    representations, available = inputs()
    available[0, 0] = False
    module = CrossAttentionFusion(5, 32, heads=4)
    with pytest.raises(ValueError):
        module(representations, available)


def test_cross_attention_validates_rank() -> None:
    module = CrossAttentionFusion(5, 32, heads=4)
    with pytest.raises(ValueError):
        module(torch.randn(3, 32), torch.ones(3, 5, dtype=torch.bool))


def test_late_fusion_shape() -> None:
    representations, available = inputs()
    module = LateFusion(5, 32, 2)
    assert module(representations, available).shape == (3, 2)


def test_gated_fusion_shape() -> None:
    representations, available = inputs()
    module = GatedFusion(5, 32)
    output = module(representations, available)
    assert output.representation.shape == (3, 32)


def test_gated_fusion_masks_missing() -> None:
    representations, available = inputs()
    module = GatedFusion(5, 32)
    output = module(representations, available)
    assert output.attention[2, 0] == 1.0


def test_modality_dropout_preserves_t1() -> None:
    representations, available = inputs()
    module = ModalityDropout(1.0)
    module.train()
    _, retained = module(representations, available)
    assert retained[:, 0].all()
    assert not retained[:, 1:].any()


def test_modality_dropout_disabled_for_evaluation() -> None:
    representations, available = inputs()
    module = ModalityDropout(1.0)
    module.eval()
    result, retained = module(representations, available)
    assert torch.equal(result, representations)
    assert torch.equal(retained, available)


def test_fusion_consistency_identity() -> None:
    values = torch.randn(4, 32)
    assert fusion_consistency(values, values).item() == pytest.approx(0.0, abs=1e-6)


def test_modality_contribution_shape() -> None:
    representations, available = inputs()
    module = GatedFusion(5, 32)
    contribution = modality_contribution(module, representations, available)
    assert contribution.shape == (3, 5)


def test_enumerate_modality_sets() -> None:
    combinations = enumerate_modality_sets(("t1", "flair", "pet"))
    assert len(combinations) == 4
    assert all("t1" in combination for combination in combinations)


def test_enumerate_requires_t1() -> None:
    with pytest.raises(ValueError):
        enumerate_modality_sets(("flair", "pet"))
