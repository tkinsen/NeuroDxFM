import torch
from neurodxfm.losses import (
    ClassBalancedCrossEntropy,
    ConcordanceCorrelationLoss,
    CoxPartialLikelihood,
    FocalLoss,
    anatomical_loss,
    binary_anchor_loss,
    dirichlet_kl,
    evidential_classification_loss,
    info_nce,
    masked_future_loss,
    modality_reconstruction_loss,
)


def test_info_nce_prefers_aligned_pairs() -> None:
    aligned = torch.eye(8)
    shuffled = aligned.roll(1, 0)
    good = info_nce(aligned, aligned)
    bad = info_nce(aligned, shuffled)
    assert good < bad


def test_info_nce_is_symmetric() -> None:
    first = torch.randn(5, 16)
    second = torch.randn(5, 16)
    assert torch.allclose(info_nce(first, second), info_nce(second, first))


def test_masked_future_loss_ignores_unselected_values() -> None:
    prediction = torch.tensor([[1.0, 10.0]])
    target = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, False]])
    assert masked_future_loss(prediction, target, mask).item() == 1.0


def test_masked_future_loss_detaches_target() -> None:
    prediction = torch.randn(3, 4, requires_grad=True)
    target = torch.randn(3, 4, requires_grad=True)
    masked_future_loss(prediction, target).backward()
    assert prediction.grad is not None
    assert target.grad is None


def test_modality_loss_uses_missing_channels() -> None:
    prediction = torch.ones(2, 3, 2, 2, 2)
    target = torch.zeros_like(prediction)
    missing = torch.tensor([[False, True, False], [True, False, False]])
    value = modality_reconstruction_loss(prediction, target, missing)
    assert value.item() > 0.0


def test_modality_loss_zero_without_missing_channels() -> None:
    prediction = torch.ones(2, 3, 2, 2, 2)
    target = torch.zeros_like(prediction)
    missing = torch.zeros(2, 3, dtype=torch.bool)
    assert modality_reconstruction_loss(prediction, target, missing).item() == 0.0


def test_anatomical_loss_accepts_validity_mask() -> None:
    prediction = torch.ones(2, 68)
    target = torch.ones(2, 68)
    valid = torch.tensor([[True] * 68, [False] * 68])
    assert anatomical_loss(prediction, target, valid).item() == 0.0


def test_binary_anchor_masks_unavailable_labels() -> None:
    logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
    labels = torch.tensor([0, 0])
    valid = torch.tensor([True, False])
    assert binary_anchor_loss(logits, labels, valid).item() < 0.001


def test_dirichlet_kl_is_zero_at_uniform_prior() -> None:
    alpha = torch.ones(4, 2)
    assert torch.allclose(dirichlet_kl(alpha, 2), torch.tensor(0.0))


def test_evidential_loss_has_gradient() -> None:
    alpha = torch.full((4, 2), 2.0, requires_grad=True)
    target = torch.tensor([0, 1, 0, 1])
    loss = evidential_classification_loss(alpha, target)
    loss.backward()
    assert alpha.grad is not None


def test_class_balanced_cross_entropy() -> None:
    objective = ClassBalancedCrossEntropy(torch.tensor([100, 10]))
    logits = torch.randn(8, 2, requires_grad=True)
    target = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    objective(logits, target).backward()
    assert logits.grad is not None


def test_focal_loss() -> None:
    objective = FocalLoss()
    good = objective(torch.tensor([[10.0, -10.0]]), torch.tensor([0]))
    bad = objective(torch.tensor([[-10.0, 10.0]]), torch.tensor([0]))
    assert good < bad


def test_cox_loss_orders_risk() -> None:
    objective = CoxPartialLikelihood()
    time = torch.tensor([1.0, 2.0, 3.0])
    event = torch.ones(3)
    correct = objective(torch.tensor([3.0, 2.0, 1.0]), time, event)
    wrong = objective(torch.tensor([1.0, 2.0, 3.0]), time, event)
    assert correct < wrong


def test_concordance_correlation_identity() -> None:
    values = torch.arange(10).float()
    assert torch.allclose(ConcordanceCorrelationLoss()(values, values), torch.tensor(0.0))
