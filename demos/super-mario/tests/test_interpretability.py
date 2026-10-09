"""Attribution completeness and causal-hook cleanup tests."""
import numpy as np
import pytest
import torch
from torch import nn
from mario_rl.interpretability import activation_intervention, grad_cam, gradient_attribution, heatmap_rgb, integrated_gradients
from mario_rl.model import ActorCritic


class LinearPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiplier = nn.Parameter(torch.tensor(2.0))
    def forward(self, observations):
        value = observations.flatten(1).sum(dim=1) * self.multiplier
        return torch.stack((value, -value), dim=1), value


def test_integrated_gradients_linear_completeness():
    model = LinearPolicy()
    observation = np.array([[[1, 2], [3, 4]]], dtype=np.uint8)
    result = integrated_gradients(model, observation, 0, 8)
    assert np.array_equal(result.attribution, observation * 2)
    assert result.score == 20
    assert result.baseline_score == 0
    assert result.completeness_residual == 0
    gradient = gradient_attribution(model, observation, 0, False)
    assert np.all(gradient.attribution == 2)


def test_identity_and_zero_intervention_cleanup():
    torch.set_num_threads(2)
    torch.manual_seed(8)
    model = ActorCritic((4, 84, 84), 7)
    observation = torch.full((1, 4, 84, 84), 128, dtype=torch.uint8)
    original = model(observation)[0].detach()
    with activation_intervention(model, "trunk.1", scale=1):
        assert torch.equal(model(observation)[0], original)
    with activation_intervention(model, "trunk.1", scale=0):
        changed = model(observation)[0]
        assert not torch.equal(changed, original)
    assert torch.equal(model(observation)[0], original)
    assert not model.trunk[1]._forward_hooks


def test_invalid_intervention_removes_hook_after_failure():
    torch.set_num_threads(2)
    model = ActorCritic((4, 84, 84), 7)
    with pytest.raises(ValueError, match="outside"):
        with activation_intervention(model, "actor", channels=(999,)):
            model(torch.zeros((1, 4, 84, 84), dtype=torch.uint8))
    assert not model.actor._forward_hooks


def test_grad_cam_is_finite_and_bounded_output_shape():
    torch.set_num_threads(2)
    model = ActorCritic((4, 84, 84), 7)
    result = grad_cam(model, np.full((4, 84, 84), 128, dtype=np.uint8), 0)
    assert result.attribution.shape == (1, 84, 84)
    assert np.isfinite(result.attribution).all()
    assert (result.attribution >= 0).all()
    image = heatmap_rgb(result.attribution)
    assert image.shape == (84, 84, 3) and image.dtype == np.uint8
    assert not model.encoder[4]._forward_hooks
