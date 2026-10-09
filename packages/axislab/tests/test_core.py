import pytest
import torch
from torch import nn

from axislab_poc import Axis, capture_activations, fit_axis, temporary_steering


def axis_for_dim(d=3):
    v = torch.zeros(d)
    v[0] = 1.0
    return Axis(direction=v, control_mean=torch.zeros(d), removed_components=0)


def test_fit_axis_recovers_clean_direction():
    positive = torch.tensor([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    control = torch.tensor([[-1., 0., 0.], [-2., 0., 0.], [-3., 0., 0.]])
    axis = fit_axis(positive, control, denoise_variance=0)
    assert torch.allclose(axis.direction, torch.tensor([1., 0., 0.]))
    assert axis.score(positive).mean() > axis.score(control).mean()


def test_steering_effect_and_cleanup():
    model = nn.Sequential(nn.Identity(), nn.Linear(3, 1, bias=False))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[1., 0., 0.]]))
    x = torch.zeros(1, 3)
    with temporary_steering(model, '0', axis_for_dim(), coefficient=2):
        assert model(x).item() == pytest.approx(2.0)
    assert model(x).item() == pytest.approx(0.0)


def test_hook_cleanup_after_error():
    model = nn.Sequential(nn.Identity())
    with pytest.raises(RuntimeError):
        with temporary_steering(model, '0', axis_for_dim(), coefficient=1):
            raise RuntimeError('test failure')
    assert len(model[0]._forward_hooks) == 0


def test_capture_and_last_token():
    model = nn.Sequential(nn.Identity())
    x = torch.zeros(2, 4, 3)
    out, acts = capture_activations(model, '0', x)
    assert torch.allclose(out, acts)
    with temporary_steering(model, '0', axis_for_dim(), coefficient=3, position='last_token'):
        y = model(x)
        assert torch.allclose(y[:, :-1, :], x[:, :-1, :])
        assert torch.allclose(y[:, -1, 0], torch.full((2,), 3.))
    assert torch.allclose(model(x), x)


def test_invalid_shapes():
    with pytest.raises(ValueError):
        fit_axis(torch.ones(2, 3), torch.ones(2, 4))
    with pytest.raises(ValueError):
        fit_axis(torch.ones(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError):
        fit_axis(torch.ones(2, 3), torch.zeros(2, 3), denoise_variance=1.)


def test_tuple_output():
    class Inner(nn.Module):
        def forward(self, x):
            return x, torch.tensor(1)
    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()
        def forward(self, x):
            return self.inner(x)[0]
    model = Outer()
    x = torch.zeros(1, 3)
    with temporary_steering(model, 'inner', axis_for_dim(), coefficient=1):
        assert model(x)[0, 0].item() == pytest.approx(1.0)
