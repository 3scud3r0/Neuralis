"""Run a reproducible CPU toy experiment with no downloads."""
from collections import OrderedDict

import torch
from torch import nn

from axislab_poc import capture_activations, fit_axis, temporary_steering


def main():
    torch.manual_seed(7)

    # x[0] is a deliberately planted, causal feature. The other entries are noise.
    positive = torch.randn(256, 8) * 0.08
    negative = torch.randn(256, 8) * 0.08
    positive[:, 0] += 1.0
    negative[:, 0] -= 1.0
    positive[:, 1] *= 25.0
    negative[:, 1] *= 25.0

    model = nn.Sequential(OrderedDict([
        ("representation", nn.Identity()),
        ("decision", nn.Linear(8, 1, bias=False)),
    ]))
    with torch.no_grad():
        model.decision.weight.zero_()
        model.decision.weight[0, 0] = 1.0

    axis = fit_axis(positive, negative, denoise_variance=0.5)
    probe = torch.zeros(1, 8)
    base_output, base_activation = capture_activations(model, "representation", probe)
    with temporary_steering(model, "representation", axis, coefficient=2.0):
        edited_output = model(probe)
    restored_output = model(probe)

    print("Direções PCA removidas:", axis.removed_components)
    print("Projeção da referência:", round(float(axis.score(base_activation)[0]), 4))
    print("Saída antes:", round(float(base_output.item()), 4))
    print("Saída durante a intervenção:", round(float(edited_output.item()), 4))
    print("Saída após remover o hook:", round(float(restored_output.item()), 4))


if __name__ == "__main__":
    main()
