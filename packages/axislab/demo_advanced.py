"""AxisLab 0.2 demo: representation, sparse features, causal and memory edits.

No downloads, deterministic synthetic examples, CPU only.
"""
import torch
from torch import nn

from axislab_poc import (
    AxisBank, TopKSAE, activation_patch, batched_ridge_update,
    evaluate_axis, fit_paired_axis, fit_ridge_probe, forward_moments,
    multi_axis_steering, projection_ablation, rank_one_update,
    sae_delta_intervention, train_topk_sae, weight_edit,
)


def main():
    torch.manual_seed(42)
    # Construct paired examples with a ground-truth direction on feature 0.
    baseline = torch.randn(100, 4)
    positive = baseline.clone()
    positive[:, 0] += 3.0
    axis = fit_paired_axis(positive[:60], baseline[:60])
    held_out = evaluate_axis(axis, positive[60:], baseline[60:])
    print("[Representations] Held-out AUC:", round(held_out['auc'], 4))

    # Causal steering of a minimal network, with automatic hook cleanup.
    network = nn.Sequential(nn.Identity(), nn.Linear(4, 1, bias=False))
    with torch.no_grad():
        network[1].weight.copy_(torch.tensor([[1., 0., 0., 0.]]))
    probe = torch.zeros(1, 4)
    first = network(probe).item()
    bank = AxisBank({"semantic": axis})
    dirs = torch.stack([bank.axes["semantic"].direction])
    with multi_axis_steering(network, '0', dirs, torch.tensor([2.0])):
        during = network(probe).item()
    after = network(probe).item()
    print("[Causal steering] baseline / edited / restored:", first, round(during, 4), after)

    # A trainable, deliberately small SAE. This is not a pretrained SAE Lens model.
    observations = torch.randn(160, 4)
    sae = TopKSAE(d_model=4, d_sae=12, k=3)
    report = train_topk_sae(sae, observations, steps=80, batch_size=32, lr=0.01)
    activation = observations[:1]
    edited_activation = sae_delta_intervention(sae, activation, feature_index=0, coefficient=1.0)
    print("[SAE] final reconstruction MSE:", round(report.final_mse, 4))
    print("[SAE] feature edit changed activation:", not torch.allclose(activation, edited_activation))

    # Exact linear editing with rollback. This is not complete ROME/MEMIT.
    layer = nn.Linear(4, 2, bias=False)
    key = torch.tensor([1.0, 0.0, 0.0, 0.0])
    target = torch.tensor([2.0, -1.0])
    original = layer(key).detach().clone()
    delta = rank_one_update(layer, key, target)
    with weight_edit(layer, delta, max_relative_norm=100):
        print("[Memory edit] target reached:", torch.allclose(layer(key), target, atol=1e-5))
    print("[Memory edit] weights restored:", torch.allclose(layer(key), original))

    # Do not keep [batch,tokens,width] caches: aggregate simple online statistics.
    moments = forward_moments(nn.Sequential(nn.Identity()), '0',
                              [torch.randn(2, 3, 4), torch.randn(2, 3, 4)])
    print("[Streaming] observations:", moments.count)


if __name__ == "__main__":
    main()
