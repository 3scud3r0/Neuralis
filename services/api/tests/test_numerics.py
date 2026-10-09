import torch
from torch import nn

from neuralis.analysis.experiments import attribution, capture_clean, execute_experiment
from neuralis.core.schemas import AttributionRequest, ExperimentRequest, InferenceRequest, InterventionSpec
from neuralis.datasets.synthetic import DataSplit, DatasetBundle, dataset_for
from neuralis.models.registry import CausalAttention, build_model
from neuralis.runtime.capture import CaptureSession, summarize_tensor
from neuralis.core.schemas import CaptureConfig


def controlled_data():
    x = torch.tensor([[1., 2.], [2., 1.]])
    y = torch.tensor([0, 1])
    split = DataSplit(x, y)
    return DatasetBundle("controlled", split, split, ("a", "b"), "classification", "test")


def linear_model():
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[2., 1.], [-1., 3.]]))
    return model


def test_summary_matches_exact_tensor():
    tensor = torch.arange(12.).reshape(3, 4)
    summary = summarize_tensor(tensor, max_elements=32, bins=4)
    assert summary["shape"] == [3, 4]
    assert summary["mean"] == float(tensor.mean())
    assert summary["std"] == float(tensor.std(unbiased=False))
    assert summary["values"] == tensor.flatten().tolist()
    assert not summary["sampled"]
    assert sum(summary["histogram"]["counts"]) == 12


def test_summary_nonfinite_and_budget():
    tensor = torch.arange(1000.)
    tensor[0] = float("nan")
    summary = summarize_tensor(tensor, max_elements=17)
    assert summary["nonfinite"] == 1
    assert summary["captured_elements"] <= 17
    assert summary["sampled"]
    assert all(torch.isfinite(torch.tensor(summary["values"])))


def test_hooks_capture_real_activation_and_gradient_and_cleanup():
    model = linear_model()
    inputs = torch.tensor([[1., 2.]], requires_grad=True)
    with CaptureSession(model, CaptureConfig(every=1, max_elements=100)) as capture:
        capture.begin(1)
        output = model(inputs)
        output.sum().backward()
        records = capture.finish()["tensors"]
        activation = next(record for record in records if record["kind"] == "activation")
        gradient = next(record for record in records if record["kind"] == "activation-gradient")
        assert activation["values"] == output.detach().flatten().tolist()
        assert gradient["values"] == [1., 1.]
    assert all(not module._forward_hooks for module in model.modules())


def test_capture_is_bounded():
    model = build_model("cnn-shapes")
    with CaptureSession(model, CaptureConfig(every=1, max_elements=32, max_tensors=2)) as capture:
        capture.begin(1)
        model(torch.rand(4, 1, 16, 16)).sum().backward()
        result = capture.finish()
        assert len(result["tensors"]) == 2
        assert result["dropped"] > 0
        assert not result["complete"]


def test_causal_attention_matches_mask_and_no_future_dependency():
    attention = CausalAttention(12, 3).eval()
    captured = []
    handle = attention.attention_probs.register_forward_hook(lambda mod, inputs, output: captured.append(output.detach()))
    inputs = torch.randn(2, 6, 12)
    first = attention(inputs)
    probabilities = captured[0]
    assert torch.allclose(probabilities.sum(-1), torch.ones(2, 3, 6), atol=1e-6)
    assert torch.equal(probabilities.triu(1), torch.zeros_like(probabilities))
    changed = inputs.clone()
    changed[:, 4:] += 100
    second = attention(changed)
    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-6)
    handle.remove()


def test_integrated_gradients_linear_completeness():
    model = linear_model()
    data = controlled_data()
    result = attribution(model, data, AttributionRequest(input=InferenceRequest(sample_indices=[0, 1]),
                                                        target_class=0, integration_steps=8))
    assert torch.allclose(torch.tensor(result["values"]), torch.tensor([[2., 2.], [4., 1.]]))
    assert max(abs(value) for value in result["completeness"]["residual"]) < 1e-6


def test_ablation_changes_only_isolated_model_output():
    model = linear_model()
    before = {name: parameter.clone() for name, parameter in model.state_dict().items()}
    result = execute_experiment(model, controlled_data(), ExperimentRequest(run_id="test",
                                input=InferenceRequest(sample_indices=[0]),
                                interventions=[InterventionSpec(module="0", operation="ablate", indices=[0])]))
    assert result["intervention"]["logits"][0][0] == 0
    assert result["intervention"]["logits"][0][1] == result["baseline"]["logits"][0][1]
    assert result["metrics"]["logit_l2"] == 4
    assert all(torch.equal(before[name], parameter) for name, parameter in model.state_dict().items())
    assert not model[0]._forward_hooks


def test_activation_patch_exact_recovery_control():
    model = linear_model()
    result = execute_experiment(model, controlled_data(), ExperimentRequest(run_id="test",
                                input=InferenceRequest(sample_indices=[0]), clean_input=InferenceRequest(sample_indices=[1]),
                                interventions=[InterventionSpec(module="0", operation="patch")]))
    assert result["metrics"]["patch_recovery"] == 1
    assert result["intervention"]["logits"] == result["clean"]["logits"]
    assert not model[0]._forward_hooks


def test_dataset_seed_reproducibility_and_independent_splits():
    first = dataset_for("mlp-spiral", 41)
    second = dataset_for("mlp-spiral", 41)
    assert torch.equal(first.train.inputs, second.train.inputs)
    assert not torch.equal(first.train.inputs[:len(first.validation)], first.validation.inputs)
