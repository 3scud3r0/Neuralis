import pytest
import torch

from neuralis.models.graph_surgery import GraphDefinition, compile_graph


def residual_definition():
    return {"input_width": 2, "seed": 42, "nodes": [
        {"id": "main", "operation": "linear", "inputs": ["input"], "width": 8},
        {"id": "activation", "operation": "tanh", "inputs": ["main"]},
        {"id": "skip", "operation": "linear", "inputs": ["input"], "width": 8},
        {"id": "residual", "operation": "add", "inputs": ["activation", "skip"]},
        {"id": "output", "operation": "linear", "inputs": ["residual"], "width": 2}], "output": "output"}


def test_residual_forward_matches_explicit_modules_and_gradients():
    model = compile_graph(residual_definition())
    inputs = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)
    output = model(inputs)
    expected = model.operations["output"](torch.tanh(model.operations["main"](inputs)) + model.operations["skip"](inputs))
    assert torch.equal(output, expected)
    gradients = torch.autograd.grad(output.square().sum(), inputs, retain_graph=True)[0]
    assert torch.equal(gradients, torch.autograd.grad(expected.square().sum(), inputs)[0])
    assert model.analysis["parameters"] == sum(parameter.numel() for parameter in model.parameters())


def test_graph_is_trainable_and_does_not_perturb_rng():
    state = torch.random.get_rng_state().clone()
    model = compile_graph(residual_definition())
    assert torch.equal(state, torch.random.get_rng_state())
    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([0, 1, 1, 0])
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    initial = float(torch.nn.functional.cross_entropy(model(inputs), targets).detach())
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial * .2


@pytest.mark.parametrize("mutation", ["cycle", "unknown", "mismatch", "duplicate", "disconnected"])
def test_invalid_graphs_are_rejected_before_allocation(mutation):
    definition = residual_definition()
    if mutation == "cycle":
        definition["nodes"][0]["inputs"] = ["residual"]
    elif mutation == "unknown":
        definition["nodes"][0]["inputs"] = ["missing"]
    elif mutation == "mismatch":
        definition["nodes"][2]["width"] = 3
    elif mutation == "duplicate":
        definition["nodes"][1]["id"] = "main"
    else:
        definition["nodes"].append({"id": "unused", "operation": "identity", "inputs": ["input"]})
    with pytest.raises(ValueError):
        GraphDefinition.model_validate(definition)


def test_recompile_removes_skip_preserves_weights_and_source():
    model = compile_graph(residual_definition())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    definition = residual_definition()
    definition["nodes"] = [node for node in definition["nodes"] if node["id"] not in {"skip", "residual"}]
    definition["nodes"][-1]["inputs"] = ["activation"]
    replacement, report = model.recompile(GraphDefinition.model_validate(definition))
    assert report["source_unchanged"] and report["retained_parameters"]
    for name, value in replacement.state_dict().items():
        assert torch.equal(value, before[name])
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    values = torch.tensor([[.3, -.4]])
    assert not torch.equal(model(values), replacement(values))


def test_concat_normalization_and_out_of_order_input_definition():
    definition = {"input_width": 2, "nodes": [
        {"id": "output", "operation": "layernorm", "inputs": ["joined"]},
        {"id": "joined", "operation": "concat", "inputs": ["main", "input"]},
        {"id": "main", "operation": "linear", "inputs": ["input"], "width": 3}], "output": "output"}
    graph = compile_graph(definition)
    result = graph(torch.tensor([[0., 1.]]))
    assert result.shape == (1, 5)
    assert torch.allclose(result.mean(-1), torch.zeros(1), atol=1e-6)
    assert graph.analysis["order"] == ["main", "joined", "output"]
