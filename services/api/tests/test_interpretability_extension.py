"""Numerical and lifecycle tests using local PyTorch models, without weights downloads."""

import pytest
import torch
from torch import nn

from neuralis.models.registry import TinyTransformer
from neuralis.interpretability.activation_cache import ActivationCache
from neuralis.interpretability.sae.dictionary_learning import TopKSAE, fit_dictionary
from neuralis.interpretability.interventions.steering_vectors import SteeringSession
from neuralis.interpretability.circuit_discovery.activation_patching import (
    ActivationPatcher, run_patching_experiment,
)
from neuralis.interpretability.lenses import infer_tokens, logit_lens, generate_tokens, axislab_logit_lens


class TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        self.block = nn.Identity()
        self.output = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor([[0., 0.], [1., 0.], [2., 0.], [3., 0.]]))
            self.output.weight.copy_(torch.tensor([[1., 0.], [-1., 0.]]))

    def forward(self, tokens):
        return self.output(self.block(self.embedding(tokens)))


def tiny_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return TinyTransformer(vocabulary=6, width=8, depth=1, heads=2, context=8)


def test_cache_is_selective_detached_cpu_and_dynamic():
    model = TokenModel()
    inputs = torch.tensor([[1, 2]])
    with ActivationCache(model, ["embedding"], positions=[-1]) as cache:
        baseline = model(inputs)
        record = cache.records[0]
        assert record.tensor.device.type == "cpu"
        assert not record.tensor.requires_grad and record.tensor.grad_fn is None
        assert record.positions == (1,)
        torch.testing.assert_close(record.tensor, torch.tensor([[[2., 0.]]]))
        cache.set_modules(["block"])
        model(inputs)
        assert [record.module for record in cache.records] == ["embedding", "block"]
        assert len(model.embedding._forward_hooks) == 0
    assert len(model.block._forward_hooks) == 0
    torch.testing.assert_close(model(inputs), baseline)
    with torch.no_grad():
        model.embedding.weight.zero_()
    torch.testing.assert_close(cache.latest("embedding"), torch.tensor([[[2., 0.]]]))
    cache.clear()
    assert not cache.records and cache.retained_bytes == 0


def test_cache_budget_skips_whole_tensors_and_records_cleanup_on_failure():
    model = TokenModel()
    with ActivationCache(model, ["block"], max_bytes=8, max_records=1) as cache:
        model(torch.tensor([[1]]))
        model(torch.tensor([[2]]))
        assert cache.retained_bytes == 8 and cache.dropped == 1
        assert cache.snapshot()["complete"] is False
    with pytest.raises(RuntimeError):
        with ActivationCache(model, ["block"]) as failed:
            model(torch.tensor([[1]]))
            raise RuntimeError("simulated downstream error")
    assert not failed.active and len(model.block._forward_hooks) == 0
    with ActivationCache(model, ["block"], max_bytes=7) as too_small:
        model(torch.tensor([[1]]))
    assert not too_small.records and too_small.dropped == 1


def test_invalid_dynamic_selection_preserves_existing_hooks():
    model = TokenModel()
    with ActivationCache(model, ["block"]) as cache:
        with pytest.raises(ValueError, match="Unknown"):
            cache.set_modules(["missing"])
        model(torch.tensor([[1]]))
        assert cache.active and len(cache.records) == 1
    assert len(model.block._forward_hooks) == 0


def test_cache_position_error_removes_hooks():
    model = TokenModel()
    with pytest.raises(ValueError, match="outside"):
        with ActivationCache(model, ["block"], positions=[2]):
            model(torch.tensor([[1]]))
    assert not model.block._forward_hooks


def test_topk_dictionary_enforces_sparsity_and_preserves_rng():
    torch.manual_seed(30)
    state = torch.random.get_rng_state().clone()
    model = TopKSAE(4, 8, 2, seed=99)
    assert torch.equal(state, torch.random.get_rng_state())
    data = torch.ones(10, 4)
    reconstructed, encoded = model(data)
    assert reconstructed.shape == data.shape
    assert ((encoded > 0).sum(-1) <= 2).all()
    torch.testing.assert_close(model.decoder.weight.norm(dim=0), torch.ones(8))
    assert all(entry["user_label"] is None for entry in model.feature_catalog())
    model.set_user_labels({2: "Candidate feature supplied by researcher"})
    assert model.feature_catalog()[2]["label_source"] == "user"
    assert model.feature_catalog()[0]["name"] == "Feature 0"
    with pytest.raises(ValueError):
        model.set_user_labels({8: "bad"})


def test_dictionary_training_improves_actual_heldout_reconstruction():
    generator = torch.Generator().manual_seed(5)
    latent = torch.randn(300, 2, generator=generator)
    mixing = torch.tensor([[1., 0., .5, -.2], [0., 1., .2, .7]])
    data = latent @ mixing
    model, report = fit_dictionary(data, features=8, k=4, steps=160, seed=12,
                                   user_labels={1: "User hypothesis"})
    metrics = report["metrics"]
    assert metrics["validation_mse"] < metrics["initial_validation_mse"] * .25
    assert metrics["variance_explained"] > .9
    assert metrics["training_observations"] == 240 and metrics["validation_observations"] == 60
    assert metrics["mean_active_features"] <= 4
    assert len(report["features"]) == 8
    assert sum(item["user_label"] is not None for item in report["features"]) == 1
    torch.testing.assert_close(model.direction(1), model.decoder.weight[:, 1].detach())
    second, repeated = fit_dictionary(data, features=8, k=4, steps=160, seed=12)
    assert repeated["metrics"] == metrics
    torch.testing.assert_close(second.decoder.weight, model.decoder.weight)


@pytest.mark.parametrize("kwargs", [{"k": 9, "features": 8}, {"steps": 2001}, {"learning_rate": float("nan")}])
def test_dictionary_rejects_unbounded_or_invalid_training(kwargs):
    with pytest.raises(ValueError):
        fit_dictionary(torch.ones(10, 4), **kwargs)


def test_position_steering_sums_features_and_restores_exact_baseline():
    model = TokenModel()
    tokens = torch.tensor([[1, 2]])
    baseline = model(tokens).detach()
    vectors = {0: torch.tensor([1., 0.]), 1: torch.tensor([0., 2.])}
    with SteeringSession(model, "block", vectors=vectors, strengths={0: .5, 1: -.25}, positions=[-1]) as steering:
        with ActivationCache(model, ["block"]) as cache:
            changed = model(tokens)
        torch.testing.assert_close(changed[0, 0], baseline[0, 0])
        torch.testing.assert_close(changed[0, 1], baseline[0, 1] + torch.tensor([.5, -.5]))
        torch.testing.assert_close(cache.latest("block")[0, 1], torch.tensor([2.5, -.5]))
        steering.update_strengths({0: -1})
        torch.testing.assert_close(model(tokens)[0, 1], baseline[0, 1] + torch.tensor([-1., 1.]))
    assert steering.applications == 2 and not model.block._forward_hooks
    torch.testing.assert_close(model(tokens), baseline)


def test_steering_error_does_not_leak_hooks_or_mutate_weights():
    model = TokenModel()
    weights = model.embedding.weight.detach().clone()
    with pytest.raises(ValueError):
        with SteeringSession(model, "block", vectors={0: torch.ones(3)}, strengths={0: 1}):
            model(torch.tensor([[1]]))
    assert not model.block._forward_hooks
    torch.testing.assert_close(model.embedding.weight, weights)


def test_patching_measures_partial_recovery_and_restores_modes():
    model = TokenModel().train()
    model.embedding.eval()
    report = run_patching_experiment(model, torch.tensor([[1, 2]]), torch.tensor([[0, 0]]),
                                    "block", lambda logits: logits[..., 0].sum(), positions=[0])
    assert report["clean_score"] == 3 and report["corrupt_score"] == 0
    assert report["patched_score"] == 1 and report["restored_score"] == 0
    assert report["recovery"] == pytest.approx(1 / 3)
    assert model.training and not model.embedding.training
    assert not model.block._forward_hooks


def test_patching_maps_source_positions_and_handles_zero_difference():
    model = TokenModel()
    clean = model.embedding(torch.tensor([[1, 2]])).detach()
    with ActivationPatcher(model, "block", clean, positions=[0], source_positions=[1]):
        output = model(torch.tensor([[0, 0]]))
    assert output[0, 0, 0] == 2 and output[0, 1, 0] == 0
    report = run_patching_experiment(model, torch.tensor([[0]]), torch.tensor([[0]]),
                                    "block", lambda logits: logits.sum())
    assert report["recovery"] is None
    with pytest.raises(ValueError):
        with ActivationPatcher(model, "block", clean):
            model(torch.tensor([[0]]))
    assert not model.block._forward_hooks


def test_inference_and_final_residual_lens_match_actual_logits():
    model = tiny_model().train()
    model.embedding.eval()
    result = infer_tokens(model, [[1, 2, 3]], modules=["blocks.0"])
    record = result["activations"]["records"][0]
    residual = torch.tensor(record["values"])
    lens = logit_lens(model, residual, top_k=3)
    torch.testing.assert_close(lens["logits"], result["logits"])
    torch.testing.assert_close(lens["probabilities"].sum(-1), torch.ones(1, 3))
    assert lens["top_token_ids"] == result["probabilities"].topk(3, -1).indices.tolist()
    assert model.training and not model.embedding.training
    assert not model.blocks[0]._forward_hooks


def test_generation_probabilities_are_real_and_capture_budget_is_total():
    model = tiny_model()
    state = torch.random.get_rng_state().clone()
    result = generate_tokens(model, [1, 2], max_new_tokens=3, modules=["blocks.0"],
                             temperature=0, max_capture_bytes=32)
    assert torch.equal(state, torch.random.get_rng_state())
    assert result["capture"]["retained_bytes"] == 32
    assert result["capture"]["dropped"] == 2 and not result["capture"]["complete"]
    for step in result["steps"]:
        actual = infer_tokens(model, [step["context_token_ids"]])
        probs = actual["probabilities"][0, -1]
        torch.testing.assert_close(torch.tensor(step["probabilities"]), probs)
        assert step["token_id"] == int(probs.argmax())
        assert step["model_probability"] == pytest.approx(float(probs[step["token_id"]]))
        assert step["sample_probability"] == 1
    assert not model.blocks[0]._forward_hooks


def test_seeded_sampling_and_topk_distribution_are_reproducible():
    model = tiny_model()
    first = generate_tokens(model, [1, 2], max_new_tokens=5, seed=22, top_k=1)
    second = generate_tokens(model, [1, 2], max_new_tokens=5, seed=22, top_k=1)
    assert first == second
    assert all(sum(p > 0 for p in step["sampling_probabilities"]) == 1 for step in first["steps"])


def test_tuple_module_output_preserves_auxiliary_values():
    class TupleBlock(nn.Module):
        def forward(self, value):
            return value, "auxiliary"

    class TupleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = TupleBlock()

        def forward(self, value):
            return self.block(value)

    model = TupleModel()
    with SteeringSession(model, "block", vectors={0: torch.ones(2)}, strengths={0: 2}):
        with ActivationCache(model, ["block"]) as cache:
            output, extra = model(torch.zeros(1, 1, 2))
    torch.testing.assert_close(output, torch.full((1, 1, 2), 2.))
    torch.testing.assert_close(cache.latest("block"), output)
    assert extra == "auxiliary" and not model.block._forward_hooks


def test_axislab_optional_bridge_matches_actual_readout_and_target_rank():
    pytest.importorskip("axislab_poc.lenses")
    model = tiny_model()
    tokens = torch.tensor([[1, 2, 3]])
    with ActivationCache(model, ["blocks.0"]) as cache:
        logits = model(tokens)
    target = logits.argmax(-1)
    readout = axislab_logit_lens(model, cache.latest("blocks.0"), top_k=3, target_ids=target)
    top_probs, top_ids = logits.softmax(-1).topk(3, -1)
    assert readout["token_ids"] == top_ids.tolist()
    torch.testing.assert_close(torch.tensor(readout["probabilities"]), top_probs)
    assert readout["target_rank"] == [[1, 1, 1]]
    assert torch.isfinite(torch.tensor(readout["entropy"])).all()


def test_reused_module_patching_is_rejected_and_hooks_removed():
    class ReusedModel(TokenModel):
        def forward(self, tokens):
            hidden = self.block(self.embedding(tokens))
            return self.output(self.block(hidden))

    model = ReusedModel()
    with pytest.raises(ValueError, match="one bounded invocation"):
        run_patching_experiment(model, torch.tensor([[1]]), torch.tensor([[0]]),
                                "block", lambda logits: logits.sum())
    assert not model.block._forward_hooks
