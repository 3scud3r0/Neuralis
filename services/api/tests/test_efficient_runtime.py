import copy
from dataclasses import replace
import hmac
import json
import struct
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from neuralis.runtime.acceleration.backends import backend_capabilities, resolve_backend
from neuralis.runtime.quantization.nf4_quantizer import (
    NF4_VALUES, NF4Linear, PackedNF4, bitsandbytes_linear, dequantize_nf4, quantize_nf4,
)
from neuralis.runtime.training.qlora_engine import LoRAConfig, LoRALinear, QLoRAEngine, inject_lora
from neuralis.cluster.memory_offloading import activation_offload, selective_cpu_offload
from neuralis.cluster import lan_discovery
from neuralis.cluster.distributed.lan_pipeline_parallelism import (
    LANPipeline, PipelineClient, PipelineServer, PipelineStage, ProtocolError, _decode, _encode,
)


TOKEN = b"a-test-secret-with-at-least-32-bytes"


def test_cpu_dispatch_and_honest_optional_backends():
    capabilities = backend_capabilities()
    assert capabilities["backends"]["cpu"]["available"]
    assert set(capabilities["backends"]) == {"cpu", "cuda", "rocm", "mps"}
    assert not (capabilities["backends"]["cuda"]["available"] and capabilities["backends"]["rocm"]["available"])
    backend = resolve_backend("cpu")
    with backend.autocast():
        assert torch.mm(torch.eye(2), torch.eye(2)).dtype == torch.float32
    backend.synchronize()
    assert backend.memory()["allocated_bytes"] is None
    for name, information in capabilities["backends"].items():
        if not information["available"]:
            with pytest.raises(ValueError, match="unavailable"):
                resolve_backend(name)


def test_cpu_bfloat16_autocast_executes_real_matrix_kernel():
    inputs = torch.randn(8, 8, generator=torch.Generator().manual_seed(6))
    with resolve_backend("cpu", "bfloat16").autocast():
        actual = inputs @ inputs.T
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), inputs @ inputs.T, rtol=0.02, atol=0.05)


@pytest.mark.parametrize("backend,precision", [("gpu", "float32"), ("cpu:0", "float32"),
                                               ("cpu", "float16"), ("cpu:0:1", "float32")])
def test_dispatch_rejects_unsupported_requests(backend, precision):
    with pytest.raises(ValueError):
        resolve_backend(backend, precision)


def test_nf4_packs_exact_codebook_nibbles():
    tensor = torch.tensor(NF4_VALUES)
    value = quantize_nf4(tensor, block_size=16)
    assert value.packed.numel() == 8
    assert value.packed.tolist() == [low | ((low + 1) << 4) for low in range(0, 16, 2)]
    assert torch.equal(value.scales, torch.ones(1))
    torch.testing.assert_close(dequantize_nf4(value), tensor, rtol=0, atol=0)


def test_nf4_odd_zero_empty_and_dtype_roundtrips():
    zeros = quantize_nf4(torch.zeros(17), block_size=8)
    assert zeros.packed.numel() == 9
    assert int(zeros.packed[-1] >> 4) == 7
    assert torch.count_nonzero(dequantize_nf4(zeros)) == 0
    assert dequantize_nf4(quantize_nf4(torch.empty(0, 3))).shape == (0, 3)
    scalar = quantize_nf4(torch.tensor(-3.0, dtype=torch.float64))
    assert dequantize_nf4(scalar).shape == ()
    assert dequantize_nf4(scalar).dtype == torch.float64
    assert dequantize_nf4(scalar).item() == -3.0


def test_nf4_random_error_storage_and_invalid_input():
    tensor = torch.randn(128, 64, generator=torch.Generator().manual_seed(17))
    value = quantize_nf4(tensor)
    recovered = dequantize_nf4(value)
    relative_error = float((tensor - recovered).square().mean() / tensor.square().mean())
    assert relative_error < 0.02
    assert value.storage_bytes == tensor.numel() // 2 + tensor.numel() // 64 * 4
    assert value.compression_ratio > 7
    for invalid in (torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.ones(2, dtype=torch.int64)):
        with pytest.raises(ValueError):
            quantize_nf4(invalid)
    with pytest.raises(ValueError):
        quantize_nf4(tensor, block_size=0)
    with pytest.raises(ValueError, match="scale count"):
        PackedNF4(torch.zeros(4, dtype=torch.uint8), torch.ones(2), (8,), 8)


def test_nf4_linear_matches_reconstructed_reference_and_input_gradient():
    torch.manual_seed(12)
    base = nn.Linear(7, 3)
    layer = NF4Linear(base, 8)
    x = torch.randn(5, 7, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    expected = F.linear(reference_x, dequantize_nf4(layer.packed), base.bias.detach())
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    assert not list(layer.parameters())
    scales = layer.weight_scales.clone()
    layer.half()
    assert layer.packed_weight.dtype == torch.uint8
    assert layer.weight_scales.dtype == torch.float32
    torch.testing.assert_close(layer.weight_scales, scales, rtol=0, atol=0)


def test_optional_bitsandbytes_is_not_required_for_cpu():
    if not torch.cuda.is_available() or torch.version.hip:
        with pytest.raises(RuntimeError, match="CUDA"):
            bitsandbytes_linear(nn.Linear(3, 2))


def test_lora_initial_identity_frozen_base_targets_and_merge():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    reference = copy.deepcopy(model)
    inputs = torch.randn(5, 4)
    names = inject_lora(model, LoRAConfig(rank=4, alpha=8, target_modules=("0",)))
    assert names == ["0"]
    assert isinstance(model[0], LoRALinear)
    assert isinstance(model[2], nn.Linear)
    torch.testing.assert_close(model(inputs), reference(inputs), rtol=0, atol=0)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"0.lora_A", "0.lora_B"}
    model(inputs).sum().backward()
    assert model[0].base.weight.grad is None
    assert torch.count_nonzero(model[0].lora_B.grad) > 0
    with torch.no_grad():
        model[0].lora_B.normal_()
    torch.testing.assert_close(model[0](inputs), model[0].merged_linear()(inputs), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="already"):
        inject_lora(model)


def test_lora_overcomplete_rank_and_tied_or_unknown_target_rejection():
    model = nn.Sequential(nn.Linear(4, 2))
    inject_lora(model, LoRAConfig(rank=4))
    assert model[0].lora_B.shape == (2, 4)
    with pytest.raises(ValueError, match="match"):
        inject_lora(nn.Sequential(nn.Linear(4, 2)), LoRAConfig(target_modules=("missing",)))
    shared = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="shared"):
        inject_lora(nn.Sequential(shared, shared))
    first, second = nn.Linear(4, 4), nn.Linear(4, 4)
    second.weight = first.weight
    with pytest.raises(ValueError, match="tied"):
        inject_lora(nn.Sequential(first, second))


def test_seeded_lora_injection_preserves_rng_and_matches_adapter_initialization():
    original = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 2))
    first, second = copy.deepcopy(original), copy.deepcopy(original)
    previous = torch.get_rng_state().clone()
    inject_lora(first, seed=45)
    assert torch.equal(torch.get_rng_state(), previous)
    torch.randn(25)  # Other jobs may draw from the global RNG between injections.
    inject_lora(second, seed=45)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_failed_nf4_injection_preserves_original_parameter_flags_and_structure():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    with torch.no_grad():
        model[1].weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        inject_lora(model, quantize_base=True)
    assert all(isinstance(module, nn.Linear) for module in model)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_quantized_adapter_training_decreases_actual_loss_and_freezes_base():
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(4, 12), nn.Tanh(), nn.Linear(12, 2))
    engine = QLoRAEngine(model, LoRAConfig(rank=4), learning_rate=0.06)
    x = torch.randn(96, 4)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    frozen = {name: tensor.clone() for name, tensor in model.state_dict().items()
              if name.rsplit(".", 1)[-1] not in {"lora_A", "lora_B"}}
    initial = engine.evaluate_batch(x, y)["loss"]
    for _ in range(45):
        metrics = engine.train_batch(x, y)
    assert metrics["optimizer_step"] == 45
    assert engine.evaluate_batch(x, y)["loss"] < initial * 0.45
    for name, tensor in frozen.items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)


def test_engine_optimizer_accumulation_checkpoint_resume_and_wrong_base(tmp_path):
    torch.manual_seed(21)
    original = nn.Sequential(nn.Linear(4, 7), nn.Tanh(), nn.Linear(7, 2))
    model = copy.deepcopy(original)
    config = LoRAConfig(rank=3, dropout=0.2)
    engine = QLoRAEngine(model, config, gradient_accumulation=2, learning_rate=0.02)
    x, y = torch.randn(12, 4), torch.arange(12) % 2
    for _ in range(6):
        engine.train_batch(x, y)
    checkpoint = tmp_path / "adapter.pt"
    info = engine.save_checkpoint(checkpoint)
    assert info["optimizer_step"] == 3
    engine.train_batch(x, y)
    with pytest.raises(ValueError, match="boundary"):
        engine.save_checkpoint(checkpoint)
    engine.train_batch(x, y)
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    resumed = QLoRAEngine(copy.deepcopy(original), config, gradient_accumulation=2, learning_rate=0.8)
    assert resumed.load_checkpoint(checkpoint)["optimizer_step"] == 3
    resumed.train_batch(x, y)
    resumed.train_batch(x, y)
    for name, tensor in expected.items():
        torch.testing.assert_close(resumed.model.state_dict()[name], tensor, rtol=0, atol=0)
    wrong = QLoRAEngine(nn.Sequential(nn.Linear(4, 7), nn.Tanh(), nn.Linear(7, 2)), config, gradient_accumulation=2)
    with pytest.raises(ValueError, match="fingerprint"):
        wrong.load_checkpoint(checkpoint)


def test_engine_freezes_batchnorm_buffers_and_rejects_nonfinite_loss():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
    engine = QLoRAEngine(model, quantize_base=False)
    mean = model[1].running_mean.clone()
    x, y = torch.randn(8, 4), torch.arange(8) % 2
    engine.train_batch(x, y)
    torch.testing.assert_close(model[1].running_mean, mean, rtol=0, atol=0)
    engine.loss_fn = lambda logits, target: logits.sum() * float("nan")
    with pytest.raises(ValueError, match="finite"):
        engine.train_batch(x, y)
    assert all(parameter.grad is None for parameter in engine.parameters)


def test_selective_offload_inference_hooks_restore_and_preserve_results():
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
    inputs = torch.randn(3, 4)
    with torch.no_grad():
        reference = model(inputs)
    with selective_cpu_offload(model, ["0", "2"]) as offload:
        with pytest.raises(RuntimeError, match="inference_mode"):
            model(inputs)
        with torch.inference_mode():
            torch.testing.assert_close(model(inputs), reference)
        assert offload.stats.loads == offload.stats.evictions == 2
        assert offload.stats.selected_bytes > 0
    assert not model[0]._forward_hooks and not model[0]._forward_pre_hooks
    with pytest.raises(ValueError, match="nonoverlapping"):
        selective_cpu_offload(nn.Sequential(model), ["0", "0.0"])


def test_activation_offload_keeps_cpu_autograd_exact():
    x = torch.randn(6, 4, requires_grad=True)
    expected = x.detach().clone().requires_grad_(True)
    with activation_offload(min_bytes=0) as stats:
        loss = torch.sin(x).square().sum()
    loss.backward()
    torch.sin(expected).square().sum().backward()
    torch.testing.assert_close(x.grad, expected.grad, rtol=0, atol=0)
    assert stats["offloaded_bytes"] == 0


def test_lan_two_stage_inference_backward_and_parameter_updates_match_unsplit_model():
    torch.manual_seed(15)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 2))
    reference = copy.deepcopy(model)
    first = nn.Sequential(model[0], model[1])
    second = nn.Sequential(model[2])
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    stages = [PipelineStage(first, torch.optim.SGD(first.parameters(), lr=0.03), max_grad_norm=100),
              PipelineStage(second, torch.optim.SGD(second.parameters(), lr=0.03), max_grad_norm=100)]
    with PipelineServer(stages[0], TOKEN) as server0, PipelineServer(stages[1], TOKEN) as server1:
        with PipelineClient(*server0.address, TOKEN) as client0, PipelineClient(*server1.address, TOKEN) as client1:
            pipeline = LANPipeline([client0, client1])
            x, y = torch.randn(10, 4), torch.arange(10) % 2
            torch.testing.assert_close(pipeline.forward(x), reference(x))
            for step in range(3):
                reference_optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(reference(x), y)
                loss.backward()
                reference_optimizer.step()
                actual = pipeline.train_batch(x, y, F.cross_entropy)
                assert actual["loss"] == pytest.approx(float(loss.detach()), abs=1e-7)
                assert all(stage["optimizer_step"] == step + 1 for stage in actual["stages"])
                for a, b in zip(model.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
            assert client0.status()["pending_microbatches"] == 0


def test_stage_pending_gradient_shape_lease_and_step_boundaries():
    model = nn.Linear(3, 2)
    stage = PipelineStage(model, torch.optim.SGD(model.parameters(), lr=0.01), max_pending=2)
    x = torch.randn(4, 3)
    output, first = stage.forward("owner", x, training=True)
    _, second = stage.forward("owner", x, training=True)
    with pytest.raises(ProtocolError, match="budget"):
        stage.forward("owner", x, training=True)
    with pytest.raises(ProtocolError, match="leased"):
        stage.forward("foreign", x)
    with pytest.raises(ProtocolError, match="shape"):
        stage.backward("owner", first, torch.zeros(1))
    gradient = stage.backward("owner", first, torch.ones_like(output))
    torch.testing.assert_close(gradient, model.weight.detach().sum(0).expand_as(x))
    with pytest.raises(ProtocolError, match="outstanding"):
        stage.step("owner")
    stage.backward("owner", second, torch.ones_like(output))
    assert stage.step("owner")["microbatches"] == 2
    with pytest.raises(ProtocolError, match="ticket"):
        stage.backward("owner", first, torch.ones_like(output))


def test_stage_abort_expiry_and_activation_budget():
    model = nn.Linear(3, 2)
    stage = PipelineStage(model, lease_seconds=1, max_activation_bytes=1024)
    stage.forward("one", torch.randn(2, 3), training=True)
    stage.last_active -= 2
    _, ticket = stage.forward("two", torch.randn(2, 3), training=True)
    assert ticket and len(stage.pending) == 1
    stage.abort("one")
    assert len(stage.pending) == 1
    stage.abort("two")
    assert stage.owner is None and not stage.pending
    with pytest.raises(ProtocolError, match="budget"):
        stage.forward("two", torch.randn(100, 3))


def test_lan_rejects_bad_auth_oversized_and_replayed_frames():
    stage = PipelineStage(nn.Linear(3, 2))
    with PipelineServer(stage, TOKEN, timeout=1, max_frame_bytes=1024) as server:
        with pytest.raises((ConnectionError, OSError)):
            PipelineClient(*server.address, b"incorrect-secret-of-more-than-32-bytes", timeout=1)
        with PipelineClient(*server.address, TOKEN, timeout=1, max_frame_bytes=1024) as client:
            with pytest.raises(ProtocolError, match="budget"):
                client.forward(torch.zeros(512, 3))
        with PipelineClient(*server.address, TOKEN, timeout=1) as client:
            client.channel.connection.sendall(struct.pack("!I", 2048))
            with pytest.raises((ConnectionError, OSError)):
                client.channel.receive()
        with PipelineClient(*server.address, TOKEN, timeout=1) as client:
            client.status()
            client.channel.sent -= 1
            with pytest.raises((ConnectionError, OSError)):
                client.status()


def test_lan_malformed_dtype_shape_and_sequence_rejected_before_allocation():
    for descriptor in ({"name": "x", "dtype": [], "shape": [1], "bytes": 4},
                       {"name": "x", "dtype": "float32", "shape": [2 ** 60], "bytes": 0}):
        header = json.dumps({"sequence": 0, "metadata": {}, "tensors": [descriptor]}).encode()
        with pytest.raises(ProtocolError):
            _decode(struct.pack("!I", len(header)) + header, 0)
    header = json.dumps({"sequence": True, "metadata": {}, "tensors": []}).encode()
    with pytest.raises(ProtocolError, match="sequence"):
        _decode(struct.pack("!I", len(header)) + header, 1)


def test_lan_frame_tampering_is_rejected_by_hmac():
    with PipelineServer(PipelineStage(nn.Linear(3, 2)), TOKEN, timeout=1) as server:
        with PipelineClient(*server.address, TOKEN, timeout=1) as client:
            body = _encode({"command": "status"}, {}, 0, 1024)
            signature = hmac.digest(TOKEN, client.channel.challenge + b"client" + body, "sha256")
            corrupted = body[:-1] + bytes([body[-1] ^ 1]) + signature
            client.channel.connection.sendall(struct.pack("!I", len(corrupted)) + corrupted)
            with pytest.raises((ConnectionError, OSError)):
                client.channel.receive()


def test_lan_failed_loss_aborts_all_training_leases():
    stage = PipelineStage(nn.Linear(3, 2))
    with PipelineServer(stage, TOKEN) as server, PipelineClient(*server.address, TOKEN) as client:
        pipeline = LANPipeline([client])
        with pytest.raises(ValueError, match="finite"):
            pipeline.train_batch(torch.randn(3, 3), torch.zeros(3, dtype=torch.long),
                                 lambda outputs, targets: outputs.sum() * float("nan"))
        assert client.status()["pending_microbatches"] == 0
        assert not client.status()["training_leased"]


def test_optional_mdns_configuration_advertisements_bounded_parsing_and_cleanup(monkeypatch):
    class FakeZeroconf:
        def __init__(self):
            self.registered = []
            self.removed = []
            self.closed = False
            self.info = None

        def register_service(self, info):
            self.registered.append(info)

        def unregister_service(self, info):
            self.removed.append(info)

        def get_service_info(self, *args, **kwargs):
            return self.info

        def close(self):
            self.closed = True

    class FakeBrowser:
        def __init__(self, *args):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(lan_discovery, "_zeroconf", lambda: SimpleNamespace(
        Zeroconf=FakeZeroconf, ServiceBrowser=FakeBrowser,
        ServiceInfo=lambda *args, **kwargs: SimpleNamespace(**kwargs)))
    discovery = lan_discovery.LANDiscovery(max_workers=1)
    with pytest.raises(ValueError):
        discovery.advertise("bad.name", 1234, ["192.168.1.2"])
    discovery.advertise("worker-a", 1234, ["192.168.1.2"])
    properties = discovery._advertisement.properties
    assert set(properties) == {"protocol", "backends"}  # No credentials in mDNS.
    discovery._zc.info = SimpleNamespace(port=1234,
        properties={b"protocol": b"neuralis-lan-v1", b"backends": b"cpu,imaginary,cuda"},
        parsed_addresses=lambda: ["192.168.1.2"])
    discovery.add_service(discovery._zc, lan_discovery.SERVICE_TYPE, "worker-a")
    discovery.add_service(discovery._zc, lan_discovery.SERVICE_TYPE, "worker-b")
    assert len(discovery.workers()) == 1
    assert discovery.workers()[0]["name"] == "worker-b"
    assert discovery.workers()[0]["backends"] == ("cpu", "cuda")
    discovery._workers["worker-b"] = replace(discovery._workers["worker-b"], last_seen=0)
    assert not discovery.workers(max_age=1)
    discovery.remove_service(discovery._zc, lan_discovery.SERVICE_TYPE, "worker-b")
    discovery.close()
    discovery.close()
    assert discovery._zc.closed and discovery._browser.cancelled
    assert discovery._zc.removed == discovery._zc.registered
