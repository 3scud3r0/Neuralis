import base64
import struct
import zlib

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from neuralis.api.frame_transport import decode_frame_png
from neuralis.core.schemas import CommandRequest, RunConfig
from neuralis.main import create_app
from neuralis.rl_arena.frames import encode_frame_png


@pytest.fixture
def extension_client(tmp_path, monkeypatch):
    monkeypatch.delenv("NEURALIS_TOKEN", raising=False)
    monkeypatch.delenv("NEURALIS_OIDC_ISSUER", raising=False)
    monkeypatch.setenv("NEURALIS_HOST", "127.0.0.1")
    with TestClient(create_app(tmp_path)) as client:
        yield client


def test_axis_held_out_separation_and_dimension_validation(extension_client):
    body = {"positive": [[2, 1], [2, -1]], "negative": [[0, 1], [0, -1]],
            "eval_positive": [[1.5, 4], [2.5, -4]], "eval_negative": [[0.5, 3], [-0.5, -3]]}
    response = extension_client.post("/api/v1/extensions/axis/evaluate", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["direction"] == pytest.approx([1, 0])
    assert result["auc"] == 1 and result["evaluation"] == "held-out"
    body["negative"][0] = [0]
    assert extension_client.post("/api/v1/extensions/axis/evaluate", json=body).status_code == 422


@pytest.mark.parametrize("quantized", [False, True])
def test_adapter_supervisor_checkpoint_and_resume_preserve_frozen_base(supervisor, quantized):
    config = RunConfig(steps=10, adapter={"enabled": True, "rank": 2, "quantize_base": quantized}, capture_every=1)
    row = supervisor.create(config)
    live = supervisor.get_live(row["id"])
    initial = {name: value.clone() for name, value in live.model.state_dict().items()}
    optimizer_parameters = {id(value) for group in live.optimizer.param_groups for value in group["params"]}
    assert optimizer_parameters == {id(value) for value in live.model.parameters() if value.requires_grad}
    for _ in range(3):
        supervisor.command(live.id, CommandRequest(command="step"))
    for name, value in live.model.state_dict().items():
        if not name.endswith(("lora_A", "lora_B")):
            assert torch.equal(value, initial[name]), name
    assert any(not torch.equal(value, initial[name]) for name, value in live.model.state_dict().items() if name.endswith("lora_B"))
    checkpoint = supervisor.save_checkpoint(live)
    resumed_row = supervisor.restore(live.id, checkpoint["id"])
    resumed = supervisor.get_live(resumed_row["id"])
    assert resumed.step == 3
    supervisor.command(live.id, CommandRequest(command="step"))
    supervisor.command(resumed.id, CommandRequest(command="step"))
    assert resumed.step == live.step == 4
    for name, value in live.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name


def test_adapter_http_demo_computes_losses_and_real_compression(extension_client):
    result = extension_client.post("/api/v1/extensions/training/demo", json={"steps": 20, "rank": 2}).json()
    assert len(result["losses"]) == 20 and all(np.isfinite(result["losses"]))
    assert result["max_base_change"] == 0
    assert 0 < result["trainable_parameters"] < result["base_parameters"]
    assert result["quantization"]["packed_bytes"] < result["quantization"]["original_bytes"]
    assert result["quantization"]["mse"] > 0


def test_encrypted_rag_http_retrieval_and_restart(tmp_path, monkeypatch):
    monkeypatch.delenv("NEURALIS_TOKEN", raising=False)
    text = "Neuralis experiments use explicit seeds. AxisLab learns contrastive activation directions."
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/v1/extensions/rag/index", json={"text": text, "source": "private-source"})
        assert response.status_code == 200, response.text
        response = client.post("/api/v1/extensions/rag/search", json={"query": "activation directions"})
        assert response.status_code == 200, response.text
        assert response.json()["results"][0]["text"] == text
    for path in (tmp_path / "extensions").glob("rag-*"):
        assert text.encode() not in path.read_bytes()
        assert b"private-source" not in path.read_bytes()
    assert (tmp_path / "extensions" / "rag.key").stat().st_mode & 0o077 == 0
    with TestClient(create_app(tmp_path)) as client:
        result = client.post("/api/v1/extensions/rag/search", json={"query": "AxisLab activation"}).json()
        assert result["encrypted"] and result["results"][0]["source"] == "private-source"


def test_foundry_exports_only_verified_math(extension_client):
    response = extension_client.post("/api/v1/extensions/foundry/generate", json={"count": 12, "seed": 9})
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["records"]) == 12
    assert result["statistics"]
    assert result["rejections"]


def test_arena_real_demo_and_frame_recording(extension_client):
    response = extension_client.post("/api/v1/extensions/arena/demo", json={"steps": 40})
    assert response.status_code == 200, response.text
    demo = response.json()
    assert demo["terminated"] and demo["storage_backend"] == "duckdb"
    steps = extension_client.get(f"/api/v1/extensions/arena/episodes/{demo['episode_id']}/steps").json()
    assert len(steps) == demo["steps"] and len(steps[0]["policy"]) == 4
    frame = np.full((16, 20, 3), [70, 150, 200], dtype=np.uint8)
    endpoint = "/api/v1/extensions/arena/episodes/recording-test/frames"
    body = {"step": 0, "frame_png": base64.b64encode(encode_frame_png(frame)).decode(),
            "policy": [-1.0, 2.0], "action": 1, "reward": 3.0,
            "metadata": {"environment": "test", "policy_kind": "q-values"}}
    result = extension_client.post(endpoint, json=body)
    assert result.status_code == 200, result.text
    assert result.json()["stored"]
    # A duplicate frame cannot silently overwrite an observed policy.
    assert extension_client.post(endpoint, json=body).status_code == 422
    png = extension_client.get(endpoint + "/0")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert np.array_equal(decode_frame_png(base64.b64encode(png.content).decode()), frame)
    observation = extension_client.get("/api/v1/extensions/arena/episodes/recording-test/records/0").json()
    assert observation["policy"] == [-1, 2] and observation["metadata"]["policy_kind"] == "q-values"
    branch = extension_client.post(f"/api/v1/extensions/arena/episodes/{demo['episode_id']}/branch", json={"step": 0, "actions": [2, 1]})
    assert branch.status_code == 200, branch.text
    assert branch.json()["outputs"][-1]["position"] == [1, 1]
    unsupported = extension_client.post("/api/v1/extensions/arena/episodes/recording-test/branch", json={"step": 0, "actions": [1]})
    assert unsupported.status_code == 422


def test_graph_http_compiles_actual_residual_and_rejects_cycle(extension_client):
    from test_graph_surgery import residual_definition
    payload = {"graph": residual_definition(), "inputs": [[0., 1.], [1., 0.]]}
    result = extension_client.post("/api/v1/extensions/graph/evaluate", json=payload)
    assert result.status_code == 200, result.text
    value = result.json()
    assert value["compiled"] and value["graph"]["parameters"] > 0
    assert len(value["outputs"]) == 2 and len(value["outputs"][0]) == 2
    payload["graph"]["nodes"][0]["inputs"] = ["residual"]
    assert extension_client.post("/api/v1/extensions/graph/evaluate", json=payload).status_code == 422


def test_sae_dictionary_train_and_reversible_http_steering(extension_client):
    run = extension_client.post("/api/v1/runs", json={"steps": 10}).json()
    identifier = run["id"]
    endpoint = f"/api/v1/runs/{identifier}/dictionary"
    response = extension_client.post(endpoint + "/train", json={"module": "activation1", "features": 12, "k": 3, "steps": 10})
    assert response.status_code == 200, response.text
    learned = response.json()
    assert learned["metrics"]["training_observations"] > 0
    directions = learned["features"]
    feature = max(directions, key=lambda item: item["activation_frequency"])["id"]
    response = extension_client.post(endpoint + "/steer", json={"dictionary_id": learned["dictionary_id"], "feature": feature, "strength": 2})
    assert response.status_code == 200, response.text
    assert response.json()["output_delta_norm"] > 0
    assert response.json()["weights_mutated"] is False


def test_token_xray_measures_actual_tiny_transformer(extension_client):
    run = extension_client.post("/api/v1/runs", json={"model_id": "tiny-transformer", "steps": 5}).json()
    response = extension_client.post(f"/api/v1/runs/{run['id']}/xray/generate", json={"prompt": "neural", "max_new_tokens": 3})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["snapshot_step"] == 0 and result["vocabulary"]
    assert result["steps"] and len(result["steps"]) == 3


def test_extension_auth_and_safe_export_recipe(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURALIS_TOKEN", "test-only-token")
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/v1/extensions/training/demo", json={}).status_code == 401
        headers = {"Authorization": "Bearer test-only-token"}
        response = client.post("/api/v1/extensions/export/recipe", json={"model_name": "research-model", "gguf_path": "model.gguf"}, headers=headers)
        assert response.status_code == 200 and response.json()["executed"] is False
        assert "FROM" in response.json()["modelfile"]
        invalid = client.post("/api/v1/extensions/export/recipe", json={"gguf_path": "../secret.gguf"}, headers=headers)
        assert invalid.status_code == 422


def _png_chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)


def test_png_decoder_rejects_corruption_and_inflation_bomb():
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    png = bytearray(encode_frame_png(frame))
    png[20] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        decode_frame_png(base64.b64encode(png).decode())
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    bomb = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(b"x" * 100000)) + _png_chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="pixel length"):
        decode_frame_png(base64.b64encode(bomb).decode())
