"""Local service integration tests: real storage, replay, filtering and crypto."""

import asyncio
import json
import struct
import sys
import zlib

import numpy as np
import pytest

from neuralis.export.gguf import GGUFConverter, inspect_gguf, write_ollama_modelfile
from neuralis.rag.chunking import chunk_text
from neuralis.rag.embeddings import HashedEmbedder
from neuralis.rag.local import LocalRAG
from neuralis.rl_arena.cot_engine import policy_diagnostics
from neuralis.rl_arena.frames import encode_frame_png
from neuralis.rl_arena.gridworld import GridWorld, record_gridworld_demo
from neuralis.rl_arena.temporal_debugger import TemporalDebugger
from neuralis.rl_arena.trajectory_db import TrajectoryStore
from neuralis.rl_arena.webrtc_streamer import FrameQueue, WebRTCStreamer, capabilities
from neuralis.synthetic.foundry import Foundry, JudgeResult, exact_math_teacher, export_jsonl, run_math_demo


@pytest.mark.parametrize("backend", ["sqlite", "duckdb"])
def test_trajectory_persistence_and_real_replay(tmp_path, backend):
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    path = tmp_path / f"capture.{backend}"
    with TrajectoryStore(path, backend=backend) as store:
        outcome = record_gridworld_demo(store, size=4, episode_id="actual-grid")
        assert outcome["terminated"] and outcome["steps"] == 6
        assert outcome["return"] == pytest.approx(0.95)
        first = store.get("actual-grid", 0)
        assert first.frame.dtype == np.uint8 and first.policy.shape == (4,)
        assert first.snapshot is not None
        world = GridWorld(4)
        debugger = TemporalDebugger(store, restore=world.restore, advance=world.advance)
        result = debugger.replay("actual-grid", 0, end_step=6)
        assert result["outputs"][-1]["terminated"]
        assert world.position == (3, 3)
        branch = debugger.replay("actual-grid", 0, actions=[2, 2])
        assert branch["outputs"][-1]["position"] == [0, 2]
        assert store.get("actual-grid", 1).metadata["next_position"] == [2, 0]
        assert store.summary()["backend"] == backend
    with TrajectoryStore(path, backend=backend) as reopened:
        assert reopened.summary()["records"] == 6
        np.testing.assert_array_equal(reopened.get("actual-grid", 0).frame, first.frame)
        assert reopened.episodes()[0]["terminated"]


def test_trajectory_budgets_and_atomic_validation(tmp_path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    policy = np.array([0.0, 1.0], dtype=np.float32)
    with TrajectoryStore(tmp_path / "capture.db", max_records=2) as store:
        with pytest.raises(ValueError, match="contiguous"):
            store.append("run", 1, frame, policy, 1, 0.0)
        assert store.summary()["records"] == 0
        store.append("run", 0, frame, policy, 1, 0.0)
        with pytest.raises(ValueError, match="finite"):
            store.append("run", 1, frame, [float("nan")], 0, 0.0)
        store.append("run", 1, frame, policy, 1, 1.0, terminated=True)
        with pytest.raises(ValueError, match="budget"):
            store.append("other", 0, frame, policy, 1, 0.0)
        assert store.summary()["records"] == 2


def test_replay_requires_actual_snapshot_and_adapter(tmp_path):
    with TrajectoryStore(tmp_path / "capture.db") as store:
        store.append("run", 0, np.zeros((1, 1, 3), np.uint8), [1.0], 0, 0.0)
        with pytest.raises(RuntimeError, match="adapter"):
            TemporalDebugger(store).replay("run", 0, actions=[])
        with pytest.raises(ValueError, match="snapshot"):
            TemporalDebugger(store, restore=lambda x: None, advance=lambda x: x).replay("run", 0, actions=[])


def test_capture_png_contains_actual_pixels():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    encoded = encode_frame_png(image)
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
    offset, compressed = 8, b""
    while offset < len(encoded):
        length = struct.unpack(">I", encoded[offset:offset + 4])[0]
        kind, payload = encoded[offset + 4:offset + 8], encoded[offset + 8:offset + 8 + length]
        crc = struct.unpack(">I", encoded[offset + 8 + length:offset + 12 + length])[0]
        assert crc == zlib.crc32(kind + payload) & 0xffffffff
        if kind == b"IDAT":
            compressed += payload
        offset += 12 + length
    assert zlib.decompress(compressed) == b"\0" + image[0].tobytes() + b"\0" + image[1].tobytes()


def test_policy_diagnostics_are_observable_and_numerically_stable():
    result = policy_diagnostics([10000, 9999, -10000], chosen_action=1)
    assert sum(result["probabilities"]) == pytest.approx(1)
    assert result["top_action"] == 0
    assert result["public_rationale"] is None and result["rationale_source"] is None
    assert result["entropy_nats"] > 0
    public = policy_diagnostics([0, 0], public_rationale="I selected the public action label.")
    assert public["rationale_source"] == "caller-provided"
    assert public["normalized_entropy"] == pytest.approx(1)


def test_actual_math_rejection_sampling_and_training_export(tmp_path):
    result = run_math_demo(12, seed=11)
    assert len(result["records"]) == 12
    assert result["statistics"]["rejected"] > 0
    assert result["statistics"]["attempted"] == 15
    for record in result["records"]:
        assert record["response"] == exact_math_teacher(record["prompt"])
    repeated = run_math_demo(12, seed=11)
    assert result["records"] == repeated["records"]
    text = export_jsonl(result["records"], tmp_path / "training.jsonl")
    assert len(text.splitlines()) == 12
    assert json.loads(text.splitlines()[0])["messages"][1]["role"] == "assistant"
    with pytest.raises(ValueError, match="approved"):
        export_jsonl(result["rejections"])


def test_foundry_caller_judge_and_attempt_bound():
    calls = []
    foundry = Foundry(lambda x: "teacher", lambda x: "student",
                      lambda *x: calls.append(x) or JudgeResult(0.4, True, "Public score"), threshold=0.5)
    result = foundry.run(["one", "two", "three"], count=1, max_attempts=2)
    assert result["statistics"]["attempted"] == 2
    assert not result["statistics"]["complete"]
    assert len(calls) == 2 and not result["records"]


def test_encrypted_rag_reopen_search_and_wrong_key(tmp_path):
    cryptography = pytest.importorskip("cryptography.fernet")
    key, other = cryptography.Fernet.generate_key(), cryptography.Fernet.generate_key()
    path = tmp_path / "private.db"
    secret = "privatezebra453 telescope observations of Jupiter moons"
    with LocalRAG(path, key=key) as rag:
        outcome = rag.add_document(secret, source="confidential astronomy notes")
        rag.add_document("garden soil watering basil tomato roots", source="gardening")
        matches = rag.search("Jupiter telescope observations", limit=2)["matches"]
        assert matches[0]["document_id"] == outcome["document_id"]
        assert matches[0]["text"] == secret
        assert matches[0]["score"] > matches[1]["score"]
        assert rag.status()["documents"] == 2
        for source in tmp_path.glob("private.db*"):
            raw = source.read_bytes()
            assert b"privatezebra453" not in raw and b"confidential astronomy" not in raw
        answer = rag.retrieve_and_answer("Jupiter", lambda query, chunks: chunks[0]["text"])
        assert answer["answer_source"] == "caller-provided-generator"
    with LocalRAG(path, key=key) as reopened:
        assert reopened.search("Jupiter")["matches"][0]["text"] == secret
    with pytest.raises(ValueError, match="authenticate"):
        LocalRAG(path, key=other)
    with pytest.raises(ValueError, match="different embedding"):
        LocalRAG(path, key=key, embedder=HashedEmbedder(128))


def test_rag_budget_failure_is_atomic(tmp_path):
    fernet = pytest.importorskip("cryptography.fernet")
    with LocalRAG(tmp_path / "private.db", key=fernet.Fernet.generate_key(), max_chunks=1) as rag:
        with pytest.raises(ValueError, match="budget"):
            rag.add_document("word " * 100, chunk_size=64, overlap=0)
        assert rag.status()["chunks"] == 0
        document = rag.add_document("small document")
        assert rag.delete_document(document["document_id"])["deleted_chunks"] == 1
        assert not rag.search("small")["matches"]


def test_chunk_coverage_and_embedding_determinism():
    text = "alpha beta gamma delta epsilon " * 30
    chunks = chunk_text(text, chunk_size=80, overlap=10)
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == len(text)
    assert all(c["text"] == text[c["start"]:c["end"]] for c in chunks)
    assert all(b["start"] < a["end"] for a, b in zip(chunks, chunks[1:]))
    embedder = HashedEmbedder()
    np.testing.assert_array_equal(embedder.embed("A telescope!"), HashedEmbedder().embed("a telescope"))
    assert np.linalg.norm(embedder.embed("A telescope!")) == pytest.approx(1)


def _gguf_fixture() -> bytes:
    def string(text):
        encoded = text.encode()
        return struct.pack("<Q", len(encoded)) + encoded
    # Structurally valid GGUF container with one scalar; not a runnable LLM.
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
    data += string("general.architecture") + struct.pack("<I", 8) + string("test-fixture")
    data += string("scalar") + struct.pack("<IQIQ", 1, 1, 0, 0)
    data += b"\0" * ((-len(data)) % 32)
    return data + struct.pack("<f", 1.0)


def test_gguf_validation_and_ollama_reference(tmp_path):
    model = tmp_path / "actual-container.gguf"
    model.write_bytes(_gguf_fixture())
    result = inspect_gguf(model)
    assert result["tensor_count"] == 1 and result["architecture"] == "test-fixture"
    output = write_ollama_modelfile(model, tmp_path / "Modelfile")
    assert str(model.resolve()) in output["modelfile"]
    assert "Not imported" in output["runtime_status"]
    model.write_bytes(_gguf_fixture()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        inspect_gguf(model)
    model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
    with pytest.raises(ValueError, match="architecture"):
        write_ollama_modelfile(model)


def test_external_converter_subprocess_contract(tmp_path):
    source = tmp_path / "hf-model"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"architectures": ["Fixture"]}))
    (source / "weights.safetensors").write_bytes(b"fixture-only")
    converter = tmp_path / "converter.py"
    converter.write_text("import pathlib,sys\npathlib.Path(sys.argv[sys.argv.index('--outfile')+1]).write_bytes(" + repr(_gguf_fixture()) + ")\n")
    output = tmp_path / "converted.gguf"
    result = GGUFConverter(converter, python_executable=sys.executable).convert(source, output)
    assert output.read_bytes() == _gguf_fixture()
    assert result["source_architectures"] == ["Fixture"]
    with pytest.raises(ValueError, match="preserved"):
        GGUFConverter(converter).convert(source, output)


def test_frame_queue_drop_oldest_and_optional_transport():
    async def exercise():
        frames = FrameQueue(2)
        for number in range(3):
            frames.push(np.full((2, 2, 3), number, dtype=np.uint8))
        assert frames.statistics()["dropped"] == 1
        assert (await frames.next())[0][0, 0, 0] == 1
        frames.close()
        with pytest.raises(EOFError):
            await frames.next()
        if not capabilities()["webrtc_available"]:
            with pytest.raises(RuntimeError, match="optional"):
                await WebRTCStreamer(FrameQueue()).answer({"type": "offer", "sdp": ""})
    asyncio.run(exercise())


@pytest.mark.skipif(not capabilities()["webrtc_available"], reason="optional aiortc/av not installed")
@pytest.mark.parametrize("explicit", [False, True])
def test_webrtc_default_is_host_only_and_preserves_explicit_configuration(monkeypatch, explicit):
    async def exercise():
        import aiortc
        original = aiortc.RTCPeerConnection
        supplied = aiortc.RTCConfiguration(iceServers=[]) if explicit else None
        observed = []

        def construct(*, configuration):
            observed.append(configuration)
            return original(configuration=configuration)

        viewer = original(aiortc.RTCConfiguration(iceServers=[]))
        streamer = WebRTCStreamer(FrameQueue(), configuration=supplied)
        monkeypatch.setattr(aiortc, "RTCPeerConnection", construct)
        viewer.addTransceiver("video", direction="recvonly")
        try:
            await viewer.setLocalDescription(await viewer.createOffer())
            await streamer.answer({"type": viewer.localDescription.type, "sdp": viewer.localDescription.sdp})
            assert len(observed) == 1
            assert observed[0].iceServers == []
            if explicit:
                assert observed[0] is supplied
        finally:
            await viewer.close()
            await streamer.close()
            streamer.frames.close()
    asyncio.run(exercise())


@pytest.mark.skipif(not capabilities()["webrtc_available"], reason="optional aiortc/av not installed")
def test_webrtc_real_local_handshake_and_video_frame():
    async def exercise():
        from aiortc import RTCConfiguration, RTCPeerConnection
        frames = FrameQueue(3)
        streamer = WebRTCStreamer(frames, configuration=RTCConfiguration(iceServers=[]))
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        received = asyncio.get_running_loop().create_future()

        @viewer.on("track")
        def on_track(track):
            async def receive():
                frame = await track.recv()
                if not received.done():
                    received.set_result(frame)
            asyncio.create_task(receive())

        viewer.addTransceiver("video", direction="recvonly")
        try:
            await viewer.setLocalDescription(await viewer.createOffer())
            offer = {"type": viewer.localDescription.type, "sdp": viewer.localDescription.sdp}
            answer = await streamer.answer(offer)
            from aiortc import RTCSessionDescription
            await viewer.setRemoteDescription(RTCSessionDescription(**answer))
            for _ in range(30):
                frames.push(np.full((32, 32, 3), (120, 60, 200), dtype=np.uint8))
                await asyncio.sleep(0.05)
                if received.done():
                    break
            frame = await asyncio.wait_for(received, timeout=10)
            assert frame.width == 32 and frame.height == 32
            rgb = frame.to_ndarray(format="rgb24")
            assert np.abs(rgb.astype(float).mean(axis=(0, 1)) - [120, 60, 200]).max() < 10
        finally:
            await viewer.close()
            await streamer.close()
            frames.close()
    asyncio.run(exercise())
