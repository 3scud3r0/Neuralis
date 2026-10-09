"""Bounded research operations and optional integrations for the local studio.

Public HTTP operations accept data, never executable Python, arbitrary converter
paths or downloaded model identifiers. Larger jobs use the existing supervisor.
"""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import threading
import uuid
import base64

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import Field, FiniteFloat, model_validator
import torch
from torch import nn

from neuralis.analysis.experiments import prepare_input
from neuralis.core.schemas import InferenceRequest, StrictModel
from neuralis.lab.auth import Principal, principal_for
from neuralis.models.registry import build_model
from neuralis.models.graph_surgery import GraphDefinition, compile_graph
from neuralis.runtime.devices import module_device


def installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class AxisEvaluationRequest(StrictModel):
    positive: list[list[FiniteFloat]] = Field(min_length=2, max_length=512)
    negative: list[list[FiniteFloat]] = Field(min_length=2, max_length=512)
    eval_positive: list[list[FiniteFloat]] | None = Field(default=None, min_length=2, max_length=512)
    eval_negative: list[list[FiniteFloat]] | None = Field(default=None, min_length=2, max_length=512)

    @model_validator(mode="after")
    def check_dimensions(self):
        width = len(self.positive[0])
        groups = [self.positive, self.negative, self.eval_positive, self.eval_negative]
        if not 1 <= width <= 1024 or any(len(row) != width for group in groups if group for row in group):
            raise ValueError("All axis observations must have the same width (1..1024)")
        if len(self.positive) != len(self.negative):
            raise ValueError("Paired axis fitting needs equal positive and negative observation counts")
        if (self.eval_positive is None) != (self.eval_negative is None):
            raise ValueError("Supply both held-out groups or neither")
        if sum(len(group) * width for group in groups if group) > 262144:
            raise ValueError("Axis request exceeds the 262144-element budget")
        return self


class GraphEvaluationRequest(StrictModel):
    graph: GraphDefinition
    inputs: list[list[FiniteFloat]] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def check_inputs(self):
        if any(len(row) != self.graph.input_width for row in self.inputs):
            raise ValueError("All graph inputs must match its declared input width")
        return self


class AdapterDemoRequest(StrictModel):
    steps: int = Field(default=40, ge=1, le=100)
    rank: int = Field(default=4, ge=1, le=8)
    quantize_base: bool = True
    seed: int = Field(default=42, ge=0, le=2147483647)


class FoundryRequest(StrictModel):
    count: int = Field(default=24, ge=1, le=100)
    seed: int = Field(default=42, ge=0, le=2147483647)


class RAGIndexRequest(StrictModel):
    text: str = Field(min_length=1, max_length=200000)
    source: str = Field(default="Local document", min_length=1, max_length=200)
    collection: str = Field(default="local", pattern=r"^[A-Za-z0-9_-]{1,64}$")


class RAGQueryRequest(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    collection: str = Field(default="local", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    limit: int = Field(default=5, ge=1, le=20)


class ExportRecipeRequest(StrictModel):
    model_name: str = Field(default="neuralis-local", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    gguf_path: str = Field(default="model.gguf", min_length=6, max_length=200)

    @model_validator(mode="after")
    def validate_recipe_path(self):
        # Recipes are inert text. Keep their example commands unambiguous anyway.
        if not re.fullmatch(r"[A-Za-z0-9_./ -]+\.gguf", self.gguf_path) or ".." in Path(self.gguf_path).parts:
            raise ValueError("Use a simple GGUF filename/path without traversal or shell syntax")
        return self


class DictionaryRequest(StrictModel):
    module: str = Field(min_length=1, max_length=200)
    input: InferenceRequest = Field(default_factory=lambda: InferenceRequest(sample_indices=list(range(64)), split="train"))
    features: int = Field(default=48, ge=2, le=256)
    k: int = Field(default=4, ge=1, le=32)
    steps: int = Field(default=80, ge=1, le=200)
    seed: int = Field(default=42, ge=0, le=2147483647)


class DictionarySteeringRequest(StrictModel):
    dictionary_id: str
    feature: int = Field(ge=0, le=255)
    strength: FiniteFloat = Field(default=1, ge=-20, le=20)
    input: InferenceRequest = Field(default_factory=InferenceRequest)
    positions: list[int] | None = Field(default=None, max_length=128)


class ArenaDemoRequest(StrictModel):
    seed: int = Field(default=42, ge=0, le=2147483647)
    steps: int = Field(default=40, ge=1, le=256)


class ArenaFrameRequest(StrictModel):
    step: int = Field(ge=0, le=1000000)
    frame_png: str = Field(min_length=24, max_length=1400000)
    policy: list[FiniteFloat] = Field(min_length=1, max_length=4096)
    action: int = Field(ge=0, le=4095)
    reward: FiniteFloat = Field(ge=-1000000, le=1000000)
    terminated: bool = False
    metadata: dict = Field(default_factory=dict)
    snapshot: str | None = Field(default=None, max_length=1400000)

    @model_validator(mode="after")
    def validate_policy(self):
        if self.action >= len(self.policy) or any(abs(value) > 1000000 for value in self.policy):
            raise ValueError("Action index or policy magnitude is outside the recording budget")
        if len(json.dumps(self.metadata, allow_nan=False).encode()) > 8192:
            raise ValueError("Trajectory metadata must fit eight KiB")
        return self


class WebRTCOfferRequest(StrictModel):
    type: str = Field(pattern="^offer$")
    sdp: str = Field(min_length=1, max_length=65536)


class ArenaBranchRequest(StrictModel):
    step: int = Field(ge=0, le=1000000)
    actions: list[int] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def check_actions(self):
        if any(isinstance(action, bool) or not 0 <= action <= 3 for action in self.actions):
            raise ValueError("Gridworld branch actions must use indices zero to three")
        return self


class XRayRequest(StrictModel):
    prompt: str = Field(default="neural", min_length=1, max_length=128)
    max_new_tokens: int = Field(default=12, ge=1, le=32)
    modules: list[str] = Field(default_factory=lambda: ["blocks.0", "blocks.1"], min_length=1, max_length=8)
    temperature: FiniteFloat = Field(default=0.8, ge=0, le=2)
    seed: int = Field(default=42, ge=0, le=2147483647)


class ExtensionResources:
    """A single instance owns file handles, crypto keys and bounded CPU slots."""

    def __init__(self, directory: Path):
        self.directory = directory / "extensions"
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.compute_slots = threading.BoundedSemaphore(2)
        self.lock = threading.RLock()
        self.rag_stores: dict[str, object] = {}
        self.trajectory = None
        self.streamers: dict[str, object] = {}

    @contextmanager
    def compute(self):
        if not self.compute_slots.acquire(blocking=False):
            raise HTTPException(429, "Two research operations are active; retry after they finish")
        try:
            yield
        finally:
            self.compute_slots.release()

    def rag(self, collection: str):
        from cryptography.fernet import Fernet
        from neuralis.rag.vector_db_engine import LocalRAG
        with self.lock:
            if collection not in self.rag_stores:
                if len(self.rag_stores) >= 16:
                    raise ValueError("At most sixteen local RAG collections may be open")
                key_path = self.directory / "rag.key"
                if key_path.is_symlink():
                    raise ValueError("A RAG key must be a regular private file")
                try:
                    descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    pass
                else:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(Fernet.generate_key())
                        stream.flush()
                        os.fsync(stream.fileno())
                if key_path.stat().st_mode & 0o077:
                    raise ValueError("RAG key permissions must restrict access to its owner (0600)")
                self.rag_stores[collection] = LocalRAG(self.directory / f"rag-{collection}.sqlite", key=key_path.read_bytes())
            return self.rag_stores[collection]

    def arena(self):
        from neuralis.rl_arena.trajectory_db import TrajectoryStore
        with self.lock:
            if self.trajectory is None:
                backend = "duckdb" if installed("duckdb") else "sqlite"
                self.trajectory = TrajectoryStore(self.directory / f"arena.{backend}", backend=backend)
            return self.trajectory

    def close(self):
        for store in [*self.rag_stores.values(), self.trajectory]:
            if store is not None and hasattr(store, "close"):
                store.close()

    async def close_streams(self):
        for streamer in self.streamers.values():
            await streamer.close()
            streamer.frames.close()
        self.streamers.clear()

    def streamer(self, episode_id):
        from neuralis.rl_arena.webrtc_streamer import FrameQueue, WebRTCStreamer
        if episode_id not in self.streamers:
            if len(self.streamers) >= 8:
                raise ValueError("At most eight live episode streams may be open; close a completed stream")
            self.streamers[episode_id] = WebRTCStreamer(FrameQueue(capacity=3), max_peers=2)
        return self.streamers[episode_id]


def extension_state(request: Request) -> ExtensionResources:
    return request.app.state.extensions


def register_extension_routes(router, run_access, require_admin):
    @router.get("/extensions/capabilities")
    def capabilities():
        from neuralis.runtime.acceleration.backends import backend_capabilities
        axislab = installed("axislab_poc")
        return {
            "axislab": {"available": axislab, "version": "0.3.0" if axislab else None},
            "backends": backend_capabilities(),
            "nf4": {"available": True, "implementation": "packed CPU reference; no fused kernel"},
            "lora": {"available": True, "supervisor": True, "base_frozen": True},
            "rag": {"available": installed("cryptography"), "encrypted": installed("cryptography"),
                    "embedding": "deterministic hashed-token baseline", "key_storage": "private local file"},
            "webrtc": {"available": installed("aiortc"), "reason": "Requires the streaming extra and a connected frame provider",
                       "hardware_encoding_validated": False},
            "trajectory": {"storage": "duckdb" if installed("duckdb") else "sqlite", "replay": "provider-supplied state only"},
            "gguf": {"available": False, "reason": "Configure a trusted llama.cpp converter for a compatible Hugging Face model"},
            "limitations": ["GPU hardware, LLM adapters and multi-machine execution require validation on that hardware",
                            "Policy diagnostics and public rationales are observable records; hidden thoughts are not extracted",
                            "SAE features carry numerical IDs; semantic labels require researcher evidence"],
        }

    @router.post("/extensions/axis/evaluate")
    def evaluate_axis(body: AxisEvaluationRequest, request: Request):
        if not installed("axislab_poc"):
            raise HTTPException(503, "Install the research extra to use the bundled AxisLab library")
        from axislab_poc import evaluate_axis as score_axis, fit_paired_axis
        with extension_state(request).compute():
            positive = torch.tensor(body.positive, dtype=torch.float32)
            negative = torch.tensor(body.negative, dtype=torch.float32)
            axis = fit_paired_axis(positive, negative)
            evaluation_positive = torch.tensor(body.eval_positive, dtype=torch.float32) if body.eval_positive is not None else positive
            evaluation_negative = torch.tensor(body.eval_negative, dtype=torch.float32) if body.eval_negative is not None else negative
            metrics = score_axis(axis, evaluation_positive, evaluation_negative)
            return {"direction": axis.direction.tolist(), "control_mean": axis.control_mean.tolist(),
                    "scores": {"positive": axis.score(evaluation_positive).tolist(), "negative": axis.score(evaluation_negative).tolist()},
                    **metrics, "training_samples": len(positive) + len(negative),
                    "evaluation_samples": len(evaluation_positive) + len(evaluation_negative),
                    "evaluation": "held-out" if body.eval_positive is not None else "training data; optimistic",
                    "method": "AxisLab 0.3.0 paired mean contrast"}

    @router.post("/extensions/graph/evaluate")
    def graph_evaluate(body: GraphEvaluationRequest, request: Request):
        with extension_state(request).compute():
            model = compile_graph(body.graph)
            with torch.no_grad():
                output = model(torch.tensor(body.inputs, dtype=torch.float32))
            return {"outputs": output.tolist(), "graph": model.analysis, "compiled": True,
                    "validation": "bounded acyclic tensor graph; fresh seeded parameters",
                    "training_state": "fresh graph; no optimizer or previous model weights implicitly transferred"}

    @router.post("/extensions/training/demo")
    def adapter_demo(body: AdapterDemoRequest, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        from neuralis.datasets.synthetic import dataset_for
        from neuralis.runtime.training.qlora_engine import LoRAConfig, inject_lora
        from neuralis.runtime.quantization.nf4_quantizer import quantize_nf4, dequantize_nf4
        with extension_state(request).compute():
            model = build_model("mlp-xor", body.seed)
            original_parameters = sum(parameter.numel() for parameter in model.parameters())
            weight = model.hidden.weight.detach().clone()
            packed = quantize_nf4(weight)
            quantization = {"original_bytes": weight.numel() * weight.element_size(),
                            "packed_bytes": packed.storage_bytes,
                            "mse": float((dequantize_nf4(packed) - weight).square().mean())}
            inject_lora(model, LoRAConfig(rank=body.rank, alpha=body.rank * 2),
                        quantize_base=body.quantize_base, seed=body.seed + 2000)
            frozen = {name: value.clone() for name, value in model.state_dict().items()
                      if not name.endswith(("lora_A", "lora_B"))}
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.Adam(parameters, lr=0.03)
            data = dataset_for("mlp-xor", body.seed)
            generator = torch.Generator().manual_seed(body.seed + 1000)
            losses = []
            for _ in range(body.steps):
                indices = torch.randint(len(data.train), (64,), generator=generator)
                inputs, targets = data.train.batch(indices)
                loss = nn.functional.cross_entropy(model(inputs), targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            with torch.no_grad():
                accuracy = float((model(data.validation.inputs).argmax(-1) == data.validation.targets).float().mean())
                base_change = max(float((model.state_dict()[name].float() - value.float()).abs().max()) for name, value in frozen.items())
            return {"losses": losses, "base_parameters": original_parameters,
                    "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                    "quantization": quantization, "quantize_base": body.quantize_base,
                    "max_base_change": base_change, "validation_accuracy": accuracy,
                    "method": "LoRA on frozen NF4 weights" if body.quantize_base else "LoRA on frozen FP32 weights"}

    @router.post("/extensions/foundry/generate")
    def foundry(body: FoundryRequest, request: Request):
        from neuralis.synthetic.llm_as_a_judge import run_math_demo
        with extension_state(request).compute():
            return run_math_demo(count=body.count, seed=body.seed)

    @router.post("/extensions/rag/index")
    def rag_index(body: RAGIndexRequest, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        if not installed("cryptography"):
            raise HTTPException(503, "Install the research extra for encrypted local RAG")
        with extension_state(request).compute():
            store = extension_state(request).rag(body.collection)
            outcome = store.add_document(body.text, source=body.source)
            return {"collection": body.collection, "encrypted": True, "result": outcome}

    @router.post("/extensions/rag/search")
    def rag_search(body: RAGQueryRequest, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        if not installed("cryptography"):
            raise HTTPException(503, "Install the research extra for encrypted local RAG")
        with extension_state(request).compute():
            store = extension_state(request).rag(body.collection)
            outcome = store.search(body.query, limit=body.limit)
            return {"results": outcome["matches"], "collection": body.collection, "encrypted": True,
                    "embedding": outcome.get("embedding", "hashed-token baseline")}

    @router.post("/extensions/export/recipe")
    def export_recipe(body: ExportRecipeRequest):
        # No local files are read or converters launched by this endpoint.
        path = body.gguf_path.replace("\\", "/")
        return {"modelfile": f'FROM "{path}"\nPARAMETER temperature 0.7\nPARAMETER num_ctx 2048\n',
                "commands": [f'ollama create {body.model_name} -f Modelfile', f'ollama run {body.model_name}'],
                "requirements": ["A real compatible GGUF artifact must exist at FROM before importing",
                                 "Convert supported Hugging Face weights with your trusted llama.cpp converter",
                                 "Registered browser toy models are not Ollama-compatible language models"],
                "executed": False}

    @router.post("/runs/{run_id}/dictionary/train")
    def dictionary(run_id: str, body: DictionaryRequest, request: Request, principal: Principal = Depends(principal_for)):
        row = run_access(request, principal, run_id, "intervene")
        from neuralis.interpretability.activation_cache import ActivationCache
        from neuralis.interpretability.sae.dictionary_learning import fit_dictionary
        from safetensors.torch import save_file
        if body.k > body.features:
            raise ValueError("TopK cannot exceed dictionary feature count")
        with extension_state(request).compute():
            model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
            inputs, _ = prepare_input(body.input, data)
            with ActivationCache(model, [body.module], max_bytes=4 * 1024 * 1024) as cache:
                with torch.no_grad():
                    model(inputs.to(module_device(model)))
                captured = cache.latest(body.module)
            if captured is None:
                raise ValueError("Requested module did not produce a captured activation")
            values = captured.reshape(-1, captured.shape[-1]).float()
            if values.shape[1] > 512:
                raise ValueError("HTTP dictionaries are limited to 512 channels; use the SDK for larger data")
            values = values[:4096]
            learned, report = fit_dictionary(values, features=body.features, k=body.k, steps=body.steps, seed=body.seed)
            identifier = str(uuid.uuid4())
            destination = extension_state(request).directory / "dictionaries"
            destination.mkdir(exist_ok=True, mode=0o700)
            if sum(1 for _ in destination.glob("*.safetensors")) >= 128:
                raise ValueError("Dictionary artifact quota reached; remove old local artifacts before training more")
            metadata = {"run_id": run_id, "module": body.module, "step": str(step),
                        "width": str(values.shape[1]), "features": str(body.features), "k": str(body.k)}
            save_file({name: tensor.detach().cpu().contiguous() for name, tensor in learned.state_dict().items()},
                      str(destination / f"{identifier}.safetensors"), metadata=metadata)
            request.app.state.repository.audit(principal.subject, row["project_id"], "train-dictionary", identifier)
            return {"dictionary_id": identifier, "module": body.module, "snapshot_step": step, **report}

    @router.post("/runs/{run_id}/dictionary/steer")
    def dictionary_steer(run_id: str, body: DictionarySteeringRequest, request: Request, principal: Principal = Depends(principal_for)):
        row = run_access(request, principal, run_id, "intervene")
        from safetensors import safe_open
        from neuralis.interpretability.interventions.steering_vectors import SteeringSession
        identifier = str(uuid.UUID(body.dictionary_id))
        path = extension_state(request).directory / "dictionaries" / f"{identifier}.safetensors"
        if not path.is_file():
            raise KeyError(identifier)
        with safe_open(path, framework="pt", device="cpu") as stream:
            metadata = stream.metadata()
            if metadata["run_id"] != run_id:
                raise ValueError("Dictionary belongs to a different run")
            directions = stream.get_tensor("decoder.weight")
        if body.feature >= directions.shape[1]:
            raise ValueError("Requested dictionary feature is outside its feature count")
        with extension_state(request).compute():
            model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
            inputs, _ = prepare_input(body.input, data)
            inputs = inputs.to(module_device(model))
            with torch.no_grad():
                baseline = model(inputs).detach().cpu()
                with SteeringSession(model, metadata["module"], vectors={body.feature: directions[:, body.feature]},
                                     strengths={body.feature: body.strength}, positions=body.positions):
                    intervention = model(inputs).detach().cpu()
            request.app.state.repository.audit(principal.subject, row["project_id"], "dictionary-steer", identifier)
            return {"run_id": run_id, "snapshot_step": step, "dictionary_step": int(metadata["step"]),
                    "dictionary_id": identifier, "feature": body.feature, "strength": body.strength,
                    "baseline": baseline.tolist(), "intervention": intervention.tolist(),
                    "output_delta_norm": float((intervention - baseline).norm()), "weights_mutated": False}

    @router.get("/extensions/arena/episodes")
    def arena_episodes(request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        return extension_state(request).arena().episodes()

    @router.get("/extensions/arena/episodes/{episode_id}/steps")
    def arena_steps(episode_id: str, request: Request, limit: int = Query(256, ge=1, le=1000), principal: Principal = Depends(principal_for)):
        require_admin(principal)
        return [{**record.public(), "policy": record.policy.tolist()}
                for record in extension_state(request).arena().steps(episode_id, limit=limit)]

    @router.get("/extensions/arena/episodes/{episode_id}/frames/{step}")
    def arena_frame(episode_id: str, step: int, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        record = extension_state(request).arena().get(episode_id, step)
        if record is None:
            raise KeyError(episode_id)
        from neuralis.rl_arena.frames import encode_frame_png
        return Response(encode_frame_png(record.frame), media_type="image/png")

    @router.get("/extensions/arena/episodes/{episode_id}/records/{step}")
    def arena_record(episode_id: str, step: int, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        from neuralis.rl_arena.frames import encode_frame_png
        record = extension_state(request).arena().get(episode_id, step)
        return {**record.public(), "policy": record.policy.tolist(),
                "frame_png": base64.b64encode(encode_frame_png(record.frame)).decode("ascii")}

    @router.post("/extensions/arena/episodes/{episode_id}/branch")
    def arena_branch(episode_id: str, body: ArenaBranchRequest, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        from neuralis.rl_arena.gridworld import GridWorld
        from neuralis.rl_arena.temporal_debugger import TemporalDebugger
        store = extension_state(request).arena()
        record = store.get(episode_id, body.step)
        if record.metadata.get("environment") != "deterministic-gridworld" or record.snapshot is None:
            raise ValueError("This episode has no supported gridworld snapshot; emulator state cannot be reconstructed from pixels")
        if len(record.snapshot) > 8192:
            raise ValueError("Gridworld snapshot exceeds the expected data budget")
        state = json.loads(record.snapshot)
        world = GridWorld(size=state["size"])
        debugger = TemporalDebugger(store, restore=world.restore, advance=world.advance)
        with extension_state(request).compute():
            return debugger.replay(episode_id, body.step, actions=body.actions)

    @router.post("/extensions/arena/demo")
    def arena_demo(body: ArenaDemoRequest, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        from neuralis.rl_arena.gridworld import record_gridworld_demo
        with extension_state(request).compute():
            return record_gridworld_demo(extension_state(request).arena(), seed=body.seed, steps=body.steps)

    @router.post("/extensions/arena/episodes/{episode_id}/frames")
    async def arena_ingest(episode_id: str, body: ArenaFrameRequest, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        import asyncio
        import numpy as np
        from neuralis.api.frame_transport import decode_frame_png
        store = extension_state(request).arena()
        store._episode(episode_id)
        try:
            snapshot = base64.b64decode(body.snapshot, validate=True) if body.snapshot is not None else None
        except ValueError as error:
            raise ValueError("Snapshot must contain valid base64 bytes") from error
        frame = await asyncio.to_thread(decode_frame_png, body.frame_png)
        record = await asyncio.to_thread(store.append, episode_id, body.step, frame,
                                        np.asarray(body.policy, dtype=np.float32), body.action, body.reward,
                                        terminated=body.terminated, metadata=body.metadata, snapshot=snapshot)
        # FrameQueue is owned by the API event loop, not the recording worker.
        streamer = extension_state(request).streamers.get(episode_id)
        if streamer is not None:
            streamer.frames.push(frame)
        return {**record.public(), "stored": True, "storage_backend": store.backend}

    @router.post("/extensions/arena/episodes/{episode_id}/webrtc")
    async def arena_offer(episode_id: str, body: WebRTCOfferRequest, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        if not installed("aiortc"):
            raise HTTPException(503, "Install the streaming extra for WebRTC")
        resources = extension_state(request)
        resources.arena()._episode(episode_id)
        return await resources.streamer(episode_id).answer(body.model_dump())

    @router.delete("/extensions/arena/episodes/{episode_id}/webrtc")
    @router.post("/extensions/arena/episodes/{episode_id}/webrtc/close")
    async def arena_close_stream(episode_id: str, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        streamer = extension_state(request).streamers.pop(episode_id, None)
        if streamer is not None:
            await streamer.close()
            streamer.frames.close()
        return {"closed": True}

    @router.post("/runs/{run_id}/xray/generate")
    def xray(run_id: str, body: XRayRequest, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        from neuralis.datasets.synthetic import VOCABULARY
        from neuralis.interpretability.lenses import generate_tokens
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        if config.model_id != "tiny-transformer":
            raise ValueError("Token X-ray requires the registered tiny transformer")
        vocabulary = {char: index for index, char in enumerate(VOCABULARY)}
        if any(char not in vocabulary for char in body.prompt):
            raise ValueError("Prompt contains characters outside the model's training vocabulary")
        with extension_state(request).compute():
            result = generate_tokens(model, [vocabulary[char] for char in body.prompt], modules=body.modules,
                                     max_new_tokens=body.max_new_tokens, temperature=body.temperature,
                                     seed=body.seed, context_length=config.architecture.context_length,
                                     max_capture_bytes=4 * 1024 * 1024)
        return {"run_id": run_id, "snapshot_step": step, "vocabulary": VOCABULARY, **result}


def init_extensions(app, directory: Path):
    app.state.extensions = ExtensionResources(directory)
