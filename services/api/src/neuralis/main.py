import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import platform
import time
import uuid
import zipfile

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
import psutil
import torch

from neuralis import __version__
from neuralis.analysis.experiments import attribution, circuit_candidates, execute_experiment, infer, prepare_input
from neuralis.analysis.sae import train_sae
from neuralis.analysis.timeseries import align_curves, minmax_decimate
from neuralis.core.schemas import AttributionRequest, CommandRequest, ExperimentRequest, InferenceRequest, RunConfig
from neuralis.datasets.synthetic import dataset_for
from neuralis.datasets.imports import CSVImport, DatasetImport, parse_csv
from neuralis.api.middleware import BodyLimitMiddleware
from neuralis.api.research import register_research_routes
from neuralis.lab.auth import Authenticator, Principal, principal_for
from neuralis.lab.coordinator import CompletionRequest, Coordinator, HeartbeatRequest, LeaseConflict, LeaseRequest, QueueRequest, WorkerRegistration
from neuralis.models.registry import DEFINITIONS, build_model, model_graph
from neuralis.runtime.supervisor import StateConflict, Supervisor
from neuralis.runtime.devices import container_resources, device_capabilities, module_device
from neuralis.runtime.sweeps import SweepManager
from neuralis.storage.repository import Repository


class ProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    quota: int = Field(default=4, ge=1, le=64)


class MemberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str = Field(min_length=1, max_length=200)
    role: str = Field(pattern=r"^(viewer|researcher|admin)$")


class RestoreRequest(BaseModel):
    checkpoint_id: str


class SAERequest(BaseModel):
    input: InferenceRequest = Field(default_factory=lambda: InferenceRequest(sample_indices=list(range(64))))
    module: str
    features: int = Field(default=64, ge=1, le=512)
    steps: int = Field(default=100, ge=1, le=500)
    l1_coefficient: float = Field(default=0.005, ge=0, le=1)


def create_app(data_directory: Path | None = None):
    @asynccontextmanager
    async def lifespan(app):
        directory = data_directory or Path(os.getenv("NEURALIS_DATA", "/tmp/neuralis-data"))
        app.state.repository = Repository(directory)
        app.state.supervisor = Supervisor(app.state.repository)
        app.state.coordinator = Coordinator(app.state.repository)
        app.state.sweeps = SweepManager(app.state.supervisor)
        app.state.auth = Authenticator()
        app.state.started_at = time.time()
        yield
        await asyncio.to_thread(app.state.sweeps.shutdown)
        await asyncio.to_thread(app.state.supervisor.shutdown)

    app = FastAPI(title="Neuralis Studio research API", version=__version__, lifespan=lifespan)
    origins = [origin.strip() for origin in os.getenv("NEURALIS_CORS_ORIGINS",
               "http://localhost:5173,http://127.0.0.1:5173,http://127.0.0.1:4173,http://localhost:4173,tauri://localhost,http://tauri.localhost").split(",") if origin.strip()]
    if "*" in origins:
        raise ValueError("Set explicit CORS origins; wildcard origins are not permitted")
    app.state.origins = set(origins)
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "DELETE", "PATCH"], allow_headers=["Authorization", "Content-Type"])

    @app.exception_handler(StateConflict)
    async def conflict(request, error):
        return JSONResponse(status_code=409, content={"category": "state-conflict", "message": str(error),
                                                      "action": "Refresh the authoritative state before sending another command."})

    @app.exception_handler(LeaseConflict)
    async def lease_conflict(request, error):
        return JSONResponse(status_code=409, content={"category": "lease-conflict", "message": str(error),
                                                      "action": "Stop work on an expired lease; enqueue a new attempt explicitly."})

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse(status_code=404, content={"category": "not-found", "message": "Requested resource was not found",
                                                      "action": "Refresh the resource list."})

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse(status_code=422, content={"category": "invalid-input", "message": str(error),
                                                      "action": "Check shapes, indices, and the registered model capabilities."})

    @app.get("/health")
    @app.get("/api/v1/health")
    def health(request: Request):
        return {"status": "ok", "service": "neuralis-api", "version": __version__, "protocol_version": "1.0",
                "uptime_seconds": time.time() - request.app.state.started_at}

    router = APIRouter(dependencies=[Depends(principal_for)])

    def authorize(request, principal, project_id, permission="read"):
        request.app.state.auth.authorize(request.app.state.repository, principal, project_id, permission)

    def run_access(request, principal, run_id, permission="read"):
        row = request.app.state.repository.get_run(run_id)
        authorize(request, principal, row["project_id"], permission)
        return row

    def require_admin(principal):
        if not principal.instance_admin:
            raise HTTPException(403, "Instance administrator permission required")

    @router.get("/capabilities")
    def capabilities(request: Request):
        auth = request.app.state.auth
        return {"version": __version__, "protocol_version": "1.0", "mode": "local",
                "devices": device_capabilities(), "configurable_architectures": True,
                "dataset_import": ["numeric-json", "numeric-csv"], "weight_import": "registered-architecture safetensors <= eight MiB",
                "sweeps": True, "cross_validation": "bounded stratified classification folds",
                "representation_analysis": ["pca", "linear-cka"], "sdk": "trusted Python instrumentation + JSONL recorder + torch.profiler",
                "optimizers": ["adamw", "adam", "sgd", "rmsprop"], "schedules": ["constant", "cosine", "step", "exponential"],
                "models": list(DEFINITIONS), "training": True, "inference": True,
                "capture": {"activations": True, "gradients": True, "weights": True, "bounded": True,
                            "attention": "tiny-transformer.attention_probs", "full_tensor_download": False},
                "interventions": ["ablate", "scale", "steer", "patch"],
                "attribution": ["gradient-input", "integrated-gradients"], "circuit_candidates": True,
                "sparse_autoencoder": "experimental", "checkpoints": "safetensors + weights_only optimizer state",
                "persistence": "sqlite", "websocket": True, "replay": True, "projects": True, "rbac": True,
                "authentication": "oidc" if auth.issuer else "bearer" if auth.token else "loopback-single-user",
                "worker_leases": True, "distributed_training": "external CPU DDP demonstration worker",
                "integrations": {"postgresql": {"available": False, "reason": "The validated local runtime uses SQLite"},
                                 "s3": {"available": False, "reason": "The validated artifact store is local disk"},
                                 "kubernetes": {"available": False, "reason": "No cluster execution adapter configured"},
                                 "slurm": {"available": False, "reason": "No scheduler connection configured"},
                                 "transformerlens": {"available": False, "reason": "No tested optional adapter installed"},
                                 "circuit_tracer": {"available": False, "reason": "Own gradient candidates are not Circuit Tracer"}},
                "limitations": ["Registered bounded architectures only; CUDA/MPS are conditional and unvalidated on this CPU host", "Cooperative threaded training; hung kernels require backend restart",
                                "OIDC must be configured and validated against the laboratory provider",
                                "Shared instance bearer tokens are administrative; use OIDC for project-separated users",
                                "No demonstrated large-laboratory scale or production cluster integration"]}

    @router.get("/models")
    def models():
        return [{**asdict(definition), "capabilities": {"train": True, "capture": True, "intervene": True,
                                                       "input_attribution": definition.family != "transformer"}}
                for definition in DEFINITIONS.values()]

    @router.get("/models/{model_id}/graph")
    def architecture(model_id: str, seed: int = 42):
        if model_id not in DEFINITIONS:
            raise KeyError(model_id)
        model = build_model(model_id, seed)
        data = dataset_for(model_id, seed)
        return model_graph(model, data.validation.inputs[:1])

    @router.get("/datasets")
    def datasets():
        result = []
        for model_id in DEFINITIONS:
            bundle = dataset_for(model_id, 42)
            result.append({"id": bundle.id, "model_id": model_id, "description": bundle.description, "task": bundle.task,
                           "training_examples": len(bundle.train), "validation_examples": len(bundle.validation),
                           "input_shape": list(bundle.train.inputs.shape[1:]), "labels": list(bundle.labels),
                           "source": "deterministic local generator", "requires_download": False})
        return result

    @router.get("/datasets/{dataset_id}/samples")
    def dataset_samples(dataset_id: str, split: str = "validation", start: int = Query(0, ge=0), count: int = Query(8, ge=1, le=64), seed: int = 42):
        definition = next((definition for definition in DEFINITIONS.values() if definition.dataset == dataset_id), None)
        if definition is None:
            raise KeyError(dataset_id)
        bundle = dataset_for(definition.id, seed)
        selected = bundle.train if split == "train" else bundle.validation
        return {"dataset_id": dataset_id, "split": split, "start": start,
                "inputs": selected.inputs[start:start + count].tolist(), "targets": selected.targets[start:start + count].tolist()}

    @router.post("/runs", status_code=201)
    def create_run(config: RunConfig, request: Request, principal: Principal = Depends(principal_for)):
        authorize(request, principal, config.project_id, "train")
        with request.app.state.repository.transaction() as connection:
            project = connection.execute("SELECT quota FROM projects WHERE id=?", (config.project_id,)).fetchone()
            if project is None:
                raise KeyError(config.project_id)
        row = request.app.state.supervisor.create(config)
        request.app.state.repository.audit(principal.subject, config.project_id, "create-run", row["id"])
        return row

    @router.post("/datasets/import", status_code=201)
    def import_dataset(body: DatasetImport, request: Request, principal: Principal = Depends(principal_for)):
        authorize(request, principal, body.project_id, "train")
        result = request.app.state.supervisor.datasets.import_numeric(body)
        request.app.state.repository.audit(principal.subject, body.project_id, "import-dataset", result["id"])
        return result

    @router.post("/datasets/import-csv", status_code=201)
    def import_csv(body: CSVImport, request: Request, principal: Principal = Depends(principal_for)):
        authorize(request, principal, body.project_id, "train")
        return request.app.state.supervisor.datasets.import_numeric(parse_csv(body))

    @router.get("/datasets/imported/{dataset_id}")
    def imported_dataset(dataset_id: str, request: Request, principal: Principal = Depends(principal_for)):
        metadata = request.app.state.supervisor.datasets.metadata(dataset_id)
        authorize(request, principal, metadata["project_id"])
        return metadata

    @router.get("/projects/{project_id}/datasets")
    def project_datasets(project_id: str, request: Request, principal: Principal = Depends(principal_for)):
        authorize(request, principal, project_id)
        return request.app.state.supervisor.datasets.list(project_id)

    @router.get("/runs")
    def runs(request: Request, project_id: str | None = None, limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0), principal: Principal = Depends(principal_for)):
        if project_id:
            authorize(request, principal, project_id)
        elif not principal.instance_admin:
            raise HTTPException(422, "Choose a project_id when listing laboratory runs")
        return request.app.state.repository.list_runs(project_id, limit, offset)

    @router.get("/runs/{run_id}")
    def run(run_id: str, request: Request, principal: Principal = Depends(principal_for)):
        return run_access(request, principal, run_id)

    @router.post("/runs/{run_id}/commands")
    def command(run_id: str, body: CommandRequest, request: Request, principal: Principal = Depends(principal_for)):
        row = run_access(request, principal, run_id, "train")
        result = request.app.state.supervisor.command(run_id, body)
        request.app.state.repository.audit(principal.subject, row["project_id"], body.command, run_id)
        return result

    @router.get("/runs/{run_id}/metrics")
    def metrics(run_id: str, request: Request, after: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=20000), budget: int | None = Query(None, ge=4, le=20000), principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        page = request.app.state.repository.events(run_id, after, limit, "metrics")
        points = [{"step": item["step"], "sequence": item["sequence"], "timestamp": item["timestamp"], **item["payload"]} for item in page["events"]]
        return {"metrics": minmax_decimate(points, budget) if budget else points, "sampled": budget is not None and len(points) > budget,
                "original_count": len(points), "gap": page["gap"], "next_sequence": page["next_sequence"]}

    @router.get("/runs/{run_id}/events")
    def events(run_id: str, request: Request, after: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=5000), principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        return request.app.state.repository.events(run_id, after, limit)

    @router.get("/runs/{run_id}/tensors")
    def tensors(run_id: str, request: Request, module: str | None = None, kind: str | None = None, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        live = request.app.state.supervisor.runs.get(run_id)
        value = live.latest_tensors if live else {}
        if not value:
            history = request.app.state.repository.events(run_id, 0, 20000, "tensors")["events"]
            value = history[-1]["payload"] if history else {"tensors": [], "reason": "No capture has been recorded"}
        return {**value, "tensors": [tensor for tensor in value.get("tensors", []) if (module is None or tensor["module"] == module)
                                    and (kind is None or tensor["kind"] == kind)]}

    @router.get("/runs/{run_id}/graph")
    def run_graph(run_id: str, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        return {**model_graph(model, data.validation.inputs[:1]), "step": step, "model_id": config.model_id}

    @router.post("/runs/{run_id}/inference")
    def inference(run_id: str, body: InferenceRequest, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        return {"run_id": run_id, "step": step, **infer(model, data, body)}

    @router.get("/runs/{run_id}/checkpoints")
    def checkpoints(run_id: str, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        return request.app.state.repository.checkpoints(run_id)

    @router.post("/runs/{run_id}/restore", status_code=201)
    def restore(run_id: str, body: RestoreRequest, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id, "train")
        return request.app.state.supervisor.restore(run_id, body.checkpoint_id)

    @router.post("/experiments", status_code=201)
    def experiment(body: ExperimentRequest, request: Request, principal: Principal = Depends(principal_for)):
        row = run_access(request, principal, body.run_id, "intervene")
        model, data, config, step = request.app.state.supervisor.snapshot_model(body.run_id)
        result = {"snapshot_step": step, **execute_experiment(model, data, body)}
        saved = request.app.state.repository.save_experiment(body.run_id, body.model_dump(), result)
        request.app.state.repository.audit(principal.subject, row["project_id"], "intervention", saved["id"])
        return saved

    @router.get("/experiments")
    def experiments(request: Request, run_id: str | None = None, principal: Principal = Depends(principal_for)):
        if run_id:
            run_access(request, principal, run_id)
        else:
            require_admin(principal)
        return request.app.state.repository.experiments(run_id)

    @router.post("/runs/{run_id}/attribution")
    def attribute(run_id: str, body: AttributionRequest, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id, "intervene")
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        return {"run_id": run_id, "step": step, **attribution(model, data, body)}

    @router.post("/runs/{run_id}/circuits")
    def circuits(run_id: str, body: InferenceRequest, request: Request, target_class: int | None = Query(None, ge=0), limit: int = Query(32, ge=1, le=128), principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id, "intervene")
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        return {"run_id": run_id, "step": step, **circuit_candidates(model, data, body, target_class, limit)}

    @router.post("/runs/{run_id}/sae")
    def sae(run_id: str, body: SAERequest, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id, "intervene")
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        modules = dict(model.named_modules())
        if body.module not in modules:
            raise ValueError("Unknown SAE activation module")
        observations = []
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                raise ValueError("SAE requires tensor activations")
            observations.append(output.detach().reshape(-1, output.shape[-1]))
        handle = modules[body.module].register_forward_hook(hook)
        try:
            inputs, _ = prepare_input(body.input, data)
            with torch.no_grad():
                model(inputs.to(module_device(model)))
        finally:
            handle.remove()
        if not observations:
            raise ValueError("Selected module was not executed")
        _, report = train_sae(torch.cat(observations).cpu(), body.features, body.steps, body.l1_coefficient, config.seed)
        return {"run_id": run_id, "module": body.module, "snapshot_step": step, "axis": "last", **report}

    @router.get("/replays/{run_id}")
    def replay(run_id: str, request: Request, after: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=5000), principal: Principal = Depends(principal_for)):
        row = run_access(request, principal, run_id)
        return {"run": row, **request.app.state.repository.events(run_id, after, limit), "retention": "bounded event history"}

    @router.get("/runs/{run_id}/compare/{other_id}")
    def compare(run_id: str, other_id: str, request: Request, metric: str = "loss", principal: Principal = Depends(principal_for)):
        first = run_access(request, principal, run_id)
        second = run_access(request, principal, other_id)
        if first["config"]["model_id"] != second["config"]["model_id"]:
            raise ValueError("Only runs of the same registered task can be directly compared")
        repository = request.app.state.repository
        points = lambda identifier: [{"step": event["step"], **event["payload"]} for event in repository.events(identifier, 0, 20000, "metrics")["events"]]
        return {"metric": metric, "first": first, "second": second, "points": align_curves(points(run_id), points(other_id), metric)}

    @router.get("/runs/{run_id}/export")
    @router.get("/exports/{run_id}")
    def export(run_id: str, request: Request, principal: Principal = Depends(principal_for)):
        row = run_access(request, principal, run_id, "export")
        repository = request.app.state.repository
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("run.json", json.dumps(row, indent=2))
            archive.writestr("events.json", json.dumps(repository.events(run_id, 0, 20000), indent=2))
            archive.writestr("experiments.json", json.dumps(repository.experiments(run_id), indent=2))
            archive.writestr("checkpoints.json", json.dumps(repository.checkpoints(run_id), indent=2))
            archive.writestr("README.txt", "Neuralis research export. Values are measured. Tensor values may be sampled; inspect sampling metadata.\nCheckpoint manifests reference server-local artifacts; this export contains metadata, not model weight files.\n")
        stream.seek(0)
        repository.audit(principal.subject, row["project_id"], "export", run_id)
        return StreamingResponse(stream, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="neuralis-{run_id}.zip"'})

    @router.get("/diagnostics")
    def diagnostics(request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        process = psutil.Process()
        disk = psutil.disk_usage(request.app.state.repository.directory)
        memory = psutil.virtual_memory()
        return {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.system(),
                "cpu_threads": torch.get_num_threads(), "cpu_logical": psutil.cpu_count(),
                "process_rss_bytes": process.memory_info().rss, "host_memory_available_bytes": memory.available,
                "disk_free_bytes": disk.free, "loaded_models": len(request.app.state.supervisor.runs),
                "active_jobs": sum(bool(run.thread and run.thread.is_alive()) for run in request.app.state.supervisor.runs.values()),
                "devices": device_capabilities(), "container_resources": container_resources(), "host_metrics_may_exceed_container_quota": True}

    @router.get("/settings")
    def settings(request: Request):
        return {"profile": "local", "max_parallel_jobs": request.app.state.supervisor.max_parallel,
                "max_events_per_run": request.app.state.repository.max_events_per_run,
                "capture_max_elements": 65536, "capture_max_tensors": 256, "protocol_version": "1.0"}

    @router.get("/auth")
    def identity(request: Request, principal: Principal = Depends(principal_for)):
        return {"subject": principal.subject, "instance_admin": principal.instance_admin,
                "oidc_configured": bool(request.app.state.auth.issuer), "tokens_are_never_returned": True}

    @router.get("/projects")
    def projects(request: Request, principal: Principal = Depends(principal_for)):
        with request.app.state.repository.transaction() as connection:
            if principal.instance_admin:
                rows = connection.execute("SELECT * FROM projects ORDER BY created_at").fetchall()
            else:
                rows = connection.execute("SELECT p.* FROM projects p JOIN members m ON p.id=m.project_id WHERE m.subject=?", (principal.subject,)).fetchall()
        return [dict(row) for row in rows]

    @router.post("/projects", status_code=201)
    def create_project(body: ProjectRequest, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        project_id = str(uuid.uuid4())
        with request.app.state.repository.transaction() as connection:
            connection.execute("INSERT INTO projects VALUES(?,?,?,?,?)", (project_id, body.name, body.description, time.time(), body.quota))
            connection.execute("INSERT INTO members VALUES(?,?,?)", (project_id, principal.subject, "admin"))
        request.app.state.repository.audit(principal.subject, project_id, "create-project", project_id)
        return {"id": project_id, **body.model_dump()}

    @router.get("/projects/{project_id}/members")
    def members(project_id: str, request: Request, principal: Principal = Depends(principal_for)):
        authorize(request, principal, project_id)
        with request.app.state.repository.transaction() as connection:
            rows = connection.execute("SELECT * FROM members WHERE project_id=?", (project_id,)).fetchall()
        return [dict(row) for row in rows]

    @router.post("/projects/{project_id}/members")
    def add_member(project_id: str, body: MemberRequest, request: Request, principal: Principal = Depends(principal_for)):
        authorize(request, principal, project_id, "manage")
        with request.app.state.repository.transaction() as connection:
            connection.execute("INSERT INTO members VALUES(?,?,?) ON CONFLICT(project_id,subject) DO UPDATE SET role=excluded.role", (project_id, body.subject, body.role))
        request.app.state.repository.audit(principal.subject, project_id, "set-member-role", body.subject)
        return {"project_id": project_id, **body.model_dump()}

    @router.get("/audit")
    def audit(request: Request, project_id: str | None = None, principal: Principal = Depends(principal_for)):
        if project_id:
            authorize(request, principal, project_id, "manage")
        else:
            require_admin(principal)
        return request.app.state.repository.audit_records(project_id)

    @router.get("/quotas")
    def quotas(request: Request, project_id: str = "local", principal: Principal = Depends(principal_for)):
        authorize(request, principal, project_id)
        with request.app.state.repository.transaction() as connection:
            row = connection.execute("SELECT quota FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(project_id)
        return {"project_id": project_id, "concurrent_queued_or_running_worker_jobs": row["quota"],
                "local_instance_parallel_jobs": request.app.state.supervisor.max_parallel}

    @router.post("/workers/register")
    def worker_register(body: WorkerRegistration, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        return request.app.state.coordinator.register(body)

    @router.get("/workers")
    def workers(request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        return request.app.state.coordinator.workers()

    @router.post("/workers/{worker_id}/lease")
    def worker_lease(worker_id: str, body: LeaseRequest, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        return request.app.state.coordinator.lease(worker_id, body)

    @router.post("/workers/{worker_id}/heartbeat")
    def worker_heartbeat(worker_id: str, body: HeartbeatRequest, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        return request.app.state.coordinator.heartbeat(worker_id, body)

    @router.post("/workers/{worker_id}/jobs/{job_id}/complete")
    def worker_complete(worker_id: str, job_id: str, body: CompletionRequest, request: Request, principal: Principal = Depends(principal_for)):
        require_admin(principal)
        return request.app.state.coordinator.complete(worker_id, job_id, body)

    @router.get("/queue")
    def queue(request: Request, project_id: str | None = None, principal: Principal = Depends(principal_for)):
        if project_id:
            authorize(request, principal, project_id)
        else:
            require_admin(principal)
        return request.app.state.coordinator.jobs(project_id)

    @router.post("/queue", status_code=201)
    def enqueue(body: QueueRequest, request: Request, principal: Principal = Depends(principal_for)):
        authorize(request, principal, body.project_id, "train")
        return request.app.state.coordinator.enqueue(body)

    @router.post("/queue/{job_id}/cancel")
    def cancel_queued(job_id: str, request: Request, principal: Principal = Depends(principal_for)):
        with request.app.state.repository.transaction() as connection:
            row = connection.execute("SELECT project_id FROM queue_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        authorize(request, principal, row["project_id"], "train")
        return request.app.state.coordinator.cancel(job_id)

    register_research_routes(router, run_access, authorize, require_admin)
    app.include_router(router, prefix="/api/v1")
    app.include_router(router, include_in_schema=False)

    async def websocket_events(websocket: WebSocket, run_id: str):
        origin = websocket.headers.get("origin")
        if origin and origin not in app.state.origins:
            await websocket.close(code=1008, reason="Origin is not allowed")
            return
        try:
            # Browsers cannot set WS headers; use a negotiated subprotocol bearing the token.
            authorization = websocket.headers.get("authorization")
            protocols = websocket.headers.get("sec-websocket-protocol", "").split(",")
            credential_protocol = next((protocol.strip() for protocol in protocols if protocol.strip().startswith("neuralis-token.")), None)
            if authorization is None and credential_protocol:
                authorization = "Bearer " + credential_protocol.removeprefix("neuralis-token.")
            principal = app.state.auth.authenticate(authorization)
            row = app.state.repository.get_run(run_id)
            app.state.auth.authorize(app.state.repository, principal, row["project_id"], "read")
            cursor = int(websocket.query_params.get("after", "0"))
            if cursor < 0:
                raise ValueError("Negative event sequence")
        except (HTTPException, KeyError, ValueError):
            await websocket.close(code=1008, reason="Unauthorized or invalid subscription")
            return
        await websocket.accept(subprotocol=credential_protocol)
        try:
            await websocket.send_json({"type": "snapshot", "protocol_version": "1.0", "run_id": run_id, "payload": row})
            idle = 0
            while True:
                page = await asyncio.to_thread(app.state.repository.events, run_id, cursor, 100)
                if page["gap"]:
                    await websocket.send_json({"type": "gap", "run_id": run_id, "payload": {"after": cursor, "oldest_sequence": page["oldest_sequence"]}})
                    cursor = page["oldest_sequence"] - 1
                for event in page["events"]:
                    await asyncio.wait_for(websocket.send_json(event), timeout=10)
                    cursor = event["sequence"]
                idle = 0 if page["events"] else idle + 1
                if idle >= 40:
                    await websocket.send_json({"type": "heartbeat", "run_id": run_id, "sequence": cursor})
                    idle = 0
                await asyncio.sleep(0.25)
        except (WebSocketDisconnect, RuntimeError, asyncio.TimeoutError):
            return

    app.websocket("/api/v1/ws/runs/{run_id}")(websocket_events)
    app.websocket("/ws/runs/{run_id}")(websocket_events)
    return app


app = create_app()
