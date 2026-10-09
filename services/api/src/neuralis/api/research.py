import json
from typing import Literal

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load, save
import torch

from neuralis.analysis.cross_validation import CrossValidationRequest, cross_validate
from neuralis.analysis.experiments import prepare_input
from neuralis.analysis.representations import attention_details, compare_representations, principal_components
from neuralis.analysis.statistics import classification_metrics, descriptive_statistics, holm_adjust, paired_bootstrap, paired_permutation_test
from neuralis.core.schemas import ArchitectureConfig, InferenceRequest
from neuralis.datasets.synthetic import dataset_for
from neuralis.lab.auth import Principal, principal_for
from neuralis.models.generation import GenerationRequest, generate
from neuralis.models.registry import DEFINITIONS, build_model, model_graph
from neuralis.runtime.devices import module_device, optimizer_for, scheduled_learning_rate
from neuralis.runtime.supervisor import StateConflict
from neuralis.runtime.sweeps import SweepRequest


class RepresentationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: InferenceRequest = Field(default_factory=InferenceRequest)
    modules: list[str] = Field(min_length=1, max_length=12)


class AttentionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: InferenceRequest = Field(default_factory=lambda: InferenceRequest(sample_indices=[0]))
    layer: int = Field(default=0, ge=0, le=11)
    head: int = Field(default=0, ge=0, le=15)
    example: int = Field(default=0, ge=0, le=63)
    token: int = Field(default=0, ge=0, le=127)


class HyperparametersRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float | None = Field(default=None, gt=0, le=1)
    weight_decay: float | None = Field(default=None, ge=0, le=1)
    gradient_clip: float | None = Field(default=None, ge=0, le=1000)


class PairedStatisticsRequest(BaseModel):
    baseline: list[float] = Field(min_length=2, max_length=100000)
    intervention: list[float] = Field(min_length=2, max_length=100000)
    repetitions: int = Field(default=2000, ge=100, le=10000)
    confidence: float = Field(default=0.95, ge=0.5, lt=1)
    seed: int = Field(default=42, ge=0, le=2147483647)


class SummaryStatisticsRequest(BaseModel):
    values: list[float] = Field(min_length=2, max_length=100000)


class PValuesRequest(BaseModel):
    p_values: list[float] = Field(min_length=1, max_length=10000)


def register_research_routes(router, run_access, authorize, require_admin):
    @router.post("/models/{model_id}/graph")
    def configured_graph(model_id: str, body: ArchitectureConfig, seed: int = 42):
        if model_id not in DEFINITIONS:
            raise KeyError(model_id)
        model = build_model(model_id, seed, body)
        data = dataset_for(model_id, seed, body.context_length)
        return {"architecture": body.model_dump(), **model_graph(model, data.validation.inputs[:1])}

    @router.patch("/runs/{run_id}/hyperparameters")
    def hyperparameters(run_id: str, body: HyperparametersRequest, request: Request, principal: Principal = Depends(principal_for)):
        row = run_access(request, principal, run_id, "train")
        supervisor = request.app.state.supervisor
        live = supervisor.get_live(run_id)
        updates = body.model_dump(exclude_none=True)
        if not updates:
            raise ValueError("Specify at least one hyperparameter to update")
        with live.condition:
            if live.state not in {"created", "paused"} or live.in_flight:
                raise StateConflict("Pause and wait for an optimizer boundary before changing hyperparameters")
            with live.model_lock:
                for name, value in updates.items():
                    setattr(live.config, name, value)
                for group in live.optimizer.param_groups:
                    group["lr"] = scheduled_learning_rate(live.config, live.step)
                    group["weight_decay"] = live.config.weight_decay
                with supervisor.repository.transaction() as connection:
                    connection.execute("UPDATE runs SET config=? WHERE id=?", (json.dumps(live.config.model_dump()), run_id))
                supervisor.emit(live, "hyperparameters", {"updates": updates, "applied_at_step": live.step})
        supervisor.repository.audit(principal.subject, row["project_id"], "update-hyperparameters", run_id)
        return {"run_id": run_id, "step": live.step, "updates": updates, "config": live.config.model_dump()}

    @router.get("/runs/{run_id}/weights")
    def export_weights(run_id: str, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id, "export")
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        weights = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
        encoded = save(weights, metadata={"model_id": config.model_id, "step": str(step), "protocol_version": "1.0",
                                          "architecture": json.dumps(config.architecture.model_dump())})
        return Response(encoded, media_type="application/octet-stream", headers={
            "Content-Disposition": f'attachment; filename="{run_id}-step-{step}.safetensors"', "X-Neuralis-Step": str(step)})

    @router.post("/runs/{run_id}/weights")
    async def import_weights(run_id: str, request: Request, principal: Principal = Depends(principal_for)):
        row = run_access(request, principal, run_id, "train")
        supervisor = request.app.state.supervisor
        live = supervisor.get_live(run_id)
        encoded = await request.body()
        if not encoded or len(encoded) > 8 * 1024 * 1024:
            raise ValueError("Safetensors upload must fit the eight-MiB import budget")
        try:
            imported = load(encoded)
        except Exception as error:
            raise ValueError("Upload is not a valid safetensors weight file") from error
        with live.condition, live.model_lock:
            if live.state != "created" or live.step:
                raise StateConflict("Import weights into a fresh created run before training")
            current = live.model.state_dict()
            if set(imported) != set(current):
                raise ValueError("Safetensors parameter names do not match the registered architecture")
            for name, value in imported.items():
                if value.shape != current[name].shape or value.dtype != current[name].dtype:
                    raise ValueError(f"Safetensors shape/dtype mismatch: {name}")
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f"Safetensors contains nonfinite values: {name}")
            live.model.load_state_dict(imported, strict=True)
            live.optimizer = optimizer_for(live.model, live.config)
            supervisor.emit(live, "weights-imported", {"parameters": len(imported), "bytes": len(encoded), "optimizer": "fresh"})
        supervisor.repository.audit(principal.subject, row["project_id"], "import-weights", run_id)
        return {"run_id": run_id, "imported": True, "parameters": len(imported), "optimizer_reset": True}

    @router.post("/runs/{run_id}/generate")
    def generation(run_id: str, body: GenerationRequest, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        return {"run_id": run_id, "snapshot_step": step, **generate(model, body)}

    @router.post("/runs/{run_id}/attention")
    def attention(run_id: str, body: AttentionRequest, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        return {"run_id": run_id, "snapshot_step": step,
                **attention_details(model, data, body.input, body.layer, body.head, body.example, body.token)}

    @router.post("/runs/{run_id}/representations")
    def representations(run_id: str, body: RepresentationRequest, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        return {"run_id": run_id, "snapshot_step": step, **compare_representations(model, data, body.input, body.modules)}

    @router.get("/runs/{run_id}/embeddings")
    def embeddings(run_id: str, request: Request, dimensions: int = Query(3, ge=1, le=3), principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        if not hasattr(model, "embedding"):
            raise ValueError("Embedding projection requires the registered tiny transformer")
        values = model.embedding.weight.detach().cpu()
        return {"run_id": run_id, "snapshot_step": step, "labels": list(data.labels), **principal_components(values, dimensions)}

    @router.post("/runs/{run_id}/classification-report")
    def classification_report(run_id: str, body: InferenceRequest, request: Request, principal: Principal = Depends(principal_for)):
        run_access(request, principal, run_id)
        model, data, config, step = request.app.state.supervisor.snapshot_model(run_id)
        inputs, targets = prepare_input(body, data)
        if targets is None or targets.ndim != 1:
            raise ValueError("Classification report requires labeled classification dataset samples")
        with torch.no_grad():
            probabilities = model(inputs.to(module_device(model))).softmax(-1).cpu().numpy()
        return {"run_id": run_id, "snapshot_step": step, **classification_metrics(probabilities, targets.numpy())}

    @router.post("/statistics/summary")
    def summary(body: SummaryStatisticsRequest):
        return descriptive_statistics(body.values)

    @router.post("/statistics/paired-effects")
    def paired_effects(body: PairedStatisticsRequest):
        return {"bootstrap": paired_bootstrap(body.baseline, body.intervention, body.repetitions, body.confidence, body.seed),
                "permutation": paired_permutation_test(body.baseline, body.intervention, body.repetitions, body.seed)}

    @router.post("/statistics/holm")
    def adjust_pvalues(body: PValuesRequest):
        return {"original": body.p_values, "adjusted": holm_adjust(body.p_values), "method": "Holm family-wise error correction"}

    @router.post("/sweeps", status_code=201)
    def create_sweep(body: SweepRequest, request: Request, principal: Principal = Depends(principal_for)):
        authorize(request, principal, body.base.project_id, "train")
        result = request.app.state.sweeps.create(body)
        request.app.state.repository.audit(principal.subject, body.base.project_id, "create-sweep", result["id"])
        return result

    @router.get("/sweeps")
    def sweeps(request: Request, project_id: str = "local", principal: Principal = Depends(principal_for)):
        authorize(request, principal, project_id)
        return request.app.state.sweeps.list(project_id)

    @router.get("/sweeps/{sweep_id}")
    def sweep(sweep_id: str, request: Request, principal: Principal = Depends(principal_for)):
        value = request.app.state.sweeps.get(sweep_id)
        authorize(request, principal, value["project_id"])
        return value

    @router.post("/sweeps/{sweep_id}/cancel")
    def cancel_sweep(sweep_id: str, request: Request, principal: Principal = Depends(principal_for)):
        value = request.app.state.sweeps.get(sweep_id)
        authorize(request, principal, value["project_id"], "train")
        return request.app.state.sweeps.cancel(sweep_id)

    @router.post("/cross-validation")
    def cross_validation(body: CrossValidationRequest, request: Request, principal: Principal = Depends(principal_for)):
        metadata = request.app.state.supervisor.datasets.metadata(body.dataset_id)
        authorize(request, principal, metadata["project_id"], "train")
        result = cross_validate(request.app.state.supervisor.datasets, body)
        request.app.state.repository.audit(principal.subject, metadata["project_id"], "cross-validation", body.dataset_id)
        return result
