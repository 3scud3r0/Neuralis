from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


TERMINAL_STATES = {RunState.COMPLETED, RunState.STOPPED, RunState.FAILED, RunState.INTERRUPTED}


class CaptureConfig(StrictModel):
    modules: list[str] = Field(default_factory=list, max_length=128)
    every: int = Field(default=10, ge=1, le=10000)
    max_elements: int = Field(default=1024, ge=16, le=65536)
    max_tensors: int = Field(default=64, ge=1, le=256)
    histogram_bins: int = Field(default=32, ge=4, le=128)
    activations: bool = True
    gradients: bool = True
    weights: bool = True


class RunConfig(StrictModel):
    model_id: Literal["mlp-xor", "mlp-spiral", "cnn-shapes", "tiny-transformer"] = "mlp-xor"
    name: str = Field(default="Untitled experiment", min_length=1, max_length=120)
    project_id: str = "local"
    seed: int = Field(default=42, ge=0, le=2147483647)
    steps: int = Field(default=500, ge=1, le=100000)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    weight_decay: float = Field(default=0, ge=0, le=1)
    gradient_clip: float = Field(default=5, ge=0, le=1000)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    capture_every: int | None = Field(default=None, ge=1, le=10000)
    checkpoint_every: int = Field(default=0, ge=0, le=100000)
    validation_every: int = Field(default=10, ge=1, le=10000)
    step_delay_ms: int = Field(default=0, ge=0, le=1000)
    device: Literal["cpu", "cuda", "mps", "rocm", "auto"] = "cpu"
    architecture: "ArchitectureConfig" = Field(default_factory=lambda: ArchitectureConfig())
    optimizer: "OptimizerConfig" = Field(default_factory=lambda: OptimizerConfig())
    schedule: "ScheduleConfig" = Field(default_factory=lambda: ScheduleConfig())
    dataset_id: str | None = Field(default=None, max_length=128)
    adapter: "AdapterConfig" = Field(default_factory=lambda: AdapterConfig())

    @model_validator(mode="after")
    def compatibility_capture_frequency(self):
        if self.capture_every is not None:
            self.capture.every = self.capture_every
        return self


class CommandRequest(StrictModel):
    command: Literal["start", "pause", "step", "resume", "stop", "checkpoint"]
    command_id: str = Field(default="", max_length=128)


class InferenceRequest(StrictModel):
    inputs: list[Any] | None = None
    sample_indices: list[int] = Field(default_factory=lambda: [0, 1, 2, 3], max_length=64)
    split: Literal["train", "validation"] = "validation"


class InterventionSpec(StrictModel):
    module: str = Field(min_length=1, max_length=200)
    operation: Literal["ablate", "scale", "steer", "patch"]
    strength: float = Field(default=1, ge=-100, le=100)
    indices: list[int] | None = Field(default=None, max_length=4096)
    vector: list[float] | None = Field(default=None, max_length=65536)


class ExperimentRequest(StrictModel):
    run_id: str
    input: InferenceRequest = Field(default_factory=InferenceRequest)
    interventions: list[InterventionSpec] = Field(min_length=1, max_length=16)
    clean_input: InferenceRequest | None = None
    target_class: int | None = Field(default=None, ge=0, le=4096)


class AttributionRequest(StrictModel):
    input: InferenceRequest = Field(default_factory=InferenceRequest)
    method: Literal["gradient-input", "integrated-gradients"] = "integrated-gradients"
    target_class: int | None = Field(default=None, ge=0, le=4096)
    integration_steps: int = Field(default=32, ge=2, le=256)


class Event(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    run_id: str
    sequence: int
    timestamp: float
    step: int
    type: str
    payload: dict[str, Any]
    rank: int = 0
    world_size: int = 1


class Problem(BaseModel):
    category: str
    message: str
    action: str


class ArchitectureConfig(StrictModel):
    hidden_widths: list[int] = Field(default_factory=list, max_length=8)
    activation: Literal["tanh", "relu", "gelu", "silu"] = "tanh"
    convolution_channels: list[int] = Field(default_factory=lambda: [12, 24], min_length=2, max_length=2)
    transformer_width: int = Field(default=48, ge=8, le=512)
    transformer_depth: int = Field(default=2, ge=1, le=12)
    transformer_heads: int = Field(default=4, ge=1, le=16)
    context_length: int = Field(default=32, ge=8, le=128)

    @model_validator(mode="after")
    def validate_dimensions(self):
        if any(width < 2 or width > 1024 for width in self.hidden_widths):
            raise ValueError("Hidden layer widths must be between 2 and 1024")
        if any(width < 2 or width > 256 for width in self.convolution_channels):
            raise ValueError("Convolution channels must be between 2 and 256")
        if self.transformer_width % self.transformer_heads:
            raise ValueError("Transformer width must be divisible by the number of heads")
        return self


class OptimizerConfig(StrictModel):
    name: Literal["adamw", "adam", "sgd", "rmsprop"] = "adamw"
    momentum: float = Field(default=0.9, ge=0, lt=1)
    beta1: float = Field(default=0.9, ge=0, lt=1)
    beta2: float = Field(default=0.999, ge=0, lt=1)
    epsilon: float = Field(default=1e-8, gt=0, le=0.1)
    nesterov: bool = False

    @model_validator(mode="after")
    def validate_nesterov(self):
        if self.nesterov and (self.name != "sgd" or self.momentum <= 0):
            raise ValueError("Nesterov requires SGD with positive momentum")
        return self


class AdapterConfig(StrictModel):
    """Opt-in low-rank training; registered architecture and data stay unchanged."""
    enabled: bool = False
    rank: int = Field(default=4, ge=1, le=64)
    alpha: float = Field(default=8, gt=0, le=256)
    target_modules: list[str] = Field(default_factory=list, max_length=32)
    quantize_base: bool = False

    @model_validator(mode="after")
    def validate_adapter(self):
        if self.quantize_base and not self.enabled:
            raise ValueError("Quantized-base training requires an enabled low-rank adapter")
        if any(not name or len(name) > 200 for name in self.target_modules):
            raise ValueError("Adapter target modules must be nonempty bounded module paths")
        if len(self.target_modules) != len(set(self.target_modules)):
            raise ValueError("Adapter target modules must be unique")
        return self


class ScheduleConfig(StrictModel):
    name: Literal["constant", "cosine", "exponential", "step"] = "constant"
    warmup_steps: int = Field(default=0, ge=0, le=10000)
    minimum_ratio: float = Field(default=0.05, ge=0, le=1)
    gamma: float = Field(default=0.95, gt=0, le=1)
    step_size: int = Field(default=100, ge=1, le=100000)


RunConfig.model_rebuild()
