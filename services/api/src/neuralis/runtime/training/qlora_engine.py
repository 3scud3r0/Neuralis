"""Frozen-base LoRA training, with optional portable NF4 storage.

The portable quantized base reconstructs weights for each linear call. For large
CUDA language models, the optional HF/PEFT loader uses its own optimized stack.
"""

from dataclasses import asdict, dataclass
import fnmatch
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from neuralis.runtime.acceleration.backends import resolve_backend
from neuralis.runtime.quantization.nf4_quantizer import NF4Linear


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ()
    block_size: int = 64

    def __post_init__(self):
        if not isinstance(self.rank, int) or not 1 <= self.rank <= 1024:
            raise ValueError("LoRA rank must be an integer between 1 and 1024")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be between zero and one")
        if not isinstance(self.block_size, int) or not 2 <= self.block_size <= 4096:
            raise ValueError("NF4 block size must be between 2 and 4096")
        if isinstance(self.target_modules, str) or len(self.target_modules) > 128:
            raise ValueError("target_modules must be a sequence of at most 128 module names or glob patterns")
        if any(not isinstance(name, str) or not name or len(name) > 256 for name in self.target_modules):
            raise ValueError("Invalid LoRA target module name")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: LoRAConfig, *, quantize_base: bool = False,
                 generator: torch.Generator | None = None):
        super().__init__()
        if base.weight.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValueError("LoRA requires real floating base weights")
        if (base.in_features + base.out_features) * config.rank > 10_000_000:
            raise ValueError("LoRA adapter exceeds the per-layer parameter budget")
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        self.base = NF4Linear(base, config.block_size) if quantize_base else base
        self.base.requires_grad_(False)
        self.dropout = nn.Dropout(config.dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(config.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)
        self.config = config
        self.quantize_base = quantize_base

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base(inputs) + update * self.scaling

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def merged_linear(self) -> nn.Linear:
        """Export a floating inference layer; NF4 reconstruction remains lossy."""
        if isinstance(self.base, NF4Linear):
            from neuralis.runtime.quantization.nf4_quantizer import dequantize_nf4
            weight = dequantize_nf4(self.base.packed, dtype=self.lora_A.dtype, device=self.lora_A.device)
        else:
            weight = self.base.weight.detach()
        bias = self.base.bias
        result = nn.Linear(self.in_features, self.out_features, bias=bias is not None,
                           device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            result.weight.copy_(weight + self.scaling * self.lora_B @ self.lora_A)
            if bias is not None:
                result.bias.copy_(bias)
        return result


def _target_matches(name: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(name, pattern) or name == pattern or name.endswith("." + pattern)


def inject_lora(model: nn.Module, config: LoRAConfig | None = None, *, quantize_base: bool = False,
                seed: int | None = None) -> list[str]:
    """Replace targeted linear children in place and freeze every other parameter.

    Shared/tied selected modules are rejected so injection cannot silently break
    weight tying. Root Linear modules must be wrapped in an nn.Sequential first.
    """
    config = config or LoRAConfig()
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("Model already contains LoRA adapters")
    if isinstance(model, nn.Linear):
        raise ValueError("Wrap a root Linear in nn.Sequential before injecting adapters")
    modules = list(model.named_modules(remove_duplicate=False))
    candidates = [(name, module) for name, module in modules if name and isinstance(module, nn.Linear)]
    selected = [(name, module) for name, module in candidates
                if not config.target_modules or any(_target_matches(name, pattern) for pattern in config.target_modules)]
    if not selected:
        raise ValueError("No linear layers match LoRA targets")
    adapter_parameters = sum((module.in_features + module.out_features) * config.rank for _, module in selected)
    if len(selected) > 256 or adapter_parameters > 20_000_000:
        raise ValueError("LoRA injection exceeds the 256-layer / 20-million-adapter-parameter budget")
    if quantize_base and sum(module.weight.numel() for _, module in selected) > 100_000_000:
        raise ValueError("Portable NF4 injection exceeds the 100-million-base-element budget; use the optional HF loader")
    for pattern in config.target_modules:
        if not any(_target_matches(name, pattern) for name, _ in candidates):
            raise ValueError(f"LoRA target {pattern!r} does not match a linear layer")
    parameter_paths: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_paths.setdefault(id(parameter), []).append(name)
    for name, module in selected:
        if sum(other is module for _, other in modules) != 1:
            raise ValueError(f"Selected module {name} is shared; tied-module injection is unsupported")
        for parameter in module.parameters():
            if len(parameter_paths[id(parameter)]) != 1:
                raise ValueError(f"Selected module {name} has tied parameters; injection is unsupported")
    # Prepare replacements before mutating the caller's model.
    original_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    if seed is not None and (not isinstance(seed, int) or not 0 <= seed < 2 ** 63):
        raise ValueError("LoRA seed must be a nonnegative 63-bit integer")
    generators = {}
    if seed is not None:
        for _, module in selected:
            device = module.weight.device
            if device.type not in {"cpu", "cuda"}:
                raise ValueError("Seeded adapter injection must run on CPU or CUDA before accelerator transfer")
            if device not in generators:
                generators[device] = torch.Generator(device=device).manual_seed(seed)
    try:
        replacements = [(name, LoRALinear(module, config, quantize_base=quantize_base,
                                         generator=generators.get(module.weight.device))) for name, module in selected]
    except BaseException:
        for parameter, flag in original_flags:
            parameter.requires_grad_(flag)
        raise
    model.requires_grad_(False)
    for name, replacement in replacements:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child] = replacement
        replacement.lora_A.requires_grad_(True)
        replacement.lora_B.requires_grad_(True)
    return [name for name, _ in selected]


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if name.rsplit(".", 1)[-1] in {"lora_A", "lora_B"}}


def _base_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    states = {name: tensor for name, tensor in model.state_dict().items()
              if name.rsplit(".", 1)[-1] not in {"lora_A", "lora_B"}}
    for name, tensor in sorted(states.items()):
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        flat = tensor.detach().reshape(-1)
        for begin in range(0, flat.numel(), 1_048_576):
            # Byte views support bfloat16 without NumPy dtype conversions.
            digest.update(flat[begin:begin + 1_048_576].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class QLoRAEngine:
    """Single-owner synchronous training engine with optimizer-boundary snapshots.

    Checkpoints store adapters/optimizer state, not the frozen model. Resume
    requires identical frozen base tensors and adapter configuration. Callers
    provide loss_fn for tasks other than classification/causal token prediction.
    """

    def __init__(self, model: nn.Module, config: LoRAConfig | None = None, *,
                 quantize_base: bool = True, backend: str = "cpu", learning_rate: float = 1e-3,
                 gradient_accumulation: int = 1, max_grad_norm: float = 1.0,
                 loss_fn: Callable | None = None):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not isinstance(gradient_accumulation, int) or not 1 <= gradient_accumulation <= 1024:
            raise ValueError("gradient_accumulation must be between 1 and 1024")
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        self.backend = resolve_backend(backend)
        self.model = model
        existing = [module for module in model.modules() if isinstance(module, LoRALinear)]
        if not existing:
            inject_lora(model, config, quantize_base=quantize_base)
        elif config is not None and any(module.config != config for module in existing):
            raise ValueError("Existing adapters do not match the requested LoRA configuration")
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)
        self.model.to(self.backend.device)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError("No trainable adapter parameters")
        self.optimizer = torch.optim.AdamW(self.parameters, lr=learning_rate)
        self.gradient_accumulation = gradient_accumulation
        self.max_grad_norm = max_grad_norm
        self.loss_fn = loss_fn or self._classification_loss
        self.micro_step = 0
        self.optimizer_step = 0
        self.optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def _classification_loss(logits, targets):
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    def train_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        if inputs.numel() > 16_000_000 or targets.numel() > 16_000_000:
            raise ValueError("Training batch exceeds the 16-million-element tensor budget")
        # Frozen BatchNorm buffers and base dropout must remain deterministic.
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.train()
        inputs, targets = inputs.to(self.backend.device), targets.to(self.backend.device)
        logits = self.model(inputs)
        if hasattr(logits, "logits"):
            logits = logits.logits
        loss = self.loss_fn(logits, targets)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            self.optimizer.zero_grad(set_to_none=True)
            self.micro_step -= self.micro_step % self.gradient_accumulation
            raise ValueError("Training loss must be a finite scalar")
        (loss / self.gradient_accumulation).backward()
        self.micro_step += 1
        applied = self.micro_step % self.gradient_accumulation == 0
        grad_norm = None
        if applied:
            try:
                grad_norm = float(nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm, error_if_nonfinite=True))
            except RuntimeError:
                self.optimizer.zero_grad(set_to_none=True)
                self.micro_step = self.optimizer_step * self.gradient_accumulation
                raise
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_step += 1
        return {"loss": float(loss.detach()), "micro_step": self.micro_step,
                "optimizer_step": self.optimizer_step, "optimizer_applied": applied, "gradient_norm": grad_norm,
                "trainable_parameters": sum(p.numel() for p in self.parameters)}

    @torch.no_grad()
    def evaluate_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        previous = self.model.training
        self.model.eval()
        try:
            logits = self.model(inputs.to(self.backend.device))
            if hasattr(logits, "logits"):
                logits = logits.logits
            loss = self.loss_fn(logits, targets.to(self.backend.device))
            return {"loss": float(loss), "examples": inputs.shape[0]}
        finally:
            self.model.train(previous)

    def _manifest(self) -> dict:
        return {name: {"config": asdict(module.config), "quantize_base": module.quantize_base}
                for name, module in self.model.named_modules() if isinstance(module, LoRALinear)}

    def save_checkpoint(self, path: str | Path) -> dict:
        if self.micro_step % self.gradient_accumulation:
            raise ValueError("Checkpoint must be saved at an optimizer boundary")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "neuralis-lora-v1", "adapters": adapter_state(self.model),
                   "manifest": self._manifest(), "base_sha256": _base_fingerprint(self.model),
                   "optimizer": self.optimizer.state_dict(), "micro_step": self.micro_step,
                   "optimizer_step": self.optimizer_step, "gradient_accumulation": self.gradient_accumulation,
                   "max_grad_norm": self.max_grad_norm, "cpu_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if self.backend.device.type == "cuda" else []}
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {"path": str(path), "optimizer_step": self.optimizer_step,
                "base_sha256": payload["base_sha256"], "bytes": path.stat().st_size}

    def load_checkpoint(self, path: str | Path, *, restore_rng: bool = True) -> dict:
        if self.micro_step % self.gradient_accumulation:
            raise ValueError("Resume must occur at an optimizer boundary")
        path = Path(path)
        if path.stat().st_size > 512 * 1024 * 1024:
            raise ValueError("Adapter checkpoint exceeds the 512 MiB budget")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "neuralis-lora-v1" or payload.get("manifest") != self._manifest():
            raise ValueError("Checkpoint format or adapter configuration does not match")
        if payload["base_sha256"] != _base_fingerprint(self.model):
            raise ValueError("Checkpoint frozen base fingerprint does not match this model")
        if payload["gradient_accumulation"] != self.gradient_accumulation:
            raise ValueError("Checkpoint gradient accumulation does not match")
        expected = adapter_state(self.model)
        incoming = payload["adapters"]
        if expected.keys() != incoming.keys() or any(
            incoming[name].shape != value.shape or incoming[name].dtype != value.dtype
            or not bool(torch.isfinite(incoming[name]).all()) for name, value in expected.items()
        ):
            raise ValueError("Invalid checkpoint adapter tensors")
        if payload["micro_step"] % self.gradient_accumulation or payload["optimizer_step"] != payload["micro_step"] // self.gradient_accumulation:
            raise ValueError("Checkpoint step counters are inconsistent")
        self.model.load_state_dict(incoming, strict=False)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.micro_step, self.optimizer_step = payload["micro_step"], payload["optimizer_step"]
        self.max_grad_norm = payload["max_grad_norm"]
        self.optimizer.zero_grad(set_to_none=True)
        if restore_rng:
            torch.set_rng_state(payload["cpu_rng"])
            if self.backend.device.type == "cuda" and payload["cuda_rng"]:
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return {"optimizer_step": self.optimizer_step, "micro_step": self.micro_step,
                "base_sha256": payload["base_sha256"]}


def load_hf_qlora(model_id: str, config: LoRAConfig | None = None, *,
                  backend: str = "cuda", local_files_only: bool = True):
    """Explicit optional integration; never download weights unless opted in.

    Requires transformers, peft, bitsandbytes and a compatible CUDA installation.
    Model repositories are loaded with trust_remote_code=False.
    """
    selected = resolve_backend(backend)
    if selected.name != "cuda":
        raise RuntimeError("The optional HF four-bit loader requires CUDA; use the portable CPU engine otherwise")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
        import bitsandbytes  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Install optional transformers, peft and bitsandbytes packages for HF QLoRA") from error
    config = config or LoRAConfig()
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float32, bnb_4bit_use_double_quant=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization,
                                                device_map={"": str(selected.device)},
                                                local_files_only=local_files_only, trust_remote_code=False)
    model = prepare_model_for_kbit_training(model)
    target_modules = list(config.target_modules) if config.target_modules else "all-linear"
    model = get_peft_model(model, PeftLoraConfig(r=config.rank, lora_alpha=config.alpha,
                                               lora_dropout=config.dropout, target_modules=target_modules,
                                               bias="none", task_type="CAUSAL_LM"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only, trust_remote_code=False)
    return model, tokenizer
