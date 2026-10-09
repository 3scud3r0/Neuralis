"""Local activation capture, including recurrent modules invoked multiple times.

Stores tensors, not natural-language chain-of-thought. Never forwards captures to
third-party endpoints. File output must be explicitly requested by the caller.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any
import torch
from torch import Tensor, nn
from .core import _tensor_from_output


@dataclass(frozen=True)
class TraceEntry:
    stage: int
    module: str
    activation: Tensor

    def summary(self) -> dict[str, Any]:
        x = self.activation.float()
        return {"stage": self.stage, "module": self.module,
                "shape": list(x.shape), "dtype": str(self.activation.dtype), "l2_norm": float(x.norm()),
                "mean": float(x.mean()), "std": float(x.std(unbiased=False))}


class LocalActivationTrace:
    """Context-managed recorder for PyTorch modules with bounded CPU captures.

    mode='last_token' expects [batch,tokens,d]; mode='all' keeps the full output.
    On maximum capacity, excess activations are discarded rather than allocating.
    Do not run concurrently on a shared model without coordinating hooks.
    """
    def __init__(self, model: nn.Module, module_names: list[str], *,
                 mode: str = "last_token", max_entries: int = 128,
                 max_bytes: int = 64 * 1024 * 1024):
        modules = dict(model.named_modules())
        if not module_names or len(set(module_names)) != len(module_names):
            raise ValueError("module_names must be nonempty and distinct")
        if any(name not in modules for name in module_names):
            raise KeyError("Unknown module in module_names")
        if mode not in ("last_token", "all") or max_entries < 1 or max_bytes < 1:
            raise ValueError("Invalid recording options")
        self.model = model
        self.names = list(module_names)
        self.mode = mode
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries: list[TraceEntry] = []
        self.dropped = 0
        self._used_bytes = 0
        self._handles = []
        self._active = False

    def _capture(self, name: str):
        def hook(_module, _inputs, output):
            x = _tensor_from_output(output)
            if self.mode == "last_token":
                if x.ndim != 3:
                    raise ValueError("last_token requires [batch,tokens,features]")
                x = x[:, -1, :]
            nbytes = x.numel() * x.element_size()
            if len(self.entries) >= self.max_entries or nbytes + self._used_bytes > self.max_bytes:
                self.dropped += 1
                return
            self.entries.append(TraceEntry(len(self.entries), name, x.detach().cpu().clone()))
            self._used_bytes += nbytes
        return hook

    def __enter__(self):
        if self._active:
            raise RuntimeError("Already recording")
        self.entries.clear()
        self._used_bytes = 0
        self.dropped = 0
        self._active = True
        modules = dict(self.model.named_modules())
        try:
            for name in self.names:
                self._handles.append(modules[name].register_forward_hook(self._capture(name)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False
        return False

    def summaries(self) -> list[dict[str, Any]]:
        return [entry.summary() for entry in self.entries]

    def export_npz(self, filepath: str | Path) -> Path:
        """Explicit local persistence; requires numpy (normally installed with PyTorch)."""
        import numpy as np
        file = Path(filepath)
        file.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"activation_{idx:04d}": e.activation.float().numpy() if e.activation.dtype == torch.bfloat16 else e.activation.numpy() for idx, e in enumerate(self.entries)}
        np.savez_compressed(file, **arrays)
        file.with_suffix(".json").write_text(json.dumps({"entries": self.summaries(),
                                  "dropped": self.dropped,
                                  "warning": "Activations are not hidden reasoning transcripts"},
                                  indent=2, ensure_ascii=False), encoding="utf8")
        return file


def capture_hf_hidden_states(model: Any, input_ids: Tensor, *,
                             attention_mask: Tensor | None = None,
                             last_token: bool = True) -> tuple[Any, list[Tensor]]:
    """Inspect a LOCAL transformers model with output_hidden_states support.

    Does not work with ordinary hosted text-only API calls. HF may return input
    embedding state as element 0, so number of states can be #blocks+1.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch,tokens]")
    kwargs = {"input_ids": input_ids, "output_hidden_states": True,
              "return_dict": True}
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape mismatch")
        kwargs["attention_mask"] = attention_mask
    with torch.no_grad():
        out = model(**kwargs)
    if not hasattr(out, "hidden_states") or out.hidden_states is None:
        raise ValueError("Model did not return hidden_states")
    states = []
    for hidden in out.hidden_states:
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise ValueError("Unexpected hidden state shape")
        if last_token:
            if attention_mask is None:
                selected = hidden[:, -1, :]
            else:
                # Select rightmost unmasked token (works with left or right padding).
                b, t = torch.where(attention_mask.to(hidden.device).bool())
                if len(b) == 0:
                    raise ValueError("attention_mask has no active tokens")
                last = torch.full((len(input_ids),), -1, dtype=torch.long, device=hidden.device)
                last.scatter_reduce_(0, b, t, reduce="amax", include_self=True)
                if (last < 0).any():
                    raise ValueError("An input has no active tokens")
                selected = hidden[torch.arange(len(input_ids), device=hidden.device), last]
        else:
            selected = hidden
        states.append(selected.detach().cpu().clone())
    return out, states
