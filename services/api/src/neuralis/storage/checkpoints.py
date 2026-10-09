import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from safetensors.torch import load_file, save_file
import torch


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class CheckpointStore:
    def __init__(self, directory: Path):
        self.directory = directory / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, run_id, config, model, optimizer, generator, step):
        checkpoint_id = str(uuid.uuid4())
        temporary = self.directory / (".tmp-" + checkpoint_id)
        destination = self.directory / checkpoint_id
        temporary.mkdir()
        try:
            weights = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
            save_file(weights, str(temporary / "weights.safetensors"))
            # Only tensors and basic Python structures; reader always uses weights_only=True.
            state = {"optimizer": optimizer.state_dict(), "generator": generator.get_state(), "step": step}
            torch.save(state, temporary / "training.pt")
            manifest = {"version": 1, "id": checkpoint_id, "run_id": run_id, "config": config.model_dump(),
                        "step": step, "created_at": time.time(), "torch_version": str(torch.__version__),
                        "files": {name: digest(temporary / name) for name in ("weights.safetensors", "training.pt")},
                        "shapes": {name: list(value.shape) for name, value in weights.items()}}
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temporary, destination)
            return manifest
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def load(self, checkpoint_id, model, optimizer=None, generator=None):
        # UUID validation prevents access to arbitrary paths.
        canonical = str(uuid.UUID(checkpoint_id))
        directory = self.directory / canonical
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != 1:
            raise ValueError("Unsupported checkpoint version")
        expected_names = {"weights.safetensors", "training.pt"}
        if set(manifest["files"]) != expected_names:
            raise ValueError("Invalid checkpoint file manifest")
        for name, expected in manifest["files"].items():
            if digest(directory / name) != expected:
                raise ValueError(f"Checksum mismatch: {name}")
        weights = load_file(str(directory / "weights.safetensors"), device="cpu")
        current = model.state_dict()
        if set(weights) != set(current):
            raise ValueError("Checkpoint architecture parameter names do not match")
        for name, value in weights.items():
            if value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError(f"Checkpoint tensor mismatch: {name}")
        training = torch.load(directory / "training.pt", map_location="cpu", weights_only=True)
        if set(training) != {"optimizer", "generator", "step"}:
            raise ValueError("Invalid checkpoint training state")
        model.load_state_dict(weights, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(training["optimizer"])
        if generator is not None:
            generator.set_state(training["generator"])
        return manifest
