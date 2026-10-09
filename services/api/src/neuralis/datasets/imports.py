import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Literal
import uuid

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from neuralis.datasets.synthetic import DataSplit, DatasetBundle
from neuralis.models.registry import DEFINITIONS


class DatasetImport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    project_id: str = "local"
    model_id: Literal["mlp-xor", "mlp-spiral", "cnn-shapes"]
    inputs: list[Any] = Field(min_length=4, max_length=10000)
    targets: list[int] = Field(min_length=4, max_length=10000)
    validation_fraction: float = Field(default=0.2, ge=0.1, le=0.5)
    seed: int = Field(default=42, ge=0, le=2147483647)
    preprocessing: Literal["none", "standardize"] = "none"
    description: str = Field(default="User supplied numeric classification data", max_length=2000)


class CSVImport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    project_id: str = "local"
    model_id: Literal["mlp-xor", "mlp-spiral"]
    csv_text: str = Field(min_length=1, max_length=2000000)
    feature_columns: list[str] = Field(min_length=2, max_length=2)
    target_column: str = Field(min_length=1, max_length=120)
    validation_fraction: float = Field(default=0.2, ge=0.1, le=0.5)
    preprocessing: Literal["none", "standardize"] = "none"
    seed: int = 42


def parse_csv(request: CSVImport):
    reader = csv.DictReader(io.StringIO(request.csv_text))
    if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("CSV needs a header with unique column names")
    required = {*request.feature_columns, request.target_column}
    if not required.issubset(reader.fieldnames):
        raise ValueError("Selected feature/target columns are absent from the CSV header")
    inputs, targets = [], []
    for index, row in enumerate(reader):
        if index >= 10000:
            raise ValueError("CSV import is limited to 10,000 rows")
        try:
            inputs.append([float(row[column]) for column in request.feature_columns])
            label = float(row[request.target_column])
            if not label.is_integer():
                raise ValueError("Target labels must be integers")
            targets.append(int(label))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid numeric CSV value on row {index + 2}") from error
    return DatasetImport(name=request.name, project_id=request.project_id, model_id=request.model_id,
                         inputs=inputs, targets=targets, validation_fraction=request.validation_fraction,
                         seed=request.seed, preprocessing=request.preprocessing)


def stratified_indices(targets: np.ndarray, fraction: float, seed: int):
    generator = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(targets):
        indices = np.flatnonzero(targets == label)
        if len(indices) < 2:
            raise ValueError("Each represented class needs at least two observations for independent splits")
        generator.shuffle(indices)
        validation_count = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        validation.extend(indices[:validation_count])
        train.extend(indices[validation_count:])
    generator.shuffle(train)
    generator.shuffle(validation)
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


class DatasetStore:
    def __init__(self, repository):
        self.repository = repository
        self.directory = repository.directory / "datasets"
        self.directory.mkdir(exist_ok=True)
        with repository.transaction() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS imported_datasets(id TEXT PRIMARY KEY,project_id TEXT NOT NULL,metadata TEXT NOT NULL)")

    def import_numeric(self, request: DatasetImport):
        definition = DEFINITIONS[request.model_id]
        try:
            inputs = np.asarray(request.inputs, dtype=np.float32)
            targets = np.asarray(request.targets, dtype=np.int64)
        except (TypeError, ValueError) as error:
            raise ValueError("Dataset inputs must be a rectangular numeric array") from error
        if inputs.ndim != len(definition.input_shape) + 1 or tuple(inputs.shape[1:]) != definition.input_shape:
            raise ValueError(f"Dataset inputs must match [N,{','.join(map(str,definition.input_shape))}]")
        if len(inputs) != len(targets) or inputs.size > 2000000:
            raise ValueError("Dataset lengths differ or the two-million-element import budget was exceeded")
        if not np.isfinite(inputs).all() or np.abs(inputs).max() > 1e6:
            raise ValueError("Dataset inputs must be finite and have magnitude at most one million")
        if targets.min() < 0 or targets.max() >= definition.output_classes:
            raise ValueError("Target labels do not match the registered output classes")
        if len(np.unique(targets)) != definition.output_classes:
            raise ValueError("Imported classification data must represent every registered output class")
        training, validation = stratified_indices(targets, request.validation_fraction, request.seed)
        preprocessing = {"method": request.preprocessing, "fitted_on": "training split only"}
        if request.preprocessing == "standardize":
            mean = inputs[training].mean(axis=0)
            scale = inputs[training].std(axis=0).clip(min=1e-6)
            inputs = (inputs - mean) / scale
            preprocessing.update(mean=mean.tolist(), scale=scale.tolist())
        dataset_id = str(uuid.uuid4())
        temporary = self.directory / (".tmp-" + dataset_id + ".npz")
        destination = self.directory / (dataset_id + ".npz")
        np.savez_compressed(temporary, inputs=inputs, targets=targets, training=training, validation=validation)
        os.replace(temporary, destination)
        fingerprint = hashlib.sha256(destination.read_bytes()).hexdigest()
        metadata = {"id": dataset_id, "name": request.name, "project_id": request.project_id,
                    "model_id": request.model_id, "description": request.description, "seed": request.seed,
                    "training_examples": len(training), "validation_examples": len(validation),
                    "shape": list(inputs.shape), "labels": list(range(definition.output_classes)),
                    "preprocessing": preprocessing, "sha256": fingerprint, "format": "numpy-npz-no-pickle"}
        try:
            with self.repository.transaction() as connection:
                connection.execute("INSERT INTO imported_datasets VALUES(?,?,?)", (dataset_id, request.project_id, json.dumps(metadata)))
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return metadata

    def metadata(self, dataset_id):
        canonical = str(uuid.UUID(dataset_id))
        with self.repository.transaction() as connection:
            row = connection.execute("SELECT metadata FROM imported_datasets WHERE id=?", (canonical,)).fetchone()
        if row is None:
            raise KeyError(dataset_id)
        return json.loads(row["metadata"])

    def list(self, project_id):
        with self.repository.transaction() as connection:
            rows = connection.execute("SELECT metadata FROM imported_datasets WHERE project_id=?", (project_id,)).fetchall()
        return [json.loads(row["metadata"]) for row in rows]

    def load(self, dataset_id, model_id):
        metadata = self.metadata(dataset_id)
        if metadata["model_id"] != model_id:
            raise ValueError("Imported dataset belongs to a different registered task")
        path = self.directory / (metadata["id"] + ".npz")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Imported dataset checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"inputs", "targets", "training", "validation"}:
                raise ValueError("Invalid numeric dataset archive")
            inputs = torch.from_numpy(archive["inputs"].copy())
            targets = torch.from_numpy(archive["targets"].copy())
            training = archive["training"]
            validation = archive["validation"]
        train_split = DataSplit(inputs[training], targets[training])
        validation_split = DataSplit(inputs[validation], targets[validation])
        return DatasetBundle(metadata["id"], train_split, validation_split,
                             tuple(str(label) for label in metadata["labels"]), "classification", metadata["description"])
