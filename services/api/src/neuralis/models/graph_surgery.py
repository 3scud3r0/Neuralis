"""Compile a bounded declarative tensor DAG without executing supplied code."""
from collections import deque
import math
from typing import Literal

from pydantic import Field, model_validator
import torch
from torch import nn

from neuralis.core.schemas import StrictModel


class GraphNode(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,47}$")
    operation: Literal["linear", "relu", "tanh", "gelu", "silu", "identity", "add", "concat", "layernorm"]
    inputs: list[str] = Field(min_length=1, max_length=8)
    width: int | None = Field(default=None, ge=1, le=1024)
    bias: bool = True

    @model_validator(mode="after")
    def check_operation(self):
        if self.id == "input":
            raise ValueError("The input node name is reserved")
        if len(self.inputs) != len(set(self.inputs)):
            raise ValueError("Node inputs must be unique")
        if self.operation in {"add", "concat"}:
            if len(self.inputs) < 2:
                raise ValueError("Add and concat require at least two input nodes")
        elif len(self.inputs) != 1:
            raise ValueError("Unary graph operations require exactly one input")
        if (self.width is not None) != (self.operation == "linear"):
            raise ValueError("Only a linear node requires an explicit output width")
        return self


class GraphDefinition(StrictModel):
    input_width: int = Field(ge=1, le=256)
    nodes: list[GraphNode] = Field(min_length=1, max_length=32)
    output: str = Field(min_length=1, max_length=48)
    seed: int = Field(default=42, ge=0, le=2147483647)

    @model_validator(mode="after")
    def verify_graph(self):
        analyze_graph(self)
        return self


def analyze_graph(definition: GraphDefinition) -> dict:
    nodes = {node.id: node for node in definition.nodes}
    if len(nodes) != len(definition.nodes):
        raise ValueError("Graph node identifiers must be unique")
    if definition.output not in nodes:
        raise ValueError("Output must identify an explicit graph node")
    references = set(nodes) | {"input"}
    if any(source not in references for node in nodes.values() for source in node.inputs):
        raise ValueError("Graph contains an unknown source node")
    adjacency = {name: [] for name in references}
    indegree = {name: len(node.inputs) for name, node in nodes.items()}
    for node in nodes.values():
        for source in node.inputs:
            adjacency[source].append(node.id)
    # Input is the only external value; all other nodes must be resolved from it.
    queue = deque(["input"])
    order, dimensions = [], {"input": definition.input_width}
    parameters = 0
    while queue:
        source = queue.popleft()
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target]:
                continue
            node = nodes[target]
            widths = [dimensions[name] for name in node.inputs]
            if node.operation == "linear":
                width = node.width
                parameters += widths[0] * width + (width if node.bias else 0)
            elif node.operation == "concat":
                width = sum(widths)
                if width > 1024:
                    raise ValueError("Concatenation exceeds the 1024-channel graph budget")
            else:
                width = widths[0]
                if node.operation == "add" and any(value != width for value in widths):
                    raise ValueError("Residual addition requires matching channel widths")
                if node.operation == "layernorm":
                    parameters += 2 * width
            dimensions[target] = width
            order.append(target)
            queue.append(target)
    if len(order) != len(nodes):
        raise ValueError("Graph has a cycle or cannot be reached from its input")
    ancestors = set()
    pending = [definition.output]
    while pending:
        node_id = pending.pop()
        if node_id in ancestors:
            continue
        ancestors.add(node_id)
        if node_id != "input":
            pending.extend(nodes[node_id].inputs)
    if set(nodes) - ancestors:
        raise ValueError("Every graph node must contribute to its output; remove disconnected branches")
    if parameters > 2_000_000:
        raise ValueError("Graph exceeds the two-million parameter CPU budget")
    return {"order": order, "dimensions": dimensions, "parameters": parameters,
            "input_width": definition.input_width, "output": definition.output}


class SeededLinear(nn.Module):
    """Use an explicit generator, keeping compilation out of the global RNG."""
    def __init__(self, inputs, outputs, bias, generator):
        super().__init__()
        limit = 1 / math.sqrt(inputs)
        self.weight = nn.Parameter(torch.empty(outputs, inputs).uniform_(-limit, limit, generator=generator))
        self.bias = nn.Parameter(torch.empty(outputs).uniform_(-limit, limit, generator=generator)) if bias else None

    def forward(self, value):
        return nn.functional.linear(value, self.weight, self.bias)


class TensorGraph(nn.Module):
    """A real autograd graph with named nodes, residuals and concatenation.

    Inputs use [batch, channels]. Recurrent cycles are rejected; recurrence needs
    an explicitly unrolled definition and remains subject to the same limits.
    """
    def __init__(self, definition: GraphDefinition):
        super().__init__()
        self.definition = definition.model_copy(deep=True)
        self.analysis = analyze_graph(self.definition)
        self.nodes = {node.id: node for node in self.definition.nodes}
        operations = {}
        generator = torch.Generator().manual_seed(self.definition.seed)
        nonlinear = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU,
                     "silu": nn.SiLU, "identity": nn.Identity}
        for name in self.analysis["order"]:
            node = self.nodes[name]
            width = self.analysis["dimensions"][node.inputs[0]]
            if node.operation == "linear":
                operations[name] = SeededLinear(width, node.width, node.bias, generator)
            elif node.operation == "layernorm":
                operations[name] = nn.LayerNorm(width)
            elif node.operation in nonlinear:
                operations[name] = nonlinear[node.operation]()
        self.operations = nn.ModuleDict(operations)

    def forward(self, inputs, *, return_nodes: bool = False):
        if inputs.ndim != 2 or inputs.shape[1] != self.definition.input_width or not 1 <= inputs.shape[0] <= 512:
            raise ValueError("Graph inputs must use [1..512 examples, declared input channels]")
        if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
            raise ValueError("Graph inputs must contain finite floating values")
        values = {"input": inputs}
        for name in self.analysis["order"]:
            node = self.nodes[name]
            incoming = [values[source] for source in node.inputs]
            if node.operation == "add":
                value = torch.stack(incoming, dim=0).sum(dim=0)
            elif node.operation == "concat":
                value = torch.cat(incoming, dim=-1)
            else:
                value = self.operations[name](incoming[0])
            if not torch.isfinite(value).all():
                raise ValueError(f"Graph node {name} produced nonfinite values")
            values[name] = value
        output = values[self.definition.output]
        return (output, values) if return_nodes else output

    def recompile(self, definition: GraphDefinition, *, preserve_weights: bool = True):
        """Create a separate graph; preserve only exact name/shape/dtype matches.

        The source stays unchanged. Optimizer state is intentionally not copied
        into a structurally edited graph; construct a new optimizer explicitly.
        """
        replacement = TensorGraph(definition)
        retained, initialized = [], []
        source = self.state_dict()
        destination = replacement.state_dict()
        if preserve_weights:
            for name, value in destination.items():
                if name in source and source[name].shape == value.shape and source[name].dtype == value.dtype:
                    destination[name] = source[name].detach().cpu().clone()
                    retained.append(name)
                else:
                    initialized.append(name)
            replacement.load_state_dict(destination)
        else:
            initialized = list(destination)
        return replacement, {"retained_parameters": retained, "initialized_parameters": initialized,
                             "source_unchanged": True, "optimizer": "new optimizer required"}


def compile_graph(definition: GraphDefinition | dict) -> TensorGraph:
    return TensorGraph(GraphDefinition.model_validate(definition))
