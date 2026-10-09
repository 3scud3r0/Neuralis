export type ArchitectureOperation =
  | "input"
  | "dense"
  | "relu"
  | "tanh"
  | "sigmoid"
  | "softmax"
  | "layernorm"
  | "add"
  | "concat"
  | "output";
export interface ArchitectureNode {
  id: string;
  name: string;
  operation: ArchitectureOperation;
  shape?: number[];
  units?: number;
  axis?: number;
  bias?: boolean;
}
export interface ArchitectureEdge {
  source: string;
  target: string;
}
export interface ArchitectureGraph {
  format: "neuralis.architecture";
  version: 1;
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
}
export interface ArchitectureIssue {
  severity: "error" | "warning";
  message: string;
  node?: string;
  edge?: number;
}
export interface ArchitectureAnalysis {
  issues: ArchitectureIssue[];
  order: string[];
  shapes: Map<string, number[]>;
  parameters: Map<string, number>;
  totalParameters: number;
  depth: Map<string, number>;
}
export interface ArchitecturePosition {
  id: string;
  x: number;
  y: number;
  depth: number;
}
const OPERATIONS: ArchitectureOperation[] = [
  "input",
  "dense",
  "relu",
  "tanh",
  "sigmoid",
  "softmax",
  "layernorm",
  "add",
  "concat",
  "output",
];

function equalShape(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((size, axis) => size === b[axis]);
}
function validShape(shape: number[]): boolean {
  return (
    shape.length > 0 &&
    shape.length <= 8 &&
    shape.every(
      (size) => Number.isInteger(size) && size > 0 && size <= 1000000,
    ) &&
    shape.reduce((size, value) => size * value, 1) <= 100000000
  );
}

export function graphAdjacency(graph: ArchitectureGraph): {
  incoming: Map<string, string[]>;
  outgoing: Map<string, string[]>;
} {
  const incoming = new Map(
    graph.nodes.map((node) => [node.id, [] as string[]]),
  );
  const outgoing = new Map(
    graph.nodes.map((node) => [node.id, [] as string[]]),
  );
  for (const edge of graph.edges) {
    if (!incoming.has(edge.target) || !outgoing.has(edge.source)) continue;
    incoming.get(edge.target)!.push(edge.source);
    outgoing.get(edge.source)!.push(edge.target);
  }
  return { incoming, outgoing };
}

export function topologicalOrder(graph: ArchitectureGraph): string[] {
  const { incoming, outgoing } = graphAdjacency(graph);
  const degrees = new Map(
    Array.from(incoming, ([id, nodes]) => [id, nodes.length]),
  );
  const queue = graph.nodes
    .filter((node) => degrees.get(node.id) === 0)
    .map((node) => node.id);
  const order: string[] = [];
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const id = queue[cursor];
    order.push(id);
    for (const target of outgoing.get(id) || []) {
      const degree = degrees.get(target)! - 1;
      degrees.set(target, degree);
      if (!degree) queue.push(target);
    }
  }
  if (order.length !== graph.nodes.length)
    throw new Error(
      "Architecture contains a directed cycle or duplicate node IDs.",
    );
  return order;
}

export function analyzeArchitecture(
  graph: ArchitectureGraph,
): ArchitectureAnalysis {
  const issues: ArchitectureIssue[] = [];
  const shapes = new Map<string, number[]>();
  const parameters = new Map<string, number>();
  const depth = new Map<string, number>();
  const result: ArchitectureAnalysis = {
    issues,
    order: [],
    shapes,
    parameters,
    totalParameters: 0,
    depth,
  };
  if (graph.nodes.length > 256 || graph.edges.length > 2048) {
    issues.push({
      severity: "error",
      message:
        "Architecture exceeds the preview budget of 256 nodes and 2,048 edges.",
    });
    return result;
  }
  const nodes = new Map<string, ArchitectureNode>();
  for (const node of graph.nodes) {
    if (!node.id || node.id.length > 120)
      issues.push({
        severity: "error",
        message: "Node ID must have between 1 and 120 characters.",
        node: node.id,
      });
    if (nodes.has(node.id))
      issues.push({
        severity: "error",
        message: "Node IDs must be unique.",
        node: node.id,
      });
    if (!OPERATIONS.includes(node.operation))
      issues.push({
        severity: "error",
        message: "Unknown architecture operation.",
        node: node.id,
      });
    nodes.set(node.id, node);
  }
  const edges = new Set<string>();
  graph.edges.forEach((edge, index) => {
    if (!nodes.has(edge.source) || !nodes.has(edge.target))
      issues.push({
        severity: "error",
        message: "Edge refers to a missing node.",
        edge: index,
      });
    if (edge.source === edge.target)
      issues.push({
        severity: "error",
        message: "Self connections are not allowed in an acyclic architecture.",
        edge: index,
      });
    const key = JSON.stringify([edge.source, edge.target]);
    if (edges.has(key))
      issues.push({
        severity: "error",
        message: "Duplicate edges are not allowed.",
        edge: index,
      });
    edges.add(key);
  });
  if (issues.some((issue) => issue.severity === "error")) return result;
  try {
    result.order = topologicalOrder(graph);
  } catch (error) {
    issues.push({
      severity: "error",
      message: error instanceof Error ? error.message : "Invalid graph.",
    });
    return result;
  }
  const { incoming, outgoing } = graphAdjacency(graph);
  for (const id of result.order) {
    const node = nodes.get(id)!;
    const sources = incoming.get(id)!;
    const sourceShapes = sources.map((source) => shapes.get(source));
    const error = (message: string) =>
      issues.push({ severity: "error", message, node: id });
    const warning = (message: string) =>
      issues.push({ severity: "warning", message, node: id });
    depth.set(
      id,
      sources.reduce(
        (max, source) => Math.max(max, (depth.get(source) || 0) + 1),
        0,
      ),
    );
    parameters.set(id, 0);
    if (node.operation === "input") {
      if (sources.length) error("Input nodes cannot have incoming edges.");
      if (!node.shape || !validShape(node.shape))
        error(
          "Input shape must contain positive dimensions within the preview budget.",
        );
      else shapes.set(id, [...node.shape]);
      continue;
    }
    if (!sources.length) {
      error("Non-input operations require an incoming connection.");
      continue;
    }
    if (sourceShapes.some((shape) => !shape)) {
      error("An upstream shape is unavailable. Fix the preceding node first.");
      continue;
    }
    const inputs = sourceShapes as number[][];
    const first = inputs[0];
    if (
      node.operation !== "add" &&
      node.operation !== "concat" &&
      sources.length !== 1
    ) {
      error("This operation accepts exactly one input.");
      continue;
    }
    if (node.operation === "dense") {
      if (
        !Number.isInteger(node.units) ||
        !node.units ||
        node.units < 1 ||
        node.units > 1000000
      ) {
        error("Dense units must be a positive integer at most 1,000,000.");
        continue;
      }
      const output = [...first];
      output[output.length - 1] = node.units;
      shapes.set(id, output);
      const count =
        first[first.length - 1] * node.units +
        (node.bias === false ? 0 : node.units);
      parameters.set(id, count);
    } else if (node.operation === "add") {
      if (inputs.some((shape) => !equalShape(shape, first))) {
        error(
          "Add requires identical shapes; implicit broadcasting is not supported.",
        );
        continue;
      }
      if (inputs.length < 2) warning("Add currently receives only one branch.");
      shapes.set(id, [...first]);
    } else if (node.operation === "concat") {
      const axis = node.axis ?? first.length - 1;
      if (!Number.isInteger(axis) || axis < 0 || axis >= first.length) {
        error("Concatenation axis is outside the input rank.");
        continue;
      }
      if (
        inputs.some(
          (shape) =>
            shape.length !== first.length ||
            shape.some(
              (size, index) => index !== axis && size !== first[index],
            ),
        )
      ) {
        error("Concat inputs must match on every non-concatenated axis.");
        continue;
      }
      const output = [...first];
      output[axis] = inputs.reduce((sum, shape) => sum + shape[axis], 0);
      shapes.set(id, output);
    } else {
      shapes.set(id, [...first]);
      if (node.operation === "layernorm")
        parameters.set(id, first[first.length - 1] * 2);
    }
    if (!outgoing.get(id)?.length && node.operation !== "output")
      warning("This operation is a terminal node without an explicit output.");
  }
  if (!graph.nodes.some((node) => node.operation === "input"))
    issues.push({
      severity: "error",
      message: "Architecture requires at least one input node.",
    });
  if (!graph.nodes.some((node) => node.operation === "output"))
    issues.push({
      severity: "warning",
      message: "Architecture has no explicit output node.",
    });
  const inputs = graph.nodes
    .filter((node) => node.operation === "input")
    .map((node) => node.id);
  const reachable = new Set<string>(inputs);
  const queue = [...inputs];
  for (let i = 0; i < queue.length; i++)
    for (const target of outgoing.get(queue[i]) || [])
      if (!reachable.has(target)) {
        reachable.add(target);
        queue.push(target);
      }
  for (const node of graph.nodes)
    if (!reachable.has(node.id))
      issues.push({
        severity: "warning",
        message: "Node is unreachable from every input.",
        node: node.id,
      });
  result.totalParameters = Array.from(parameters.values()).reduce(
    (sum, count) => sum + count,
    0,
  );
  return result;
}

/** Layered DAG layout with barycentric sweeps to reduce crossings, deterministic for identical graph order. */
export function layoutArchitecture(
  graph: ArchitectureGraph,
  horizontalGap = 190,
  verticalGap = 94,
): ArchitecturePosition[] {
  const analysis = analyzeArchitecture(graph);
  if (!analysis.order.length)
    return graph.nodes.map((node, index) => ({
      id: node.id,
      x: 45 + (index % 4) * horizontalGap,
      y: 40 + Math.floor(index / 4) * verticalGap,
      depth: 0,
    }));
  const layers = new Map<number, string[]>();
  for (const id of analysis.order) {
    const depth = analysis.depth.get(id) || 0;
    if (!layers.has(depth)) layers.set(depth, []);
    layers.get(depth)!.push(id);
  }
  const { incoming, outgoing } = graphAdjacency(graph);
  const positions = new Map<string, number>();
  for (const nodes of layers.values())
    nodes.forEach((id, index) => positions.set(id, index));
  const sortLayer = (nodes: string[], adjacency: Map<string, string[]>) => {
    const average = (id: string) => {
      const neighbors = adjacency.get(id) || [];
      return neighbors.length
        ? neighbors.reduce(
            (sum, neighbor) => sum + (positions.get(neighbor) || 0),
            0,
          ) / neighbors.length
        : positions.get(id) || 0;
    };
    nodes.sort(
      (a, b) =>
        average(a) - average(b) ||
        graph.nodes.findIndex((node) => node.id === a) -
          graph.nodes.findIndex((node) => node.id === b),
    );
    nodes.forEach((id, index) => positions.set(id, index));
  };
  const depths = Array.from(layers.keys()).sort((a, b) => a - b);
  for (let iteration = 0; iteration < 4; iteration++) {
    for (const depth of depths) sortLayer(layers.get(depth)!, incoming);
    for (const depth of depths.slice().reverse())
      sortLayer(layers.get(depth)!, outgoing);
  }
  const maxNodes = Math.max(
    1,
    ...Array.from(layers.values(), (nodes) => nodes.length),
  );
  return analysis.order.map((id) => {
    const depth = analysis.depth.get(id) || 0;
    const nodes = layers.get(depth)!;
    return {
      id,
      depth,
      x: 45 + depth * horizontalGap,
      y: 40 + ((maxNodes - nodes.length) / 2 + nodes.indexOf(id)) * verticalGap,
    };
  });
}

export function connectArchitecture(
  graph: ArchitectureGraph,
  source: string,
  target: string,
): ArchitectureGraph {
  if (
    !graph.nodes.some((node) => node.id === source) ||
    !graph.nodes.some((node) => node.id === target)
  )
    throw new Error("Both connection endpoints must exist.");
  if (
    graph.edges.some((edge) => edge.source === source && edge.target === target)
  )
    throw new Error("Connection already exists.");
  const next = { ...graph, edges: [...graph.edges, { source, target }] };
  topologicalOrder(next);
  return next;
}

export function removeArchitectureNode(
  graph: ArchitectureGraph,
  id: string,
): ArchitectureGraph {
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => node.id !== id),
    edges: graph.edges.filter(
      (edge) => edge.source !== id && edge.target !== id,
    ),
  };
}

export function renameArchitectureNode(
  graph: ArchitectureGraph,
  id: string,
  name: string,
): ArchitectureGraph {
  if (!name.trim() || name.length > 200)
    throw new Error("Node name must have between 1 and 200 characters.");
  return {
    ...graph,
    nodes: graph.nodes.map((node) =>
      node.id === id ? { ...node, name: name.trim() } : node,
    ),
  };
}

export function parseArchitecture(text: string): ArchitectureGraph {
  if (text.length > 2000000)
    throw new Error("Architecture file exceeds the import budget.");
  const parsed: unknown = JSON.parse(text);
  if (!parsed || typeof parsed !== "object")
    throw new Error("Expected an architecture document.");
  const record = parsed as Record<string, unknown>;
  if (
    record.format !== "neuralis.architecture" ||
    record.version !== 1 ||
    !Array.isArray(record.nodes) ||
    !Array.isArray(record.edges)
  )
    throw new Error("Expected neuralis.architecture version 1.");
  const nodes = record.nodes.map((value: unknown) => {
    if (!value || typeof value !== "object")
      throw new Error("Invalid architecture node.");
    const node = value as Record<string, unknown>;
    if (
      typeof node.id !== "string" ||
      typeof node.name !== "string" ||
      typeof node.operation !== "string" ||
      !OPERATIONS.includes(node.operation as ArchitectureOperation)
    )
      throw new Error("Invalid node identity, name or operation.");
    if (
      node.shape !== undefined &&
      (!Array.isArray(node.shape) ||
        !node.shape.every((size) => typeof size === "number"))
    )
      throw new Error("Node shape must be a numeric array.");
    if (node.units !== undefined && typeof node.units !== "number")
      throw new Error("Dense units must be numeric.");
    if (node.axis !== undefined && typeof node.axis !== "number")
      throw new Error("Axis must be numeric.");
    if (node.bias !== undefined && typeof node.bias !== "boolean")
      throw new Error("Bias flag must be boolean.");
    return {
      id: node.id,
      name: node.name,
      operation: node.operation,
      ...(node.shape ? { shape: [...(node.shape as number[])] } : {}),
      ...(node.units !== undefined ? { units: node.units } : {}),
      ...(node.axis !== undefined ? { axis: node.axis } : {}),
      ...(node.bias !== undefined ? { bias: node.bias } : {}),
    } as ArchitectureNode;
  });
  const edges = record.edges.map((value: unknown) => {
    if (!value || typeof value !== "object")
      throw new Error("Invalid architecture edge.");
    const edge = value as Record<string, unknown>;
    if (typeof edge.source !== "string" || typeof edge.target !== "string")
      throw new Error("Architecture endpoints must be node IDs.");
    return { source: edge.source, target: edge.target };
  });
  const graph: ArchitectureGraph = {
    format: "neuralis.architecture",
    version: 1,
    nodes,
    edges,
  };
  const analysis = analyzeArchitecture(graph);
  const structural = analysis.issues.filter(
    (issue) =>
      issue.severity === "error" &&
      (!issue.node || /IDs|Edge|cycle|Duplicate|Self/.test(issue.message)),
  );
  if (structural.length)
    throw new Error(structural.map((issue) => issue.message).join(" "));
  return graph;
}

export function sequentialArchitecture(
  sizes: readonly number[],
  names?: readonly string[],
): ArchitectureGraph {
  if (
    !sizes.length ||
    sizes.some((size) => !Number.isInteger(size) || size < 1)
  )
    throw new Error("Architecture sizes must be positive integers.");
  const nodes: ArchitectureNode[] = sizes.map((size, index) =>
    index === 0
      ? {
          id: "input",
          name: names?.[index] || "Input",
          operation: "input",
          shape: [size],
        }
      : {
          id: `dense-${index}`,
          name: names?.[index] || `Dense ${index}`,
          operation: "dense",
          units: size,
        },
  );
  nodes.push({ id: "output", name: "Output", operation: "output" });
  return {
    format: "neuralis.architecture",
    version: 1,
    nodes,
    edges: nodes
      .slice(1)
      .map((node, index) => ({ source: nodes[index].id, target: node.id })),
  };
}
