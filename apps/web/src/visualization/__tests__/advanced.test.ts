import { describe, expect, it } from "vitest";
import {
  analyzeArchitecture,
  connectArchitecture,
  layoutArchitecture,
  parseArchitecture,
  removeArchitectureNode,
  sequentialArchitecture,
  topologicalOrder,
  type ArchitectureGraph,
} from "../architecture";
import {
  attentionHeadDivergence,
  attentionRollout,
  checkCausalMask,
  fuseAttentionHeads,
  headDivergenceMatrix,
  normalizeAttention,
} from "../attention";
import {
  contiguousStrides,
  denseTensor,
  flattenIndex,
  largestTensorElements,
  parseTensor,
  reduceTensor,
  reshapeTensor,
  serializeTensor,
  sliceTensor,
  tensorChannelStatistics,
  tensorDifference,
  transposeTensor,
  unravelIndex,
} from "../tensors";
import {
  cameraPreset,
  defaultVisualizationSettings,
  parseViewSettings,
} from "../viewState";

describe("exact multidimensional tensor transforms", () => {
  const tensor = denseTensor(
    Array.from({ length: 24 }, (_, i) => i),
    [2, 3, 4],
  );
  it("inverts row-major indices for every element", () => {
    expect(contiguousStrides([2, 3, 4])).toEqual([12, 4, 1]);
    for (let i = 0; i < 24; i++)
      expect(flattenIndex(unravelIndex(i, tensor.shape), tensor.shape)).toBe(i);
    expect(() => flattenIndex([2, 0, 0], tensor.shape)).toThrow();
    expect(() => unravelIndex(24, tensor.shape)).toThrow();
  });
  it("extracts arbitrary axis pairs with fixed coordinates", () => {
    expect(sliceTensor(tensor, 0, 2, [0, 1, 0]).matrix).toEqual([
      [4, 5, 6, 7],
      [16, 17, 18, 19],
    ]);
    expect(sliceTensor(tensor, 2, 0, [0, 2, 0]).matrix).toEqual([
      [8, 20],
      [9, 21],
      [10, 22],
      [11, 23],
    ]);
    expect(() => sliceTensor(tensor, 1, 1)).toThrow();
  });
  it("transposes values and recomputes contiguous strides", () => {
    const transpose = transposeTensor(tensor, [2, 0, 1]);
    expect(transpose.shape).toEqual([4, 2, 3]);
    expect(transpose.data[flattenIndex([3, 1, 2], transpose.shape)]).toBe(23);
    const roundTrip = transposeTensor(transpose, [1, 2, 0]);
    expect(Array.from(roundTrip.data)).toEqual(Array.from(tensor.data));
    expect(() => transposeTensor(tensor, [0, 0, 2])).toThrow();
  });
  it("preserves data buffer during reshape and rejects changes in size", () => {
    expect(reshapeTensor(tensor, [6, 4]).data).toBe(tensor.data);
    expect(() => reshapeTensor(tensor, [10, 4])).toThrow();
    expect(() => denseTensor([1, 2], [3])).toThrow();
  });
  it("reduces each axis against a hand-computed reference", () => {
    const small = denseTensor([1, 2, 3, 4, 5, 6], [2, 3]);
    expect(Array.from(reduceTensor(small, 0, "sum").data)).toEqual([5, 7, 9]);
    expect(Array.from(reduceTensor(small, 1, "mean").data)).toEqual([2, 5]);
    expect(Array.from(reduceTensor(small, 1, "max").data)).toEqual([3, 6]);
    expect(Array.from(reduceTensor(small, 0, "min").data)).toEqual([1, 2, 3]);
    expect(reduceTensor(small, 1, "l2").data[0]).toBeCloseTo(Math.sqrt(14));
  });
  it("computes channel statistics and reports original top-element coordinates", () => {
    expect(
      tensorChannelStatistics(tensor, 0).map((stats) => stats.mean),
    ).toEqual([5.5, 17.5]);
    const largest = largestTensorElements(tensor, 2);
    expect(largest.map((element) => element.index)).toEqual([
      [1, 2, 3],
      [1, 2, 2],
    ]);
    expect(largestTensorElements(tensor, 0)).toEqual([]);
  });
  it("round-trips validated imports and preserves exact elementwise differences", () => {
    const restored = parseTensor(serializeTensor(tensor));
    expect(Array.from(restored.data)).toEqual(Array.from(tensor.data));
    expect(
      Array.from(tensorDifference(tensor, restored).data).every(
        (value) => value === 0,
      ),
    ).toBe(true);
    expect(() =>
      parseTensor(
        '{"format":"neuralis.tensor","version":1,"shape":[3],"data":[1]}',
      ),
    ).toThrow();
    expect(() => serializeTensor(denseTensor([NaN], [1]))).toThrow();
  });
});

describe("architecture DAG validation and layout", () => {
  it("infers sequential dense parameters including biases", () => {
    const graph = sequentialArchitecture([3, 4, 2]);
    const result = analyzeArchitecture(graph);
    expect(result.issues).toEqual([]);
    expect(result.totalParameters).toBe(3 * 4 + 4 + 4 * 2 + 2);
    expect(result.shapes.get("output")).toEqual([2]);
    expect(result.depth.get("output")).toBe(3);
  });
  it("validates residual addition shapes and topological order", () => {
    const graph: ArchitectureGraph = {
      format: "neuralis.architecture",
      version: 1,
      nodes: [
        { id: "x", name: "x", operation: "input", shape: [5, 4] },
        { id: "p", name: "p", operation: "dense", units: 4 },
        { id: "add", name: "add", operation: "add" },
        { id: "out", name: "out", operation: "output" },
      ],
      edges: [
        { source: "x", target: "p" },
        { source: "x", target: "add" },
        { source: "p", target: "add" },
        { source: "add", target: "out" },
      ],
    };
    const result = analyzeArchitecture(graph);
    expect(result.shapes.get("add")).toEqual([5, 4]);
    expect(result.totalParameters).toBe(20);
    expect(result.issues).toEqual([]);
    const mismatch = {
      ...graph,
      nodes: graph.nodes.map((node) =>
        node.id === "p" ? { ...node, units: 3 } : node,
      ),
    };
    expect(
      analyzeArchitecture(mismatch).issues.some((issue) =>
        issue.message.includes("identical shapes"),
      ),
    ).toBe(true);
  });
  it("rejects cycles, dangling endpoints and duplicate edges", () => {
    const graph = sequentialArchitecture([2, 3, 1]);
    expect(() => connectArchitecture(graph, "output", "dense-1")).toThrow(
      /cycle/,
    );
    expect(() => connectArchitecture(graph, "missing", "output")).toThrow();
    expect(() => connectArchitecture(graph, "input", "dense-1")).toThrow(
      /exists/,
    );
    const duplicate = { ...graph, edges: [...graph.edges, graph.edges[0]] };
    expect(
      analyzeArchitecture(duplicate).issues.some((issue) =>
        issue.message.includes("Duplicate"),
      ),
    ).toBe(true);
  });
  it("concatenates only compatible non-axis shapes", () => {
    const graph: ArchitectureGraph = {
      format: "neuralis.architecture",
      version: 1,
      nodes: [
        { id: "a", name: "a", operation: "input", shape: [8, 3] },
        { id: "b", name: "b", operation: "input", shape: [8, 5] },
        { id: "c", name: "c", operation: "concat", axis: 1 },
        { id: "o", name: "o", operation: "output" },
      ],
      edges: [
        { source: "a", target: "c" },
        { source: "b", target: "c" },
        { source: "c", target: "o" },
      ],
    };
    expect(analyzeArchitecture(graph).shapes.get("c")).toEqual([8, 8]);
    expect(
      analyzeArchitecture({
        ...graph,
        nodes: graph.nodes.map((node) =>
          node.id === "c" ? { ...node, axis: 0 } : node,
        ),
      }).issues.some((issue) => issue.severity === "error"),
    ).toBe(true);
  });
  it("lays out all nodes deterministically and preserves upstream ordering", () => {
    const graph = sequentialArchitecture([2, 3, 1]);
    expect(layoutArchitecture(graph)).toEqual(layoutArchitecture(graph));
    const positions = layoutArchitecture(graph);
    expect(positions.map((position) => position.id)).toEqual(
      topologicalOrder(graph),
    );
    expect(positions[1].x).toBeGreaterThan(positions[0].x);
  });
  it("removes incident edges and round-trips architecture documents", () => {
    const graph = sequentialArchitecture([2, 3, 1]);
    expect(parseArchitecture(JSON.stringify(graph))).toEqual(graph);
    const removed = removeArchitectureNode(graph, "dense-1");
    expect(
      removed.edges.some(
        (edge) => edge.target === "dense-1" || edge.source === "dense-1",
      ),
    ).toBe(false);
    expect(() => parseArchitecture('{"format":"other"}')).toThrow();
  });
});

describe("attention propagation from measured heads", () => {
  const a = [
    [0.8, 0.2],
    [0.1, 0.9],
  ];
  const b = [
    [0.6, 0.4],
    [0.3, 0.7],
  ];
  it("fuses matching heads without altering source arrays", () => {
    expect(fuseAttentionHeads([a, b])).toEqual([
      [0.7, 0.30000000000000004],
      [0.2, 0.8],
    ]);
    expect(fuseAttentionHeads([a, b], "max")).toEqual([
      [0.8, 0.4],
      [0.3, 0.9],
    ]);
    expect(a[0]).toEqual([0.8, 0.2]);
    expect(() => fuseAttentionHeads([a, [[1]]])).toThrow();
  });
  it("multiplies rollout transitions in chronological order", () => {
    const result = attentionRollout([[a], [b]], { residualWeight: 0 });
    expect(result[0][0]).toBeCloseTo(0.52);
    expect(result[0][1]).toBeCloseTo(0.48);
    expect(result[1][0]).toBeCloseTo(0.31);
    expect(result[1][1]).toBeCloseTo(0.69);
    expect(
      result.every(
        (row) =>
          Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) < 1e-12,
      ),
    ).toBe(true);
  });
  it("includes identity residuals and restricts selected layers", () => {
    const result = attentionRollout([[a], [b]], {
      startLayer: 1,
      endLayer: 1,
      residualWeight: 1,
    });
    expect(result[0][0]).toBeCloseTo(0.8);
    expect(result[0][1]).toBeCloseTo(0.2);
    expect(attentionRollout([])).toEqual([]);
    expect(() =>
      attentionRollout([
        [
          [
            [1, 0, 0],
            [0, 1, 0],
          ],
        ],
      ]),
    ).toThrow(/square/);
  });
  it("computes symmetric bounded head divergence", () => {
    expect(attentionHeadDivergence(a, a)).toBe(0);
    expect(attentionHeadDivergence([[1, 0]], [[0, 1]])).toBeCloseTo(1);
    expect(attentionHeadDivergence(a, b)).toBeCloseTo(
      attentionHeadDivergence(b, a),
    );
    const matrix = headDivergenceMatrix([a, b]);
    expect(matrix[0][1]).toEqual(matrix[1][0]);
    expect(matrix[0][0]).toBe(0);
    expect(normalizeAttention([[0, 0]])).toEqual([[0, 0]]);
  });
  it("detects future-token leakage", () => {
    expect(
      checkCausalMask([
        [1, 0],
        [0.2, 0.8],
      ]).valid,
    ).toBe(true);
    expect(checkCausalMask(a)).toMatchObject({
      valid: false,
      violations: 1,
      leakage: 0.2,
    });
  });
});

describe("portable camera and filter settings", () => {
  it("round-trips filters and safely bounds imported budgets", () => {
    const defaults = defaultVisualizationSettings();
    expect(parseViewSettings(JSON.stringify(defaults))).toEqual(defaults);
    expect(
      parseViewSettings(JSON.stringify({ ...defaults, edgeBudget: 1000000 }))
        .edgeBudget,
    ).toBe(20000);
    expect(() =>
      parseViewSettings(
        '{"format":"neuralis.view","version":1,"edgeOpacity":"bad"}',
      ),
    ).toThrow();
  });
  it("validates camera vectors and real unit annotations", () => {
    const settings = {
      ...defaultVisualizationSettings(),
      bookmarks: [
        {
          id: "1",
          name: "Front",
          ...cameraPreset("front", 4),
          createdAt: "2026-01-01",
        },
      ],
      annotations: [
        {
          id: "a",
          layer: 1,
          neuron: 2,
          text: "Measured activation changed.",
          createdAt: "2026-01-01",
        },
      ],
    };
    expect(parseViewSettings(JSON.stringify(settings))).toEqual(settings);
    expect(() =>
      parseViewSettings(
        JSON.stringify({
          ...settings,
          bookmarks: [{ ...settings.bookmarks[0], position: [0, 1] }],
        }),
      ),
    ).toThrow();
    expect(cameraPreset("top", 8).position[1]).toBe(24);
  });
});
