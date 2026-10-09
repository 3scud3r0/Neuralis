import { describe, expect, it } from "vitest";
import type { LayerData } from "../../types";
import {
  attentionEntropy,
  compareVectors,
  cosineSimilarity,
  divergingColor,
  histogram,
  neuronPosition,
  pearsonCorrelation,
  projectPCA,
  sampledIndices,
  selectEdges,
  summarize,
  validateMatrix,
} from "../numerics";

function layer(id: string, size: number, weights: number[][] = []): LayerData {
  return {
    id,
    name: id,
    size,
    shape: [size],
    weights,
    activations: Array(size).fill(0),
    biases: Array(size).fill(0),
    gradients: Array(size).fill(0),
  };
}

describe("captured tensor statistics", () => {
  it("uses population variance and counts invalid data separately", () => {
    const stats = summarize([0, 1, 2, 3, NaN, Infinity]);
    expect(stats.count).toBe(4);
    expect(stats.invalid).toBe(2);
    expect(stats.mean).toBe(1.5);
    expect(stats.median).toBe(1.5);
    expect(stats.variance).toBe(1.25);
    expect(stats.l2).toBeCloseTo(Math.sqrt(14));
    expect(stats.sparsity).toBe(0.25);
  });
  it("has a finite empty state", () => {
    expect(summarize([]).count).toBe(0);
    expect(Object.values(summarize([NaN])).every(Number.isFinite)).toBe(true);
  });
  it("keeps the upper endpoint and conserves observation counts", () => {
    const bins = histogram([0, 1, 2, 3, 4], 4, [0, 4]);
    expect(bins.map((bin) => bin.count)).toEqual([1, 1, 1, 2]);
    expect(bins.reduce((sum, bin) => sum + bin.count, 0)).toBe(5);
    expect(
      bins.reduce((sum, bin) => sum + bin.density * (bin.upper - bin.lower), 0),
    ).toBeCloseTo(1);
  });
  it("supports constant and finite-range histograms", () => {
    expect(histogram([3, 3, 3]).reduce((sum, bin) => sum + bin.count, 0)).toBe(
      3,
    );
    expect(
      histogram([-2, 0, 2], 2, [-1, 1]).reduce(
        (sum, bin) => sum + bin.count,
        0,
      ),
    ).toBe(1);
    expect(() => histogram([0], 2, [1, -1])).toThrow();
  });
});

describe("bounded connection geometry", () => {
  it("ranks actual output-by-input weights and preserves indices", () => {
    const layers = [
      layer("input", 2),
      layer("hidden", 2, [
        [0.1, -0.9],
        [0.4, 0.2],
      ]),
      layer("output", 1, [[0.7, 0.3]]),
    ];
    const edges = selectEdges(layers, 3);
    expect(edges.map((edge) => edge.weight)).toEqual([-0.9, 0.7, 0.4]);
    expect(edges[0]).toMatchObject({
      sourceLayer: 0,
      source: 1,
      targetLayer: 1,
      target: 0,
    });
  });
  it("matches a full-sort reference across many candidate edges", () => {
    const weights = Array.from({ length: 24 }, (_, i) =>
      Array.from({ length: 18 }, (_, j) => Math.sin((i * 18 + j + 1) * 0.37)),
    );
    const layers = [layer("a", 18), layer("b", 24, weights)];
    const actual = selectEdges(layers, 53).map((edge) => Math.abs(edge.weight));
    const expected = weights
      .flat()
      .map(Math.abs)
      .sort((a, b) => b - a)
      .slice(0, 53);
    expect(actual).toEqual(expected);
  });
  it("honors render visibility and excludes invalid or subthreshold weights", () => {
    const layers = [
      layer("input", 3),
      layer("output", 2, [
        [Infinity, -0.8, 0.7],
        [0.2, 0.1, NaN],
      ]),
    ];
    const visible = new Map([
      [0, new Set([1])],
      [1, new Set([0])],
    ]);
    expect(
      selectEdges(layers, 10, 0.5, visible).map((edge) => edge.weight),
    ).toEqual([-0.8]);
    expect(selectEdges(layers, 0)).toEqual([]);
  });
  it("samples deterministically without duplicate indices", () => {
    expect(sampledIndices(1000, 4)).toEqual([0, 333, 666, 999]);
    expect(sampledIndices(3, 8)).toEqual([0, 1, 2]);
    expect(sampledIndices(3, 0)).toEqual([]);
    expect(new Set(sampledIndices(57, 32)).size).toBe(32);
  });
  it("centers layers and places units at distinct positions", () => {
    expect(neuronPosition(0, 0, 1, 3)).toEqual([-4, 0, 0]);
    expect(neuronPosition(2, 0, 1, 3)).toEqual([4, 0, 0]);
    expect(neuronPosition(1, 0, 16, 3)).not.toEqual(
      neuronPosition(1, 1, 16, 3),
    );
    expect(
      divergingColor(-100, 1).every((value) => value >= 0 && value <= 1),
    ).toBe(true);
  });
});

describe("matrix-free PCA", () => {
  it("recovers the principal axis of collinear data and retains centered distances", () => {
    const projection = projectPCA([
      [1, 2, 3],
      [2, 4, 6],
      [3, 6, 9],
      [4, 8, 12],
    ]);
    expect(projection.components.length).toBe(1);
    expect(projection.explainedVariance[0]).toBeCloseTo(1);
    expect(
      Math.abs(projection.points[1][0] - projection.points[0][0]),
    ).toBeCloseTo(Math.sqrt(14), 5);
    expect(
      projection.points.reduce((sum, point) => sum + point[0], 0),
    ).toBeCloseTo(0);
  });
  it("produces orthogonal components and the correct eigenvalue fractions", () => {
    const projection = projectPCA([
      [3, 0, 0],
      [-3, 0, 0],
      [0, 2, 0],
      [0, -2, 0],
      [0, 0, 1],
      [0, 0, -1],
    ]);
    expect(projection.explainedVariance[0]).toBeCloseTo(9 / 14, 4);
    expect(projection.explainedVariance[1]).toBeCloseTo(4 / 14, 4);
    expect(projection.explainedVariance[2]).toBeCloseTo(1 / 14, 4);
    expect(
      cosineSimilarity(projection.components[0], projection.components[1]),
    ).toBeCloseTo(0, 5);
    expect(
      cosineSimilarity(projection.components[0], projection.components[2]),
    ).toBeCloseTo(0, 5);
  });
  it("handles constants, one sample, invalid shapes and non-finite observations", () => {
    expect(
      projectPCA([
        [3, 3],
        [3, 3],
      ]).points,
    ).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
    expect(projectPCA([[1, 2]]).components).toEqual([]);
    expect(() => projectPCA([[1], [1, 2]])).toThrow();
    expect(() => projectPCA([[NaN]])).toThrow();
  });
});

describe("attention and captured tensor comparison", () => {
  it("computes attention entropy in bits after row normalization", () => {
    expect(attentionEntropy([0.5, 0.5])).toBeCloseTo(1);
    expect(attentionEntropy([1, 1, 1, 1])).toBeCloseTo(2);
    expect(attentionEntropy([0, 0])).toBe(0);
    expect(() => attentionEntropy([-1, 2])).toThrow();
  });
  it("returns finite zero-norm comparisons and detects shape mismatch", () => {
    expect(cosineSimilarity([0, 0], [0, 0])).toBe(0);
    expect(compareVectors([1, 2], [1, 4]).rmse).toBeCloseTo(Math.sqrt(2));
    expect(pearsonCorrelation([1, 2, 3], [6, 4, 2])).toBeCloseTo(-1);
    expect(() => compareVectors([1], [1, 2])).toThrow();
    expect(() => validateMatrix([[1], [1, 2]])).toThrow();
  });
});
