import type { LayerData } from "../types";

export type Vector3Tuple = [number, number, number];
export interface NumericSummary {
  count: number;
  invalid: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  variance: number;
  standardDeviation: number;
  l1: number;
  l2: number;
  sparsity: number;
}
export interface HistogramBin {
  lower: number;
  upper: number;
  count: number;
  density: number;
}
export interface EdgeRecord {
  sourceLayer: number;
  source: number;
  targetLayer: number;
  target: number;
  weight: number;
  score: number;
}
export interface Projection {
  points: Vector3Tuple[];
  components: number[][];
  mean: number[];
  explainedVariance: number[];
  totalVariance: number;
}
export const finite = (value: unknown, fallback = 0): number =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;
export const clamp = (value: number, lower: number, upper: number): number =>
  Math.max(lower, Math.min(upper, value));

/** Stable two-pass statistics. Non-finite values are counted, never silently presented as observations. */
export function summarize(
  values: readonly number[],
  zeroThreshold = 1e-8,
): NumericSummary {
  const valid = values.filter(Number.isFinite);
  const count = valid.length;
  if (!count)
    return {
      count: 0,
      invalid: values.length,
      min: 0,
      max: 0,
      mean: 0,
      median: 0,
      variance: 0,
      standardDeviation: 0,
      l1: 0,
      l2: 0,
      sparsity: 0,
    };
  let mean = 0;
  let m2 = 0;
  let l1 = 0;
  let squareSum = 0;
  let zeros = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < count; i++) {
    const value = valid[i];
    const delta = value - mean;
    mean += delta / (i + 1);
    m2 += delta * (value - mean);
    min = Math.min(min, value);
    max = Math.max(max, value);
    l1 += Math.abs(value);
    squareSum += value * value;
    if (Math.abs(value) <= zeroThreshold) zeros++;
  }
  const ordered = valid.slice().sort((a, b) => a - b);
  const midpoint = Math.floor(count / 2);
  const median =
    count % 2
      ? ordered[midpoint]
      : (ordered[midpoint - 1] + ordered[midpoint]) / 2;
  const variance = Math.max(0, m2 / count);
  return {
    count,
    invalid: values.length - count,
    min,
    max,
    mean,
    median,
    variance,
    standardDeviation: Math.sqrt(variance),
    l1,
    l2: Math.sqrt(squareSum),
    sparsity: zeros / count,
  };
}

export function histogram(
  values: readonly number[],
  binCount = 32,
  range?: [number, number],
): HistogramBin[] {
  const valid = values.filter(Number.isFinite);
  const count = clamp(Math.floor(finite(binCount, 32)), 1, 512);
  if (!valid.length) return [];
  const stats = summarize(valid);
  let lower = range ? range[0] : stats.min;
  let upper = range ? range[1] : stats.max;
  if (!Number.isFinite(lower) || !Number.isFinite(upper) || lower > upper)
    throw new Error("Histogram range must be finite and ordered.");
  if (lower === upper) {
    const padding = Math.max(Math.abs(lower) * 0.01, 0.5);
    lower -= padding;
    upper += padding;
  }
  const width = (upper - lower) / count;
  const bins = Array.from({ length: count }, (_, i) => ({
    lower: lower + width * i,
    upper: lower + width * (i + 1),
    count: 0,
    density: 0,
  }));
  let included = 0;
  for (const value of valid) {
    if (value < lower || value > upper) continue;
    bins[Math.min(count - 1, Math.floor((value - lower) / width))].count++;
    included++;
  }
  for (const bin of bins)
    bin.density = included ? bin.count / (included * width) : 0;
  return bins;
}

export function quantile(values: readonly number[], fraction: number): number {
  const sorted = values
    .filter(Number.isFinite)
    .slice()
    .sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const position = clamp(fraction, 0, 1) * (sorted.length - 1);
  const index = Math.floor(position);
  const remainder = position - index;
  return (
    sorted[index] * (1 - remainder) +
    sorted[Math.min(index + 1, sorted.length - 1)] * remainder
  );
}

export function symmetricDomain(
  values: readonly number[],
  percentile = 0.98,
): number {
  const magnitude = quantile(
    values.filter(Number.isFinite).map(Math.abs),
    percentile,
  );
  return magnitude > 1e-12 ? magnitude : 1;
}

const COLOR_STOPS: Vector3Tuple[] = [
  [0.11, 0.32, 0.72],
  [0.19, 0.82, 0.96],
  [0.11, 0.15, 0.25],
  [0.64, 0.31, 0.94],
  [1, 0.4, 0.55],
];
export function divergingColor(value: number, domain = 1): Vector3Tuple {
  const normalized = clamp(
    finite(value) / Math.max(Math.abs(domain), 1e-12),
    -1,
    1,
  );
  const position = (normalized + 1) * 2;
  const index = Math.min(3, Math.floor(position));
  const mix = position - index;
  return COLOR_STOPS[index].map(
    (channel, i) => channel + (COLOR_STOPS[index + 1][i] - channel) * mix,
  ) as Vector3Tuple;
}
export function rgbCss(rgb: Vector3Tuple, alpha = 1): string {
  return `rgba(${rgb.map((value) => Math.round(clamp(value, 0, 1) * 255)).join(",")},${clamp(alpha, 0, 1)})`;
}

export function neuronPosition(
  layer: number,
  neuron: number,
  size: number,
  layerCount: number,
): Vector3Tuple {
  const rows = Math.max(1, Math.ceil(Math.sqrt(Math.max(size, 1))));
  const columns = Math.max(1, Math.ceil(size / rows));
  const row = neuron % rows;
  const column = Math.floor(neuron / rows);
  const spacing = Math.min(0.72, 5.8 / rows);
  return [
    (layer - (layerCount - 1) / 2) * 4,
    (row - (rows - 1) / 2) * spacing,
    (column - (columns - 1) / 2) * spacing,
  ];
}

/** Uniform deterministic sampling preserves endpoint identities for very wide layers. */
export function sampledIndices(size: number, budget: number): number[] {
  const length = Math.max(0, Math.floor(finite(size)));
  const count = Math.min(length, Math.max(0, Math.floor(finite(budget))));
  if (!count) return [];
  if (count === 1) return [0];
  return Array.from({ length: count }, (_, i) =>
    Math.round((i * (length - 1)) / (count - 1)),
  );
}

export function layerValues(
  layer: LayerData,
  mode: "activations" | "weights" | "gradients",
): number[] {
  if (mode === "activations") return layer.activations;
  if (mode === "gradients") return layer.gradients;
  return Array.from({ length: layer.size }, (_, i) => {
    const row = layer.weights[i];
    if (!row?.length) return 0;
    return row.reduce((sum, weight) => sum + finite(weight), 0) / row.length;
  });
}

/** Top-k by absolute weight, using a bounded min-heap rather than allocating/sorting the complete graph. */
export function selectEdges(
  layers: LayerData[],
  budget = 1600,
  threshold = 0,
  visible?: Map<number, Set<number>>,
): EdgeRecord[] {
  const limit = clamp(Math.floor(finite(budget)), 0, 50000);
  if (!limit) return [];
  const heap: EdgeRecord[] = [];
  const swap = (a: number, b: number) => {
    const value = heap[a];
    heap[a] = heap[b];
    heap[b] = value;
  };
  const insert = (edge: EdgeRecord) => {
    if (heap.length < limit) {
      heap.push(edge);
      let i = heap.length - 1;
      while (i > 0) {
        const parent = (i - 1) >> 1;
        if (heap[parent].score <= heap[i].score) break;
        swap(i, parent);
        i = parent;
      }
      return;
    }
    if (edge.score <= heap[0].score) return;
    heap[0] = edge;
    let i = 0;
    while (true) {
      const left = i * 2 + 1;
      const right = left + 1;
      let smallest = i;
      if (left < heap.length && heap[left].score < heap[smallest].score)
        smallest = left;
      if (right < heap.length && heap[right].score < heap[smallest].score)
        smallest = right;
      if (smallest === i) break;
      swap(i, smallest);
      i = smallest;
    }
  };
  for (let targetLayer = 1; targetLayer < layers.length; targetLayer++) {
    const sourceSize = layers[targetLayer - 1].size;
    const target = layers[targetLayer];
    for (
      let neuron = 0;
      neuron < Math.min(target.size, target.weights.length);
      neuron++
    ) {
      if (visible && !visible.get(targetLayer)?.has(neuron)) continue;
      const row = target.weights[neuron];
      for (
        let source = 0;
        source < Math.min(sourceSize, row.length);
        source++
      ) {
        if (visible && !visible.get(targetLayer - 1)?.has(source)) continue;
        const weight = row[source];
        if (!Number.isFinite(weight) || Math.abs(weight) < threshold) continue;
        insert({
          sourceLayer: targetLayer - 1,
          source,
          targetLayer,
          target: neuron,
          weight,
          score: Math.abs(weight),
        });
      }
    }
  }
  return heap.sort(
    (a, b) =>
      b.score - a.score ||
      a.targetLayer - b.targetLayer ||
      a.target - b.target ||
      a.source - b.source,
  );
}

export function cosineSimilarity(
  a: readonly number[],
  b: readonly number[],
): number {
  if (a.length !== b.length)
    throw new Error("Cosine similarity requires vectors of equal dimensions.");
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const x = finite(a[i]);
    const y = finite(b[i]);
    dot += x * y;
    normA += x * x;
    normB += y * y;
  }
  return normA > 0 && normB > 0
    ? clamp(dot / Math.sqrt(normA * normB), -1, 1)
    : 0;
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}
function normalize(vector: number[]): number[] {
  const norm = Math.sqrt(dot(vector, vector));
  return norm > 1e-12
    ? vector.map((value) => value / norm)
    : vector.map(() => 0);
}

/** Centered PCA with matrix-free power iteration and orthogonal deflation. O(n*d*k), no d² covariance. */
export function projectPCA(
  vectors: readonly (readonly number[])[],
  iterations = 80,
): Projection {
  if (!vectors.length)
    return {
      points: [],
      components: [],
      mean: [],
      explainedVariance: [],
      totalVariance: 0,
    };
  const dimension = vectors[0].length;
  if (
    !dimension ||
    vectors.some(
      (vector) =>
        vector.length !== dimension ||
        vector.some((value) => !Number.isFinite(value)),
    )
  )
    throw new Error("PCA requires non-empty, equal-length finite vectors.");
  if (vectors.length > 10000 || dimension > 8192)
    throw new Error(
      "PCA preview budget exceeded. Sample at most 10,000 vectors with at most 8,192 dimensions.",
    );
  if (vectors.length * dimension > 2000000)
    throw new Error(
      "PCA preview supports at most two million captured values. Sample representations before projecting.",
    );
  const iterationBudget = Math.min(
    clamp(iterations, 1, 250),
    Math.max(
      8,
      Math.floor(48000000 / Math.max(1, vectors.length * dimension * 3)),
    ),
  );
  const mean = Array.from(
    { length: dimension },
    (_, j) =>
      vectors.reduce((sum, vector) => sum + vector[j], 0) / vectors.length,
  );
  const centered = vectors.map((vector) =>
    vector.map((value, j) => value - mean[j]),
  );
  const denominator = Math.max(1, vectors.length - 1);
  const totalVariance =
    centered.reduce((sum, vector) => sum + dot(vector, vector), 0) /
    denominator;
  const components: number[][] = [];
  const explainedVariance: number[] = [];
  for (
    let component = 0;
    component < Math.min(3, dimension, Math.max(0, vectors.length - 1));
    component++
  ) {
    let axis = normalize(
      Array.from(
        { length: dimension },
        (_, j) =>
          Math.sin((j + 1) * (component + 1) * 1.618) + Math.cos(j * 0.71),
      ),
    );
    for (let iteration = 0; iteration < iterationBudget; iteration++) {
      const next = Array<number>(dimension).fill(0);
      for (const vector of centered) {
        const projection = dot(vector, axis);
        for (let j = 0; j < dimension; j++) next[j] += vector[j] * projection;
      }
      for (const previous of components) {
        const projection = dot(next, previous);
        for (let j = 0; j < dimension; j++) next[j] -= previous[j] * projection;
      }
      const normalized = normalize(next);
      if (Math.abs(dot(axis, normalized)) > 1 - 1e-10) {
        axis = normalized;
        break;
      }
      axis = normalized;
    }
    const variance =
      centered.reduce((sum, vector) => {
        const projection = dot(vector, axis);
        return sum + projection * projection;
      }, 0) / denominator;
    if (variance <= 1e-12) break;
    components.push(axis);
    explainedVariance.push(totalVariance > 0 ? variance / totalVariance : 0);
  }
  const points = centered.map(
    (vector) =>
      [0, 1, 2].map((i) =>
        components[i] ? dot(vector, components[i]) : 0,
      ) as Vector3Tuple,
  );
  return { points, components, mean, explainedVariance, totalVariance };
}

export function pearsonCorrelation(
  a: readonly number[],
  b: readonly number[],
): number {
  if (a.length !== b.length)
    throw new Error("Correlation requires equal-length vectors.");
  const pairs = a
    .map((value, i) => [value, b[i]])
    .filter((pair) => pair.every(Number.isFinite));
  if (!pairs.length) return 0;
  const averageA = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
  const averageB = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;
  let numerator = 0;
  let denominatorA = 0;
  let denominatorB = 0;
  for (const [x, y] of pairs) {
    numerator += (x - averageA) * (y - averageB);
    denominatorA += (x - averageA) ** 2;
    denominatorB += (y - averageB) ** 2;
  }
  return denominatorA > 0 && denominatorB > 0
    ? clamp(numerator / Math.sqrt(denominatorA * denominatorB), -1, 1)
    : 0;
}

export function compareVectors(
  left: readonly number[],
  right: readonly number[],
): {
  rmse: number;
  cosine: number;
  correlation: number;
  maxDifference: number;
  compared: number;
} {
  if (left.length !== right.length)
    throw new Error("Comparison requires tensors with equal sizes.");
  const differences = left.map((value, i) => value - right[i]);
  if (differences.some((value) => !Number.isFinite(value)))
    throw new Error("Comparison requires finite tensors.");
  const stats = summarize(differences);
  return {
    rmse: left.length ? stats.l2 / Math.sqrt(left.length) : 0,
    cosine: cosineSimilarity(left, right),
    correlation: pearsonCorrelation(left, right),
    maxDifference: Math.max(Math.abs(stats.min), Math.abs(stats.max)),
    compared: left.length,
  };
}

export function attentionEntropy(row: readonly number[]): number {
  if (row.some((value) => !Number.isFinite(value) || value < 0))
    throw new Error("Attention weights must be finite and non-negative.");
  const total = row.reduce((sum, value) => sum + value, 0);
  if (!total) return 0;
  return -row.reduce((sum, value) => {
    const probability = value / total;
    return probability ? sum + probability * Math.log2(probability) : sum;
  }, 0);
}

export function validateMatrix(matrix: readonly (readonly number[])[]): {
  rows: number;
  columns: number;
} {
  if (!matrix.length) return { rows: 0, columns: 0 };
  const columns = matrix[0].length;
  if (matrix.some((row) => row.length !== columns))
    throw new Error("Tensor rows must have equal widths.");
  return { rows: matrix.length, columns };
}
