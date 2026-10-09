export const EPSILON = 1e-12;

export function zeros(size: number): number[] {
  return Array.from({ length: size }, () => 0);
}

export function matrix(rows: number, columns: number, initial = 0): number[][] {
  return Array.from({ length: rows }, () => Array.from({ length: columns }, () => initial));
}

export function sum(values: number[]): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

export function mean(values: number[]): number {
  return values.length ? sum(values) / values.length : 0;
}

export function dot(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error('Dot product dimensions do not match.');
  let result = 0;
  for (let i = 0; i < a.length; i++) result += a[i] * b[i];
  return result;
}

export function norm(values: number[]): number {
  return Math.sqrt(dot(values, values));
}

export function normalize(values: number[]): number[] {
  const denominator = norm(values);
  return denominator > EPSILON ? values.map(value => value / denominator) : zeros(values.length);
}

export function argmax(values: number[]): number {
  if (!values.length) throw new Error('Cannot select a maximum from an empty vector.');
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

export function softmax(values: number[]): number[] {
  const maximum = Math.max(...values);
  const exponentials = values.map(value => Math.exp(value - maximum));
  const denominator = sum(exponentials);
  return exponentials.map(value => value / denominator);
}

export function sigmoid(value: number): number {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

export function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

export function squaredDistance(a: number[], b: number[]): number {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += (a[i] - b[i]) ** 2;
  return total;
}

export function transpose(values: number[][]): number[][] {
  if (!values.length) return [];
  return Array.from({ length: values[0].length }, (_, column) => values.map(row => row[column]));
}

export function multiplyVector(values: number[][], vector: number[]): number[] {
  return values.map(row => dot(row, vector));
}

export function finiteVector(value: unknown, expected?: number): value is number[] {
  return Array.isArray(value) && (expected === undefined || value.length === expected) &&
    value.every(item => typeof item === 'number' && Number.isFinite(item));
}

export function finiteMatrix(value: unknown, rows: number, columns: number): value is number[][] {
  return Array.isArray(value) && value.length === rows && value.every(row => finiteVector(row, columns));
}

/** KL divergence only for normalized probability vectors. */
export function klDivergence(before: number[], after: number[]): number | undefined {
  const isProbability = (values: number[]) => values.every(value => value >= 0 && value <= 1) &&
    Math.abs(sum(values) - 1) < 1e-6;
  if (!isProbability(before) || !isProbability(after)) return undefined;
  return before.reduce((total, value, i) => total +
    (value > 0 ? value * Math.log(value / Math.max(EPSILON, after[i])) : 0), 0);
}

export function covariance(samples: number[][]): { center: number[]; covariance: number[][] } {
  if (samples.length < 2) throw new Error('Covariance needs at least two samples.');
  const dimension = samples[0].length;
  const center = Array.from({ length: dimension }, (_, i) => mean(samples.map(row => row[i])));
  const result = matrix(dimension, dimension);
  for (const sample of samples) {
    for (let i = 0; i < dimension; i++) {
      for (let j = 0; j <= i; j++) {
        result[i][j] += (sample[i] - center[i]) * (sample[j] - center[j]) / (samples.length - 1);
      }
    }
  }
  for (let i = 0; i < dimension; i++) for (let j = 0; j < i; j++) result[j][i] = result[i][j];
  return { center, covariance: result };
}
