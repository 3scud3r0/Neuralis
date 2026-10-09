import type { EngineController, InterventionResult, InterventionSpec } from '../types';
import { pearsonCorrelation, summarize } from './statistics';

export interface Attribution {
  values: number[]; prediction: number[]; baselinePrediction: number[];
  target: number; completenessError: number; evaluations: number;
  method: string; baseline: number[];
}
export interface AblationObservation {
  layer: number; neuron: number; averageDelta: number; meanAbsoluteDelta: number;
  perSample: number[]; samples: number;
}
export interface CalibrationBin {
  lower: number; upper: number; count: number; confidence: number; accuracy: number; gap: number;
}
export interface CalibrationResult {
  bins: CalibrationBin[]; expectedCalibrationError: number;
  maximumCalibrationError: number; brierScore: number; negativeLogLikelihood: number;
  samples: number; classes: number;
}
type Predictor = (input: number[]) => number[];

function checkedInput(input: readonly number[]): number[] {
  if (!input.length || input.some(value => !Number.isFinite(value))) throw new Error('The model input must be a nonempty finite vector.');
  if (input.length > 512) throw new Error('Black-box analysis is limited to 512 input dimensions.');
  return [...input];
}

function checkedPrediction(predict: Predictor, input: number[], target: number): number[] {
  const result = predict(input);
  if (!Number.isInteger(target) || target < 0 || target >= result.length || result.some(value => !Number.isFinite(value))) {
    throw new Error('The target output is unavailable or the model returned nonfinite values.');
  }
  return result;
}

export function numericalGradient(predict: Predictor, input: readonly number[], target = 0, epsilon = 1e-4): number[] {
  const vector = checkedInput(input);
  if (!(epsilon > 0 && epsilon <= 0.1)) throw new Error('Finite-difference epsilon must be in (0, 0.1].');
  const gradient = new Array<number>(vector.length);
  for (let dimension = 0; dimension < vector.length; dimension++) {
    const value = vector[dimension];
    const displacement = epsilon * Math.max(1, Math.abs(value));
    vector[dimension] = value + displacement;
    const plus = checkedPrediction(predict, vector, target)[target];
    vector[dimension] = value - displacement;
    const minus = checkedPrediction(predict, vector, target)[target];
    vector[dimension] = value;
    gradient[dimension] = (plus - minus) / (2 * displacement);
  }
  return gradient;
}

export function integratedGradients(predict: Predictor, input: readonly number[], options: {
  baseline?: number[]; target?: number; steps?: number; epsilon?: number; maxEvaluations?: number;
} = {}): Attribution {
  const vector = checkedInput(input);
  const baseline = checkedInput(options.baseline ?? vector.map(() => 0));
  if (baseline.length !== vector.length) throw new Error('The baseline and input must have identical dimensions.');
  const target = options.target ?? 0;
  const steps = options.steps ?? 32;
  if (!Number.isInteger(steps) || steps < 2 || steps > 256) throw new Error('Integration steps must be between 2 and 256.');
  const evaluations = 2 * vector.length * (steps + 1) + 2;
  if (evaluations > (options.maxEvaluations ?? 50000)) throw new Error('The attribution exceeds the evaluation budget. Reduce dimensions or integration steps.');
  const accumulated = vector.map(() => 0);
  for (let index = 0; index <= steps; index++) {
    const alpha = index / steps;
    const point = vector.map((value, dimension) => baseline[dimension] + alpha * (value - baseline[dimension]));
    const gradient = numericalGradient(predict, point, target, options.epsilon ?? 1e-4);
    const weight = index === 0 || index === steps ? 0.5 : 1;
    gradient.forEach((value, dimension) => { accumulated[dimension] += weight * value; });
  }
  const values = accumulated.map((value, dimension) => value * (vector[dimension] - baseline[dimension]) / steps);
  const prediction = checkedPrediction(predict, vector, target);
  const baselinePrediction = checkedPrediction(predict, baseline, target);
  const completenessError = values.reduce((sum, value) => sum + value, 0) - (prediction[target] - baselinePrediction[target]);
  return { values, prediction, baselinePrediction, target, completenessError, evaluations, baseline,
    method: 'Integrated gradients using central finite differences and trapezoidal integration' };
}

export function requireStableEngine(engine: EngineController): void {
  if (engine.getSnapshot().state === 'running') throw new Error('Pause training before analysis so every comparison uses the same parameters.');
}

export function interventionSweep(engine: EngineController, input: number[], spec: Omit<InterventionSpec, 'value'>, values: number[]): InterventionResult[] {
  requireStableEngine(engine);
  if (values.length < 1 || values.length > 100 || values.some(value => !Number.isFinite(value) || Math.abs(value) > 100)) {
    throw new Error('Use 1–100 finite intervention strengths between -100 and 100.');
  }
  const baseline = engine.predict(input);
  const results = values.map(value => engine.intervene(input, { ...spec, value }));
  const after = engine.predict(input);
  if (baseline.length !== after.length || baseline.some((value, index) => Math.abs(value - after[index]) > 1e-10)) {
    throw new Error('The model changed during intervention analysis. Results are not a controlled comparison.');
  }
  return results;
}

export function ablationMap(engine: EngineController, inputs: number[][], target: number, options: {
  maxNeurons?: number; layers?: number[]; maxEvaluations?: number;
} = {}): AblationObservation[] {
  requireStableEngine(engine);
  if (!inputs.length || inputs.length > 128) throw new Error('Ablation analysis requires 1–128 input examples.');
  const snapshot = engine.getSnapshot();
  const selected = options.layers ?? snapshot.layers.map((_, index) => index).slice(1, -1);
  const targets = selected.flatMap(layer => {
    if (layer < 0 || layer >= snapshot.layers.length) throw new Error('An ablation target layer is invalid.');
    return Array.from({ length: Math.min(options.maxNeurons ?? 32, snapshot.layers[layer].size) }, (_, neuron) => ({ layer, neuron }));
  });
  if ((targets.length + 1) * inputs.length > (options.maxEvaluations ?? 5000)) throw new Error('The ablation map exceeds the evaluation budget.');
  const before = inputs.map(input => checkedPrediction(value => engine.predict(value), checkedInput(input), target)[target]);
  return targets.map(({ layer, neuron }) => {
    const perSample = inputs.map((input, index) => engine.intervene(input, { layer, neuron, kind: 'ablate', value: 0 }).after[target] - before[index]);
    return { layer, neuron, perSample, averageDelta: summarize(perSample).mean,
      meanAbsoluteDelta: summarize(perSample.map(Math.abs)).mean, samples: inputs.length };
  });
}

export function featureOcclusion(predict: Predictor, input: number[], baseline: number[], target = 0): number[] {
  const vector = checkedInput(input);
  if (baseline.length !== vector.length) throw new Error('Occlusion requires an aligned baseline.');
  const before = checkedPrediction(predict, vector, target)[target];
  return vector.map((_, dimension) => {
    const masked = [...vector];
    masked[dimension] = baseline[dimension];
    return checkedPrediction(predict, masked, target)[target] - before;
  });
}

export function adversarialSearch(predict: Predictor, input: number[], options: {
  target: number; direction: 'increase' | 'decrease'; radius?: number; stepSize?: number;
  steps?: number; bounds?: { minimum: number; maximum: number }; maxEvaluations?: number;
}): { input: number[]; before: number[]; after: number[]; trajectory: { iteration: number; score: number; distance: number }[]; evaluations: number } {
  const original = checkedInput(input);
  const radius = options.radius ?? 0.15;
  const steps = options.steps ?? 10;
  const stepSize = options.stepSize ?? radius / Math.max(1, steps / 2);
  if (!(radius > 0) || !(stepSize > 0) || !Number.isInteger(steps) || steps < 1 || steps > 100) throw new Error('Invalid perturbation budget.');
  const evaluations = steps * (2 * original.length + 1) + 2;
  if (evaluations > (options.maxEvaluations ?? 20000)) throw new Error('The robustness search exceeds its evaluation budget.');
  const before = checkedPrediction(predict, original, options.target);
  let current = [...original];
  const trajectory = [];
  for (let iteration = 1; iteration <= steps; iteration++) {
    const gradient = numericalGradient(predict, current, options.target);
    current = current.map((value, dimension) => {
      const moved = value + (options.direction === 'increase' ? 1 : -1) * stepSize * Math.sign(gradient[dimension]);
      const projected = Math.max(original[dimension] - radius, Math.min(original[dimension] + radius, moved));
      return options.bounds ? Math.max(options.bounds.minimum, Math.min(options.bounds.maximum, projected)) : projected;
    });
    const score = checkedPrediction(predict, current, options.target)[options.target];
    trajectory.push({ iteration, score, distance: Math.max(...current.map((value, dimension) => Math.abs(value - original[dimension]))) });
  }
  return { input: current, before, after: checkedPrediction(predict, current, options.target), trajectory, evaluations };
}

function validateMatrix(matrix: number[][], maximum = 512): void {
  if (matrix.length < 2 || matrix.length > maximum || !matrix[0]?.length || matrix[0].length > 1024 ||
      matrix.some(row => row.length !== matrix[0].length || row.some(value => !Number.isFinite(value)))) {
    throw new Error('Representations must be rectangular finite matrices with 2–512 aligned observations.');
  }
}

function centeredGram(matrix: number[][]): number[][] {
  const count = matrix.length;
  const gram = Array.from({ length: count }, () => new Array<number>(count).fill(0));
  const means = new Array<number>(count).fill(0);
  for (let first = 0; first < count; first++) {
    for (let second = 0; second <= first; second++) {
      let dot = 0;
      for (let dimension = 0; dimension < matrix[0].length; dimension++) dot += matrix[first][dimension] * matrix[second][dimension];
      gram[first][second] = dot;
      gram[second][first] = dot;
    }
  }
  for (let first = 0; first < count; first++) means[first] = gram[first].reduce((sum, value) => sum + value, 0) / count;
  const grand = means.reduce((sum, value) => sum + value, 0) / count;
  return gram.map((row, first) => row.map((value, second) => value - means[first] - means[second] + grand));
}

export function linearCKA(first: number[][], second: number[][]): number {
  validateMatrix(first);
  validateMatrix(second);
  if (first.length !== second.length) throw new Error('CKA requires the same observations in the same order.');
  const a = centeredGram(first);
  const b = centeredGram(second);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let row = 0; row < a.length; row++) for (let column = 0; column < a.length; column++) {
    dot += a[row][column] * b[row][column];
    normA += a[row][column] ** 2;
    normB += b[row][column] ** 2;
  }
  return normA > 1e-20 && normB > 1e-20 ? Math.max(0, Math.min(1, dot / Math.sqrt(normA * normB))) : 0;
}

export function representationSimilarity(first: number[][], second: number[][]): number {
  validateMatrix(first);
  validateMatrix(second);
  if (first.length !== second.length) throw new Error('RSA requires aligned observations.');
  const a: number[] = [];
  const b: number[] = [];
  for (let row = 1; row < first.length; row++) for (let column = 0; column < row; column++) {
    a.push(Math.sqrt(first[row].reduce((sum, value, dimension) => sum + (value - first[column][dimension]) ** 2, 0)));
    b.push(Math.sqrt(second[row].reduce((sum, value, dimension) => sum + (value - second[column][dimension]) ** 2, 0)));
  }
  return a.length > 1 ? pearsonCorrelation(a, b) : 0;
}

export function calibration(probabilities: number[][], labels: number[], binCount = 10): CalibrationResult {
  if (!probabilities.length || probabilities.length !== labels.length || !Number.isInteger(binCount) || binCount < 2 || binCount > 100) {
    throw new Error('Calibration needs aligned probabilities/labels and 2–100 bins.');
  }
  const classes = probabilities[0].length;
  if (classes < 2) throw new Error('Calibration is defined here for multi-class probabilities.');
  const bins: CalibrationBin[] = Array.from({ length: binCount }, (_, index) => ({ lower: index / binCount, upper: (index + 1) / binCount, count: 0, confidence: 0, accuracy: 0, gap: 0 }));
  let brierScore = 0;
  let negativeLogLikelihood = 0;
  probabilities.forEach((row, index) => {
    if (row.length !== classes || row.some(value => !Number.isFinite(value) || value < 0 || value > 1) || Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) > 1e-5 ||
        !Number.isInteger(labels[index]) || labels[index] < 0 || labels[index] >= classes) throw new Error('Invalid probability distribution or class label.');
    let predicted = 0;
    row.forEach((value, classIndex) => { if (value > row[predicted]) predicted = classIndex; });
    const confidence = row[predicted];
    const bin = bins[Math.min(binCount - 1, Math.floor(confidence * binCount))];
    bin.count++;
    bin.confidence += confidence;
    bin.accuracy += Number(predicted === labels[index]);
    negativeLogLikelihood -= Math.log(Math.max(1e-15, row[labels[index]]));
    brierScore += row.reduce((sum, value, classIndex) => sum + (value - Number(classIndex === labels[index])) ** 2, 0);
  });
  let expectedCalibrationError = 0;
  let maximumCalibrationError = 0;
  bins.forEach(bin => {
    if (!bin.count) return;
    bin.confidence /= bin.count;
    bin.accuracy /= bin.count;
    bin.gap = Math.abs(bin.accuracy - bin.confidence);
    expectedCalibrationError += bin.gap * bin.count / labels.length;
    maximumCalibrationError = Math.max(maximumCalibrationError, bin.gap);
  });
  return { bins, expectedCalibrationError, maximumCalibrationError, brierScore: brierScore / labels.length,
    negativeLogLikelihood: negativeLogLikelihood / labels.length, samples: labels.length, classes };
}

export function nearestRepresentations(query: number[], representations: number[][], count = 10, metric: 'cosine' | 'euclidean' = 'cosine'): { index: number; score: number }[] {
  const vector = checkedInput(query);
  if (!Number.isInteger(count) || count < 1 || count > 100) throw new Error('Neighbor count must be 1–100.');
  const queryNorm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return representations.map((row, index) => {
    if (row.length !== vector.length || row.some(value => !Number.isFinite(value))) throw new Error('Neighbor representations must align with the query.');
    let dot = 0;
    let norm = 0;
    let distance = 0;
    row.forEach((value, dimension) => { dot += value * vector[dimension]; norm += value * value; distance += (value - vector[dimension]) ** 2; });
    const score = metric === 'cosine' ? (norm && queryNorm ? dot / (Math.sqrt(norm) * queryNorm) : 0) : -Math.sqrt(distance);
    return { index, score };
  }).sort((a, b) => b.score - a.score || a.index - b.index).slice(0, count);
}
