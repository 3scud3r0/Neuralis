export interface ActivationPair {
  id: string;
  negative: number[];
  positive: number[];
  split: 'train' | 'evaluation';
}
export interface ConceptAxis {
  vector: number[];
  center: number[];
  trainPairs: number;
  dimension: number;
  meanDifferenceNorm: number;
  threshold: number;
}
export interface AxisEvaluation {
  accuracy: number;
  auc: number;
  meanMargin: number;
  pairOrdering: number;
  scores: { id: string; negative: number; positive: number; margin: number }[];
}
export const examplePairs: ActivationPair[] = [
  { id: 'concise-01', negative: [0.2, 1.4, -0.3, 0.8, 0.4, -0.2], positive: [1.6, 0.3, -0.1, 0.5, 0.5, -0.2], split: 'train' },
  { id: 'concise-02', negative: [-0.1, 1.2, 0.5, 0.1, -0.3, 0.2], positive: [1.2, 0.2, 0.7, -0.2, -0.2, 0.2], split: 'train' },
  { id: 'concise-03', negative: [0.5, 1.7, 0.2, -0.2, 0.1, 0.6], positive: [1.8, 0.4, 0.3, -0.4, 0.2, 0.7], split: 'train' },
  { id: 'held-out-01', negative: [0.1, 1.1, -0.2, 0.6, 0.7, 0.1], positive: [1.3, 0.1, -0.1, 0.2, 0.8, 0.2], split: 'evaluation' },
  { id: 'held-out-02', negative: [0.3, 1.5, 0.8, -0.1, 0.2, -0.2], positive: [1.4, 0.5, 0.7, -0.3, 0.4, -0.3], split: 'evaluation' },
  { id: 'held-out-03', negative: [0.8, 0.6, -0.4, 0.3, 0.5, -0.5], positive: [0.7, 0.8, -0.3, 0.4, 0.4, -0.4], split: 'evaluation' },
];
export function dot(a: number[], b: number[]): number {
  if (a.length !== b.length || !a.length || [...a, ...b].some(value => !Number.isFinite(value))) throw new Error('Vectors must have the same nonzero dimension and finite entries.');
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}
export function decodePairs(text: string): ActivationPair[] {
  if (text.length > 2_000_000) throw new Error('Activation import exceeds 2 MB.');
  const input: unknown = JSON.parse(text);
  if (!Array.isArray(input) || input.length < 2 || input.length > 2000) throw new Error('Supply between 2 and 2,000 paired captures.');
  let dimension = 0;
  const identifiers = new Set<string>();
  return input.map((item: unknown) => {
    if (!item || typeof item !== 'object') throw new Error('Each pair must be an object.');
    const pair = item as ActivationPair;
    if (typeof pair.id !== 'string' || !pair.id || identifiers.has(pair.id)) throw new Error('Pair identifiers must be nonempty and unique.');
    identifiers.add(pair.id);
    if (!Array.isArray(pair.negative) || !Array.isArray(pair.positive) || !pair.negative.length || pair.negative.length > 4096 ||
      pair.negative.length !== pair.positive.length || [...pair.negative, ...pair.positive].some(value => typeof value !== 'number' || !Number.isFinite(value))) throw new Error(`Invalid activation vectors in ${pair.id}.`);
    dimension ||= pair.negative.length;
    if (pair.negative.length !== dimension) throw new Error('All pairs must use the same activation dimension.');
    if (pair.split !== 'train' && pair.split !== 'evaluation') throw new Error('Each pair needs an explicit train or evaluation split.');
    return { id: pair.id, negative: [...pair.negative], positive: [...pair.positive], split: pair.split };
  });
}
export function fitAxis(pairs: ActivationPair[]): ConceptAxis {
  const validated = decodePairs(JSON.stringify(pairs));
  const train = validated.filter(pair => pair.split === 'train');
  if (train.length < 2) throw new Error('At least two training pairs are required.');
  const dimension = train[0].negative.length;
  const difference = Array.from({ length: dimension }, (_, i) => train.reduce((sum, pair) => sum + pair.positive[i] - pair.negative[i], 0) / train.length);
  const norm = Math.sqrt(dot(difference, difference));
  if (norm < 1e-10) throw new Error('The average paired difference is zero; no direction can be fitted.');
  const vector = difference.map(value => value / norm);
  const center = Array.from({ length: dimension }, (_, i) => train.reduce((sum, pair) => sum + (pair.negative[i] + pair.positive[i]) / 2, 0) / train.length);
  return { vector, center, trainPairs: train.length, dimension, meanDifferenceNorm: norm, threshold: dot(center, vector) };
}
export function evaluateAxis(axis: ConceptAxis, pairs: ActivationPair[]): AxisEvaluation {
  const evaluation = decodePairs(JSON.stringify(pairs)).filter(pair => pair.split === 'evaluation');
  if (!evaluation.length) throw new Error('At least one held-out evaluation pair is required.');
  const scores = evaluation.map(pair => {
    const negative = dot(pair.negative, axis.vector) - axis.threshold;
    const positive = dot(pair.positive, axis.vector) - axis.threshold;
    return { id: pair.id, negative, positive, margin: positive - negative };
  });
  const correct = scores.reduce((sum, item) => sum + Number(item.negative < 0) + Number(item.positive >= 0), 0);
  let concordance = 0;
  for (const p of scores) for (const n of scores) concordance += p.positive > n.negative ? 1 : p.positive === n.negative ? 0.5 : 0;
  return { accuracy: correct / (2 * scores.length), auc: concordance / (scores.length * scores.length),
    meanMargin: scores.reduce((sum, item) => sum + item.margin, 0) / scores.length,
    pairOrdering: scores.filter(item => item.margin > 0).length / scores.length, scores };
}
export function steerActivation(input: number[], axis: ConceptAxis, strength: number): number[] {
  dot(input, axis.vector);
  if (!Number.isFinite(strength) || Math.abs(strength) > 20) throw new Error('Steering strength must be finite and between -20 and 20.');
  return input.map((value, index) => value + strength * axis.vector[index]);
}
