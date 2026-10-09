import type { DemoDefinition, RunConfig } from '../types';
import { argmax, clamp, finiteVector, mean, softmax } from './math';
import type { Objective } from './network';
import { Random } from './random';

export interface Example {
  input: number[];
  target: number[];
  label: string;
  clean?: number[];
  tokens?: string[];
  attention?: number[][];
}

export interface Dataset {
  train: Example[];
  validation: Example[];
  objective: Objective;
  inputSize: number;
  outputSize: number;
  scoreLabel: string;
  extras: Record<string, unknown>;
}

export type CustomDataset = NonNullable<RunConfig['customDataset']>;

/** Validate uploaded numerical arrays before model allocation or checkpoint restore. */
export function validateCustomDataset(value: CustomDataset): CustomDataset {
  if (!value || !['classification', 'regression'].includes(value.task) ||
    !Array.isArray(value.inputs) || !Array.isArray(value.targets) || value.inputs.length < 2 || value.inputs.length > 10_000 ||
    value.targets.length !== value.inputs.length || !finiteVector(value.inputs[0]) || !finiteVector(value.targets[0])) {
    throw new Error('Custom data needs 2–10,000 paired numerical examples and a classification/regression task.');
  }
  const inputSize = value.inputs[0].length;
  const outputSize = value.targets[0].length;
  if (inputSize < 1 || inputSize > 128 || outputSize < 1 || outputSize > 128) throw new Error('Custom data permits 1–128 input features and 1–128 output targets.');
  const validateRows = (inputs: number[][], targets: number[][]) => {
    if (inputs.length !== targets.length || inputs.length > 10_000 ||
      inputs.some(row => !finiteVector(row, inputSize)) || targets.some(row => !finiteVector(row, outputSize))) {
      throw new Error('Custom data contains ragged, mismatched, or nonfinite numerical rows.');
    }
    if (value.task === 'classification' && targets.some(row => row.some(item => item < 0 || item > 1) ||
      (outputSize > 1 && Math.abs(row.reduce((total, item) => total + item, 0) - 1) > 1e-6))) {
      throw new Error('Classification targets must be binary probabilities or normalized one-hot/probability vectors.');
    }
  };
  validateRows(value.inputs, value.targets);
  if ((value.validationInputs === undefined) !== (value.validationTargets === undefined)) throw new Error('Held-out inputs and targets must be supplied together.');
  if (value.validationInputs && value.validationTargets) {
    if (!value.validationInputs.length) throw new Error('The explicit validation split must not be empty.');
    validateRows(value.validationInputs, value.validationTargets);
  }
  const cells = (value.inputs.length + (value.validationInputs?.length ?? 0)) * (inputSize + outputSize);
  if (cells > 1_000_000) throw new Error('Custom data exceeds the one-million numerical value browser memory budget.');
  if (value.name !== undefined && (typeof value.name !== 'string' || value.name.length > 500)) throw new Error('Dataset name is invalid.');
  if (value.labels && (!Array.isArray(value.labels) || value.labels.length > 128 || value.labels.some(label => typeof label !== 'string' || label.length > 500))) throw new Error('Dataset labels are invalid.');
  if (value.featureNames && (!Array.isArray(value.featureNames) || value.featureNames.length !== inputSize || value.featureNames.some(label => typeof label !== 'string' || label.length > 500))) throw new Error('Feature names must match the number of inputs.');
  return { name: value.name, task: value.task, inputs: value.inputs.map(row => [...row]), targets: value.targets.map(row => [...row]),
    validationInputs: value.validationInputs?.map(row => [...row]), validationTargets: value.validationTargets?.map(row => [...row]),
    labels: value.labels ? [...value.labels] : undefined, featureNames: value.featureNames ? [...value.featureNames] : undefined };
}

export function createCustomDataset(value: CustomDataset, seed: number): Dataset {
  const data = validateCustomDataset(value);
  const random = new Random((seed ^ 0x624ac19) >>> 0);
  const objective: Objective = data.task === 'regression' ? 'mse' : data.targets[0].length === 1 ? 'binary' : 'categorical';
  const example = (input: number[], target: number[]): Example => ({ input: [...input], target: [...target],
    label: data.task === 'regression' ? 'Uploaded numerical target' : data.labels?.[objective === 'binary' ? Number(target[0] >= 0.5) : argmax(target)] ?? 'Uploaded class' });
  let train: Example[];
  let validation: Example[];
  if (data.validationInputs && data.validationTargets) {
    train = data.inputs.map((input, i) => example(input, data.targets[i]));
    validation = data.validationInputs.map((input, i) => example(input, data.validationTargets![i]));
  } else {
    const indices = random.shuffle(Array.from({ length: data.inputs.length }, (_, i) => i));
    const validationCount = Math.max(1, Math.floor(data.inputs.length * 0.2));
    validation = indices.slice(0, validationCount).map(i => example(data.inputs[i], data.targets[i]));
    train = indices.slice(validationCount).map(i => example(data.inputs[i], data.targets[i]));
  }
  const originalValidationCount = validation.length;
  if (validation.length > 1024) validation = random.shuffle([...validation]).slice(0, 1024);
  return { train, validation, objective, inputSize: train[0].input.length, outputSize: train[0].target.length,
    scoreLabel: data.task === 'regression' ? 'Explained variance (R², clipped at zero)' : 'Held-out accuracy',
    extras: { dataset: data.name ?? 'Uploaded numerical dataset', datasetOrigin: 'User supplied', trainingSamples: train.length,
      validationSamples: validation.length, totalValidationSamples: originalValidationCount,
      scoreLabel: data.task === 'regression' ? 'R² (clipped to [0, 1])' : 'Held-out accuracy',
      featureNames: data.featureNames, outputLabels: data.labels,
      splitNote: data.validationInputs ? 'Explicit training and held-out splits supplied by the dataset workbench.' : 'Seeded random 80/20 holdout; preprocessing must be fitted on the training split to avoid leakage.',
      captureNote: originalValidationCount > 1024 ? 'Validation measurements use a fixed seeded 1,024-row holdout subset to bound browser capture cost.' : undefined,
      inputShape: [train[0].input.length], architectureNote: 'A dense MLP trained on uploaded numerical features; hidden-layer overrides are supported.',
    } };
}

const oneHot = (index: number, size: number) => Array.from({ length: size }, (_, i) => Number(i === index));

function binary(input: number[], label: boolean, names = ['Negative', 'Positive']): Example {
  return { input, target: [Number(label)], label: names[Number(label)] };
}

function multiclass(input: number[], label: number, names: string[]): Example {
  return { input, target: oneHot(label, names.length), label: names[label] };
}

function regression(input: number[], target: number[], label = 'Continuous target'): Example {
  return { input, target, label };
}

const DIGITS = [
  ['01110', '11011', '11011', '11011', '01110'],
  ['00100', '01100', '00100', '00100', '01110'],
  ['01110', '00011', '00110', '01100', '11111'],
  ['11110', '00011', '01110', '00011', '11110'],
  ['11010', '11010', '11111', '00010', '00010'],
  ['11111', '11000', '11110', '00011', '11110'],
  ['01110', '11000', '11110', '11011', '01110'],
  ['11111', '00011', '00110', '01100', '01100'],
  ['01110', '11011', '01110', '11011', '01110'],
  ['01110', '11011', '01111', '00011', '01110'],
].map(rows => rows.join('').split('').map(Number));

export const VOCABULARY = ['the', 'robot', 'cat', 'researcher', 'learns', 'sees', 'builds', 'a', 'model', 'pattern', 'world', '.'];
const PHRASES = [
  ['the', 'robot', 'learns', 'a', 'model', '.'],
  ['the', 'researcher', 'builds', 'a', 'model', '.'],
  ['the', 'cat', 'sees', 'a', 'pattern', '.'],
  ['the', 'robot', 'sees', 'the', 'world', '.'],
  ['the', 'researcher', 'learns', 'a', 'pattern', '.'],
];

const SENTIMENT_WORDS = ['great', 'clear', 'helpful', 'excellent', 'bad', 'confusing', 'broken', 'slow', 'model', 'result', 'very', 'the'];
const SENTIMENT_VECTORS = SENTIMENT_WORDS.map((_, index) => [
  index < 4 ? 1 : index < 8 ? -1 : 0,
  index === 10 ? 1 : 0,
  index >= 8 && index <= 9 ? 1 : 0,
  (index % 3 - 1) * 0.4,
  Math.sin(index * 0.9),
  Math.cos(index * 0.9),
]);

/** Fixed scaled dot-product attention. Dense readout is the trainable component. */
export function attentionPool(tokens: string[]): { feature: number[]; attention: number[][] } {
  const vectors = tokens.map(token => {
    const index = SENTIMENT_WORDS.indexOf(token);
    if (index < 0) throw new Error(`Unknown sentiment token: ${token}`);
    return SENTIMENT_VECTORS[index];
  });
  const queries = vectors.map((vector, position) => vector.map((value, i) => value + Math.sin((position + 1) * (i + 1)) * 0.05));
  const attention = queries.map(query => softmax(vectors.map(key =>
    query.reduce((total, value, i) => total + value * key[i], 0) / Math.sqrt(query.length))));
  const contextual = attention.map(row => Array.from({ length: vectors[0].length }, (_, dimension) =>
    row.reduce((total, weight, i) => total + weight * vectors[i][dimension], 0)));
  return { feature: contextual[0].map((_, dimension) => mean(contextual.map(vector => vector[dimension]))), attention };
}

function sampleDemo(id: string, random: Random, sampleIndex: number): Example {
  const x = random.uniform() * 2 - 1;
  const y = random.uniform() * 2 - 1;
  switch (id) {
    case 'xor': {
      const first = random.integer(2);
      const second = random.integer(2);
      return binary([first, second], first !== second, ['Equal bits', 'Different bits']);
    }
    case 'circles': {
      const inside = sampleIndex % 2 === 0;
      const angle = random.uniform() * Math.PI * 2;
      const radius = inside ? 0.22 + random.uniform() * 0.2 : 0.68 + random.uniform() * 0.23;
      return binary([Math.cos(angle) * radius, Math.sin(angle) * radius], !inside, ['Inner ring', 'Outer ring']);
    }
    case 'spirals': {
      const label = sampleIndex % 3;
      const radius = 0.1 + random.uniform() * 0.85;
      const angle = radius * 4.5 + label * Math.PI * 2 / 3 + random.normal(0, 0.1);
      return multiclass([radius * Math.sin(angle), radius * Math.cos(angle)], label, ['Spiral A', 'Spiral B', 'Spiral C']);
    }
    case 'moons': {
      const label = sampleIndex % 2;
      const angle = random.uniform() * Math.PI;
      return binary([label ? 1 - Math.cos(angle) : Math.cos(angle),
        label ? 0.5 - Math.sin(angle) : Math.sin(angle)].map(value => value + random.normal(0, 0.07)),
      Boolean(label), ['Upper moon', 'Lower moon']);
    }
    case 'gaussian-clusters': {
      const label = sampleIndex % 4;
      const centers = [[-0.6, -0.6], [-0.6, 0.6], [0.6, -0.6], [0.6, 0.6]];
      return multiclass(centers[label].map(value => value + random.normal(0, 0.18)), label,
        ['Northwest', 'Southwest', 'Northeast', 'Southeast']);
    }
    case 'parity': {
      const bits = Array.from({ length: 4 }, () => random.integer(2));
      return binary(bits, bits.reduce((total, value) => total + value, 0) % 2 === 1, ['Even parity', 'Odd parity']);
    }
    case 'sine-regression':
      return regression([x], [Math.sin(x * Math.PI) + random.normal(0, 0.025)], 'Sine signal');
    case 'surface-regression':
      return regression([x, y], [Math.sin(Math.PI * x) * Math.cos(Math.PI * y)], 'Saddle-wave surface');
    case 'linear-regression':
      return regression([x, y, random.normal(0, 0.5)], [0.55 * x - 0.35 * y + 0.12 + random.normal(0, 0.02)], 'Linear response');
    case 'forecasting': {
      const phase = random.uniform() * 2 * Math.PI;
      const frequency = 0.2 + random.uniform() * 0.12;
      const input = Array.from({ length: 8 }, (_, i) => Math.sin(phase + i * frequency));
      return regression(input, [Math.sin(phase + 8 * frequency)], 'Next signal value');
    }
    case 'image-patterns': {
      const label = sampleIndex % 4;
      const offset = random.integer(3) - 1;
      const input = Array.from({ length: 64 }, (_, index) => {
        const row = Math.floor(index / 8);
        const column = index % 8;
        const signal = label === 0 ? Math.abs(column - 3 - offset) <= 1 :
          label === 1 ? Math.abs(row - 3 - offset) <= 1 :
          label === 2 ? Math.abs(row - column - offset) <= 1 : Math.abs(row + column - 7 - offset) <= 1;
        return clamp(Number(signal) + random.normal(0, 0.12), 0, 1);
      });
      return multiclass(input, label, ['Vertical', 'Horizontal', 'Diagonal', 'Anti-diagonal']);
    }
    case 'tiny-digits': {
      const label = sampleIndex % 10;
      const input = DIGITS[label].map(value => clamp(value + random.normal(0, 0.16), 0, 1));
      return multiclass(input, label, DIGITS.map((_, i) => String(i)));
    }
    case 'autoencoder':
    case 'denoising': {
      const label = sampleIndex % DIGITS.length;
      const clean = DIGITS[label].map(value => value * 2 - 1);
      const input = clean.map(value => id === 'denoising' ? clamp(value + random.normal(0, 0.45), -1, 1) : value);
      return { input, target: clean, label: `Synthetic digit ${label}`, clean };
    }
    case 'next-token': {
      const phrase = random.choice(PHRASES);
      const position = random.integer(phrase.length - 2) + 1;
      const context = phrase.slice(position - 1, position + 1);
      const input = context.flatMap(token => oneHot(VOCABULARY.indexOf(token), VOCABULARY.length));
      const next = phrase[position + 1];
      return { input, target: oneHot(VOCABULARY.indexOf(next), VOCABULARY.length),
        label: `${context.join(' ')} → ${next}`, tokens: context };
    }
    case 'attention-sentiment': {
      const positive = sampleIndex % 2 === 0;
      const adjective = random.choice(SENTIMENT_WORDS.slice(positive ? 0 : 4, positive ? 4 : 8));
      const tokens = ['the', random.choice(['model', 'result']), random.choice(['very', 'the']), adjective];
      const pooled = attentionPool(tokens);
      return { input: pooled.feature, target: [Number(positive)], label: tokens.join(' '), tokens, attention: pooled.attention };
    }
    case 'anomaly-detection': {
      const anomaly = sampleIndex % 3 === 0;
      const angle = random.uniform() * Math.PI * 2;
      const radius = anomaly ? 0.9 + random.uniform() * 0.5 : 0.3 + random.normal(0, 0.06);
      return binary([Math.cos(angle) * radius, Math.sin(angle) * radius, random.normal(0, anomaly ? 0.65 : 0.1)],
        anomaly, ['Nominal sensor', 'Anomalous sensor']);
    }
    case 'pca': {
      const first = random.normal();
      const second = random.normal(0, 0.4);
      const input = [first + second * 0.1, first * 0.7 + second, first * -0.5 + second * 0.3 + random.normal(0, 0.02)];
      return regression(input, input, 'Correlated observation');
    }
    case 'k-means': {
      const label = sampleIndex % 4;
      const center = [[-0.8, -0.55], [-0.5, 0.65], [0.7, -0.55], [0.6, 0.75]][label];
      return multiclass(center.map(value => value + random.normal(0, 0.14)), label, ['Cluster A', 'Cluster B', 'Cluster C', 'Cluster D']);
    }
    default:
      throw new Error(`No browser dataset is registered for ${id}.`);
  }
}

export function createDataset(demo: DemoDefinition, seed: number): Dataset {
  const random = new Random((seed ^ 0x5a17c9e3) >>> 0);
  const trainCount = demo.id === 'xor' ? 128 : demo.id === 'parity' ? 512 :
    ['tiny-digits', 'autoencoder', 'denoising', 'next-token'].includes(demo.id) ? 480 : 384;
  const train = Array.from({ length: trainCount }, (_, i) => sampleDemo(demo.id, random, i));
  const validation = Array.from({ length: 128 }, (_, i) => sampleDemo(demo.id, random, i + trainCount));
  random.shuffle(train);
  const outputSize = train[0].target.length;
  const regressionDemos = ['sine-regression', 'surface-regression', 'linear-regression', 'forecasting', 'autoencoder', 'denoising', 'pca'];
  const objective: Objective = regressionDemos.includes(demo.id) ? 'mse' : outputSize === 1 ? 'binary' : 'categorical';
  return {
    train, validation, objective,
    inputSize: train[0].input.length, outputSize,
    scoreLabel: objective === 'mse' ? 'Explained variance (R², clipped at zero)' : 'Validation accuracy',
    extras: {
      dataset: 'Deterministic synthetic data; training and validation draws are separate.',
      trainingSamples: train.length,
      validationSamples: validation.length,
      scoreLabel: objective === 'mse' ? 'R² (clipped to [0, 1])' : 'Accuracy',
      inputShape: ['image-patterns'].includes(demo.id) ? [8, 8] :
        ['tiny-digits', 'autoencoder', 'denoising'].includes(demo.id) ? [5, 5] : [train[0].input.length],
      vocabulary: demo.id === 'next-token' ? VOCABULARY : undefined,
      architectureNote: demo.id === 'attention-sentiment' ?
        'Fixed scaled dot-product Q/K/V pooling; only the dense readout is trained. This is not a pretrained Transformer.' :
        ['tiny-digits', 'image-patterns'].includes(demo.id) ?
        'Flattened synthetic pixels with a dense MLP. No convolutional layers or external image dataset.' :
        demo.id === 'next-token' ? 'Two-token one-hot context → dense MLP → vocabulary softmax. No Transformer or pretrained language model.' : undefined,
    },
  };
}
