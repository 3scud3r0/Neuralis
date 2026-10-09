import type { InterventionSpec, LayerData, PredictionSample, RunConfig } from '../types';
import { VOCABULARY } from './datasets';
import {
  Affine, applyDropout, causalAttention, causalAttentionBackward, LayerNorm,
  type AttentionCache, type LayerNormCache,
} from './layers';
import { argmax, finiteVector, softmax, transpose, zeros } from './math';
import type { ResearchModel } from './model';
import type { BatchResult } from './network';
import { Parameter, ParameterBank, type ParameterBankCheckpoint } from './optimizers';
import { Random } from './random';

export const TRANSFORMER_VOCABULARY = [...VOCABULARY, '<bos>'];
const SEQUENCE_LENGTH = 4;
const DIMENSION = 8;
const HEADS = 2;
const FEED_FORWARD = 16;
const PHRASES = [
  'the robot learns a model .',
  'the researcher builds a model .',
  'the cat sees a pattern .',
  'the robot sees the world .',
  'the researcher learns a pattern .',
  'the cat sees the world .',
  'the researcher sees a model .',
  'the robot builds a model .',
].map(sentence => sentence.split(' ').map(token => TRANSFORMER_VOCABULARY.indexOf(token)));

interface LanguageExample { input: number[]; target: number[]; label: string; }

function languageExample(random: Random): LanguageExample {
  const phrase = random.choice(PHRASES);
  const position = 1 + random.integer(phrase.length - 1);
  const input = phrase.slice(Math.max(0, position - SEQUENCE_LENGTH), position);
  while (input.length < SEQUENCE_LENGTH) input.unshift(TRANSFORMER_VOCABULARY.length - 1);
  const target = Array.from({ length: TRANSFORMER_VOCABULARY.length }, (_, i) => Number(i === phrase[position]));
  return { input, target, label: `${input.map(index => TRANSFORMER_VOCABULARY[index]).join(' ')} → ${TRANSFORMER_VOCABULARY[phrase[position]]}` };
}

interface TransformerPass {
  tokens: number[];
  embedded: number[][];
  normalized1: number[][];
  norm1Cache: LayerNormCache[];
  attention: AttentionCache;
  attentionDropped: number[][];
  attentionMasks: number[][];
  residual1: number[][];
  normalized2: number[][];
  norm2Cache: LayerNormCache[];
  hidden: number[][];
  hiddenDropped: number[][];
  hiddenMasks: number[][];
  residual2: number[][];
  final: number[];
  finalCache: LayerNormCache;
  output: number[];
}

interface TransformerCheckpoint {
  bank: ParameterBankCheckpoint;
  random: ReturnType<Random['exportState']>;
  activationGradients: number[][];
}

/**
 * One complete causal pre-norm Transformer block with two trainable heads.
 * All token/position embeddings, Q/K/V/O projections, FFN, normalization and
 * vocabulary readout parameters receive exact analytic gradients.
 */
export class TransformerModel implements ResearchModel {
  readonly bank = new ParameterBank();
  readonly tokenEmbedding: Parameter;
  readonly positionEmbedding: Parameter;
  readonly norm1: LayerNorm;
  readonly query: Affine;
  readonly key: Affine;
  readonly value: Affine;
  readonly attentionOutput: Affine;
  readonly norm2: LayerNorm;
  readonly ff1: Affine;
  readonly ff2: Affine;
  readonly finalNorm: LayerNorm;
  readonly readout: Affine;
  readonly parameters: number;
  readonly trainingSize: number;
  private readonly random: Random;
  private readonly config: RunConfig;
  private readonly trainSet: LanguageExample[];
  private readonly validation: LanguageExample[];
  private activationGradients = [zeros(4), zeros(8), zeros(8), zeros(16), zeros(8), zeros(13)];

  constructor(config: RunConfig) {
    this.config = { ...config };
    this.random = new Random(config.seed);
    this.tokenEmbedding = this.bank.add(new Parameter('embedding.token', [TRANSFORMER_VOCABULARY.length, DIMENSION], this.random, 0.2));
    this.positionEmbedding = this.bank.add(new Parameter('embedding.position', [SEQUENCE_LENGTH, DIMENSION], this.random, 0.05));
    this.norm1 = new LayerNorm('norm.attention', DIMENSION, this.bank, this.random);
    this.query = new Affine('attention.query', DIMENSION, DIMENSION, this.bank, this.random);
    this.key = new Affine('attention.key', DIMENSION, DIMENSION, this.bank, this.random);
    this.value = new Affine('attention.value', DIMENSION, DIMENSION, this.bank, this.random);
    this.attentionOutput = new Affine('attention.output', DIMENSION, DIMENSION, this.bank, this.random);
    this.norm2 = new LayerNorm('norm.feedforward', DIMENSION, this.bank, this.random);
    this.ff1 = new Affine('ffn.up', DIMENSION, FEED_FORWARD, this.bank, this.random);
    this.ff2 = new Affine('ffn.down', FEED_FORWARD, DIMENSION, this.bank, this.random);
    this.finalNorm = new LayerNorm('norm.final', DIMENSION, this.bank, this.random);
    this.readout = new Affine('readout', DIMENSION, TRANSFORMER_VOCABULARY.length, this.bank, this.random);
    this.parameters = this.bank.size;
    const dataRandom = new Random((config.seed ^ 0x930ce71) >>> 0);
    this.trainSet = Array.from({ length: 256 }, () => languageExample(dataRandom));
    this.validation = Array.from({ length: 96 }, () => languageExample(dataRandom));
    this.trainingSize = this.trainSet.length;
  }

  private modify(input: number[], layer: number, intervention?: InterventionSpec): number[] {
    if (!intervention || intervention.layer !== layer) return input;
    const result = [...input];
    const i = intervention.neuron;
    result[i] = intervention.kind === 'ablate' ? 0 : intervention.kind === 'scale' ?
      result[i] * intervention.value : result[i] + intervention.value;
    return result;
  }

  private forward(tokens: number[], training = false, intervention?: InterventionSpec): TransformerPass {
    if (!finiteVector(tokens, SEQUENCE_LENGTH) || tokens.some(token => !Number.isInteger(token) || token < 0 || token >= TRANSFORMER_VOCABULARY.length)) {
      throw new Error(`Tiny Transformer expects four integer token IDs in [0, ${TRANSFORMER_VOCABULARY.length - 1}].`);
    }
    const sizes = [4, 8, 8, 16, 8, 13];
    if (intervention && (!Number.isInteger(intervention.layer) || intervention.layer < 1 || intervention.layer >= sizes.length ||
      !Number.isInteger(intervention.neuron) || intervention.neuron < 0 || intervention.neuron >= sizes[intervention.layer] ||
      !Number.isFinite(intervention.value) || !['ablate', 'scale', 'steer'].includes(intervention.kind))) {
      throw new Error('Transformer interventions target the final context token in embedding/attention/FFN/readout layers 1–5.');
    }
    const last = SEQUENCE_LENGTH - 1;
    const dropout = training ? this.config.dropout ?? 0 : 0;
    const embedded = tokens.map((token, position) => {
      const vector = Array.from({ length: DIMENSION }, (_, i) => this.tokenEmbedding.values[token * DIMENSION + i] +
        this.positionEmbedding.values[position * DIMENSION + i]);
      return position === last ? this.modify(vector, 1, intervention) : vector;
    });
    const firstNorm = embedded.map(vector => this.norm1.forward(vector));
    const normalized1 = firstNorm.map(result => result.output);
    const attention = causalAttention(normalized1.map(vector => this.query.forward(vector)),
      normalized1.map(vector => this.key.forward(vector)), normalized1.map(vector => this.value.forward(vector)), HEADS);
    const contextDropout = attention.context.map(vector => applyDropout(vector, dropout, this.random));
    const attentionDropped = contextDropout.map(result => result.output);
    const residual1 = embedded.map((vector, token) => {
      const attentionContribution = this.attentionOutput.forward(attentionDropped[token]);
      const result = vector.map((value, i) => value + attentionContribution[i]);
      return token === last ? this.modify(result, 2, intervention) : result;
    });
    const secondNorm = residual1.map(vector => this.norm2.forward(vector));
    const normalized2 = secondNorm.map(result => result.output);
    const hidden = normalized2.map((vector, token) => {
      const result = this.ff1.forward(vector).map(Math.tanh);
      return token === last ? this.modify(result, 3, intervention) : result;
    });
    const hiddenDropout = hidden.map(vector => applyDropout(vector, dropout, this.random));
    const hiddenDropped = hiddenDropout.map(result => result.output);
    const residual2 = residual1.map((vector, token) => {
      const contribution = this.ff2.forward(hiddenDropped[token]);
      const result = vector.map((value, i) => value + contribution[i]);
      return token === last ? this.modify(result, 4, intervention) : result;
    });
    const final = this.finalNorm.forward(residual2[last]);
    const output = this.modify(softmax(this.readout.forward(final.output)), 5, intervention);
    return { tokens: [...tokens], embedded, normalized1, norm1Cache: firstNorm.map(result => result.cache), attention,
      attentionDropped, attentionMasks: contextDropout.map(result => result.mask), residual1,
      normalized2, norm2Cache: secondNorm.map(result => result.cache), hidden, hiddenDropped,
      hiddenMasks: hiddenDropout.map(result => result.mask), residual2,
      final: final.output, finalCache: final.cache, output };
  }

  predict(input: number[], intervention?: InterventionSpec): number[] {
    return this.forward(input, false, intervention).output;
  }

  sampleLoss(input: number[], target: number[]): number {
    return this.predict(input).reduce((total, value, i) => total - target[i] * Math.log(Math.max(1e-12, value)), 0);
  }

  private backward(pass: TransformerPass, target: number[]): void {
    const last = SEQUENCE_LENGTH - 1;
    const outputGradient = pass.output.map((value, i) => value - target[i]);
    const finalGradient = this.readout.backward(pass.final, outputGradient);
    const residual2Gradient = this.finalNorm.backward(pass.finalCache, finalGradient);
    const residual1Gradients = Array.from({ length: SEQUENCE_LENGTH }, () => zeros(DIMENSION));
    const hiddenGradient = this.ff2.backward(pass.hiddenDropped[last], residual2Gradient)
      .map((value, i) => value * pass.hiddenMasks[last][i] * (1 - pass.hidden[last][i] ** 2));
    const secondNormGradient = this.ff1.backward(pass.normalized2[last], hiddenGradient);
    const secondResidualGradient = this.norm2.backward(pass.norm2Cache[last], secondNormGradient);
    residual1Gradients[last] = residual2Gradient.map((value, i) => value + secondResidualGradient[i]);
    const contextGradients = residual1Gradients.map((gradient, token) =>
      this.attentionOutput.backward(pass.attentionDropped[token], gradient).map((value, i) => value * pass.attentionMasks[token][i]));
    const attentionGradients = causalAttentionBackward(pass.attention, contextGradients);
    const embeddedGradients = pass.embedded.map((_, token) => {
      const queryGradient = this.query.backward(pass.normalized1[token], attentionGradients.queries[token]);
      const keyGradient = this.key.backward(pass.normalized1[token], attentionGradients.keys[token]);
      const valueGradient = this.value.backward(pass.normalized1[token], attentionGradients.values[token]);
      const normalizedGradient = queryGradient.map((value, i) => value + keyGradient[i] + valueGradient[i]);
      const directGradient = this.norm1.backward(pass.norm1Cache[token], normalizedGradient);
      return directGradient.map((value, i) => value + residual1Gradients[token][i]);
    });
    for (let token = 0; token < SEQUENCE_LENGTH; token++) {
      for (let i = 0; i < DIMENSION; i++) {
        this.tokenEmbedding.gradients[pass.tokens[token] * DIMENSION + i] += embeddedGradients[token][i];
        this.positionEmbedding.gradients[token * DIMENSION + i] += embeddedGradients[token][i];
      }
    }
    this.activationGradients = [zeros(4), embeddedGradients[last], residual1Gradients[last], hiddenGradient, residual2Gradient, outputGradient];
  }

  /** Full-network analytic derivatives; training=false makes finite differences deterministic. */
  analyticGradients(inputs: number[][], targets: number[][], training = false): BatchResult {
    if (!inputs.length || inputs.length !== targets.length || targets.some(target => !finiteVector(target, TRANSFORMER_VOCABULARY.length))) {
      throw new Error('Invalid Transformer training batch.');
    }
    this.bank.zeroGradients();
    let loss = 0;
    let correct = 0;
    for (let sample = 0; sample < inputs.length; sample++) {
      const pass = this.forward(inputs[sample], training);
      const target = targets[sample];
      loss += pass.output.reduce((total, value, i) => total - target[i] * Math.log(Math.max(1e-12, value)), 0);
      correct += Number(argmax(pass.output) === argmax(target));
      this.backward(pass, target);
    }
    this.bank.scaleGradients(1 / inputs.length);
    return { loss: loss / inputs.length, accuracy: correct / inputs.length, gradientNorm: this.bank.gradientNorm };
  }

  train(learningRate: number, batchSize: number): BatchResult {
    const batch = Array.from({ length: batchSize }, () => this.trainSet[this.random.integer(this.trainSet.length)]);
    const result = this.analyticGradients(batch.map(row => row.input), batch.map(row => row.target), true);
    this.bank.update(learningRate, { kind: this.config.optimizer, weightDecay: this.config.weightDecay, gradientClip: this.config.gradientClip });
    return result;
  }

  evaluate(): BatchResult {
    let loss = 0;
    let correct = 0;
    for (const row of this.validation) {
      const prediction = this.predict(row.input);
      loss += prediction.reduce((total, value, i) => total - row.target[i] * Math.log(Math.max(1e-12, value)), 0);
      correct += Number(argmax(prediction) === argmax(row.target));
    }
    return { loss: loss / this.validation.length, accuracy: correct / this.validation.length, gradientNorm: this.bank.gradientNorm };
  }

  capture(): LayerData[] {
    const pass = this.forward(this.validation[0].input);
    const last = SEQUENCE_LENGTH - 1;
    const activations = [pass.tokens, pass.embedded[last], pass.residual1[last], pass.hidden[last], pass.residual2[last], pass.output];
    const weights = [[], transpose(this.tokenEmbedding.rows()), this.attentionOutput.weight.rows(), this.ff1.weight.rows(), this.ff2.weight.rows(), this.readout.weight.rows()];
    const biases = [[], this.positionEmbedding.values.slice(last * DIMENSION, (last + 1) * DIMENSION),
      [...this.attentionOutput.bias.values], [...this.ff1.bias.values], [...this.ff2.bias.values], [...this.readout.bias.values]];
    const names = ['Four context token IDs', 'Learned token + position embedding', 'Causal attention + residual',
      'Tanh feed-forward expansion', 'Feed-forward + residual', 'Next-token vocabulary'];
    const shapes = [[4], [TRANSFORMER_VOCABULARY.length, DIMENSION], [HEADS, 4, 4], [FEED_FORWARD, DIMENSION], [DIMENSION, FEED_FORWARD], [TRANSFORMER_VOCABULARY.length, DIMENSION]];
    return activations.map((values, i) => ({ id: `transformer-${i}`, name: names[i], size: values.length, shape: shapes[i],
      activations: [...values], weights: weights[i], gradients: [...this.activationGradients[i]], biases: biases[i] }));
  }

  samples(): PredictionSample[] {
    return this.validation.slice(0, 8).map(row => ({ input: [...row.input], target: [...row.target], prediction: this.predict(row.input), label: row.label }));
  }

  extras(): Record<string, unknown> {
    const pass = this.forward(this.validation[0].input);
    return { algorithm: 'Trainable one-block causal pre-norm Transformer', scoreLabel: 'Held-out next-token accuracy',
      architectureNote: 'All embeddings, Q/K/V/O, FFN, layer norms and readout are trained. This tiny synthetic grammar model is not a pretrained LLM.',
      attention: pass.attention.probabilities[0].map(row => [...row]),
      attentionHeads: pass.attention.probabilities.map(head => head.map(row => [...row])),
      attentionKind: 'Actual learned causal scaled dot-product attention · head 0',
      tokens: pass.tokens.map(index => TRANSFORMER_VOCABULARY[index]), vocabulary: [...TRANSFORMER_VOCABULARY],
      headCount: HEADS, headDimension: DIMENSION / HEADS, sequenceLength: SEQUENCE_LENGTH,
      tokenEmbeddings: this.tokenEmbedding.rows(), positionEmbeddings: this.positionEmbedding.rows(),
      residualStream: { embedding: pass.embedded.map(row => [...row]), attention: pass.residual1.map(row => [...row]), feedforward: pass.residual2.map(row => [...row]) },
      q: pass.attention.queries.map(row => [...row]), k: pass.attention.keys.map(row => [...row]), v: pass.attention.values.map(row => [...row]),
      weightGradients: this.bank.parameters.map(parameter => ({ name: parameter.name, shape: [...parameter.shape], values: [...parameter.gradients] })),
      trainingSamples: this.trainSet.length, validationSamples: this.validation.length,
      dataset: 'Small explicit synthetic grammar; draws separated by seeded stream. Grammar contexts can repeat between splits.',
      gradientNote: 'Node gradients show final-context-token loss derivatives from the last training sample. Parameter gradients are minibatch means.',
      visualizationNote: 'Layer summaries show final-context-token vectors; attention mixes all four tokens. Embedding weights are vocabulary lookup values, not dense token-ID connections.',
      dropout: this.config.dropout ?? 0, optimizer: this.config.optimizer ?? 'adam',
      customArchitectureNote: 'This demo fixes dimension=8, heads=2, feed-forward=16, context=4. Dense hiddenLayers overrides are ignored.',
    };
  }

  checkpoint(): TransformerCheckpoint {
    return { bank: this.bank.checkpoint(), random: this.random.exportState(), activationGradients: this.activationGradients.map(row => [...row]) };
  }

  restore(value: unknown): void {
    const saved = value as TransformerCheckpoint;
    const sizes = [4, 8, 8, 16, 8, 13];
    if (!saved || !Array.isArray(saved.activationGradients) || saved.activationGradients.length !== sizes.length ||
      saved.activationGradients.some((row, i) => !finiteVector(row, sizes[i]))) throw new Error('Invalid Transformer checkpoint.');
    const random = new Random(0);
    random.importState(saved.random);
    this.bank.restore(saved.bank);
    this.random.importState(saved.random);
    this.activationGradients = saved.activationGradients.map(row => [...row]);
  }
}
