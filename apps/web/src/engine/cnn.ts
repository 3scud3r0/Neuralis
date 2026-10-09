import type { DemoDefinition, InterventionSpec, LayerData, PredictionSample, RunConfig } from '../types';
import { createDataset, type Dataset } from './datasets';
import { Affine, applyDropout, AveragePool2D, Convolution2D } from './layers';
import { argmax, finiteVector, softmax, zeros } from './math';
import type { ResearchModel } from './model';
import type { BatchResult } from './network';
import { ParameterBank, type ParameterBankCheckpoint } from './optimizers';
import { Random } from './random';

interface CNNPass {
  input: number[];
  convolution: number[];
  pooled: number[];
  hidden: number[];
  dropped: number[];
  dropoutMask: number[];
  output: number[];
}

interface CNNCheckpoint {
  bank: ParameterBankCheckpoint;
  random: ReturnType<Random['exportState']>;
  activationGradients: number[][];
}

/** A fully trainable small convolutional classifier for the browser. */
export class CNNModel implements ResearchModel {
  readonly bank = new ParameterBank();
  readonly convolution: Convolution2D;
  readonly pool: AveragePool2D;
  readonly hidden: Affine;
  readonly readout: Affine;
  readonly parameters: number;
  readonly trainingSize: number;
  private readonly data: Dataset;
  private readonly random: Random;
  private readonly config: RunConfig;
  private activationGradients: number[][];

  constructor(demo: DemoDefinition, config: RunConfig) {
    this.config = { ...config };
    this.data = createDataset({ ...demo, id: 'image-patterns' }, config.seed);
    this.random = new Random(config.seed);
    this.convolution = new Convolution2D('conv', { width: 8, height: 8, inputChannels: 1, outputChannels: 2, kernel: 3 }, this.bank, this.random);
    this.pool = new AveragePool2D(6, 6, 2, 2);
    this.hidden = new Affine('hidden', this.pool.outputSize, 12, this.bank, this.random);
    this.readout = new Affine('readout', 12, 4, this.bank, this.random);
    this.parameters = this.bank.size;
    this.trainingSize = this.data.train.length;
    this.activationGradients = [zeros(64), zeros(72), zeros(18), zeros(12), zeros(4)];
  }

  private modify(input: number[], layer: number, intervention?: InterventionSpec): number[] {
    if (!intervention || intervention.layer !== layer) return input;
    const result = [...input];
    const neuron = intervention.neuron;
    result[neuron] = intervention.kind === 'ablate' ? 0 : intervention.kind === 'scale' ?
      result[neuron] * intervention.value : result[neuron] + intervention.value;
    return result;
  }

  private forward(input: number[], training = false, intervention?: InterventionSpec): CNNPass {
    if (!finiteVector(input, 64)) throw new Error('CNN expects 64 finite pixels from an 8×8 image.');
    const sizes = [64, 72, 18, 12, 4];
    if (intervention && (!Number.isInteger(intervention.layer) || intervention.layer < 0 || intervention.layer >= sizes.length ||
      !Number.isInteger(intervention.neuron) || intervention.neuron < 0 || intervention.neuron >= sizes[intervention.layer] ||
      !Number.isFinite(intervention.value) || !['ablate', 'scale', 'steer'].includes(intervention.kind))) throw new Error('Invalid CNN activation intervention.');
    const modifiedInput = this.modify([...input], 0, intervention);
    const convolution = this.modify(this.convolution.forward(modifiedInput).map(Math.tanh), 1, intervention);
    const pooled = this.modify(this.pool.forward(convolution), 2, intervention);
    const hidden = this.modify(this.hidden.forward(pooled).map(Math.tanh), 3, intervention);
    const dropout = applyDropout(hidden, training ? this.config.dropout ?? 0 : 0, this.random);
    const output = this.modify(softmax(this.readout.forward(dropout.output)), 4, intervention);
    return { input: modifiedInput, convolution, pooled, hidden, dropped: dropout.output, dropoutMask: dropout.mask, output };
  }

  predict(input: number[], intervention?: InterventionSpec): number[] {
    return this.forward(input, false, intervention).output;
  }

  sampleLoss(input: number[], target: number[]): number {
    return this.predict(input).reduce((total, value, i) => total - target[i] * Math.log(Math.max(1e-12, value)), 0);
  }

  /** Exposed for meaningful analytic-gradient tests and research inspection. */
  analyticGradients(inputs: number[][], targets: number[][], training = false): BatchResult {
    if (!inputs.length || inputs.length !== targets.length || targets.some(target => !finiteVector(target, 4))) throw new Error('Invalid CNN training batch.');
    this.bank.zeroGradients();
    let loss = 0;
    let correct = 0;
    for (let sample = 0; sample < inputs.length; sample++) {
      const pass = this.forward(inputs[sample], training);
      const target = targets[sample];
      loss += pass.output.reduce((total, value, i) => total - target[i] * Math.log(Math.max(1e-12, value)), 0);
      correct += Number(argmax(pass.output) === argmax(target));
      const outputGradient = pass.output.map((value, i) => value - target[i]);
      const hiddenGradient = this.readout.backward(pass.dropped, outputGradient)
        .map((value, i) => value * pass.dropoutMask[i] * (1 - pass.hidden[i] ** 2));
      const pooledGradient = this.hidden.backward(pass.pooled, hiddenGradient);
      const convolutionGradient = this.pool.backward(pooledGradient).map((value, i) => value * (1 - pass.convolution[i] ** 2));
      const inputGradient = this.convolution.backward(pass.input, convolutionGradient);
      this.activationGradients = [inputGradient, convolutionGradient, pooledGradient, hiddenGradient, outputGradient];
    }
    this.bank.scaleGradients(1 / inputs.length);
    return { loss: loss / inputs.length, accuracy: correct / inputs.length, gradientNorm: this.bank.gradientNorm };
  }

  train(learningRate: number, batchSize: number): BatchResult {
    const batch = Array.from({ length: batchSize }, () => this.data.train[this.random.integer(this.data.train.length)]);
    const result = this.analyticGradients(batch.map(row => row.input), batch.map(row => row.target), true);
    this.bank.update(learningRate, { kind: this.config.optimizer, weightDecay: this.config.weightDecay, gradientClip: this.config.gradientClip });
    return result;
  }

  evaluate(): BatchResult {
    let loss = 0;
    let correct = 0;
    for (const row of this.data.validation) {
      const prediction = this.predict(row.input);
      loss += prediction.reduce((total, value, i) => total - row.target[i] * Math.log(Math.max(1e-12, value)), 0);
      correct += Number(argmax(prediction) === argmax(row.target));
    }
    return { loss: loss / this.data.validation.length, accuracy: correct / this.data.validation.length, gradientNorm: this.bank.gradientNorm };
  }

  capture(): LayerData[] {
    const pass = this.forward(this.data.validation[0].input);
    const activations = [pass.input, pass.convolution, pass.pooled, pass.hidden, pass.output];
    const names = ['8×8 pixels', '3×3 convolution · 2 channels', '2×2 average pooling', 'Dense tanh readout', 'Orientation softmax'];
    const weights = [[], this.convolution.expandedWeights(), this.pool.expandedWeights(), this.hidden.weight.rows(), this.readout.weight.rows()];
    const biases = [[], Array.from({ length: 72 }, (_, i) => this.convolution.bias.values[Math.floor(i / 36)]),
      zeros(18), [...this.hidden.bias.values], [...this.readout.bias.values]];
    const shapes = [[1, 8, 8], [2, 6, 6], [2, 3, 3], [12, 18], [4, 12]];
    return activations.map((values, i) => ({ id: `cnn-${i}`, name: names[i], size: values.length, shape: shapes[i],
      activations: [...values], weights: weights[i], gradients: [...this.activationGradients[i]], biases: biases[i] }));
  }

  samples(): PredictionSample[] {
    return this.data.validation.slice(0, 8).map(row => ({ input: [...row.input], target: [...row.target], prediction: this.predict(row.input), label: row.label }));
  }

  extras(): Record<string, unknown> {
    const pass = this.forward(this.data.validation[0].input);
    return { ...this.data.extras, architectureNote: 'Trainable 3×3 valid convolution (2 channels), tanh, 2×2 average pooling, dense tanh, softmax.',
      algorithm: 'CNN with exact analytic convolution/pooling backpropagation', scoreLabel: 'Validation accuracy',
      kernels: this.convolution.weight.rows(), kernelShape: [2, 1, 3, 3], featureMaps: pass.convolution,
      featureMapShape: [2, 6, 6], pooledMaps: pass.pooled, pooledShape: [2, 3, 3],
      weightSharing: 'Spatial convolution edges share 18 kernel parameters; expanded visual edges do not add parameters.',
      gradientNote: 'Node gradients are loss derivatives from the final sample of the most recent batch. Parameter gradients are batch means.',
      weightGradients: this.bank.parameters.map(parameter => ({ name: parameter.name, shape: [...parameter.shape], values: [...parameter.gradients] })),
      dropout: this.config.dropout ?? 0, optimizer: this.config.optimizer ?? 'adam',
      customArchitectureNote: 'This demo uses a fixed convolution/pooling architecture. Custom dense hiddenLayers are ignored.',
    };
  }

  checkpoint(): CNNCheckpoint {
    return { bank: this.bank.checkpoint(), random: this.random.exportState(), activationGradients: this.activationGradients.map(row => [...row]) };
  }

  restore(value: unknown): void {
    const saved = value as CNNCheckpoint;
    const sizes = [64, 72, 18, 12, 4];
    if (!saved || !Array.isArray(saved.activationGradients) || saved.activationGradients.length !== sizes.length ||
      saved.activationGradients.some((row, i) => !finiteVector(row, sizes[i]))) throw new Error('Invalid CNN checkpoint.');
    const random = new Random(0);
    random.importState(saved.random);
    this.bank.restore(saved.bank);
    this.random.importState(saved.random);
    this.activationGradients = saved.activationGradients.map(row => [...row]);
  }
}
