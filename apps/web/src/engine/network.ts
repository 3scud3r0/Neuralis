import type { InterventionSpec, LayerData } from '../types';
import { argmax, clamp, finiteMatrix, finiteVector, matrix, sigmoid, softmax, zeros } from './math';
import { Random } from './random';
import { updateScalar, type OptimizerOptions } from './optimizers';

export type Activation = 'tanh' | 'relu' | 'sigmoid' | 'linear' | 'softmax';
export type Objective = 'categorical' | 'binary' | 'mse';

export interface DenseLayer {
  input: number;
  output: number;
  activation: Activation;
  weights: number[][];
  biases: number[];
  weightGradient: number[][];
  biasGradient: number[];
  firstWeights: number[][];
  secondWeights: number[][];
  firstBiases: number[];
  secondBiases: number[];
}

export interface ForwardPass {
  activations: number[][];
  preactivations: number[][];
  dropoutMasks: number[][];
}

export interface DenseTrainOptions extends OptimizerOptions { dropout?: number; random?: Random; }

export interface BatchResult {
  loss: number;
  accuracy: number;
  gradientNorm: number;
}

export interface NetworkCheckpoint {
  architecture: number[];
  objective: Objective;
  updates: number;
  layers: DenseLayer[];
}

function activate(values: number[], activation: Activation): number[] {
  if (activation === 'softmax') return softmax(values);
  return values.map(value => activation === 'tanh' ? Math.tanh(value) :
    activation === 'relu' ? Math.max(0, value) : activation === 'sigmoid' ? sigmoid(value) : value);
}

function derivative(value: number, preactivation: number, activation: Activation): number {
  if (activation === 'tanh') return 1 - value * value;
  if (activation === 'relu') return preactivation > 0 ? 1 : 0;
  if (activation === 'sigmoid') return value * (1 - value);
  return 1;
}

/** Dense feed-forward network with exact minibatch gradients and Adam updates. */
export class DenseNetwork {
  readonly architecture: number[];
  readonly objective: Objective;
  readonly layers: DenseLayer[];
  updates = 0;

  constructor(architecture: number[], objective: Objective, random: Random, hiddenActivation: Activation = 'tanh') {
    if (architecture.length < 2 || architecture.length > 10 ||
      architecture.some(size => !Number.isInteger(size) || size < 1 || size > 256)) {
      throw new Error('Architecture needs 2–10 layers with 1–256 units each.');
    }
    if (!['tanh', 'relu', 'sigmoid', 'linear'].includes(hiddenActivation)) {
      throw new Error('Unsupported hidden-layer activation.');
    }
    this.architecture = [...architecture];
    this.objective = objective;
    this.layers = architecture.slice(1).map((output, index) => {
      const input = architecture[index];
      const activation = index === architecture.length - 2 ?
        objective === 'categorical' ? 'softmax' : objective === 'binary' ? 'sigmoid' : 'linear' : hiddenActivation;
      const deviation = hiddenActivation === 'relu' ? Math.sqrt(2 / input) : Math.sqrt(2 / (input + output));
      return {
        input, output, activation,
        weights: Array.from({ length: output }, () => Array.from({ length: input }, () => random.normal(0, deviation))),
        biases: zeros(output),
        weightGradient: matrix(output, input),
        biasGradient: zeros(output),
        firstWeights: matrix(output, input),
        secondWeights: matrix(output, input),
        firstBiases: zeros(output),
        secondBiases: zeros(output),
      };
    });
  }

  forward(input: number[], intervention?: InterventionSpec, training?: DenseTrainOptions): ForwardPass {
    if (!finiteVector(input, this.architecture[0])) throw new Error(`Expected ${this.architecture[0]} finite inputs.`);
    if (intervention && (!Number.isInteger(intervention.layer) || intervention.layer < 0 ||
      intervention.layer >= this.architecture.length || !Number.isInteger(intervention.neuron) ||
      intervention.neuron < 0 || intervention.neuron >= this.architecture[intervention.layer] ||
      !['ablate', 'scale', 'steer'].includes(intervention.kind) || !Number.isFinite(intervention.value))) {
      throw new Error('Invalid intervention location, kind, or value.');
    }
    const change = (values: number[], layer: number) => {
      if (!intervention || intervention.layer !== layer) return values;
      const result = [...values];
      const neuron = intervention.neuron;
      result[neuron] = intervention.kind === 'ablate' ? 0 : intervention.kind === 'scale' ?
        result[neuron] * intervention.value : result[neuron] + intervention.value;
      return result;
    };
    const activations: number[][] = [change([...input], 0)];
    const preactivations: number[][] = [];
    const dropoutMasks: number[][] = [];
    for (let index = 0; index < this.layers.length; index++) {
      const layer = this.layers[index];
      const previous = activations[index];
      const values = layer.weights.map((row, unit) => {
        let value = layer.biases[unit];
        for (let i = 0; i < layer.input; i++) value += row[i] * previous[i];
        return value;
      });
      preactivations.push(values);
      let output = activate(values, layer.activation);
      const dropout = training?.dropout ?? 0;
      const mask = output.map(() => dropout && index < this.layers.length - 1 ?
        (training!.random!.uniform() < dropout ? 0 : 1 / (1 - dropout)) : 1);
      dropoutMasks.push(mask);
      if (dropout) output = output.map((value, i) => value * mask[i]);
      activations.push(change(output, index + 1));
    }
    return { activations, preactivations, dropoutMasks };
  }

  predict(input: number[], intervention?: InterventionSpec): number[] {
    return this.forward(input, intervention).activations.at(-1)!;
  }

  loss(input: number[], target: number[]): number {
    return this.outputLoss(this.predict(input), target);
  }

  private outputLoss(output: number[], target: number[]): number {
    if (!finiteVector(target, output.length)) throw new Error('Target dimensions do not match the network.');
    if (this.objective === 'categorical') return output.reduce((total, value, i) =>
      total - target[i] * Math.log(Math.max(1e-12, value)), 0);
    if (this.objective === 'binary') return output.reduce((total, value, i) => total -
      (target[i] * Math.log(Math.max(1e-12, value)) +
      (1 - target[i]) * Math.log(Math.max(1e-12, 1 - value))), 0) / output.length;
    return output.reduce((total, value, i) => total + (value - target[i]) ** 2, 0) / output.length;
  }

  gradients(inputs: number[][], targets: number[][], training?: DenseTrainOptions): BatchResult {
    if (!inputs.length || inputs.length !== targets.length) throw new Error('A batch needs matching nonempty inputs and targets.');
    for (const layer of this.layers) {
      layer.weightGradient.forEach(row => row.fill(0));
      layer.biasGradient.fill(0);
    }
    let loss = 0;
    let correct = 0;
    for (let sample = 0; sample < inputs.length; sample++) {
      const pass = this.forward(inputs[sample], undefined, training);
      const output = pass.activations.at(-1)!;
      const target = targets[sample];
      loss += this.outputLoss(output, target);
      if (this.objective === 'categorical') correct += Number(argmax(output) === argmax(target));
      else if (this.objective === 'binary') correct += Number(output.every((value, i) => (value >= 0.5) === (target[i] >= 0.5)));
      let delta = output.map((value, i) => this.objective === 'mse' ?
        2 * (value - target[i]) / output.length :
        (value - target[i]) / (this.objective === 'binary' ? output.length : 1));
      for (let index = this.layers.length - 1; index >= 0; index--) {
        const layer = this.layers[index];
        const previous = pass.activations[index];
        for (let unit = 0; unit < layer.output; unit++) {
          layer.biasGradient[unit] += delta[unit] / inputs.length;
          for (let i = 0; i < layer.input; i++) {
            layer.weightGradient[unit][i] += delta[unit] * previous[i] / inputs.length;
          }
        }
        if (index > 0) {
          const hiddenLayer = this.layers[index - 1];
          const nextDelta = zeros(layer.input);
          for (let i = 0; i < layer.input; i++) {
            for (let unit = 0; unit < layer.output; unit++) nextDelta[i] += layer.weights[unit][i] * delta[unit];
            const preactivation = pass.preactivations[index - 1][i];
            const unmasked = hiddenLayer.activation === 'tanh' ? Math.tanh(preactivation) :
              hiddenLayer.activation === 'sigmoid' ? sigmoid(preactivation) : previous[i];
            nextDelta[i] *= derivative(unmasked, preactivation, hiddenLayer.activation) * pass.dropoutMasks[index - 1][i];
          }
          delta = nextDelta;
        }
      }
    }
    let normSquared = 0;
    for (const layer of this.layers) {
      for (const row of layer.weightGradient) for (const value of row) normSquared += value * value;
      for (const value of layer.biasGradient) normSquared += value * value;
    }
    return { loss: loss / inputs.length, accuracy: correct / inputs.length, gradientNorm: Math.sqrt(normSquared) };
  }

  trainBatch(inputs: number[][], targets: number[][], learningRate: number, gradientClip = 5, options: DenseTrainOptions = {}): BatchResult {
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) throw new Error('Learning rate must lie in (0, 1].');
    if (!Number.isFinite(options.dropout ?? 0) || (options.dropout ?? 0) < 0 || (options.dropout ?? 0) >= 1 ||
      ((options.dropout ?? 0) > 0 && !options.random)) throw new Error('Dropout needs a probability in [0, 1) and a seeded random stream.');
    const result = this.gradients(inputs, targets, options);
    if (!Number.isFinite(result.loss) || !Number.isFinite(result.gradientNorm)) throw new Error('Training diverged; lower the learning rate.');
    const threshold = options.gradientClip ?? gradientClip;
    const scale = result.gradientNorm > threshold ? threshold / result.gradientNorm : 1;
    this.updates++;
    const update = (parameter: number, gradient: number, first: number, second: number, decay: boolean): [number, number, number] => {
      const next = updateScalar(parameter, gradient * scale, first, second, this.updates, learningRate, options, decay);
      return [next.value, next.first, next.second];
    };
    for (const layer of this.layers) {
      for (let unit = 0; unit < layer.output; unit++) {
        [layer.biases[unit], layer.firstBiases[unit], layer.secondBiases[unit]] = update(
          layer.biases[unit], layer.biasGradient[unit], layer.firstBiases[unit], layer.secondBiases[unit], false);
        for (let i = 0; i < layer.input; i++) {
          [layer.weights[unit][i], layer.firstWeights[unit][i], layer.secondWeights[unit][i]] = update(
            layer.weights[unit][i], layer.weightGradient[unit][i], layer.firstWeights[unit][i], layer.secondWeights[unit][i], true);
        }
      }
    }
    return result;
  }

  evaluate(inputs: number[][], targets: number[][]): BatchResult {
    let loss = 0;
    let correct = 0;
    let squaredError = 0;
    const targetMeans = targets[0].map((_, i) => targets.reduce((total, row) => total + row[i], 0) / targets.length);
    let variance = 0;
    for (let i = 0; i < inputs.length; i++) {
      const output = this.predict(inputs[i]);
      loss += this.outputLoss(output, targets[i]);
      if (this.objective === 'categorical') correct += Number(argmax(output) === argmax(targets[i]));
      else if (this.objective === 'binary') correct += Number(output.every((value, j) => (value >= 0.5) === (targets[i][j] >= 0.5)));
      else for (let j = 0; j < output.length; j++) {
        squaredError += (output[j] - targets[i][j]) ** 2;
        variance += (targets[i][j] - targetMeans[j]) ** 2;
      }
    }
    return { loss: loss / inputs.length,
      accuracy: this.objective === 'mse' ? clamp(1 - squaredError / Math.max(1e-12, variance), 0, 1) : correct / inputs.length,
      gradientNorm: Math.sqrt(this.layers.reduce((total, layer) => total +
        layer.weightGradient.flat().reduce((subtotal, value) => subtotal + value * value, 0) +
        layer.biasGradient.reduce((subtotal, value) => subtotal + value * value, 0), 0)),
    };
  }

  capture(input: number[]): LayerData[] {
    const pass = this.forward(input);
    return this.architecture.map((size, index) => ({
      id: `layer-${index}`,
      name: index === 0 ? 'Input' : index === this.architecture.length - 1 ? 'Readout' : `Dense ${index}`,
      size,
      shape: index ? [size, this.architecture[index - 1]] : [size],
      activations: [...pass.activations[index]],
      weights: index ? this.layers[index - 1].weights.map(row => [...row]) : [],
      gradients: index ? [...this.layers[index - 1].biasGradient] : zeros(size),
      biases: index ? [...this.layers[index - 1].biases] : [],
    }));
  }

  get parameters(): number {
    return this.layers.reduce((total, layer) => total + layer.output * (layer.input + 1), 0);
  }

  exportCheckpoint(): NetworkCheckpoint {
    return JSON.parse(JSON.stringify({ architecture: this.architecture, objective: this.objective,
      updates: this.updates, layers: this.layers })) as NetworkCheckpoint;
  }

  importCheckpoint(checkpoint: NetworkCheckpoint): void {
    if (!checkpoint || JSON.stringify(checkpoint.architecture) !== JSON.stringify(this.architecture) ||
      checkpoint.objective !== this.objective || !Number.isInteger(checkpoint.updates) || checkpoint.updates < 0 ||
      !Array.isArray(checkpoint.layers) || checkpoint.layers.length !== this.layers.length) {
      throw new Error('Checkpoint architecture or objective does not match this demo.');
    }
    checkpoint.layers.forEach((saved, index) => {
      const layer = this.layers[index];
      if (saved.input !== layer.input || saved.output !== layer.output || saved.activation !== layer.activation ||
        !finiteMatrix(saved.weights, layer.output, layer.input) ||
        !finiteMatrix(saved.weightGradient, layer.output, layer.input) ||
        !finiteMatrix(saved.firstWeights, layer.output, layer.input) ||
        !finiteMatrix(saved.secondWeights, layer.output, layer.input) ||
        saved.secondWeights.some(row => row.some(value => value < 0)) ||
        !finiteVector(saved.biases, layer.output) || !finiteVector(saved.biasGradient, layer.output) ||
        !finiteVector(saved.firstBiases, layer.output) || !finiteVector(saved.secondBiases, layer.output) ||
        saved.secondBiases.some(value => value < 0)) throw new Error(`Checkpoint contains invalid dense layer ${index}.`);
    });
    this.updates = checkpoint.updates;
    checkpoint.layers.forEach((saved, index) => Object.assign(this.layers[index], JSON.parse(JSON.stringify(saved))));
  }
}
