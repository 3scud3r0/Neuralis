import type { DemoDefinition, InterventionSpec, LayerData, PredictionSample, RunConfig } from '../types';
import { createDataset, type Dataset } from './datasets';
import { Affine, applyDropout } from './layers';
import { clamp, finiteVector, sigmoid, zeros } from './math';
import type { ResearchModel } from './model';
import type { BatchResult } from './network';
import { ParameterBank, type ParameterBankCheckpoint } from './optimizers';
import { Random } from './random';

const HIDDEN = 8;

interface GRUStepCache {
  input: number;
  previous: number[];
  update: number[];
  reset: number[];
  candidate: number[];
  hidden: number[];
}

interface GRUPass {
  steps: GRUStepCache[];
  hiddenDropped: number[];
  dropoutMask: number[];
  output: number[];
}

interface GRUCheckpoint {
  bank: ParameterBankCheckpoint;
  random: ReturnType<Random['exportState']>;
  activationGradients: number[][];
}

/** A genuine recurrent gated cell, unrolled over eight scalar observations. */
export class GRUModel implements ResearchModel {
  readonly bank = new ParameterBank();
  readonly updateGate: Affine;
  readonly resetGate: Affine;
  readonly candidate: Affine;
  readonly readout: Affine;
  readonly parameters: number;
  readonly trainingSize: number;
  private readonly config: RunConfig;
  private readonly data: Dataset;
  private readonly random: Random;
  private activationGradients = [zeros(8), zeros(8), zeros(8), zeros(8), zeros(1)];

  constructor(demo: DemoDefinition, config: RunConfig) {
    this.config = { ...config };
    this.random = new Random(config.seed);
    this.data = createDataset({ ...demo, id: 'forecasting' }, config.seed);
    this.updateGate = new Affine('gru.update', 1 + HIDDEN, HIDDEN, this.bank, this.random);
    this.resetGate = new Affine('gru.reset', 1 + HIDDEN, HIDDEN, this.bank, this.random);
    this.candidate = new Affine('gru.candidate', 1 + HIDDEN, HIDDEN, this.bank, this.random);
    this.readout = new Affine('readout', HIDDEN, 1, this.bank, this.random);
    this.parameters = this.bank.size;
    this.trainingSize = this.data.train.length;
  }

  private modify(input: number[], layer: number, intervention?: InterventionSpec): number[] {
    if (!intervention || intervention.layer !== layer) return input;
    const result = [...input];
    const i = intervention.neuron;
    result[i] = intervention.kind === 'ablate' ? 0 : intervention.kind === 'scale' ? result[i] * intervention.value : result[i] + intervention.value;
    return result;
  }

  private forward(input: number[], training = false, intervention?: InterventionSpec): GRUPass {
    if (!finiteVector(input, 8)) throw new Error('GRU expects eight finite scalar time-series values.');
    const sizes = [8, 8, 8, 8, 1];
    if (intervention && (!Number.isInteger(intervention.layer) || intervention.layer < 0 || intervention.layer >= sizes.length ||
      !Number.isInteger(intervention.neuron) || intervention.neuron < 0 || intervention.neuron >= sizes[intervention.layer] ||
      !Number.isFinite(intervention.value) || !['ablate', 'scale', 'steer'].includes(intervention.kind))) throw new Error('Invalid GRU intervention.');
    const sequence = this.modify([...input], 0, intervention);
    let hidden = zeros(HIDDEN);
    const steps: GRUStepCache[] = [];
    for (let time = 0; time < sequence.length; time++) {
      const previous = hidden;
      const joined = [sequence[time], ...previous];
      let update = this.updateGate.forward(joined).map(sigmoid);
      let reset = this.resetGate.forward(joined).map(sigmoid);
      if (time === sequence.length - 1) {
        update = this.modify(update, 1, intervention);
        reset = this.modify(reset, 2, intervention);
      }
      const candidateInput = [sequence[time], ...previous.map((value, i) => value * reset[i])];
      const candidate = this.candidate.forward(candidateInput).map(Math.tanh);
      hidden = candidate.map((value, i) => (1 - update[i]) * value + update[i] * previous[i]);
      if (time === sequence.length - 1) hidden = this.modify(hidden, 3, intervention);
      steps.push({ input: sequence[time], previous, update, reset, candidate, hidden });
    }
    const dropout = applyDropout(hidden, training ? this.config.dropout ?? 0 : 0, this.random);
    const output = this.modify(this.readout.forward(dropout.output), 4, intervention);
    return { steps, hiddenDropped: dropout.output, dropoutMask: dropout.mask, output };
  }

  predict(input: number[], intervention?: InterventionSpec): number[] {
    return this.forward(input, false, intervention).output;
  }

  sampleLoss(input: number[], target: number[]): number {
    return (this.predict(input)[0] - target[0]) ** 2;
  }

  analyticGradients(inputs: number[][], targets: number[][], training = false): BatchResult {
    if (!inputs.length || inputs.length !== targets.length || targets.some(row => !finiteVector(row, 1))) throw new Error('Invalid GRU training batch.');
    this.bank.zeroGradients();
    let loss = 0;
    for (let sample = 0; sample < inputs.length; sample++) {
      const pass = this.forward(inputs[sample], training);
      const residual = pass.output[0] - targets[sample][0];
      loss += residual * residual;
      const outputGradient = [2 * residual];
      let hiddenGradient = this.readout.backward(pass.hiddenDropped, outputGradient).map((value, i) => value * pass.dropoutMask[i]);
      const finalHiddenGradient = [...hiddenGradient];
      const inputGradients = zeros(inputs[sample].length);
      let finalUpdateGradient = zeros(HIDDEN);
      let finalResetGradient = zeros(HIDDEN);
      for (let time = pass.steps.length - 1; time >= 0; time--) {
        const cache = pass.steps[time];
        const candidateGradient = hiddenGradient.map((value, i) => value * (1 - cache.update[i]) * (1 - cache.candidate[i] ** 2));
        const updateGradient = hiddenGradient.map((value, i) => value * (cache.previous[i] - cache.candidate[i]) * cache.update[i] * (1 - cache.update[i]));
        const previousGradient = hiddenGradient.map((value, i) => value * cache.update[i]);
        const candidateInput = [cache.input, ...cache.previous.map((value, i) => value * cache.reset[i])];
        const candidateInputGradient = this.candidate.backward(candidateInput, candidateGradient);
        const resetGradient = cache.reset.map((value, i) => candidateInputGradient[i + 1] * cache.previous[i] * value * (1 - value));
        for (let i = 0; i < HIDDEN; i++) previousGradient[i] += candidateInputGradient[i + 1] * cache.reset[i];
        const joined = [cache.input, ...cache.previous];
        const updateInputGradient = this.updateGate.backward(joined, updateGradient);
        const resetInputGradient = this.resetGate.backward(joined, resetGradient);
        inputGradients[time] = candidateInputGradient[0] + updateInputGradient[0] + resetInputGradient[0];
        hiddenGradient = previousGradient.map((value, i) => value + updateInputGradient[i + 1] + resetInputGradient[i + 1]);
        if (time === pass.steps.length - 1) {
          finalUpdateGradient = updateGradient;
          finalResetGradient = resetGradient;
        }
      }
      this.activationGradients = [inputGradients, finalUpdateGradient, finalResetGradient, finalHiddenGradient, outputGradient];
    }
    this.bank.scaleGradients(1 / inputs.length);
    return { loss: loss / inputs.length, accuracy: 0, gradientNorm: this.bank.gradientNorm };
  }

  train(learningRate: number, batchSize: number): BatchResult {
    const batch = Array.from({ length: batchSize }, () => this.data.train[this.random.integer(this.data.train.length)]);
    const result = this.analyticGradients(batch.map(row => row.input), batch.map(row => row.target), true);
    this.bank.update(learningRate, { kind: this.config.optimizer, weightDecay: this.config.weightDecay, gradientClip: this.config.gradientClip });
    return result;
  }

  evaluate(): BatchResult {
    const targetMean = this.data.validation.reduce((total, row) => total + row.target[0], 0) / this.data.validation.length;
    let error = 0;
    let variance = 0;
    for (const row of this.data.validation) {
      error += this.sampleLoss(row.input, row.target);
      variance += (row.target[0] - targetMean) ** 2;
    }
    return { loss: error / this.data.validation.length, accuracy: clamp(1 - error / Math.max(1e-12, variance), 0, 1), gradientNorm: this.bank.gradientNorm };
  }

  capture(): LayerData[] {
    const input = this.data.validation[0].input;
    const pass = this.forward(input);
    const last = pass.steps.at(-1)!;
    const activations = [input, last.update, last.reset, last.hidden, pass.output];
    const names = ['Eight scalar observations', 'Final-time update gate', 'Final-time reset gate', 'Final recurrent state', 'Next signal value'];
    const weights = [[], this.updateGate.weight.rows(), this.resetGate.weight.rows(), this.candidate.weight.rows(), this.readout.weight.rows()];
    const biases = [[], [...this.updateGate.bias.values], [...this.resetGate.bias.values], [...this.candidate.bias.values], [...this.readout.bias.values]];
    return activations.map((values, i) => ({ id: `gru-${i}`, name: names[i], size: values.length,
      shape: i ? [values.length, i === 4 ? HIDDEN : 1 + HIDDEN] : [8, 1],
      activations: [...values], weights: weights[i], gradients: [...this.activationGradients[i]], biases: biases[i] }));
  }

  samples(): PredictionSample[] {
    return this.data.validation.slice(0, 8).map(row => ({ input: [...row.input], target: [...row.target], prediction: this.predict(row.input), label: 'GRU one-step forecast' }));
  }

  extras(): Record<string, unknown> {
    const pass = this.forward(this.data.validation[0].input);
    return { ...this.data.extras, algorithm: 'Eight-step GRU with analytic backpropagation through time',
      architectureNote: 'Actual shared gated recurrence: update, reset, tanh candidate, recurrent state, linear readout. Readout-only dropout.',
      hiddenSize: HIDDEN, sequenceLength: 8, hiddenTrajectory: pass.steps.map(step => [...step.hidden]),
      gateTrajectory: { update: pass.steps.map(step => [...step.update]), reset: pass.steps.map(step => [...step.reset]), candidate: pass.steps.map(step => [...step.candidate]) },
      visualizationNote: 'Layer summaries show the final time step. Gate matrices operate on [scalar observation, previous hidden state], not on the entire input window.',
      weightGradients: this.bank.parameters.map(parameter => ({ name: parameter.name, shape: [...parameter.shape], values: [...parameter.gradients] })),
      customArchitectureNote: 'The recurrent state has eight units. Dense hiddenLayers overrides are ignored.',
      dropout: this.config.dropout ?? 0,
    };
  }

  checkpoint(): GRUCheckpoint {
    return { bank: this.bank.checkpoint(), random: this.random.exportState(), activationGradients: this.activationGradients.map(row => [...row]) };
  }

  restore(value: unknown): void {
    const saved = value as GRUCheckpoint;
    const sizes = [8, 8, 8, 8, 1];
    if (!saved || !Array.isArray(saved.activationGradients) || saved.activationGradients.length !== sizes.length ||
      saved.activationGradients.some((row, i) => !finiteVector(row, sizes[i]))) throw new Error('Invalid GRU checkpoint.');
    const random = new Random(0);
    random.importState(saved.random);
    this.bank.restore(saved.bank);
    this.random.importState(saved.random);
    this.activationGradients = saved.activationGradients.map(row => [...row]);
  }
}
