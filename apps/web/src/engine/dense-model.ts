import type { DemoDefinition, InterventionSpec, LayerData, PredictionSample, RunConfig } from '../types';
import { createCustomDataset, createDataset, type Dataset } from './datasets';
import type { ResearchModel } from './model';
import { DenseNetwork, type BatchResult, type NetworkCheckpoint } from './network';
import { Random } from './random';

export class DenseModel implements ResearchModel {
  readonly network: DenseNetwork;
  readonly data: Dataset;
  readonly random: Random;
  readonly parameters: number;
  readonly trainingSize: number;
  private readonly demoId: string;
  private readonly config: RunConfig;
  private readonly previewIndices: number[];
  private recentBatch: BatchResult = { loss: 0, accuracy: 0, gradientNorm: 0 };

  constructor(demo: DemoDefinition, config: RunConfig) {
    this.demoId = demo.id;
    this.config = { ...config };
    this.data = demo.id === 'custom-data' ? createCustomDataset(config.customDataset!, config.seed) : createDataset(demo, config.seed);
    this.random = new Random(config.seed);
    const architecture = config.hiddenLayers ?
      [this.data.inputSize, ...config.hiddenLayers, this.data.outputSize] : demo.id === 'custom-data' ?
      [this.data.inputSize, 24, 16, this.data.outputSize] : demo.architecture;
    if (architecture[0] !== this.data.inputSize || architecture.at(-1) !== this.data.outputSize) {
      throw new Error(`Architecture does not match the ${demo.id} dataset.`);
    }
    this.network = new DenseNetwork(architecture, this.data.objective, this.random);
    this.parameters = this.network.parameters;
    this.trainingSize = this.data.train.length;
    this.previewIndices = Array.from({ length: 8 }, (_, i) => Math.floor(i * this.data.validation.length / 8));
  }

  train(learningRate: number, batchSize: number): BatchResult {
    const batch = Array.from({ length: batchSize }, () => this.data.train[this.random.integer(this.data.train.length)]);
    this.recentBatch = this.network.trainBatch(batch.map(row => row.input), batch.map(row => row.target), learningRate,
      this.config.gradientClip ?? 5, { kind: this.config.optimizer, weightDecay: this.config.weightDecay,
        dropout: this.config.dropout, gradientClip: this.config.gradientClip, random: this.random });
    return this.recentBatch;
  }

  evaluate(): BatchResult {
    return this.network.evaluate(this.data.validation.map(row => row.input), this.data.validation.map(row => row.target));
  }

  predict(input: number[], intervention?: InterventionSpec): number[] {
    return this.network.predict(input, intervention);
  }

  capture(): LayerData[] {
    return this.network.capture(this.data.validation[0].input);
  }

  samples(): PredictionSample[] {
    return this.previewIndices.map(index => {
      const row = this.data.validation[index];
      return { input: [...row.input], target: [...row.target], prediction: this.predict(row.input), label: row.label };
    });
  }

  extras(): Record<string, unknown> {
    const sample = this.data.validation[0];
    const pass = this.network.forward(sample.input);
    const bottleneck = this.network.architecture.indexOf(Math.min(...this.network.architecture.slice(1, -1)), 1);
    return {
      ...this.data.extras,
      objective: this.data.objective,
      optimizer: this.config.optimizer ?? 'adam', dropout: this.config.dropout ?? 0, weightDecay: this.config.weightDecay ?? 0,
      trainingLoss: this.recentBatch.loss,
      batchAccuracy: this.recentBatch.accuracy,
      gradientNorm: this.recentBatch.gradientNorm,
      weightGradients: this.network.layers.map(layer => layer.weightGradient.map(row => [...row])),
      preactivations: pass.preactivations.map(row => [...row]),
      attention: sample.attention?.map(row => [...row]),
      tokens: sample.tokens ? [...sample.tokens] : undefined,
      attentionKind: sample.attention ? 'Fixed scaled dot-product pooling; actual probabilities' : undefined,
      latent: ['autoencoder', 'denoising'].includes(this.demoId) && bottleneck > 0 ?
        this.previewIndices.map(index => this.network.forward(this.data.validation[index].input).activations[bottleneck]) : undefined,
      trainingPreview: this.data.train.slice(0, 64).map(row => ({ input: [...row.input], target: [...row.target] })),
      validationPreview: this.data.validation.slice(0, 128).map(row => ({ input: [...row.input], target: [...row.target], prediction: this.predict(row.input) })),
    };
  }

  checkpoint(): unknown {
    return { network: this.network.exportCheckpoint(), random: this.random.exportState(), recentBatch: { ...this.recentBatch } };
  }

  restore(value: unknown): void {
    const checkpoint = value as { network: NetworkCheckpoint; random: ReturnType<Random['exportState']>; recentBatch: BatchResult };
    if (!checkpoint || !checkpoint.recentBatch ||
      !Number.isFinite(checkpoint.recentBatch.loss) || !Number.isFinite(checkpoint.recentBatch.accuracy) ||
      !Number.isFinite(checkpoint.recentBatch.gradientNorm)) throw new Error('Invalid dense model checkpoint.');
    const random = new Random(0);
    random.importState(checkpoint.random);
    this.network.importCheckpoint(checkpoint.network);
    this.random.importState(checkpoint.random);
    this.recentBatch = { ...checkpoint.recentBatch };
  }
}
