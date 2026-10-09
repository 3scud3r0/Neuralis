import type { DemoDefinition, InterventionSpec, LayerData, PredictionSample, RunConfig } from '../types';
import { createDataset, type Dataset } from './datasets';
import { clamp, covariance, dot, finiteMatrix, matrix, multiplyVector, norm, normalize, squaredDistance, zeros } from './math';
import type { ResearchModel } from './model';
import type { BatchResult } from './network';
import { Random } from './random';

interface PCACheckpoint {
  axes: number[][];
  displacement: number[][];
  iterations: number;
}

/** Orthogonal block power iteration on the centered sample covariance. */
export class PCAModel implements ResearchModel {
  readonly parameters = 6;
  readonly trainingSize: number;
  private readonly data: Dataset;
  private readonly center: number[];
  private readonly covariance: number[][];
  private axes: number[][];
  private displacement: number[][] = matrix(2, 3);
  private iterations = 0;

  constructor(demo: DemoDefinition, config: RunConfig) {
    this.data = createDataset(demo, config.seed);
    this.trainingSize = this.data.train.length;
    const result = covariance(this.data.train.map(row => row.input));
    this.center = result.center;
    this.covariance = result.covariance;
    const random = new Random(config.seed);
    const first = normalize(Array.from({ length: 3 }, () => random.normal()));
    const second = Array.from({ length: 3 }, () => random.normal());
    const projection = dot(first, second);
    this.axes = [first, normalize(second.map((value, i) => value - projection * first[i]))];
  }

  train(_learningRate: number, _batchSize: number): BatchResult {
    const first = normalize(multiplyVector(this.covariance, this.axes[0]));
    const secondCandidate = multiplyVector(this.covariance, this.axes[1]);
    const projection = dot(first, secondCandidate);
    const second = normalize(secondCandidate.map((value, i) => value - projection * first[i]));
    const next = [first, second];
    this.displacement = next.map((axis, i) => axis.map((value, j) => value - this.axes[i][j]));
    this.axes = next;
    this.iterations++;
    return this.evaluate();
  }

  private project(input: number[]): number[] {
    const centered = input.map((value, i) => value - this.center[i]);
    return this.axes.map(axis => dot(axis, centered));
  }

  predict(input: number[], intervention?: InterventionSpec): number[] {
    if (input.length !== 3 || input.some(value => !Number.isFinite(value))) throw new Error('PCA expects three finite coordinates.');
    const latent = this.project(input);
    if (intervention) {
      if (intervention.layer !== 1 || ![0, 1].includes(intervention.neuron) ||
        !['ablate', 'scale', 'steer'].includes(intervention.kind) || !Number.isFinite(intervention.value)) {
        throw new Error('PCA interventions target one of the two principal components in layer 1.');
      }
      const i = intervention.neuron;
      latent[i] = intervention.kind === 'ablate' ? 0 : intervention.kind === 'scale' ? latent[i] * intervention.value : latent[i] + intervention.value;
    }
    return this.center.map((value, i) => value + latent.reduce((total, coefficient, j) => total + coefficient * this.axes[j][i], 0));
  }

  evaluate(): BatchResult {
    const residual = this.data.validation.reduce((total, row) => total + squaredDistance(row.input, this.predict(row.input)), 0);
    const variance = this.data.validation.reduce((total, row) => total + squaredDistance(row.input, this.center), 0);
    return { loss: residual / (this.data.validation.length * 3), accuracy: clamp(1 - residual / variance, 0, 1),
      gradientNorm: norm(this.displacement.flat()) };
  }

  capture(): LayerData[] {
    const input = this.data.validation[0].input;
    const encoded = this.project(input);
    return [
      { id: 'pca-input', name: 'Centered coordinates', size: 3, shape: [3], activations: input.map((value, i) => value - this.center[i]),
        weights: [], gradients: zeros(3), biases: this.center.map(value => -value) },
      { id: 'pca-projection', name: 'Principal axes', size: 2, shape: [2, 3], activations: encoded,
        weights: this.axes.map(axis => [...axis]), gradients: zeros(2), biases: zeros(2) },
      { id: 'pca-reconstruction', name: 'Linear reconstruction', size: 3, shape: [3, 2], activations: this.predict(input),
        weights: Array.from({ length: 3 }, (_, i) => this.axes.map(axis => axis[i])), gradients: zeros(3), biases: [...this.center] },
    ];
  }

  samples(): PredictionSample[] {
    return this.data.validation.slice(0, 8).map(row => ({ input: [...row.input], target: [...row.target], prediction: this.predict(row.input), label: row.label }));
  }

  extras(): Record<string, unknown> {
    const eigenvalues = this.axes.map(axis => dot(axis, multiplyVector(this.covariance, axis)));
    const totalVariance = this.covariance.reduce((total, row, i) => total + row[i], 0);
    return { algorithm: 'Orthogonal power iteration', scoreLabel: 'Explained variance',
      gradientNote: 'This eigensolver has no backpropagated gradient. Node gradients are zero; axis displacements are shown separately.',
      axes: this.axes.map(axis => [...axis]), axisDisplacement: this.displacement.map(row => [...row]),
      covariance: this.covariance.map(row => [...row]), eigenvalues,
      explainedVarianceRatio: eigenvalues.map(value => value / totalVariance),
      orthogonalityError: Math.abs(dot(this.axes[0], this.axes[1])), iterations: this.iterations,
      learningRateNote: 'Power iteration ignores learning rate and minibatch size.',
      embedding: this.data.validation.map(row => ({ original: [...row.input], projected: this.project(row.input) })),
    };
  }

  checkpoint(): PCACheckpoint {
    return { axes: this.axes.map(axis => [...axis]), displacement: this.displacement.map(row => [...row]), iterations: this.iterations };
  }

  restore(value: unknown): void {
    const saved = value as PCACheckpoint;
    if (!saved || !finiteMatrix(saved.axes, 2, 3) || !finiteMatrix(saved.displacement, 2, 3) ||
      !Number.isInteger(saved.iterations) || saved.iterations < 0 ||
      saved.axes.some(axis => Math.abs(norm(axis) - 1) > 1e-5) ||
      Math.abs(dot(saved.axes[0], saved.axes[1])) > 1e-5) throw new Error('Invalid PCA checkpoint.');
    this.axes = saved.axes.map(axis => [...axis]);
    this.displacement = saved.displacement.map(row => [...row]);
    this.iterations = saved.iterations;
  }
}
