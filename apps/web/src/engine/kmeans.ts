import type { DemoDefinition, InterventionSpec, LayerData, PredictionSample, RunConfig } from '../types';
import { createDataset, type Dataset } from './datasets';
import { argmax, finiteMatrix, matrix, softmax, squaredDistance, zeros } from './math';
import type { ResearchModel } from './model';
import type { BatchResult } from './network';
import { Random } from './random';

interface KMeansCheckpoint { centroids: number[][]; gradients: number[][]; iterations: number; }

function permutations(values: number[]): number[][] {
  if (!values.length) return [[]];
  return values.flatMap((value, i) => permutations(values.filter((_, j) => i !== j)).map(rest => [value, ...rest]));
}

const LABEL_PERMUTATIONS = permutations([0, 1, 2, 3]);

/** Four-centroid Lloyd clustering initialized with the k-means++ distribution. */
export class KMeansModel implements ResearchModel {
  readonly parameters = 8;
  readonly trainingSize: number;
  private readonly data: Dataset;
  private centroids: number[][];
  private gradients: number[][] = matrix(4, 2);
  private iterations = 0;

  constructor(demo: DemoDefinition, config: RunConfig) {
    this.data = createDataset(demo, config.seed);
    this.trainingSize = this.data.train.length;
    const random = new Random(config.seed);
    const points = this.data.train.map(row => row.input);
    this.centroids = [];
    let bestInertia = Infinity;
    // A bounded set of independent k-means++ starts avoids a fragile single seed.
    for (let restart = 0; restart < 8; restart++) {
      const candidate = [[...random.choice(points)]];
      while (candidate.length < 4) {
        const distances = points.map(point => Math.min(...candidate.map(center => squaredDistance(point, center))));
        const total = distances.reduce((sum, value) => sum + value, 0);
        let threshold = random.uniform() * total;
        let index = points.length - 1;
        for (let i = 0; i < points.length; i++) {
          threshold -= distances[i];
          if (threshold <= 0) { index = i; break; }
        }
        candidate.push([...points[index]]);
      }
      const inertia = points.reduce((total, point) => total + Math.min(...candidate.map(center => squaredDistance(point, center))), 0);
      if (inertia < bestInertia) { this.centroids = candidate; bestInertia = inertia; }
    }
  }

  private cluster(input: number[]): number {
    const distances = this.centroids.map(center => squaredDistance(input, center));
    return argmax(distances.map(value => -value));
  }

  train(_learningRate: number, _batchSize: number): BatchResult {
    const counts = zeros(4);
    const sums = matrix(4, 2);
    this.gradients = matrix(4, 2);
    for (const row of this.data.train) {
      const cluster = this.cluster(row.input);
      counts[cluster]++;
      for (let dimension = 0; dimension < 2; dimension++) {
        sums[cluster][dimension] += row.input[dimension];
        this.gradients[cluster][dimension] += 2 * (this.centroids[cluster][dimension] - row.input[dimension]) / this.data.train.length;
      }
    }
    this.centroids = this.centroids.map((center, i) => counts[i] ? sums[i].map(value => value / counts[i]) : center);
    this.iterations++;
    return this.evaluate();
  }

  predict(input: number[], intervention?: InterventionSpec): number[] {
    if (input.length !== 2 || input.some(value => !Number.isFinite(value))) throw new Error('K-means expects two finite coordinates.');
    if (intervention) throw new Error('K-means uses centroids rather than intervenable neural activations.');
    return softmax(this.centroids.map(center => -squaredDistance(input, center) / 0.12));
  }

  evaluate(): BatchResult {
    const assignments = this.data.validation.map(row => this.cluster(row.input));
    const inertia = this.data.validation.reduce((total, row, i) => total + squaredDistance(row.input, this.centroids[assignments[i]]), 0);
    const accuracy = Math.max(...LABEL_PERMUTATIONS.map(permutation => this.data.validation.reduce((total, row, i) =>
      total + Number(permutation[assignments[i]] === argmax(row.target)), 0))) / this.data.validation.length;
    return { loss: inertia / this.data.validation.length, accuracy,
      gradientNorm: Math.sqrt(this.gradients.flat().reduce((total, value) => total + value * value, 0)) };
  }

  capture(): LayerData[] {
    const input = this.data.validation[0].input;
    return [
      { id: 'cluster-input', name: 'Point coordinates', size: 2, shape: [2], activations: [...input], weights: [], gradients: zeros(2), biases: [] },
      { id: 'cluster-centroids', name: 'Centroid affinities', size: 4, shape: [4, 2], activations: this.predict(input),
        weights: this.centroids.map(center => [...center]), gradients: this.gradients.map(row => Math.sqrt(row[0] ** 2 + row[1] ** 2)), biases: zeros(4) },
    ];
  }

  samples(): PredictionSample[] {
    const assignments = this.data.validation.map(row => this.cluster(row.input));
    const permutation = LABEL_PERMUTATIONS.reduce((best, candidate) => {
      const score = (mapping: number[]) => this.data.validation.reduce((total, row, i) => total + Number(mapping[assignments[i]] === argmax(row.target)), 0);
      return score(candidate) > score(best) ? candidate : best;
    });
    return this.data.validation.slice(0, 8).map(row => {
      const clusterProbabilities = this.predict(row.input);
      const prediction = zeros(4);
      permutation.forEach((label, cluster) => { prediction[label] = clusterProbabilities[cluster]; });
      return { input: [...row.input], target: [...row.target], prediction, label: row.label };
    });
  }

  extras(): Record<string, unknown> {
    return { algorithm: 'Best of eight k-means++ starts followed by full-batch Lloyd updates', scoreLabel: 'Permutation-matched cluster accuracy',
      lossLabel: 'Validation mean squared distance to nearest centroid', iterations: this.iterations,
      centroids: this.centroids.map(row => [...row]), centroidGradients: this.gradients.map(row => [...row]),
      learningRateNote: 'Lloyd centroid updates ignore learning rate and minibatch size.',
      architectureNote: 'Centroid affinities are a visualization of distances, not neural hidden units or calibrated class probabilities.',
      embedding: this.data.validation.map(row => ({ input: [...row.input], cluster: this.cluster(row.input), label: argmax(row.target) })),
    };
  }

  checkpoint(): KMeansCheckpoint {
    return { centroids: this.centroids.map(row => [...row]), gradients: this.gradients.map(row => [...row]), iterations: this.iterations };
  }

  restore(value: unknown): void {
    const saved = value as KMeansCheckpoint;
    if (!saved || !finiteMatrix(saved.centroids, 4, 2) || !finiteMatrix(saved.gradients, 4, 2) ||
      !Number.isInteger(saved.iterations) || saved.iterations < 0) throw new Error('Invalid k-means checkpoint.');
    this.centroids = saved.centroids.map(row => [...row]);
    this.gradients = saved.gradients.map(row => [...row]);
    this.iterations = saved.iterations;
  }
}
