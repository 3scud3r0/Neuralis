import { describe, expect, it } from 'vitest';
import type { RunConfig } from '../../types';
import { getDemo } from '../catalog';
import { attentionPool } from '../datasets';
import { GridworldModel, gridTransition, GRID } from '../gridworld';
import { KMeansModel } from '../kmeans';
import { PCAModel } from '../pca';
import { dot } from '../math';
import { Random } from '../random';

const config: RunConfig = { demoId: 'xor', learningRate: 0.1, maxSteps: 1000, batchSize: 32, seed: 42, captureEvery: 10 };

describe('specialized numerical algorithms', () => {
  it('orthogonal PCA axes converge to low reconstruction error', () => {
    const model = new PCAModel(getDemo('pca'), config);
    const initial = model.evaluate().loss;
    for (let i = 0; i < 12; i++) model.train(0.1, 32);
    const axes = model.extras().axes as number[][];
    expect(Math.abs(dot(axes[0], axes[1]))).toBeLessThan(1e-10);
    expect(model.evaluate().loss).toBeLessThan(initial * 0.05);
    expect(model.evaluate().accuracy).toBeGreaterThan(0.99);
  });

  it('Lloyd updates decrease actual distance and recover four clusters', () => {
    const model = new KMeansModel(getDemo('k-means'), config);
    const initial = model.evaluate().loss;
    for (let i = 0; i < 8; i++) model.train(0.1, 32);
    expect(model.evaluate().loss).toBeLessThan(initial);
    expect(model.evaluate().accuracy).toBeGreaterThan(0.95);
  });

  it('Q-learning learns a goal-reaching route using environment transitions', () => {
    const model = new GridworldModel(config);
    const initial = model.evaluate().accuracy;
    for (let i = 0; i < 1500; i++) model.train(0.15, 32);
    const result = model.evaluate();
    expect(result.accuracy).toBeGreaterThan(initial);
    expect(result.accuracy).toBeGreaterThan(0.9);
    const grid = model.extras().grid as { path: number[] };
    expect(grid.path.at(-1)).toBe(GRID.goal);
    expect(gridTransition(0, 0).next).toBe(0);
    expect(gridTransition(7, 1).next).toBe(7);
  });

  it('computes normalized, input-dependent attention probabilities', () => {
    const positive = attentionPool(['the', 'model', 'very', 'great']);
    const negative = attentionPool(['the', 'model', 'very', 'bad']);
    expect(positive.attention).not.toEqual(negative.attention);
    for (const row of positive.attention) expect(row.reduce((total, value) => total + value, 0)).toBeCloseTo(1, 12);
    expect(positive.feature[0]).toBeGreaterThan(negative.feature[0]);
  });

  it('serializes the cached Box–Muller normal as well as the uniform state', () => {
    const a = new Random(17);
    a.normal();
    const b = new Random(99);
    b.importState(a.exportState());
    expect(Array.from({ length: 20 }, () => b.normal())).toEqual(Array.from({ length: 20 }, () => a.normal()));
  });
});
