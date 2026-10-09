import { afterEach, describe, expect, it, vi } from 'vitest';
import type { EngineController, RunConfig } from '../../types';
import { DEMOS } from '../catalog';
import { createEngine } from '../controller';

const config: RunConfig = { demoId: 'xor', learningRate: 0.02, maxSteps: 100, batchSize: 16, seed: 42, captureEvery: 5 };
const active: EngineController[] = [];
const make = (options: Partial<RunConfig> = {}) => {
  const engine = createEngine({ ...config, ...options }, () => undefined);
  active.push(engine);
  return engine;
};

afterEach(() => { active.splice(0).forEach(engine => engine.dispose()); vi.useRealTimers(); });

describe('browser run lifecycle and persistence', () => {
  it('initializes actual predictions and network tensors before start', () => {
    const snapshot = make().getSnapshot();
    expect(snapshot.state).toBe('idle');
    expect(snapshot.step).toBe(0);
    expect(snapshot.layers.length).toBe(4);
    expect(snapshot.samples.length).toBe(8);
    expect(snapshot.parameters).toBe(105);
    expect(snapshot.loss).toBeGreaterThan(0);
  });

  it('can single-step, resume, pause, and stop without background mutation', async () => {
    vi.useFakeTimers();
    const engine = make();
    engine.step();
    expect(engine.getSnapshot().step).toBe(1);
    expect(engine.getSnapshot().state).toBe('paused');
    engine.resume();
    await vi.advanceTimersByTimeAsync(1);
    engine.pause();
    const step = engine.getSnapshot().step;
    await vi.advanceTimersByTimeAsync(100);
    expect(engine.getSnapshot().step).toBe(step);
    expect(engine.getSnapshot().state).toBe('paused');
    engine.stop();
    expect(engine.getSnapshot().state).toBe('stopped');
    expect(() => engine.resume()).toThrow();
  });

  it('reproduces the next minibatch after export/import, including PRNG state', () => {
    const first = make();
    const second = make({ demoId: 'circles', seed: 55 });
    for (let i = 0; i < 12; i++) first.step();
    second.importCheckpoint(first.exportCheckpoint());
    first.step();
    second.step();
    expect(second.getSnapshot().layers).toEqual(first.getSnapshot().layers);
    expect(second.getSnapshot().loss).toEqual(first.getSnapshot().loss);
    expect(second.getSnapshot().samples).toEqual(first.getSnapshot().samples);
  });

  it('rejects corrupted checkpoint tensors without changing the current model', () => {
    const engine = make();
    engine.step();
    const before = engine.getSnapshot();
    const checkpoint = JSON.parse(engine.exportCheckpoint());
    checkpoint.model.network.layers[0].weights[0][0] = 'corrupted';
    expect(() => engine.importCheckpoint(JSON.stringify(checkpoint))).toThrow();
    expect(engine.getSnapshot()).toEqual(before);
  });

  it('returns isolated immutable snapshot copies', () => {
    const engine = make();
    const external = engine.getSnapshot();
    external.layers[1].weights[0][0] = 999;
    external.metrics.push({ step: 99, loss: 99, accuracy: 99, timestamp: 99 });
    expect(engine.getSnapshot().layers[1].weights[0][0]).not.toBe(999);
    expect(engine.getSnapshot().metrics).toHaveLength(1);
  });

  it('completes exactly at the configured budget', () => {
    const engine = make({ maxSteps: 3 });
    engine.step(); engine.step(); engine.step();
    expect(engine.getSnapshot().state).toBe('completed');
    expect(engine.getSnapshot().step).toBe(3);
    expect(() => engine.step()).toThrow();
  });

  for (const demo of DEMOS.filter(candidate => !['super-mario', 'custom-data'].includes(candidate.id))) {
    it(`produces finite real tensors and checkpoint round-trips for ${demo.id}`, () => {
      const engine = make({ demoId: demo.id, maxSteps: 5 });
      engine.step(); engine.step();
      const before = engine.getSnapshot();
      expect(Number.isFinite(before.loss)).toBe(true);
      expect(Number.isFinite(before.accuracy)).toBe(true);
      expect(before.accuracy).toBeGreaterThanOrEqual(0);
      expect(before.accuracy).toBeLessThanOrEqual(1);
      expect(before.layers.flatMap(layer => layer.weights.flat()).every(Number.isFinite)).toBe(true);
      const checkpoint = engine.exportCheckpoint();
      engine.importCheckpoint(checkpoint);
      expect(engine.getSnapshot().layers).toEqual(before.layers);
      expect(engine.getSnapshot().samples).toEqual(before.samples);
    });
  }

  it('learns user numerical arrays with explicit holdout and dynamic dimensions', () => {
    const engine = make({ demoId: 'custom-data', hiddenLayers: [], learningRate: 0.05, maxSteps: 200,
      customDataset: { name: 'Linear uploaded data', task: 'regression',
        inputs: [[-1], [-0.5], [0], [0.5], [1]], targets: [[-1.7], [-0.7], [0.3], [1.3], [2.3]],
        validationInputs: [[-0.75], [0.25], [0.75]], validationTargets: [[-1.2], [0.8], [1.8]], featureNames: ['x'] } });
    const before = engine.getSnapshot().loss;
    for (let i = 0; i < 200; i++) engine.step();
    expect(engine.getSnapshot().loss).toBeLessThan(before * 0.05);
    expect(engine.getSnapshot().layers.map(layer => layer.size)).toEqual([1, 1]);
    expect(engine.getSnapshot().extras?.dataset).toBe('Linear uploaded data');
    const restored = make();
    restored.importCheckpoint(engine.exportCheckpoint());
    expect(restored.predict([0.25])).toEqual(engine.predict([0.25]));
  });

  it('rejects malformed custom datasets before allocating a model', () => {
    expect(() => make({ demoId: 'custom-data', customDataset: { task: 'classification', inputs: [[0], [1]], targets: [[2], [-1]] } })).toThrow();
    expect(() => make({ demoId: 'custom-data', customDataset: { task: 'regression', inputs: [[0], [1, 2]], targets: [[0], [1]] } })).toThrow();
    expect(() => make({ demoId: 'custom-data' })).toThrow();
  });
});
