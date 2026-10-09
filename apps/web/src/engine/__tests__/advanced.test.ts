import { describe, expect, it } from 'vitest';
import type { RunConfig } from '../../types';
import { getDemo } from '../catalog';
import { CNNModel } from '../cnn';
import { Affine, AveragePool2D, causalAttention, causalAttentionBackward, Convolution2D, LayerNorm } from '../layers';
import { DenseNetwork } from '../network';
import { ParameterBank, scheduledLearningRate, updateScalar } from '../optimizers';
import { Random } from '../random';
import { TransformerModel } from '../transformer';

const config: RunConfig = { demoId: 'tiny-cnn', learningRate: 0.01, maxSteps: 200, batchSize: 16, seed: 42, captureEvery: 10 };

describe('complete CNN and Transformer numerical gradients', () => {
  for (const architecture of ['cnn', 'transformer']) {
    it(`matches loss finite differences across every parameter tensor in ${architecture}`, () => {
      const model = architecture === 'cnn' ? new CNNModel(getDemo('tiny-cnn'), config) : new TransformerModel(config);
      const sample = model.samples()[0];
      model.analyticGradients([sample.input], [sample.target]);
      const epsilon = 1e-5;
      for (const parameter of model.bank.parameters) {
        const indices = [...new Set([0, Math.floor(parameter.values.length / 2), parameter.values.length - 1])];
        for (const index of indices) {
          const original = parameter.values[index];
          parameter.values[index] = original + epsilon;
          const positive = model.sampleLoss(sample.input, sample.target);
          parameter.values[index] = original - epsilon;
          const negative = model.sampleLoss(sample.input, sample.target);
          parameter.values[index] = original;
          const measured = (positive - negative) / (2 * epsilon);
          expect(parameter.gradients[index], `${parameter.name}[${index}]`).toBeCloseTo(measured, 5);
        }
      }
    });
  }

  it('actually improves convolutional held-out classification', () => {
    const model = new CNNModel(getDemo('tiny-cnn'), config);
    const initial = model.evaluate();
    for (let i = 0; i < 100; i++) model.train(0.01, 16);
    const result = model.evaluate();
    expect(result.loss).toBeLessThan(initial.loss * 0.2);
    expect(result.accuracy).toBeGreaterThan(0.9);
  });

  it('actually learns causal next-token grammar and exposes masked attention', () => {
    const model = new TransformerModel(config);
    const initial = model.evaluate();
    for (let i = 0; i < 100; i++) model.train(0.008, 16);
    const result = model.evaluate();
    expect(result.loss).toBeLessThan(initial.loss * 0.35);
    expect(result.accuracy).toBeGreaterThan(0.7);
    const heads = model.extras().attentionHeads as number[][][];
    for (const head of heads) for (let row = 0; row < head.length; row++) {
      expect(head[row].reduce((total, value) => total + value, 0)).toBeCloseTo(1, 12);
      for (let column = row + 1; column < head.length; column++) expect(head[row][column]).toBe(0);
    }
  });
});

describe('primitive derivatives and optimizer families', () => {
  it('checks every Q/K/V derivative of masked attention against a directional finite difference', () => {
    const queries = [[0.1, 0.2, 0.3, 0.4], [0.5, -0.1, 0.2, -0.3], [0.2, 0.6, -0.2, 0.1]];
    const keys = queries.map(row => row.map(value => value * 0.7));
    const values = queries.map(row => row.map(value => value + 0.2));
    const upstream = queries.map(row => row.map(value => value * 0.3 - 0.1));
    const gradient = causalAttentionBackward(causalAttention(queries, keys, values, 2), upstream);
    const objective = () => causalAttention(queries, keys, values, 2).context.reduce((total, row, i) =>
      total + row.reduce((subtotal, value, j) => subtotal + value * upstream[i][j], 0), 0);
    const epsilon = 1e-5;
    for (const field of ['queries', 'keys', 'values'] as const) {
      const tensor = field === 'queries' ? queries : field === 'keys' ? keys : values;
      for (let row = 0; row < tensor.length; row++) for (let column = 0; column < tensor[row].length; column++) {
        const original = tensor[row][column];
        tensor[row][column] = original + epsilon;
        const positive = objective();
        tensor[row][column] = original - epsilon;
        const negative = objective();
        tensor[row][column] = original;
        expect(gradient[field][row][column]).toBeCloseTo((positive - negative) / (2 * epsilon), 8);
      }
    }
  });

  it('checks convolution input derivatives and average-pooling transpose', () => {
    const bank = new ParameterBank();
    const convolution = new Convolution2D('test', { width: 4, height: 4, inputChannels: 2, outputChannels: 2, kernel: 3 }, bank, new Random(7));
    const input = Array.from({ length: 32 }, (_, i) => Math.sin(i));
    const upstream = Array.from({ length: 8 }, (_, i) => Math.cos(i));
    const gradient = convolution.backward(input, upstream);
    const objective = () => convolution.forward(input).reduce((total, value, i) => total + value * upstream[i], 0);
    for (let i = 0; i < input.length; i++) {
      const original = input[i];
      input[i] = original + 1e-5;
      const positive = objective();
      input[i] = original - 1e-5;
      const negative = objective();
      input[i] = original;
      expect(gradient[i]).toBeCloseTo((positive - negative) / 2e-5, 8);
    }
    const pool = new AveragePool2D(4, 4, 2);
    const pooled = pool.forward(input);
    const transposeGradient = pool.backward(upstream);
    expect(pooled.reduce((total, value, i) => total + value * upstream[i], 0)).toBeCloseTo(
      input.reduce((total, value, i) => total + value * transposeGradient[i], 0), 10);
  });

  it('layer normalization preserves the derivative of a constant shift', () => {
    const bank = new ParameterBank();
    const normalization = new LayerNorm('test', 4, bank, new Random(2));
    const pass = normalization.forward([0.2, -0.3, 0.8, 0.4]);
    const inputGradient = normalization.backward(pass.cache, [0.1, 0.6, -0.2, 0.3]);
    expect(inputGradient.reduce((total, value) => total + value, 0)).toBeCloseTo(0, 12);
  });

  for (const kind of ['sgd', 'momentum', 'adam', 'adamw', 'rmsprop'] as const) {
    it(`${kind} reduces a convex objective with finite optimizer state`, () => {
      let value = 2;
      let first = 0;
      let second = 0;
      for (let step = 1; step <= 300; step++) {
        const result = updateScalar(value, 2 * value, first, second, step, 0.03, { kind });
        value = result.value; first = result.first; second = result.second;
      }
      expect(Math.abs(value)).toBeLessThan(0.1);
      expect([value, first, second].every(Number.isFinite)).toBe(true);
    });
  }

  it('warmup/cosine schedule has the documented endpoints', () => {
    const scheduled = { ...config, schedule: 'warmup-cosine' as const, warmupSteps: 20 };
    expect(scheduledLearningRate(scheduled, 0)).toBeCloseTo(config.learningRate / 20);
    expect(scheduledLearningRate(scheduled, 20)).toBeCloseTo(config.learningRate);
    expect(scheduledLearningRate(scheduled, 200)).toBeCloseTo(config.learningRate * 0.05);
  });

  it('seeded dropout resumes identically from a checkpoint', () => {
    const first = new CNNModel(getDemo('tiny-cnn'), { ...config, dropout: 0.3 });
    const second = new CNNModel(getDemo('tiny-cnn'), { ...config, dropout: 0.3 });
    for (let i = 0; i < 3; i++) first.train(0.01, 8);
    second.restore(first.checkpoint());
    first.train(0.01, 8); second.train(0.01, 8);
    expect(second.checkpoint()).toEqual(first.checkpoint());
  });
});
