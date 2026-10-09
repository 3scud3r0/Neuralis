import { describe, expect, it } from 'vitest';
import { DenseNetwork } from '../network';
import { Random } from '../random';

describe('exact dense-network derivatives', () => {
  for (const objective of ['mse', 'binary', 'categorical'] as const) {
    it(`matches centered finite differences for ${objective}`, () => {
      const network = new DenseNetwork([2, 3, objective === 'categorical' ? 3 : 1], objective, new Random(12));
      const input = [0.25, -0.6];
      const target = objective === 'categorical' ? [0, 1, 0] : [0.7];
      network.gradients([input], [target]);
      const epsilon = 1e-5;
      for (const layer of network.layers) {
        for (let unit = 0; unit < layer.output; unit++) {
          for (let coordinate = 0; coordinate < layer.input; coordinate++) {
            const original = layer.weights[unit][coordinate];
            layer.weights[unit][coordinate] = original + epsilon;
            const positive = network.loss(input, target);
            layer.weights[unit][coordinate] = original - epsilon;
            const negative = network.loss(input, target);
            layer.weights[unit][coordinate] = original;
            expect(layer.weightGradient[unit][coordinate]).toBeCloseTo((positive - negative) / (2 * epsilon), 6);
          }
          const original = layer.biases[unit];
          layer.biases[unit] = original + epsilon;
          const positive = network.loss(input, target);
          layer.biases[unit] = original - epsilon;
          const negative = network.loss(input, target);
          layer.biases[unit] = original;
          expect(layer.biasGradient[unit]).toBeCloseTo((positive - negative) / (2 * epsilon), 6);
        }
      }
    });
  }

  it('learns all four XOR cases rather than a fabricated loss curve', () => {
    const network = new DenseNetwork([2, 8, 8, 1], 'binary', new Random(42));
    const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const targets = [[0], [1], [1], [0]];
    const initial = network.evaluate(inputs, targets).loss;
    for (let update = 0; update < 500; update++) network.trainBatch(inputs, targets, 0.02);
    const result = network.evaluate(inputs, targets);
    expect(result.loss).toBeLessThan(initial * 0.05);
    expect(result.accuracy).toBe(1);
    expect(network.predict([0, 1])[0]).toBeGreaterThan(0.95);
    expect(network.predict([1, 1])[0]).toBeLessThan(0.05);
  });

  it('restores Adam moments for an identical next update', () => {
    const a = new DenseNetwork([2, 5, 2], 'categorical', new Random(5));
    const b = new DenseNetwork([2, 5, 2], 'categorical', new Random(99));
    const inputs = [[0.2, -0.1], [0.7, 0.6]];
    const targets = [[1, 0], [0, 1]];
    for (let i = 0; i < 7; i++) a.trainBatch(inputs, targets, 0.01);
    b.importCheckpoint(a.exportCheckpoint());
    a.trainBatch(inputs, targets, 0.01);
    b.trainBatch(inputs, targets, 0.01);
    expect(b.exportCheckpoint()).toEqual(a.exportCheckpoint());
  });

  it('does not mutate parameters while performing an activation intervention', () => {
    const network = new DenseNetwork([2, 4, 1], 'binary', new Random(31));
    const before = network.exportCheckpoint();
    const baseline = network.predict([0.5, 0.7]);
    const modified = network.predict([0.5, 0.7], { layer: 1, neuron: 0, kind: 'steer', value: 2 });
    expect(modified).not.toEqual(baseline);
    expect(network.exportCheckpoint()).toEqual(before);
  });
});
