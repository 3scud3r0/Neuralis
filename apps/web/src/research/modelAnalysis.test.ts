import { describe, expect, it } from 'vitest';
import { adversarialSearch, calibration, integratedGradients, linearCKA, nearestRepresentations, numericalGradient } from './modelAnalysis';

describe('controlled model analysis', () => {
  it('recovers analytical gradients and integrated-gradient completeness', () => {
    const model = (input: number[]) => [2 * input[0] + 3 * input[1] + 1];
    expect(numericalGradient(model, [4, 5])[0]).toBeCloseTo(2, 8);
    const result = integratedGradients(model, [4, 5], { steps: 8 });
    expect(result.values[0]).toBeCloseTo(8, 8);
    expect(result.values[1]).toBeCloseTo(15, 8);
    expect(result.completenessError).toBeCloseTo(0, 8);
  });
  it('bounds attribution work before evaluating an expensive model', () => {
    expect(() => integratedGradients(() => [1], [1, 2, 3], { maxEvaluations: 5 })).toThrow(/budget/);
  });
  it('keeps projected adversarial perturbations inside the requested norm ball', () => {
    const result = adversarialSearch(input => [input[0] + input[1]], [0, 0], { target: 0, direction: 'increase', radius: 0.1, steps: 4 });
    expect(result.input).toEqual([0.1, 0.1]);
    expect(result.after[0]).toBeCloseTo(0.2);
    expect(result.trajectory.every(point => point.distance <= 0.1 + 1e-12)).toBe(true);
  });
  it('makes linear CKA invariant to translation and isotropic scale', () => {
    const data = [[1, 2], [2, 0], [0, 1], [4, 3]];
    const transformed = data.map(row => row.map(value => value * 7 + 100));
    expect(linearCKA(data, transformed)).toBeCloseTo(1, 8);
    expect(linearCKA(data, data)).toBeCloseTo(1, 10);
  });
  it('computes zero calibration error for a perfectly confident correct classifier', () => {
    const result = calibration([[1, 0], [0, 1]], [0, 1]);
    expect(result.expectedCalibrationError).toBe(0);
    expect(result.brierScore).toBe(0);
    expect(() => calibration([[0.8, 0.8]], [0])).toThrow();
  });
  it('finds nearest representations with deterministic tie resolution', () => {
    const result = nearestRepresentations([1, 0], [[0, 1], [1, 0], [2, 0]], 2);
    expect(result.map(item => item.index)).toEqual([1, 2]);
  });
});
