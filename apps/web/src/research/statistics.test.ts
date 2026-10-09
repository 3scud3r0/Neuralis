import { describe, expect, it } from 'vitest';
import { adjustPValues, bootstrapInterval, meanInterval, pairedTTest, pearsonCorrelation, regularizedBeta, studentTCdf, studentTQuantile, summarize, wilsonInterval } from './statistics';

describe('research statistics', () => {
  it('uses stable sample variance even around a large offset', () => {
    const result = summarize([1e9 + 1, 1e9 + 2, 1e9 + 3]);
    expect(result.mean).toBe(1e9 + 2);
    expect(result.variance).toBeCloseTo(1, 8);
  });
  it('matches reference Student t values and beta endpoints', () => {
    expect(studentTCdf(0, 10)).toBeCloseTo(0.5, 12);
    expect(studentTQuantile(0.975, 10)).toBeCloseTo(2.22813885196, 8);
    expect(regularizedBeta(0.5, 1, 1)).toBeCloseTo(0.5, 12);
  });
  it('reports no difference for identical paired observations', () => {
    const result = pairedTTest([1, 2, 3, 4], [1, 2, 3, 4]);
    expect(result.difference).toBe(0);
    expect(result.pValue).toBe(1);
  });
  it('rejects silently misaligned or nonfinite data', () => {
    expect(() => pairedTTest([1, 2], [1, 2, 3])).toThrow();
    expect(() => summarize([1, NaN])).toThrow();
    expect(summarize([1, NaN, 3], true).missingCount).toBe(1);
  });
  it('produces deterministic bootstrap intervals and sensible mean confidence', () => {
    const sample = [1, 2, 3, 4, 5];
    expect(bootstrapInterval(sample, { iterations: 200, seed: 7 })).toEqual(bootstrapInterval(sample, { iterations: 200, seed: 7 }));
    const interval = meanInterval(sample);
    expect(interval.lower).toBeLessThan(3);
    expect(interval.upper).toBeGreaterThan(3);
  });
  it('keeps Wilson bounds inside probabilities even for zero or complete successes', () => {
    expect(wilsonInterval(0, 10).lower).toBe(0);
    expect(wilsonInterval(10, 10).upper).toBeCloseTo(1, 14);
    expect(wilsonInterval(10, 10).lower).toBeCloseTo(0.7224672, 6);
  });
  it('corrects multiple comparisons in original order', () => {
    expect(adjustPValues([0.04, 0.01, 0.03], 'holm')).toEqual([0.06, 0.03, 0.06]);
    expect(adjustPValues([0.04, 0.01, 0.03], 'bonferroni')).toEqual([0.12, 0.03, 0.09]);
  });
  it('computes positive and negative reference correlations', () => {
    expect(pearsonCorrelation([1, 2, 3], [2, 4, 6])).toBeCloseTo(1);
    expect(pearsonCorrelation([1, 2, 3], [6, 4, 2])).toBeCloseTo(-1);
  });
});
