import { Random } from '../engine/random';

export interface SummaryStatistics {
  count: number; mean: number; variance: number; standardDeviation: number;
  minimum: number; maximum: number; median: number; q1: number; q3: number;
  standardError: number; finiteCount: number; missingCount: number;
}
export interface ConfidenceInterval {
  estimate: number; lower: number; upper: number; confidence: number;
  method: string; samples: number; seed?: number;
}
export interface ComparisonResult {
  difference: number; statistic: number; pValue: number; degreesOfFreedom?: number;
  effectSize: number; sampleSize: number; method: string;
}

function finite(values: readonly number[], minimum = 1): number[] {
  if (values.length < minimum) throw new Error(`At least ${minimum} observations are required.`);
  if (values.some(value => !Number.isFinite(value))) throw new Error('Statistical inputs must all be finite; missing observations must be handled explicitly.');
  return [...values];
}

export function quantile(sorted: readonly number[], probability: number): number {
  if (!sorted.length || probability < 0 || probability > 1 || !Number.isFinite(probability)) {
    throw new Error('A nonempty sorted sample and a quantile between zero and one are required.');
  }
  const position = probability * (sorted.length - 1);
  const left = Math.floor(position);
  const right = Math.ceil(position);
  return sorted[left] * (right - position || 1) + (left === right ? 0 : sorted[right] * (position - left));
}

export function summarize(values: readonly number[], allowMissing = false): SummaryStatistics {
  const data = allowMissing ? values.filter(Number.isFinite) : finite(values);
  if (!data.length) throw new Error('There are no finite observations to summarize.');
  let mean = 0;
  let m2 = 0;
  let count = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of data) {
    count++;
    const delta = value - mean;
    mean += delta / count;
    m2 += delta * (value - mean);
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  const variance = count > 1 ? m2 / (count - 1) : 0;
  const sorted = [...data].sort((a, b) => a - b);
  return {
    count, mean, variance, standardDeviation: Math.sqrt(variance), minimum, maximum,
    median: quantile(sorted, 0.5), q1: quantile(sorted, 0.25), q3: quantile(sorted, 0.75),
    standardError: Math.sqrt(variance / count), finiteCount: count, missingCount: values.length - count,
  };
}

const LANCZOS = [676.5203681218851, -1259.1392167224028, 771.3234287776531,
  -176.6150291621406, 12.507343278686905, -0.13857109526572012,
  9.984369578019572e-6, 1.5056327351493116e-7];

export function logGamma(value: number): number {
  if (!(value > 0)) throw new Error('Gamma requires a positive argument.');
  if (value < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  let coefficient = 0.99999999999980993;
  const shifted = value - 1;
  for (let index = 0; index < LANCZOS.length; index++) coefficient += LANCZOS[index] / (shifted + index + 1);
  const t = shifted + LANCZOS.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(coefficient);
}

function betaFraction(a: number, b: number, x: number): number {
  const epsilon = 3e-14;
  const tiny = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - qab * x / qap;
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < epsilon) return h;
  }
  throw new Error('Incomplete beta calculation did not converge.');
}

export function regularizedBeta(x: number, a: number, b: number): number {
  if (!Number.isFinite(x) || x < 0 || x > 1 || a <= 0 || b <= 0) throw new Error('Invalid beta distribution parameters.');
  if (x === 0 || x === 1) return x;
  const front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log1p(-x));
  return x < (a + 1) / (a + b + 2) ? front * betaFraction(a, b, x) / a :
    1 - front * betaFraction(b, a, 1 - x) / b;
}

export function studentTCdf(t: number, degreesOfFreedom: number): number {
  if (!(degreesOfFreedom > 0) || Number.isNaN(t)) throw new Error('Invalid Student t parameters.');
  if (t === Infinity) return 1;
  if (t === -Infinity) return 0;
  const beta = regularizedBeta(degreesOfFreedom / (degreesOfFreedom + t * t), degreesOfFreedom / 2, 0.5);
  return t >= 0 ? 1 - beta / 2 : beta / 2;
}

export function studentTQuantile(probability: number, degreesOfFreedom: number): number {
  if (!(probability > 0 && probability < 1)) throw new Error('A finite quantile requires probability strictly between zero and one.');
  if (probability === 0.5) return 0;
  if (probability < 0.5) return -studentTQuantile(1 - probability, degreesOfFreedom);
  let low = 0;
  let high = 1;
  while (studentTCdf(high, degreesOfFreedom) < probability && high < 1e12) high *= 2;
  for (let index = 0; index < 100; index++) {
    const midpoint = (low + high) / 2;
    if (studentTCdf(midpoint, degreesOfFreedom) < probability) low = midpoint;
    else high = midpoint;
  }
  return (low + high) / 2;
}

export function meanInterval(values: readonly number[], confidence = 0.95): ConfidenceInterval {
  finite(values, 2);
  if (!(confidence > 0 && confidence < 1)) throw new Error('Confidence must be between zero and one.');
  const summary = summarize(values);
  const margin = studentTQuantile((1 + confidence) / 2, summary.count - 1) * summary.standardError;
  return { estimate: summary.mean, lower: summary.mean - margin, upper: summary.mean + margin,
    confidence, samples: summary.count, method: 'Student t mean interval (independent observations)' };
}

export function bootstrapInterval(values: readonly number[], options: {
  iterations?: number; confidence?: number; seed?: number; statistic?: (sample: number[]) => number;
} = {}): ConfidenceInterval {
  const data = finite(values, 2);
  const iterations = options.iterations ?? 2000;
  const confidence = options.confidence ?? 0.95;
  const seed = options.seed ?? 42;
  if (!Number.isInteger(iterations) || iterations < 100 || iterations > 100000 || !(confidence > 0 && confidence < 1)) {
    throw new Error('Bootstrap iterations must be 100–100000 and confidence must be between zero and one.');
  }
  const statistic = options.statistic ?? (sample => summarize(sample).mean);
  const random = new Random(seed);
  const draws = new Array<number>(iterations);
  const sample = new Array<number>(data.length);
  for (let draw = 0; draw < iterations; draw++) {
    for (let index = 0; index < data.length; index++) sample[index] = data[random.integer(data.length)];
    draws[draw] = statistic(sample);
    if (!Number.isFinite(draws[draw])) throw new Error('The bootstrap statistic returned a nonfinite value.');
  }
  draws.sort((a, b) => a - b);
  const alpha = (1 - confidence) / 2;
  return { estimate: statistic(data), lower: quantile(draws, alpha), upper: quantile(draws, 1 - alpha),
    confidence, samples: data.length, seed, method: `Percentile bootstrap (${iterations} resamples)` };
}

export function pairedTTest(before: readonly number[], after: readonly number[]): ComparisonResult {
  finite(before, 2);
  finite(after, 2);
  if (before.length !== after.length) throw new Error('Paired observations must have the same length and alignment.');
  const differences = after.map((value, index) => value - before[index]);
  const summary = summarize(differences);
  const statistic = summary.standardError > 0 ? summary.mean / summary.standardError : summary.mean === 0 ? 0 : Math.sign(summary.mean) * Infinity;
  const pValue = statistic === 0 ? 1 : 2 * (1 - studentTCdf(Math.abs(statistic), before.length - 1));
  return { difference: summary.mean, statistic, pValue: Math.max(0, Math.min(1, pValue)),
    degreesOfFreedom: before.length - 1, effectSize: summary.standardDeviation ? summary.mean / summary.standardDeviation : 0,
    sampleSize: before.length, method: 'Two-sided paired Student t test; effect size is Cohen dz' };
}

export function welchTTest(first: readonly number[], second: readonly number[]): ComparisonResult {
  finite(first, 2);
  finite(second, 2);
  const a = summarize(first);
  const b = summarize(second);
  const va = a.variance / a.count;
  const vb = b.variance / b.count;
  const error = Math.sqrt(va + vb);
  const difference = b.mean - a.mean;
  const statistic = error ? difference / error : difference === 0 ? 0 : Math.sign(difference) * Infinity;
  const denominator = va * va / (a.count - 1) + vb * vb / (b.count - 1);
  const degreesOfFreedom = denominator ? (va + vb) ** 2 / denominator : a.count + b.count - 2;
  const pooled = Math.sqrt(((a.count - 1) * a.variance + (b.count - 1) * b.variance) / (a.count + b.count - 2));
  return { difference, statistic, degreesOfFreedom, pValue: Math.max(0, 2 * (1 - studentTCdf(Math.abs(statistic), degreesOfFreedom))),
    effectSize: pooled ? difference / pooled : 0, sampleSize: a.count + b.count,
    method: 'Two-sided Welch t test; effect size uses pooled descriptive standard deviation' };
}

export function pairedPermutationTest(before: readonly number[], after: readonly number[], iterations = 10000, seed = 42): ComparisonResult {
  finite(before, 2);
  finite(after, 2);
  if (before.length !== after.length || !Number.isInteger(iterations) || iterations < 100 || iterations > 100000) throw new Error('Invalid paired permutation configuration.');
  const differences = after.map((value, index) => value - before[index]);
  const observed = summarize(differences);
  const random = new Random(seed);
  let extreme = 0;
  for (let trial = 0; trial < iterations; trial++) {
    let sum = 0;
    for (const value of differences) sum += (random.uniform() < 0.5 ? -1 : 1) * value;
    if (Math.abs(sum / differences.length) >= Math.abs(observed.mean) - 1e-14) extreme++;
  }
  return { difference: observed.mean, statistic: observed.mean, pValue: (extreme + 1) / (iterations + 1),
    effectSize: observed.standardDeviation ? observed.mean / observed.standardDeviation : 0,
    sampleSize: differences.length, method: `Paired sign-flip permutation test (${iterations} samples, seed ${seed})` };
}

export function adjustPValues(values: readonly number[], method: 'holm' | 'bonferroni' | 'benjamini-hochberg' = 'holm'): number[] {
  if (values.some(value => !Number.isFinite(value) || value < 0 || value > 1)) throw new Error('P-values must be finite probabilities.');
  const ordered = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
  const result = new Array<number>(values.length);
  if (method === 'bonferroni') return values.map(value => Math.min(1, value * values.length));
  if (method === 'holm') {
    let previous = 0;
    ordered.forEach((entry, position) => {
      previous = Math.max(previous, Math.min(1, entry.value * (values.length - position)));
      result[entry.index] = previous;
    });
  } else {
    let previous = 1;
    for (let position = ordered.length - 1; position >= 0; position--) {
      const entry = ordered[position];
      previous = Math.min(previous, entry.value * values.length / (position + 1));
      result[entry.index] = Math.min(1, previous);
    }
  }
  return result;
}

export function wilsonInterval(successes: number, trials: number, z = 1.959963984540054): ConfidenceInterval {
  if (!Number.isInteger(trials) || trials < 1 || !Number.isInteger(successes) || successes < 0 || successes > trials || !(z > 0)) {
    throw new Error('Wilson intervals require a valid success count and positive trial count.');
  }
  const proportion = successes / trials;
  const denominator = 1 + z * z / trials;
  const center = (proportion + z * z / (2 * trials)) / denominator;
  const margin = z * Math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator;
  return { estimate: proportion, lower: Math.max(0, center - margin), upper: Math.min(1, center + margin),
    confidence: 2 * normalCdf(z) - 1, samples: trials, method: `Wilson score interval (z=${z})` };
}

export function normalCdf(value: number): number {
  if (value === Infinity) return 1;
  if (value === -Infinity) return 0;
  if (!Number.isFinite(value)) throw new Error('The normal CDF requires a numeric argument.');
  const x = Math.abs(value);
  const t = 1 / (1 + 0.2316419 * x);
  const polynomial = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const upperTail = Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * polynomial;
  return value >= 0 ? 1 - upperTail : upperTail;
}

export function pearsonCorrelation(first: readonly number[], second: readonly number[]): number {
  finite(first, 2);
  finite(second, 2);
  if (first.length !== second.length) throw new Error('Correlation vectors must be aligned.');
  const a = summarize(first);
  const b = summarize(second);
  let covariance = 0;
  for (let index = 0; index < first.length; index++) covariance += (first[index] - a.mean) * (second[index] - b.mean);
  const denominator = (first.length - 1) * a.standardDeviation * b.standardDeviation;
  return denominator ? Math.max(-1, Math.min(1, covariance / denominator)) : 0;
}

export function effectiveSampleSize(values: readonly number[], maxLag = Math.min(100, Math.floor(values.length / 4))): number {
  const data = finite(values, 4);
  const summary = summarize(data);
  if (!summary.variance) return data.length;
  let correlationSum = 0;
  for (let lag = 1; lag <= maxLag; lag++) {
    let covariance = 0;
    for (let index = lag; index < data.length; index++) covariance += (data[index] - summary.mean) * (data[index - lag] - summary.mean);
    const correlation = covariance / ((data.length - lag) * summary.variance);
    if (correlation <= 0) break;
    correlationSum += correlation;
  }
  return Math.max(1, Math.min(data.length, data.length / (1 + 2 * correlationSum)));
}
