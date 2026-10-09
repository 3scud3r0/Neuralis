import type { RunConfig } from '../types';
import { finiteVector, zeros } from './math';
import { Random } from './random';

export type OptimizerKind = NonNullable<RunConfig['optimizer']>;

export interface OptimizerOptions {
  kind?: OptimizerKind;
  weightDecay?: number;
  gradientClip?: number;
}

export interface ScalarUpdate {
  value: number;
  first: number;
  second: number;
}

/** Optimizer state has one first and one second moment for every parameter. */
export function updateScalar(
  value: number, gradient: number, first: number, second: number,
  step: number, learningRate: number, options: OptimizerOptions = {}, decay = true,
): ScalarUpdate {
  const kind = options.kind ?? 'adam';
  const weightDecay = decay ? options.weightDecay ?? 0 : 0;
  const g = gradient + (kind === 'adamw' ? 0 : weightDecay * value);
  let direction: number;
  let nextFirst = first;
  let nextSecond = second;
  if (kind === 'sgd') direction = g;
  else if (kind === 'momentum') {
    nextFirst = 0.9 * first + g;
    direction = nextFirst;
  } else if (kind === 'rmsprop') {
    nextSecond = 0.99 * second + 0.01 * g * g;
    direction = g / (Math.sqrt(nextSecond) + 1e-8);
  } else {
    nextFirst = 0.9 * first + 0.1 * g;
    nextSecond = 0.999 * second + 0.001 * g * g;
    direction = (nextFirst / (1 - 0.9 ** step)) / (Math.sqrt(nextSecond / (1 - 0.999 ** step)) + 1e-8);
  }
  const nextValue = value - learningRate * direction - (kind === 'adamw' ? learningRate * weightDecay * value : 0);
  if (![nextValue, nextFirst, nextSecond].every(Number.isFinite)) throw new Error('Optimizer produced a nonfinite parameter or moment.');
  return { value: nextValue, first: nextFirst, second: nextSecond };
}

export function scheduledLearningRate(config: RunConfig, step: number): number {
  const schedule = config.schedule ?? 'constant';
  if (schedule === 'constant') return config.learningRate;
  const warmup = schedule === 'warmup-cosine' ? Math.min(config.maxSteps, config.warmupSteps ?? Math.max(1, Math.floor(config.maxSteps * 0.1))) : 0;
  if (step < warmup) return config.learningRate * (step + 1) / Math.max(1, warmup);
  const progress = Math.min(1, Math.max(0, (step - warmup) / Math.max(1, config.maxSteps - warmup)));
  return config.learningRate * (0.05 + 0.95 * (1 + Math.cos(Math.PI * progress)) / 2);
}

export interface ParameterCheckpoint {
  name: string;
  shape: number[];
  values: number[];
  gradients: number[];
  first: number[];
  second: number[];
}

export class Parameter {
  readonly name: string;
  readonly shape: number[];
  readonly decay: boolean;
  values: number[];
  gradients: number[];
  first: number[];
  second: number[];

  constructor(name: string, shape: number[], random: Random, deviation = 0, decay = true, initial = 0) {
    if (!shape.length || shape.some(size => !Number.isInteger(size) || size < 1)) throw new Error('Invalid parameter shape.');
    const size = shape.reduce((total, value) => total * value, 1);
    this.name = name;
    this.shape = [...shape];
    this.decay = decay;
    this.values = Array.from({ length: size }, () => deviation ? random.normal(initial, deviation) : initial);
    this.gradients = zeros(size);
    this.first = zeros(size);
    this.second = zeros(size);
  }

  rows(): number[][] {
    const columns = this.shape.length > 1 ? this.shape.slice(1).reduce((total, value) => total * value, 1) : 1;
    return Array.from({ length: this.shape[0] }, (_, i) => this.values.slice(i * columns, (i + 1) * columns));
  }

  gradientRows(): number[][] {
    const columns = this.shape.length > 1 ? this.shape.slice(1).reduce((total, value) => total * value, 1) : 1;
    return Array.from({ length: this.shape[0] }, (_, i) => this.gradients.slice(i * columns, (i + 1) * columns));
  }

  checkpoint(): ParameterCheckpoint {
    return { name: this.name, shape: [...this.shape], values: [...this.values], gradients: [...this.gradients], first: [...this.first], second: [...this.second] };
  }

  validate(saved: ParameterCheckpoint): void {
    if (!saved || saved.name !== this.name || JSON.stringify(saved.shape) !== JSON.stringify(this.shape) ||
      !finiteVector(saved.values, this.values.length) || !finiteVector(saved.gradients, this.values.length) ||
      !finiteVector(saved.first, this.values.length) || !finiteVector(saved.second, this.values.length) ||
      saved.second.some(value => value < 0)) throw new Error(`Invalid checkpoint tensor: ${this.name}.`);
  }

  restore(saved: ParameterCheckpoint): void {
    this.validate(saved);
    this.values = [...saved.values];
    this.gradients = [...saved.gradients];
    this.first = [...saved.first];
    this.second = [...saved.second];
  }
}

export interface ParameterBankCheckpoint { updates: number; parameters: ParameterCheckpoint[]; }

export class ParameterBank {
  readonly parameters: Parameter[] = [];
  updates = 0;

  add(parameter: Parameter): Parameter {
    if (this.parameters.some(item => item.name === parameter.name)) throw new Error('Parameter names must be unique.');
    this.parameters.push(parameter);
    return parameter;
  }

  zeroGradients(): void {
    this.parameters.forEach(parameter => parameter.gradients.fill(0));
  }

  scaleGradients(scale: number): void {
    for (const parameter of this.parameters) {
      for (let i = 0; i < parameter.gradients.length; i++) parameter.gradients[i] *= scale;
    }
  }

  get gradientNorm(): number {
    return Math.sqrt(this.parameters.reduce((total, parameter) => total +
      parameter.gradients.reduce((subtotal, value) => subtotal + value * value, 0), 0));
  }

  get size(): number {
    return this.parameters.reduce((total, parameter) => total + parameter.values.length, 0);
  }

  update(learningRate: number, options: OptimizerOptions = {}): number {
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) throw new Error('Invalid optimizer learning rate.');
    const gradientNorm = this.gradientNorm;
    if (!Number.isFinite(gradientNorm)) throw new Error('Nonfinite analytic gradient.');
    const threshold = options.gradientClip ?? 5;
    const scale = gradientNorm > threshold ? threshold / gradientNorm : 1;
    this.updates++;
    for (const parameter of this.parameters) {
      for (let i = 0; i < parameter.values.length; i++) {
        const next = updateScalar(parameter.values[i], parameter.gradients[i] * scale, parameter.first[i], parameter.second[i],
          this.updates, learningRate, options, parameter.decay);
        parameter.values[i] = next.value;
        parameter.first[i] = next.first;
        parameter.second[i] = next.second;
      }
    }
    return gradientNorm;
  }

  checkpoint(): ParameterBankCheckpoint {
    return { updates: this.updates, parameters: this.parameters.map(parameter => parameter.checkpoint()) };
  }

  restore(value: ParameterBankCheckpoint): void {
    if (!value || !Number.isInteger(value.updates) || value.updates < 0 || !Array.isArray(value.parameters) ||
      value.parameters.length !== this.parameters.length) throw new Error('Invalid parameter-bank checkpoint.');
    this.parameters.forEach((parameter, i) => parameter.validate(value.parameters[i]));
    this.parameters.forEach((parameter, i) => parameter.restore(value.parameters[i]));
    this.updates = value.updates;
  }
}
