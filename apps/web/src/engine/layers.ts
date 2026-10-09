import { mean, softmax, zeros } from './math';
import { Parameter, ParameterBank } from './optimizers';
import { Random } from './random';

/** Affine parameter orientation is [output, input], matching captured layers. */
export class Affine {
  readonly input: number;
  readonly output: number;
  readonly weight: Parameter;
  readonly bias: Parameter;

  constructor(name: string, input: number, output: number, bank: ParameterBank, random: Random) {
    this.input = input;
    this.output = output;
    this.weight = bank.add(new Parameter(`${name}.weight`, [output, input], random, Math.sqrt(2 / (input + output))));
    this.bias = bank.add(new Parameter(`${name}.bias`, [output], random, 0, false));
  }

  forward(input: number[]): number[] {
    if (input.length !== this.input) throw new Error('Affine input has an incompatible shape.');
    const output = [...this.bias.values];
    for (let unit = 0; unit < this.output; unit++) {
      const offset = unit * this.input;
      for (let i = 0; i < this.input; i++) output[unit] += this.weight.values[offset + i] * input[i];
    }
    return output;
  }

  backward(input: number[], gradient: number[]): number[] {
    if (gradient.length !== this.output) throw new Error('Affine gradient has an incompatible shape.');
    const inputGradient = zeros(this.input);
    for (let unit = 0; unit < this.output; unit++) {
      const g = gradient[unit];
      const offset = unit * this.input;
      this.bias.gradients[unit] += g;
      for (let i = 0; i < this.input; i++) {
        this.weight.gradients[offset + i] += g * input[i];
        inputGradient[i] += g * this.weight.values[offset + i];
      }
    }
    return inputGradient;
  }
}

export interface ConvolutionShape {
  width: number; height: number; inputChannels: number;
  outputChannels: number; kernel: number;
}

/** Valid, unit-stride convolution; channel-major flattened tensors. */
export class Convolution2D {
  readonly shape: ConvolutionShape;
  readonly outputWidth: number;
  readonly outputHeight: number;
  readonly weight: Parameter;
  readonly bias: Parameter;

  constructor(name: string, shape: ConvolutionShape, bank: ParameterBank, random: Random) {
    if ([shape.width, shape.height, shape.inputChannels, shape.outputChannels, shape.kernel].some(value => !Number.isInteger(value) || value < 1) ||
      shape.kernel > shape.width || shape.kernel > shape.height) throw new Error('Invalid convolution dimensions.');
    this.shape = { ...shape };
    this.outputWidth = shape.width - shape.kernel + 1;
    this.outputHeight = shape.height - shape.kernel + 1;
    this.weight = bank.add(new Parameter(`${name}.weight`, [shape.outputChannels, shape.inputChannels, shape.kernel, shape.kernel],
      random, Math.sqrt(1 / (shape.inputChannels * shape.kernel * shape.kernel))));
    this.bias = bank.add(new Parameter(`${name}.bias`, [shape.outputChannels], random, 0, false));
  }

  get outputSize(): number {
    return this.outputWidth * this.outputHeight * this.shape.outputChannels;
  }

  forward(input: number[]): number[] {
    const { width, height, inputChannels, outputChannels, kernel } = this.shape;
    if (input.length !== width * height * inputChannels) throw new Error('Convolution input has an incompatible shape.');
    const result = zeros(this.outputSize);
    for (let channel = 0; channel < outputChannels; channel++) {
      for (let row = 0; row < this.outputHeight; row++) {
        for (let column = 0; column < this.outputWidth; column++) {
          let value = this.bias.values[channel];
          for (let inputChannel = 0; inputChannel < inputChannels; inputChannel++) {
            for (let kr = 0; kr < kernel; kr++) {
              for (let kc = 0; kc < kernel; kc++) {
                const weightIndex = ((channel * inputChannels + inputChannel) * kernel + kr) * kernel + kc;
                const inputIndex = (inputChannel * height + row + kr) * width + column + kc;
                value += input[inputIndex] * this.weight.values[weightIndex];
              }
            }
          }
          result[(channel * this.outputHeight + row) * this.outputWidth + column] = value;
        }
      }
    }
    return result;
  }

  backward(input: number[], gradient: number[]): number[] {
    if (gradient.length !== this.outputSize) throw new Error('Convolution gradient has an incompatible shape.');
    const { width, height, inputChannels, outputChannels, kernel } = this.shape;
    const result = zeros(input.length);
    for (let channel = 0; channel < outputChannels; channel++) {
      for (let row = 0; row < this.outputHeight; row++) {
        for (let column = 0; column < this.outputWidth; column++) {
          const g = gradient[(channel * this.outputHeight + row) * this.outputWidth + column];
          this.bias.gradients[channel] += g;
          for (let inputChannel = 0; inputChannel < inputChannels; inputChannel++) {
            for (let kr = 0; kr < kernel; kr++) {
              for (let kc = 0; kc < kernel; kc++) {
                const weightIndex = ((channel * inputChannels + inputChannel) * kernel + kr) * kernel + kc;
                const inputIndex = (inputChannel * height + row + kr) * width + column + kc;
                this.weight.gradients[weightIndex] += input[inputIndex] * g;
                result[inputIndex] += this.weight.values[weightIndex] * g;
              }
            }
          }
        }
      }
    }
    return result;
  }

  /** The sparse spatial operator exposes real weight sharing for rendering. */
  expandedWeights(): number[][] {
    const { width, height, inputChannels, outputChannels, kernel } = this.shape;
    const inputSize = width * height * inputChannels;
    const rows: number[][] = [];
    for (let channel = 0; channel < outputChannels; channel++) {
      for (let row = 0; row < this.outputHeight; row++) {
        for (let column = 0; column < this.outputWidth; column++) {
          const weights = zeros(inputSize);
          for (let inputChannel = 0; inputChannel < inputChannels; inputChannel++) {
            for (let kr = 0; kr < kernel; kr++) {
              for (let kc = 0; kc < kernel; kc++) {
                const weightIndex = ((channel * inputChannels + inputChannel) * kernel + kr) * kernel + kc;
                weights[(inputChannel * height + row + kr) * width + column + kc] = this.weight.values[weightIndex];
              }
            }
          }
          rows.push(weights);
        }
      }
    }
    return rows;
  }
}

/** Nonoverlapping spatial average pooling, with its exact transpose backward. */
export class AveragePool2D {
  readonly width: number;
  readonly height: number;
  readonly channels: number;
  readonly window: number;
  readonly outputWidth: number;
  readonly outputHeight: number;

  constructor(width: number, height: number, channels: number, window = 2) {
    if ([width, height, channels, window].some(value => !Number.isInteger(value) || value < 1) ||
      width % window !== 0 || height % window !== 0) throw new Error('Pooling requires divisible positive spatial dimensions.');
    this.width = width; this.height = height; this.channels = channels; this.window = window;
    this.outputWidth = width / window; this.outputHeight = height / window;
  }

  get outputSize(): number { return this.outputWidth * this.outputHeight * this.channels; }

  forward(input: number[]): number[] {
    if (input.length !== this.width * this.height * this.channels) throw new Error('Pooling input shape mismatch.');
    const result = zeros(this.outputSize);
    const scale = 1 / (this.window * this.window);
    for (let channel = 0; channel < this.channels; channel++) {
      for (let row = 0; row < this.outputHeight; row++) {
        for (let column = 0; column < this.outputWidth; column++) {
          const index = (channel * this.outputHeight + row) * this.outputWidth + column;
          for (let dr = 0; dr < this.window; dr++) {
            for (let dc = 0; dc < this.window; dc++) {
              result[index] += input[(channel * this.height + row * this.window + dr) * this.width + column * this.window + dc] * scale;
            }
          }
        }
      }
    }
    return result;
  }

  backward(gradient: number[]): number[] {
    if (gradient.length !== this.outputSize) throw new Error('Pooling gradient shape mismatch.');
    const result = zeros(this.width * this.height * this.channels);
    const scale = 1 / (this.window * this.window);
    for (let channel = 0; channel < this.channels; channel++) {
      for (let row = 0; row < this.outputHeight; row++) {
        for (let column = 0; column < this.outputWidth; column++) {
          const index = (channel * this.outputHeight + row) * this.outputWidth + column;
          for (let dr = 0; dr < this.window; dr++) {
            for (let dc = 0; dc < this.window; dc++) {
              result[(channel * this.height + row * this.window + dr) * this.width + column * this.window + dc] += gradient[index] * scale;
            }
          }
        }
      }
    }
    return result;
  }

  expandedWeights(): number[][] {
    return Array.from({ length: this.outputSize }, (_, i) => {
      const gradient = zeros(this.outputSize);
      gradient[i] = 1;
      return this.backward(gradient);
    });
  }
}

export interface LayerNormCache {
  normalized: number[];
  inverseDeviation: number;
}

export class LayerNorm {
  readonly scale: Parameter;
  readonly shift: Parameter;
  readonly dimension: number;

  constructor(name: string, dimension: number, bank: ParameterBank, random: Random) {
    this.dimension = dimension;
    this.scale = bank.add(new Parameter(`${name}.scale`, [dimension], random, 0, false, 1));
    this.shift = bank.add(new Parameter(`${name}.shift`, [dimension], random, 0, false));
  }

  forward(input: number[]): { output: number[]; cache: LayerNormCache } {
    if (input.length !== this.dimension) throw new Error('Layer norm shape mismatch.');
    const center = mean(input);
    const variance = mean(input.map(value => (value - center) ** 2));
    const inverseDeviation = 1 / Math.sqrt(variance + 1e-5);
    const normalized = input.map(value => (value - center) * inverseDeviation);
    return { output: normalized.map((value, i) => value * this.scale.values[i] + this.shift.values[i]),
      cache: { normalized, inverseDeviation } };
  }

  backward(cache: LayerNormCache, gradient: number[]): number[] {
    const scaled = gradient.map((value, i) => value * this.scale.values[i]);
    const average = mean(scaled);
    const correlation = mean(scaled.map((value, i) => value * cache.normalized[i]));
    for (let i = 0; i < this.dimension; i++) {
      this.scale.gradients[i] += gradient[i] * cache.normalized[i];
      this.shift.gradients[i] += gradient[i];
    }
    return scaled.map((value, i) => cache.inverseDeviation * (value - average - cache.normalized[i] * correlation));
  }
}

export interface AttentionCache {
  queries: number[][];
  keys: number[][];
  values: number[][];
  probabilities: number[][][];
  context: number[][];
}

export interface AttentionGradients {
  queries: number[][];
  keys: number[][];
  values: number[][];
}

/** Causal scaled dot-product attention with independent contiguous heads. */
export function causalAttention(
  queries: number[][], keys: number[][], values: number[][], heads: number,
): AttentionCache {
  const length = queries.length;
  const dimension = queries[0]?.length ?? 0;
  if (!length || keys.length !== length || values.length !== length || !heads || dimension % heads !== 0 ||
    [...queries, ...keys, ...values].some(row => row.length !== dimension)) throw new Error('Attention tensors have incompatible shapes.');
  const headSize = dimension / heads;
  const scale = 1 / Math.sqrt(headSize);
  const probabilities: number[][][] = [];
  const context = Array.from({ length }, () => zeros(dimension));
  for (let head = 0; head < heads; head++) {
    const offset = head * headSize;
    const attention: number[][] = [];
    for (let query = 0; query < length; query++) {
      const scores = Array.from({ length: query + 1 }, (_, key) => {
        let score = 0;
        for (let i = 0; i < headSize; i++) score += queries[query][offset + i] * keys[key][offset + i];
        return score * scale;
      });
      const row = [...softmax(scores), ...zeros(length - query - 1)];
      attention.push(row);
      for (let key = 0; key <= query; key++) {
        for (let i = 0; i < headSize; i++) context[query][offset + i] += row[key] * values[key][offset + i];
      }
    }
    probabilities.push(attention);
  }
  return { queries, keys, values, probabilities, context };
}

/** Exact softmax Jacobian-vector product, including the causal mask. */
export function causalAttentionBackward(cache: AttentionCache, gradient: number[][]): AttentionGradients {
  const length = cache.queries.length;
  const dimension = cache.queries[0].length;
  const heads = cache.probabilities.length;
  const headSize = dimension / heads;
  const scale = 1 / Math.sqrt(headSize);
  const result: AttentionGradients = {
    queries: Array.from({ length }, () => zeros(dimension)),
    keys: Array.from({ length }, () => zeros(dimension)),
    values: Array.from({ length }, () => zeros(dimension)),
  };
  for (let head = 0; head < heads; head++) {
    const offset = head * headSize;
    for (let query = 0; query < length; query++) {
      const probabilities = cache.probabilities[head][query];
      const probabilityGradient = zeros(length);
      for (let key = 0; key <= query; key++) {
        for (let i = 0; i < headSize; i++) {
          probabilityGradient[key] += gradient[query][offset + i] * cache.values[key][offset + i];
          result.values[key][offset + i] += probabilities[key] * gradient[query][offset + i];
        }
      }
      const meanGradient = probabilities.reduce((total, value, key) => total + value * probabilityGradient[key], 0);
      for (let key = 0; key <= query; key++) {
        const scoreGradient = probabilities[key] * (probabilityGradient[key] - meanGradient) * scale;
        for (let i = 0; i < headSize; i++) {
          result.queries[query][offset + i] += scoreGradient * cache.keys[key][offset + i];
          result.keys[key][offset + i] += scoreGradient * cache.queries[query][offset + i];
        }
      }
    }
  }
  return result;
}

export function applyDropout(input: number[], probability: number, random?: Random): { output: number[]; mask: number[] } {
  if (!probability) return { output: [...input], mask: input.map(() => 1) };
  if (!random || probability < 0 || probability >= 1) throw new Error('Dropout requires a seeded stream and probability in [0, 1).');
  const mask = input.map(() => random.uniform() < probability ? 0 : 1 / (1 - probability));
  return { output: input.map((value, i) => value * mask[i]), mask };
}
