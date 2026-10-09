import { Random } from '../../engine/random';
export const NF4_LEVELS = [-1, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1];
export interface QuantizedWeights { count: number; blockSize: number; packed: number[]; scales: number[]; }
export function quantizeNF4(weights: number[], blockSize = 64): QuantizedWeights {
  if (!weights.length || weights.length > 65536 || weights.some(value => !Number.isFinite(value))) throw new Error('Quantization needs 1–65,536 finite weights.');
  if (!Number.isInteger(blockSize) || blockSize < 8 || blockSize > 256) throw new Error('Use a block size between 8 and 256.');
  const scales: number[] = [];
  const codes: number[] = [];
  for (let offset = 0; offset < weights.length; offset += blockSize) {
    const block = weights.slice(offset, offset + blockSize);
    const scale = Math.max(...block.map(Math.abs));
    scales.push(scale);
    for (const weight of block) {
      const normalized = scale ? weight / scale : 0;
      let best = 0;
      for (let code = 1; code < NF4_LEVELS.length; code++) if (Math.abs(normalized - NF4_LEVELS[code]) < Math.abs(normalized - NF4_LEVELS[best])) best = code;
      codes.push(best);
    }
  }
  const packed = Array.from({ length: Math.ceil(codes.length / 2) }, (_, index) => codes[index * 2] | ((codes[index * 2 + 1] ?? 0) << 4));
  return { count: weights.length, blockSize, packed, scales };
}
export function dequantizeNF4(weights: QuantizedWeights): number[] {
  if (!Number.isInteger(weights.count) || weights.count < 1 || weights.count > 65536 || !Number.isInteger(weights.blockSize) || weights.blockSize < 8 || weights.blockSize > 256 || weights.packed.length !== Math.ceil(weights.count / 2) || weights.scales.length !== Math.ceil(weights.count / weights.blockSize) || weights.packed.some(value => !Number.isInteger(value) || value < 0 || value > 255) || weights.scales.some(value => !Number.isFinite(value) || value < 0)) throw new Error('Invalid packed NF4 artifact.');
  return Array.from({ length: weights.count }, (_, index) => NF4_LEVELS[index % 2 ? weights.packed[Math.floor(index / 2)] >>> 4 : weights.packed[Math.floor(index / 2)] & 15] * weights.scales[Math.floor(index / weights.blockSize)]);
}
export function quantizationDemo(seed: number, count = 256, blockSize = 64) {
  const random = new Random(seed);
  const original = Array.from({ length: count }, () => random.normal(0, 0.4));
  const artifact = quantizeNF4(original, blockSize);
  const decoded = dequantizeNF4(artifact);
  const squared = original.map((value, index) => (value - decoded[index]) ** 2);
  const originalBytes = count * 4;
  const packedBytes = artifact.packed.length + artifact.scales.length * 4;
  return { original, decoded, artifact, mse: squared.reduce((sum, value) => sum + value, 0) / count,
    maxError: Math.sqrt(Math.max(...squared)), originalBytes, packedBytes, compression: originalBytes / packedBytes };
}
export function adapterBudget(input: number, output: number, rank: number) {
  if (![input, output, rank].every(value => Number.isInteger(value) && value > 0 && value <= 65536) || rank > Math.min(input, output)) throw new Error('Rank must fit positive bounded layer dimensions.');
  const baseParameters = input * output;
  const adapterParameters = rank * (input + output);
  return { baseParameters, adapterParameters, fraction: adapterParameters / baseParameters,
    fullTrainingBytes: baseParameters * 16, adapterTrainingBytes: adapterParameters * 16 + Math.ceil(baseParameters / 2) + Math.ceil(baseParameters / 64) * 4 };
}
