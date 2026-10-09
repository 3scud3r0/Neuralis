import type { GPUCommandEncoderHandle, GPUDeviceHandle, GPUPipelineHandle } from './types';

export const MAX_GPU_PARTICLES = 8192;
export function boundedDispatch(count: number, workgroupSize = 64, maximum = MAX_GPU_PARTICLES): { count: number; groups: number } {
  if (!Number.isInteger(count) || count < 0 || count > maximum) throw new Error(`Compute capture requires 0–${maximum} particles.`);
  if (!Number.isInteger(workgroupSize) || workgroupSize < 1 || workgroupSize > 256) throw new Error('Workgroup size must be an integer in [1, 256].');
  return { count, groups: Math.ceil(count / workgroupSize) };
}

export class ComputeShaderRouter {
  private kernels = new Map<string, { pipeline: GPUPipelineHandle; workgroupSize: number }>();
  constructor(private device: GPUDeviceHandle) {}
  register(name: string, source: string, workgroupSize = 64): GPUPipelineHandle {
    boundedDispatch(0, workgroupSize);
    if (!name || this.kernels.has(name)) throw new Error('Compute kernel names must be unique and non-empty.');
    const pipeline = this.device.createComputePipeline({ layout: 'auto', compute: { module: this.device.createShaderModule({ code: source }), entryPoint: 'main' } });
    this.kernels.set(name, { pipeline, workgroupSize });
    return pipeline;
  }
  dispatch(name: string, encoder: GPUCommandEncoderHandle, group: unknown, count: number): void {
    const kernel = this.kernels.get(name);
    if (!kernel) throw new Error(`Unknown compute kernel: ${name}`);
    const { groups } = boundedDispatch(count, kernel.workgroupSize);
    if (groups > this.device.limits.maxComputeWorkgroupsPerDimension) throw new Error('Dispatch exceeds adapter limits.');
    if (!groups) return;
    const pass = encoder.beginComputePass();
    pass.setPipeline(kernel.pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(groups);
    pass.end();
  }
  clear(): void { this.kernels.clear(); }
}
