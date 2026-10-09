/** Structural WebGPU subset, kept local for DOM releases without WebGPU declarations. */
export interface GPUBufferHandle { destroy(): void }
export interface GPUPipelineHandle { getBindGroupLayout(index: number): unknown }
export interface GPUComputePassHandle {
  setPipeline(pipeline: GPUPipelineHandle): void;
  setBindGroup(index: number, group: unknown): void;
  dispatchWorkgroups(count: number): void;
  end(): void;
}
export interface GPURenderPassHandle {
  setPipeline(pipeline: GPUPipelineHandle): void;
  setBindGroup(index: number, group: unknown): void;
  draw(vertices: number, instances: number): void;
  end(): void;
}
export interface GPUCommandEncoderHandle {
  beginComputePass(): GPUComputePassHandle;
  beginRenderPass(descriptor: unknown): GPURenderPassHandle;
  finish(): unknown;
}
export interface GPUDeviceHandle {
  queue: { writeBuffer(buffer: GPUBufferHandle, offset: number, data: Float32Array): void; submit(commands: unknown[]): void };
  limits: { maxStorageBufferBindingSize: number; maxComputeWorkgroupsPerDimension: number };
  lost: Promise<{ message: string }>;
  createBuffer(descriptor: unknown): GPUBufferHandle;
  createShaderModule(descriptor: { code: string }): unknown;
  createComputePipeline(descriptor: unknown): GPUPipelineHandle;
  createRenderPipeline(descriptor: unknown): GPUPipelineHandle;
  createBindGroup(descriptor: unknown): unknown;
  createCommandEncoder(): GPUCommandEncoderHandle;
  destroy(): void;
}
export interface GPUCanvasHandle {
  configure(descriptor: unknown): void;
  getCurrentTexture(): { createView(): unknown };
  unconfigure(): void;
}
export interface GPUAccess {
  requestAdapter(options?: unknown): Promise<{ requestDevice(): Promise<GPUDeviceHandle> } | null>;
  getPreferredCanvasFormat(): string;
}
export function browserGPU(): GPUAccess | undefined {
  return typeof navigator === 'undefined' ? undefined : (navigator as Navigator & { gpu?: GPUAccess }).gpu;
}
export const BufferUsage = { COPY_DST: 8, UNIFORM: 64, STORAGE: 128 } as const;
