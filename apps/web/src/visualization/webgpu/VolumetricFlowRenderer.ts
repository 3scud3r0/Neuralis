import { ComputeShaderRouter, MAX_GPU_PARTICLES } from './ComputeShaderRouter';
import { BufferUsage, browserGPU, type GPUBufferHandle, type GPUCanvasHandle, type GPUDeviceHandle, type GPUPipelineHandle } from './types';

export interface FlowPoint { position: [number, number, number]; value: number; label?: string }
export type RendererCapability = { mode: 'webgpu' | 'canvas'; message: string };

export function packFlowPoints(points: readonly FlowPoint[]): Float32Array {
  if (points.length > MAX_GPU_PARTICLES) throw new Error(`Particle capture exceeds ${MAX_GPU_PARTICLES} points.`);
  const packed = new Float32Array(Math.max(1, points.length) * 4);
  points.forEach((point, index) => {
    if (![...point.position, point.value].every(Number.isFinite)) throw new Error('Particle positions and values must be finite.');
    packed.set([...point.position, point.value], index * 4);
  });
  return packed;
}

export const FLOW_COMPUTE_WGSL = `
struct Camera { angle: f32, aspect: f32, radius: f32, count: f32 };
@group(0) @binding(0) var<storage, read> captured: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> projected: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> camera: Camera;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(camera.count)) { return; }
  let p = captured[gid.x];
  let c = cos(camera.angle); let s = sin(camera.angle);
  let x = p.x * c + p.z * s; let z = -p.x * s + p.z * c;
  let depth = max(1.0, 3.6 + z);
  projected[gid.x] = vec4<f32>(x / depth / camera.aspect * 2.0, p.y / depth * 2.0, z, p.w);
}`;

const PARTICLE_WGSL = `
struct Camera { angle: f32, aspect: f32, radius: f32, count: f32 };
@group(0) @binding(0) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> camera: Camera;
struct Vertex { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) value: f32 };
@vertex fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Vertex {
  var corners = array<vec2<f32>, 6>(vec2<f32>(-1,-1),vec2<f32>(1,-1),vec2<f32>(-1,1),vec2<f32>(-1,1),vec2<f32>(1,-1),vec2<f32>(1,1));
  let p = particles[i]; let uv = corners[v];
  var out: Vertex; out.position = vec4<f32>(p.xy + uv * vec2<f32>(camera.radius/camera.aspect,camera.radius), 0.0, 1.0);
  out.uv = uv; out.value = p.w; return out;
}
@fragment fn fragment(in: Vertex) -> @location(0) vec4<f32> {
  let r = length(in.uv); if (r > 1.0) { discard; }
  let color = select(vec3<f32>(0.18,0.80,1.0),vec3<f32>(0.72,0.44,1.0),in.value < 0.0);
  let alpha = (1.0-r*r) * (0.3 + 0.65 * min(abs(in.value),1.0));
  return vec4<f32>(color,alpha);
}`;

/** Camera projection is GPU compute; particle color is captured activation, never a simulated training value. */
export class VolumetricFlowRenderer {
  private device?: GPUDeviceHandle;
  private context?: GPUCanvasHandle;
  private router?: ComputeShaderRouter;
  private renderPipeline?: GPUPipelineHandle;
  private buffers: GPUBufferHandle[] = [];
  private computeGroup: unknown;
  private renderGroup: unknown;
  private points: FlowPoint[] = [];
  private cpuContext: CanvasRenderingContext2D | null = null;
  private disposed = false;
  capability: RendererCapability = { mode: 'canvas', message: 'Initializing renderer' };
  constructor(private canvas: HTMLCanvasElement, private onCapability?: (status: RendererCapability) => void, private fallbackCanvas?: HTMLCanvasElement) {}
  async initialize(): Promise<RendererCapability> {
    const gpu = browserGPU();
    try {
      if (!gpu) throw new Error('WebGPU is unavailable in this browser.');
      const adapter = await gpu.requestAdapter({ powerPreference: 'low-power' });
      if (!adapter) throw new Error('No WebGPU adapter is available.');
      const device = await adapter.requestDevice();
      if (this.disposed) { device.destroy(); return this.capability; }
      const context = (this.canvas.getContext as (id: string) => unknown)('webgpu') as GPUCanvasHandle | null;
      if (!context) { device.destroy(); throw new Error('WebGPU canvas context is unavailable.'); }
      this.device = device; this.context = context;
      const format = gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: 'opaque' });
      this.router = new ComputeShaderRouter(device);
      const compute = this.router.register('camera-project', FLOW_COMPUTE_WGSL);
      const bytes = MAX_GPU_PARTICLES * 16;
      if (bytes > device.limits.maxStorageBufferBindingSize) throw new Error('Adapter storage limit is too small.');
      this.buffers = [device.createBuffer({ size: bytes, usage: BufferUsage.STORAGE | BufferUsage.COPY_DST }), device.createBuffer({ size: bytes, usage: BufferUsage.STORAGE }), device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST })];
      this.computeGroup = device.createBindGroup({ layout: compute.getBindGroupLayout(0), entries: this.buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const module = device.createShaderModule({ code: PARTICLE_WGSL });
      this.renderPipeline = device.createRenderPipeline({ layout: 'auto', vertex: { module, entryPoint: 'vertex' }, fragment: { module, entryPoint: 'fragment', targets: [{ format, blend: { color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }] }, primitive: { topology: 'triangle-list' } });
      this.renderGroup = device.createBindGroup({ layout: this.renderPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.buffers[1] } }, { binding: 1, resource: { buffer: this.buffers[2] } }] });
      this.setPoints(this.points);
      this.capability = { mode: 'webgpu', message: 'WebGPU · bounded projection compute + instanced particles' };
      void device.lost.then(info => { if (!this.disposed) { this.cpuContext = (this.fallbackCanvas ?? this.canvas).getContext('2d'); this.capability = { mode: 'canvas', message: `WebGPU device lost: ${info.message}. CPU canvas projection active.` }; this.renderCPU(0, 0.018); this.onCapability?.(this.capability); } });
    } catch (error) {
      this.buffers.forEach(buffer => buffer.destroy()); this.buffers = [];
      this.device?.destroy(); this.device = undefined;
      this.cpuContext = (this.fallbackCanvas ?? this.canvas).getContext('2d');
      this.capability = { mode: 'canvas', message: `${error instanceof Error ? error.message : 'GPU initialization failed.'} CPU canvas projection active.` };
    }
    this.onCapability?.(this.capability); return this.capability;
  }
  setPoints(points: readonly FlowPoint[]): void {
    const data = packFlowPoints(points);
    this.points = points.map(point => ({ ...point, position: [...point.position] }));
    if (this.device && this.buffers.length) this.device.queue.writeBuffer(this.buffers[0], 0, data);
  }
  render(angle = 0, size = 0.018): void {
    if (this.disposed) return;
    const aspect = this.canvas.width / Math.max(1, this.canvas.height);
    if (this.device && this.context && this.router && this.renderPipeline && this.capability.mode === 'webgpu') {
      this.device.queue.writeBuffer(this.buffers[2], 0, new Float32Array([angle, aspect, size, this.points.length]));
      const encoder = this.device.createCommandEncoder();
      this.router.dispatch('camera-project', encoder, this.computeGroup, this.points.length);
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: { r: 0.018, g: 0.032, b: 0.062, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
      pass.setPipeline(this.renderPipeline); pass.setBindGroup(0, this.renderGroup); pass.draw(6, this.points.length); pass.end();
      this.device.queue.submit([encoder.finish()]);
    } else this.renderCPU(angle, size);
  }
  private renderCPU(angle: number, size: number): void {
    const ctx = this.cpuContext; if (!ctx) return;
    const target = this.fallbackCanvas ?? this.canvas;
    const w = target.width, h = target.height;
    ctx.fillStyle = '#050810'; ctx.fillRect(0, 0, w, h);
    const c = Math.cos(angle), s = Math.sin(angle);
    const projected = this.points.map(point => {
      const [x,y,z] = point.position; const depth = Math.max(1, 3.6 - x*s + z*c);
      return { x: w/2 + (x*c+z*s)/depth*h, y: h/2-y/depth*h, z: depth, value: point.value };
    }).sort((a,b) => b.z-a.z);
    for (const point of projected) {
      const radius = size*h/2; const gradient = ctx.createRadialGradient(point.x,point.y,0,point.x,point.y,radius);
      const color = point.value < 0 ? '174,109,255' : '47,205,255';
      gradient.addColorStop(0, `rgba(${color},${0.3+0.65*Math.min(1,Math.abs(point.value))})`); gradient.addColorStop(1,`rgba(${color},0)`);
      ctx.fillStyle = gradient; ctx.fillRect(point.x-radius,point.y-radius,radius*2,radius*2);
    }
  }
  dispose(): void { this.disposed = true; this.buffers.forEach(buffer => buffer.destroy()); this.context?.unconfigure(); this.router?.clear(); this.device?.destroy(); }
}
