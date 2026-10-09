import { useEffect, useRef, useState } from 'react';
import { VolumetricFlowRenderer, type FlowPoint, type RendererCapability } from '../webgpu/VolumetricFlowRenderer';
import { MAX_GPU_PARTICLES } from '../webgpu/ComputeShaderRouter';

export interface LatentSpaceExplorerProps { points: FlowPoint[]; title?: string; reducedMotion?: boolean; height?: number }
export function LatentSpaceExplorer({ points, title = 'Captured representation space', reducedMotion = false, height = 420 }: LatentSpaceExplorerProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const fallbackCanvas = useRef<HTMLCanvasElement>(null);
  const renderer = useRef<VolumetricFlowRenderer | null>(null);
  const pointsRef = useRef(points);
  const angleRef = useRef(0);
  const [angle, setAngle] = useState(0);
  const [rotate, setRotate] = useState(false);
  const [status, setStatus] = useState<RendererCapability>({ mode: 'canvas', message: 'Checking graphics capability…' });
  pointsRef.current = points;
  angleRef.current = angle;
  useEffect(() => {
    if (!canvas.current) return;
    const element = canvas.current;
    const fallback = fallbackCanvas.current;
    if (!fallback) return;
    const instance = new VolumetricFlowRenderer(element, setStatus, fallback);
    renderer.current = instance;
    let disposed = false;
    const resize = () => {
      const scale = Math.min(2, window.devicePixelRatio || 1);
      element.width = Math.max(1, Math.floor(element.clientWidth*scale));
      element.height = Math.max(1, Math.floor(height*scale));
      fallback.width = element.width;
      fallback.height = element.height;
      instance.render(angleRef.current);
    };
    const observer = new ResizeObserver(resize); observer.observe(element);
    void instance.initialize().then(() => { if (!disposed) { instance.setPoints(pointsRef.current.slice(0, MAX_GPU_PARTICLES)); resize(); } });
    return () => { disposed = true; observer.disconnect(); instance.dispose(); renderer.current = null; };
  }, [height]);
  useEffect(() => { renderer.current?.setPoints(points.slice(0,MAX_GPU_PARTICLES)); renderer.current?.render(angleRef.current); }, [points]);
  useEffect(() => { renderer.current?.render(angle); }, [angle]);
  useEffect(() => { if (status.mode === 'canvas') renderer.current?.render(angleRef.current); }, [status.mode]);
  useEffect(() => {
    if (!rotate || reducedMotion) return;
    let frame = 0, last = 0;
    const animate = (time: number) => {
      if (last) angleRef.current += Math.min(32,time-last)*0.00008;
      last = time; renderer.current?.render(angleRef.current); frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate); return () => cancelAnimationFrame(frame);
  }, [rotate,reducedMotion]);
  return <div className="advanced-space"><div className="advanced-space-toolbar"><strong>{title}</strong><span className="badge badge-cyan">{status.mode === 'webgpu' ? 'WEBGPU' : 'CPU CANVAS'}</span></div><canvas ref={canvas} style={{ width: '100%', height, display: status.mode === 'webgpu' ? 'block' : 'none' }} aria-label={`${title}: ${points.length} measured points`} /><canvas ref={fallbackCanvas} style={{ width: '100%', height, display: status.mode === 'webgpu' ? 'none' : 'block' }} aria-label={`${title}: CPU fallback projection`} /><div className="advanced-space-controls"><label>Orbit angle<input type="range" min={-3.14} max={3.14} step={0.01} value={angle} onChange={event => setAngle(Number(event.target.value))} /></label><label><input type="checkbox" checked={rotate} disabled={reducedMotion} onChange={event => setRotate(event.target.checked)} />Auto orbit</label><span>{Math.min(points.length,MAX_GPU_PARTICLES).toLocaleString()} measured points</span></div><p className="advanced-note">{status.message}. Cyan = nonnegative values; violet = negative values. Camera motion changes the view, while measured values stay fixed.</p></div>;
}
