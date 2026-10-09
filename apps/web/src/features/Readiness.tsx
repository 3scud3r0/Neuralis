import { useEffect, useState } from 'react';
import { CheckCircle2, XCircle, AlertCircle, Cpu, Play, Clock3, ShieldCheck, Download, Activity, Monitor } from 'lucide-react';
import { createEngine } from '../engine';
import type { RunSnapshot } from '../types';
import { useStudio } from '../app/StudioContext';
import { downloadText } from '../app/storage';
import { Panel, PageHeader, Button, Badge, KeyValue, Callout, Progress, number } from '../components/ui';

interface ReadinessCheck {
  id: string;
  label: string;
  supported: boolean;
  required: boolean;
  detail: string;
}
interface BenchmarkResult {
  startedAt: string;
  durationMs: number;
  steps: number;
  stepsPerSecond: number;
  finalLoss: number;
  finalScore: number;
  seed: number;
  device: string;
}
function probe(): ReadinessCheck[] {
  let storage = false;
  try { const key = 'neuralis.readiness'; localStorage.setItem(key, 'probe'); storage = localStorage.getItem(key) === 'probe'; localStorage.removeItem(key); } catch { /* Optional persistence may be unavailable. */ }
  const canvas = document.createElement('canvas');
  let webgl = false;
  try {
    const context = canvas.getContext('webgl2') || canvas.getContext('webgl');
    webgl = Boolean(context);
    context?.getExtension('WEBGL_lose_context')?.loseContext();
  } catch { /* Renderer can use its fallback. */ }
  return [
    { id: 'js', label: 'Local numerical engine', supported: typeof Float64Array !== 'undefined', required: true, detail: 'Typed arrays are available for model parameters and numerical operations.' },
    { id: 'timing', label: 'High-resolution timing', supported: typeof performance?.now === 'function', required: true, detail: 'Used to bound optimizer scheduling and measure actual elapsed work.' },
    { id: 'storage', label: 'Local artifact storage', supported: storage, required: false, detail: storage ? 'Browser localStorage accepted a write/read/delete probe.' : 'Persistence is blocked. Use downloadable checkpoints.' },
    { id: 'webgl', label: 'WebGL scene acceleration', supported: webgl, required: false, detail: webgl ? 'A WebGL context was created successfully.' : 'WebGL is unavailable; scene rendering may use a fallback.' },
    { id: 'download', label: 'Artifact download', supported: typeof Blob !== 'undefined' && typeof URL.createObjectURL === 'function', required: true, detail: 'Browser file downloads can export model JSON and captured data.' },
    { id: 'uuid', label: 'Secure identifier generation', supported: Boolean(crypto?.randomUUID), required: false, detail: 'Secure-context UUID support is used for local artifact identifiers.' },
    { id: 'clipboard', label: 'Clipboard access', supported: Boolean(navigator.clipboard), required: false, detail: 'Availability does not guarantee permission. File export is always preferred for durable data.' },
  ];
}
export function Readiness() {
  const { notify } = useStudio();
  const [checks, setChecks] = useState<ReadinessCheck[]>([]);
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [progress, setProgress] = useState(0);
  const [running, setRunning] = useState(false);
  useEffect(() => { setChecks(probe()); }, []);
  const benchmark = async () => {
    setRunning(true); setProgress(0); setResult(null);
    let engine: ReturnType<typeof createEngine> | null = null;
    try {
      engine = createEngine({ demoId: 'xor', learningRate: .03, maxSteps: 100, batchSize: 16, seed: 42, captureEvery: 10 }, () => {});
      const start = performance.now();
      let snapshot: RunSnapshot = engine.getSnapshot();
      for (let batch = 0; batch < 10; batch++) {
        for (let index = 0; index < 10; index++) engine.step();
        snapshot = engine.getSnapshot();
        setProgress(snapshot.step);
        await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      }
      const durationMs = performance.now() - start;
      setResult({ startedAt: new Date().toISOString(), durationMs, steps: snapshot.step, stepsPerSecond: snapshot.step / Math.max(durationMs / 1000, 1e-9), finalLoss: snapshot.loss, finalScore: snapshot.accuracy, seed: 42, device: snapshot.device });
      notify('Bounded XOR benchmark completed using real model updates.', 'success');
    } catch (error) { notify(error instanceof Error ? error.message : 'Benchmark failed.', 'error'); }
    finally { engine?.dispose(); setRunning(false); }
  };
  const allRequired = checks.filter(check => check.required).every(check => check.supported);
  return <>
    <PageHeader eyebrow="ENVIRONMENT READINESS" title="Know the limits of your workspace." description="Runtime capability checks and a bounded benchmark with measured results." actions={<Button onClick={() => setChecks(probe())}><Activity size={14} />Run checks again</Button>} />
    <div className="readiness-banner"><ShieldCheck size={29} /><div><h2>{allRequired ? 'Core browser capabilities are available.' : 'A required browser capability is missing.'}</h2><p>Optional capabilities can improve your workflow; availability is checked on this device.</p></div><Badge tone={allRequired ? 'green' : 'red'}>{allRequired ? 'Core ready' : 'Review required'}</Badge></div>
    <div className="readiness-layout"><Panel title="Capability report" subtitle="Actual feature detection and a storage round trip"><div className="readiness-checks">{checks.map(check => <div className="readiness-check" key={check.id}><span className={check.supported ? 'check-pass' : check.required ? 'check-fail' : 'check-optional'}>{check.supported ? <CheckCircle2 size={19} /> : check.required ? <XCircle size={19} /> : <AlertCircle size={19} />}</span><div><strong>{check.label}</strong><p>{check.detail}</p></div><Badge tone={check.supported ? 'green' : check.required ? 'red' : 'amber'}>{check.supported ? 'Available' : 'Unavailable'}</Badge></div>)}</div></Panel><div className="readiness-detail"><Panel title="Bounded numerical benchmark" action={<Cpu size={17} />}><div className="control-stack"><div className="benchmark-icon"><Clock3 size={28} /></div><p className="muted">Run 100 actual XOR optimizer steps in an isolated model. This does not change your active experiment or save a run to history.</p><KeyValue label="Model" value="2 → 8 → 8 → 1" /><KeyValue label="Seed" value="42" /><KeyValue label="Batch size" value="16" /><KeyValue label="Step budget" value="100" /><Button variant="primary" disabled={running} onClick={() => void benchmark()}><Play size={14} />{running ? 'Benchmarking…' : 'Run benchmark'}</Button>{running && <Progress value={progress} />}{result && <div className="benchmark-result"><Badge tone="green" dot>Measured</Badge><KeyValue label="Elapsed wall time" value={`${number(result.durationMs, 1)} ms`} /><KeyValue label="Throughput" value={`${number(result.stepsPerSecond, 1)} steps/s`} /><KeyValue label="Final loss" value={number(result.finalLoss, 6)} /><KeyValue label="Final task score" value={`${(result.finalScore * 100).toFixed(1)}%`} /><Button onClick={() => downloadText('neuralis-browser-benchmark.json', JSON.stringify({ benchmark: result, checks, userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency }, null, 2))}><Download size={14} />Export measured report</Button></div>}</div></Panel><Panel title="Device observations" action={<Monitor size={17} />}><div className="info-stack"><KeyValue label="Reported logical processors" value={navigator.hardwareConcurrency ?? 'Unavailable'} /><KeyValue label="Viewport" value={`${window.innerWidth} × ${window.innerHeight}`} /><KeyValue label="Device pixel ratio" value={window.devicePixelRatio} /><KeyValue label="Online hint" value={navigator.onLine ? 'Online' : 'Offline'} /></div></Panel></div></div>
    <Callout>Benchmark throughput includes UI scheduling and snapshot evaluation overhead. It is useful for this small model and browser session, rather than a GPU benchmark or a prediction of large-model performance.</Callout>
  </>;
}
