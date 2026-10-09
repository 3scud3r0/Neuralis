import { useCallback, useEffect, useRef, useState } from 'react';
import { CheckCircle2, Cloud, Download, FlaskConical, LoaderCircle, Pause, Play, RefreshCw, Square, Unplug, Wifi, XCircle } from 'lucide-react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { BackendClient, BackendError, type BackendCapabilities, type BackendCommand, type BackendConnection, type BackendEvent, type BackendModel, type BackendRun } from './backendClient';
import './remote.css';

interface Props {
  initialUrl?: string;
  onUrlChange?: (url: string) => void;
}
interface Metric { step: number; loss: number; accuracy?: number; }

function identifier(run: BackendRun): string {
  return String(run.run_id || run.id || '');
}
function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function metricsFrom(value: unknown): Metric[] {
  const data = Array.isArray(value) ? value : asRecord(value).metrics;
  if (!Array.isArray(data)) return [];
  return data.map(item => {
    const metric = asRecord(item);
    return { step: Number(metric.step), loss: Number(metric.loss ?? metric.train_loss), accuracy: Number(metric.accuracy ?? metric.validation_accuracy) };
  }).filter(item => Number.isFinite(item.step) && Number.isFinite(item.loss));
}
function printable(value: unknown): string {
  return JSON.stringify(value, null, 2) ?? 'No response';
}
function saveJson(name: string, content: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export { BackendWorkbench };

export default function BackendWorkbench({ initialUrl = 'http://127.0.0.1:8000', onUrlChange }: Props) {
  const [url, setUrl] = useState(initialUrl);
  const [sessionToken, setSessionToken] = useState('');
  const [connection, setConnection] = useState<BackendConnection | null>(null);
  const [capabilities, setCapabilities] = useState<BackendCapabilities | null>(null);
  const [models, setModels] = useState<BackendModel[]>([]);
  const [runs, setRuns] = useState<BackendRun[]>([]);
  const [modelId, setModelId] = useState('mlp-xor');
  const [steps, setSteps] = useState(300);
  const [learningRate, setLearningRate] = useState(0.02);
  const [seed, setSeed] = useState(42);
  const [selected, setSelected] = useState<BackendRun | null>(null);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [events, setEvents] = useState<BackendEvent[]>([]);
  const [streamState, setStreamState] = useState('disconnected');
  const [captureGaps, setCaptureGaps] = useState(0);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<unknown>(null);
  const [responseTitle, setResponseTitle] = useState('Server response');
  const [module, setModule] = useState('hidden');
  const [strength, setStrength] = useState(0);
  const client = useRef<BackendClient | null>(null);
  const closeStream = useRef<(() => void) | null>(null);
  const active = useRef<string>('');
  const mounted = useRef(true);

  useEffect(() => () => {
    mounted.current = false;
    closeStream.current?.();
    client.current?.dispose();
  }, []);

  const perform = useCallback(async <T,>(name: string, action: () => Promise<T>): Promise<T | undefined> => {
    setBusy(name);
    setError(null);
    try {
      return await action();
    } catch (failure) {
      if (mounted.current) {
        const message = failure instanceof Error ? failure.message : 'The operation failed.';
        const hint = failure instanceof BackendError && failure.action ? ` ${failure.action}` : '';
        setError(`${message}${hint}`);
      }
      return undefined;
    } finally {
      if (mounted.current) setBusy(null);
    }
  }, []);

  const connect = async () => {
    await perform('connect', async () => {
      closeStream.current?.();
      client.current?.dispose();
      const next = new BackendClient(url, sessionToken || undefined);
      client.current = next;
      next.subscribe(value => { if (mounted.current) setConnection(value); });
      const found = await next.connect();
      setCapabilities(found);
      onUrlChange?.(next.baseUrl);
      const [available, previous] = await Promise.all([next.models(), next.runs()]);
      setModels(available);
      setRuns(previous);
      if (available.length) setModelId(available[0].id);
      return found;
    });
  };

  const disconnect = () => {
    closeStream.current?.();
    closeStream.current = null;
    client.current?.dispose();
    client.current = null;
    setCapabilities(null);
    setSelected(null);
    setMetrics([]);
    setEvents([]);
    setResponse(null);
    setStreamState('disconnected');
  };

  const selectRun = useCallback(async (run: BackendRun) => {
    const current = client.current;
    if (!current) return;
    const id = identifier(run);
    if (!id) return;
    closeStream.current?.();
    active.current = id;
    setSelected(run);
    setEvents([]);
    setMetrics([]);
    setCaptureGaps(0);
    closeStream.current = current.stream(id, event => {
      if (!mounted.current || active.current !== event.run_id) return;
      setEvents(previous => [event, ...previous].slice(0, 80));
      if (event.type === 'metrics') {
        const loss = Number(event.payload.loss ?? event.payload.train_loss);
        if (Number.isFinite(loss)) {
          const accuracy = Number(event.payload.accuracy ?? event.payload.validation_accuracy);
          setMetrics(previous => [...previous.filter(item => item.step !== event.step), { step: event.step, loss, accuracy }].slice(-2000));
        }
      }
      if (event.type === 'state') {
        setSelected(previous => previous ? { ...previous, state: String(event.payload.state || event.payload.status) as BackendRun['state'] } : previous);
      }
    }, {
      onState: setStreamState,
      onGap: (previous, next) => setCaptureGaps(count => count + next - previous - 1),
      onError: failure => { if (mounted.current) setError(failure.message); },
    });
    try {
      const existing = await current.metrics(id);
      if (mounted.current && active.current === id) setMetrics(metricsFrom(existing));
    } catch (failure) {
      if (mounted.current) setError(failure instanceof Error ? failure.message : 'Cannot load metrics.');
    }
  }, []);

  useEffect(() => {
    if (!selected || !client.current) return;
    const id = identifier(selected);
    let inFlight = false;
    const timer = setInterval(async () => {
      if (inFlight || !client.current) return;
      inFlight = true;
      try {
        const value = await client.current.run(id);
        if (mounted.current && active.current === id) setSelected(value);
      } catch {
        // A WebSocket disconnection is displayed independently. Polling failures
        // do not fabricate a new run state or reset the existing evidence.
      } finally {
        inFlight = false;
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [selected?.id, selected?.run_id]);

  const create = async () => {
    const current = client.current;
    if (!current) return;
    const run = await perform('create', () => current.createRun({
      model_id: modelId,
      name: `${modelId} · browser session`,
      seed, steps, batch_size: 32, learning_rate: learningRate,
      capture_every: 5, device: 'cpu', step_delay_ms: 5,
    }));
    if (run) {
      setRuns(previous => [run, ...previous]);
      await selectRun(run);
    }
  };

  const command = async (value: BackendCommand) => {
    if (!client.current || !selected) return;
    const current = client.current;
    const id = identifier(selected);
    await perform(value, async () => {
      const result = await current.command(id, value);
      const run = await current.run(id);
      setSelected(run);
      if (value === 'checkpoint') {
        setResponseTitle('Checkpoint acknowledgment');
        setResponse(result);
      }
      return result;
    });
  };

  const inspect = async (kind: 'inference' | 'tensors' | 'attribution' | 'export') => {
    if (!client.current || !selected) return;
    const current = client.current;
    const id = identifier(selected);
    await perform(kind, async () => {
      if (kind === 'export') {
        const result = await current.exportRun(id);
        saveJson(`neuralis-backend-${id}.json`, result);
        return result;
      }
      const result = kind === 'inference' ? await current.inference(id) :
        kind === 'tensors' ? await current.tensors(id) : await current.attribution(id, 'integrated-gradients', 16);
      setResponseTitle(kind === 'inference' ? 'Actual inference result' : kind === 'tensors' ? 'Captured tensors' : 'Integrated gradients result');
      setResponse(result);
      return result;
    });
  };

  const intervene = async () => {
    if (!client.current || !selected) return;
    const current = client.current;
    await perform('intervention', async () => {
      const result = await current.experiment({
        run_id: identifier(selected),
        input: { sample_indices: [0, 1, 2, 3], split: 'validation' },
        interventions: [{ module, operation: 'scale', strength }],
      });
      setResponseTitle('Baseline versus scaled activation');
      setResponse(result);
      return result;
    });
  };

  const connected = connection?.state === 'connected' && capabilities !== null;
  const state = selected?.state || selected?.status || 'created';
  const latest = metrics[metrics.length - 1];
  return <section className="remote-workbench" aria-label="Python research backend">
    <header className="remote-header">
      <div className="remote-icon"><Cloud size={22}/></div>
      <div><h2>Python research runtime</h2><p>Real PyTorch jobs, captured tensors and reproducible interventions.</p></div>
      <span className={`remote-status ${connected ? 'is-connected' : ''}`}>
        {connected ? <CheckCircle2 size={14}/> : <Unplug size={14}/>}
        {connection?.state || 'disconnected'}
      </span>
    </header>
    <form className="remote-connection" onSubmit={event => { event.preventDefault(); void connect(); }}>
      <label>Backend URL<input aria-label="Backend URL" value={url} onChange={event => setUrl(event.target.value)} placeholder="http://127.0.0.1:8000"/></label>
      <label>Session token (optional)<input aria-label="Backend session token" type="password" autoComplete="off" value={sessionToken} onChange={event => setSessionToken(event.target.value)} placeholder="Kept in memory only"/></label>
      <button type="submit" disabled={busy !== null}>{busy === 'connect' ? <LoaderCircle size={16} className="spin"/> : <Wifi size={16}/>}Connect</button>
      {connected && <button type="button" onClick={disconnect}><Unplug size={16}/>Disconnect</button>}
    </form>
    {error && <div className="remote-error" role="alert"><XCircle size={18}/><span>{error}</span><button onClick={() => setError(null)} aria-label="Dismiss backend error">×</button></div>}
    {!connected && <div className="remote-empty"><Cloud size={32}/><h3>Connect when you need more compute</h3><p>The demo library already trains in your browser. This panel connects to an actual local or remote Python runtime; no server or GPU is simulated.</p><code>uv run --project services/api python -m neuralis</code><p>For a hosted workbench, add its origin to the backend CORS configuration. Remote production access requires HTTPS and an authenticated server.</p></div>}
    {connected && <>
      <div className="remote-evidence"><span>HTTP handshake: <strong>{connection?.latencyMs} ms</strong></span><span>Protocol: <strong>{capabilities.protocol_version || 'reported below'}</strong></span><span>Source: <strong>server-reported capabilities</strong></span></div>
      <details className="remote-details"><summary>Inspect capabilities and limits</summary><pre>{printable(capabilities)}</pre></details>
      <div className="remote-grid">
        <div className="remote-card"><h3>Create a CPU experiment</h3>
          <label>Model<select aria-label="Backend model" value={modelId} onChange={event => setModelId(event.target.value)}>{models.map(model => <option key={model.id} value={model.id}>{model.name || model.title || model.id}</option>)}</select></label>
          <div className="remote-fields"><label>Steps<input type="number" min={1} max={100000} value={steps} onChange={event => setSteps(Number(event.target.value))}/></label><label>Seed<input type="number" min={0} value={seed} onChange={event => setSeed(Number(event.target.value))}/></label></div>
          <label>Learning rate<input type="number" step={0.001} min={0.00001} max={1} value={learningRate} onChange={event => setLearningRate(Number(event.target.value))}/></label>
          <button disabled={busy !== null || !models.length} onClick={() => void create()}><FlaskConical size={16}/>Create experiment</button>
        </div>
        <div className="remote-card"><div className="remote-title"><h3>Server runs</h3><button disabled={busy !== null} aria-label="Refresh server runs" onClick={() => void perform('refresh', async () => { const value = await client.current!.runs(); setRuns(value); })}><RefreshCw size={15}/></button></div>
          <div className="remote-run-list">{runs.length === 0 ? <p>No server runs yet.</p> : runs.map(run => <button key={identifier(run)} className={selected && identifier(selected) === identifier(run) ? 'selected' : ''} onClick={() => void selectRun(run)}><span>{run.config?.name || String(run.name || identifier(run).slice(0, 12))}</span><small>{run.state || run.status || 'created'}</small></button>)}</div>
        </div>
      </div>
      {selected && <div className="remote-card remote-active">
        <div className="remote-title"><div><h3>{selected.config?.name || 'Selected server experiment'}</h3><small>{identifier(selected)}</small></div><span className="remote-status">{state}</span></div>
        <div className="remote-evidence"><span>Step: <strong>{Number(selected.step || latest?.step || 0).toLocaleString()}</strong></span><span>Loss: <strong>{latest ? latest.loss.toFixed(5) : 'not captured'}</strong></span><span>Telemetry: <strong>{streamState}</strong></span><span>Missing sequences: <strong>{captureGaps}</strong></span></div>
        <div className="remote-actions">
          <button disabled={busy !== null || state !== 'created'} onClick={() => void command('start')}><Play size={15}/>Start</button>
          <button disabled={busy !== null || state !== 'running'} onClick={() => void command('pause')}><Pause size={15}/>Pause</button>
          <button disabled={busy !== null || state !== 'paused'} onClick={() => void command('resume')}><Play size={15}/>Resume</button>
          <button disabled={busy !== null || !['created', 'paused'].includes(state)} onClick={() => void command('step')}>One step</button>
          <button disabled={busy !== null || ['completed', 'stopped', 'failed'].includes(state)} onClick={() => void command('stop')}><Square size={15}/>Stop</button>
          <button disabled={busy !== null} onClick={() => void command('checkpoint')}>Checkpoint</button>
        </div>
        <div className="remote-chart" aria-label="Actual backend loss history">{metrics.length > 1 ? <ResponsiveContainer width="100%" height={200}><LineChart data={metrics}><CartesianGrid stroke="#1b2b40" strokeDasharray="3 3"/><XAxis dataKey="step" stroke="#8494a9"/><YAxis stroke="#8494a9"/><Tooltip contentStyle={{ background: '#0d1727', border: '1px solid #223550' }}/><Line type="monotone" dataKey="loss" dot={false} stroke="#43d9e3" strokeWidth={2} isAnimationActive={false}/></LineChart></ResponsiveContainer> : <p>Loss curves appear after actual training steps are captured.</p>}</div>
        <div className="remote-actions"><button disabled={busy !== null} onClick={() => void inspect('inference')}>Run inference</button><button disabled={busy !== null} onClick={() => void inspect('tensors')}>Inspect tensors</button><button disabled={busy !== null} onClick={() => void inspect('attribution')}>Integrated gradients</button><button disabled={busy !== null} onClick={() => void inspect('export')}><Download size={15}/>Export run</button></div>
        <details className="remote-details"><summary>Controlled activation intervention</summary><p>Run a baseline and a scaled-activation comparison on an isolated inference copy. A score change is evidence about this experiment, not a complete explanation of the model.</p><div className="remote-fields"><label>Module<input value={module} onChange={event => setModule(event.target.value)}/></label><label>Scale<input type="number" min={-100} max={100} step={0.1} value={strength} onChange={event => setStrength(Number(event.target.value))}/></label></div><button disabled={busy !== null} onClick={() => void intervene()}>Compare against baseline</button></details>
      </div>}
      {response !== null && <div className="remote-card"><div className="remote-title"><h3>{responseTitle}</h3><button onClick={() => saveJson('neuralis-server-result.json', printable(response))}><Download size={15}/>Save JSON</button></div><pre className="remote-result">{printable(response)}</pre></div>}
      {events.length > 0 && <div className="remote-card"><h3>Measured event stream</h3><div className="remote-event-list">{events.slice(0, 20).map(event => <div key={event.sequence}><code>#{event.sequence}</code><span>{event.type}</span><span>step {event.step}</span><small>rank {event.rank ?? 0}</small></div>)}</div></div>}
    </>}
  </section>;
}
