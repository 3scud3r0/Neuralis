import { useRef, useState } from 'react';
import { Cpu, Server, ShieldCheck, Laptop, Download, Upload, ExternalLink as LinkIcon, Monitor, Database, RotateCcw, CheckCircle2 } from 'lucide-react';
import { useStudio } from '../app/StudioContext';
import { DesktopRuntime } from './DesktopRuntime';
import { BackendWorkbench } from '../integrations/BackendWorkbench';
import { downloadText } from '../app/storage';
import { Panel, PageHeader, Badge, Button, Toggle, Field, Callout, KeyValue, ExternalLink } from '../components/ui';

export function Laboratory() {
  const { settings, updateSettings } = useStudio();
  return <>
    <PageHeader eyebrow="RESEARCH INFRASTRUCTURE" title="From your browser to your own compute." description="The local browser engine is always available. Connect optional infrastructure when you need it." />
    <div className="infrastructure-grid"><Panel title="Browser engine" action={<Badge tone="green" dot>Available</Badge>}><div className="infrastructure-icon"><Cpu size={29} /></div><h3>Private, immediate computation</h3><p>Small models execute in this browser with inspectable parameters, seeded data, and genuine optimizer updates.</p><div className="info-stack"><KeyValue label="Compute" value="Local CPU" /><KeyValue label="Remote dependency" value="None" /><KeyValue label="Persistence" value="Browser local storage" /></div></Panel><Panel title="Python research backend" action={<Badge dot>Optional</Badge>}><div className="infrastructure-icon violet"><Server size={29} /></div><h3>Your models, your hardware</h3><p>Run the included backend on your own machine to instrument PyTorch models and expose model operations through its API.</p><ExternalLink href="https://github.com/3scud3r0/Neuralis/tree/main/services/api">Backend source and setup</ExternalLink></Panel><Panel title="Desktop workspace" action={<Badge tone="violet">Tauri</Badge>}><div className="infrastructure-icon green"><Laptop size={29} /></div><h3>A dedicated research window</h3><p>The desktop shell hosts this interface with local runtime support. Browser demos remain available without a backend.</p><ExternalLink href="https://github.com/3scud3r0/Neuralis/tree/main/apps/desktop">Desktop build instructions</ExternalLink></Panel></div>
    <BackendWorkbench initialUrl={settings.backendUrl} onUrlChange={url => updateSettings({ backendUrl: url })} />
    <Callout>GitHub Pages serves a static frontend. A local API must permit the frontend origin through its CORS configuration. HTTPS pages may block HTTP endpoints; use a local frontend or an HTTPS backend when required by your browser.</Callout>
  </>;
}

export function Settings() {
  const { settings, updateSettings, config, setConfig, history, checkpoints, importCheckpoint, notify } = useStudio();
  const [urlDraft, setUrlDraft] = useState(settings.backendUrl);
  const importRef = useRef<HTMLInputElement>(null);
  const [saved, setSaved] = useState(false);
  const exportWorkspace = () => {
    downloadText('neuralis-workspace.json', JSON.stringify({ schemaVersion: 1, exportedAt: new Date().toISOString(), settings, config, history, checkpoints }, null, 2));
    notify('Workspace exported. The file contains local model data and settings.', 'success');
  };
  const saveEndpoint = () => {
    try {
      const url = new URL(urlDraft);
      if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Unsupported protocol');
      updateSettings({ backendUrl: url.toString().replace(/\/$/, '') });
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
      notify('Backend endpoint saved. Connection is established from Infrastructure.', 'success');
    } catch { notify('Enter a valid HTTP or HTTPS backend URL.', 'error'); }
  };
  return <>
    <PageHeader eyebrow="WORKSPACE SETTINGS" title="A workspace that works like you do." description="Local preferences, execution defaults, and portable model artifacts." />
    <div className="settings-layout"><div><Panel title="Appearance and accessibility" subtitle="Saved only in this browser" action={<Monitor size={17} />}><div className="settings-section"><Toggle checked={settings.reducedMotion} onChange={reducedMotion => updateSettings({ reducedMotion })} label="Reduce visual motion" detail="Disable ambient scene motion and interface transitions" /><Toggle checked={settings.dense} onChange={dense => updateSettings({ dense })} label="Compact interface" detail="Reduce spacing for information-dense research sessions" /></div></Panel><Panel title="Execution defaults" subtitle="CPU-first and bounded by design" action={<Cpu size={17} />}><div className="settings-section"><Field label="Default random seed" hint="Changing the seed reinitializes the model deterministically"><div className="inline-field"><input type="number" value={config.seed} min={0} max={2147483647} onChange={event => { const next = Number(event.target.value); if (Number.isInteger(next) && next >= 0) setConfig({ seed: next }); }} /><Badge tone="cyan">Reproducible</Badge></div></Field><Field label="Capture limit" hint="Maximum preferred captured items for future remote sessions"><input type="number" min={10} max={1000} step={10} value={settings.captureLimit} onChange={event => updateSettings({ captureLimit: Math.max(10, Math.min(1000, Number(event.target.value))) })} /></Field><Callout>Browser demo engines apply their own fixed capture bounds to keep tensor rendering responsive. Larger models should use a bounded backend instrumentation plan.</Callout></div></Panel><Panel title="Remote backend" subtitle="No credentials are stored in these settings" action={<LinkIcon size={17} />}><div className="settings-section"><Field label="API endpoint"><div className="inline-field"><input type="url" value={urlDraft} onChange={event => setUrlDraft(event.target.value)} placeholder="http://127.0.0.1:8000" /><Button onClick={saveEndpoint}>{saved ? <CheckCircle2 size={14} /> : <Server size={14} />}{saved ? 'Saved' : 'Save'}</Button></div></Field><p className="muted">Configure trusted origins and authentication on your backend. Use Infrastructure to test the connection and run actual API operations.</p></div></Panel></div><aside><DesktopRuntime /><Panel title="Workspace portability" action={<Database size={17} />}><div className="control-stack"><p className="muted">Export your local run history, preferences, and checkpoints as a JSON archive.</p><KeyValue label="Recorded runs" value={history.length} /><KeyValue label="Saved checkpoints" value={checkpoints.length} /><Button variant="primary" onClick={exportWorkspace}><Download size={14} />Export workspace</Button><Button onClick={() => importRef.current?.click()}><Upload size={14} />Import checkpoint</Button><input type="file" accept=".json,application/json" hidden ref={importRef} onChange={async event => { const file = event.target.files?.[0]; if (file) importCheckpoint(await file.text()); event.target.value = ''; }} /></div></Panel><Panel title="Your data stays local"><div className="privacy-card"><ShieldCheck size={30} /><h3>Built for private exploration</h3><p>Browser demos execute on this device. No analytics or hosted model calls are required.</p><p>Browser storage can be cleared by your browser. Export checkpoints you want to keep.</p></div></Panel><Button variant="ghost" onClick={() => { updateSettings({ reducedMotion: false, dense: false, captureLimit: 100, backendUrl: 'http://127.0.0.1:8000' }); setUrlDraft('http://127.0.0.1:8000'); notify('Workspace preferences reset. Runs and checkpoints were retained.'); }}><RotateCcw size={14} />Reset preferences</Button></aside></div>
  </>;
}
