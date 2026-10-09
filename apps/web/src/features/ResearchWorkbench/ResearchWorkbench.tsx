import { useState } from 'react';
import { desktop, isDesktop } from '../../../../desktop/bridge';
import { ArrowUpRight, Boxes, Cpu, Database, FlaskConical, Gamepad2, LibraryBig, LoaderCircle, Package, Plug, Unplug, Wifi } from 'lucide-react';
import { readStored, writeStored, type WorkspaceSettings } from '../../app/storage';
import { AxisPanel } from './AxisPanel';
import { ArenaPanel } from './ArenaPanel';
import { ExportPanel } from './ExportPanel';
import { FoundryPanel } from './FoundryPanel';
import { RagPanel } from './RagPanel';
import { TrainingPanel } from './TrainingPanel';
import { ExtensionClient, type ExtensionCapabilities } from './extensionClient';
import { ErrorNotice, ResultJson, errorMessage } from './shared';
import './researchWorkbench.css';

type Tab = 'axis' | 'training' | 'arena' | 'foundry' | 'rag' | 'export';
const tabs = [
  { id: 'axis', name: 'AxisLab', icon: FlaskConical, description: 'Concept directions' },
  { id: 'training', name: 'Efficient training', icon: Cpu, description: 'NF4 & low-rank adapters' },
  { id: 'arena', name: 'RL Arena', icon: Gamepad2, description: 'Policy trajectories' },
  { id: 'foundry', name: 'Data Foundry', icon: Database, description: 'Verifiable datasets' },
  { id: 'rag', name: 'Local RAG', icon: LibraryBig, description: 'Source-grounded retrieval' },
  { id: 'export', name: 'Export', icon: Package, description: 'GGUF & Ollama recipes' },
] as const;
interface Props { initialUrl?: string; onUrlChange?: (url: string) => void; initialTab?: Tab; }
export function ResearchWorkbench({ initialUrl, onUrlChange, initialTab = 'axis' }: Props) {
  const [tab, setTab] = useState<Tab>(initialTab);
  const [url, setUrl] = useState(initialUrl ?? readStored<Partial<WorkspaceSettings>>('settings', {}).backendUrl ?? 'http://127.0.0.1:8000');
  const [token, setToken] = useState('');
  const [client, setClient] = useState<ExtensionClient | null>(null);
  const [capabilities, setCapabilities] = useState<ExtensionCapabilities | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showConnection, setShowConnection] = useState(false);
  const connect = async (useNative = false) => {
    setBusy(true); setError(null); setClient(null); setCapabilities(null);
    try {
      let endpoint = url;
      if (useNative) {
        setToken('');
        const status = await desktop.status();
        if (status.phase !== 'ready' || !status.endpoint) throw new Error('The desktop runtime is not ready. Start or configure it in Desktop runtime settings, then try again.');
        endpoint = status.endpoint;
      }
      const next = new ExtensionClient(endpoint, useNative ? '' : token, fetch, useNative);
      const response = await next.capabilities();
      for (const key of ['axislab', 'nf4', 'lora', 'rag', 'webrtc', 'gguf'] as const) if (typeof response?.[key]?.available !== 'boolean') throw new Error(`The backend extension capability ${key} is malformed.`);
      setClient(next); setCapabilities(response); setUrl(next.baseUrl);
      const settings = readStored<Partial<WorkspaceSettings>>('settings', {});
      writeStored('settings', { ...settings, backendUrl: next.baseUrl });
      onUrlChange?.(next.baseUrl);
    } catch (failure) { setError(errorMessage(failure)); } finally { setBusy(false); }
  };
  const context = { client, capabilities };
  return <section className="research-workbench" aria-label="Research extensions workbench">
    <header className="rw-header"><div className="rw-header-icon"><Boxes size={25}/></div><div><span className="rw-eyebrow">NEURALIS / RESEARCH EXTENSIONS</span><h1>Research workbench</h1><p>From concept vectors to portable experiments.</p></div><div className="rw-header-actions"><span className="rw-chip good"><span className="rw-status-dot"/>Browser engines ready</span><button className={client ? 'rw-connected' : ''} onClick={() => setShowConnection(value => !value)}><Wifi size={15}/>{client?.useNative ? 'Desktop runtime connected' : client ? 'Backend connected' : 'Connect backend'}<ArrowUpRight size={14}/></button></div></header>
    {showConnection && <div className="rw-card rw-connection"><div className="rw-card-heading"><h3><Plug size={17}/>Optional extension backend</h3><span className="rw-chip">{client?.useNative ? 'Authentication managed by desktop runtime' : 'Token stays in this page session'}</span></div><p>Browser experiments work on GitHub Pages. Connect your own backend for installed AxisLab, CPU adaptation, encrypted retrieval and export capabilities.</p><div className="rw-actions"><label className="rw-url">Backend URL<input aria-label="Extension backend URL" type="url" value={url} readOnly={client?.useNative} onChange={event => setUrl(event.target.value)}/></label>{!client?.useNative && <label>Session bearer token<input aria-label="Extension session token" type="password" autoComplete="off" value={token} onChange={event => setToken(event.target.value)} placeholder="Optional for public capabilities"/></label>}<button className="rw-primary" disabled={busy || client?.useNative} onClick={() => void connect()}>{busy ? <LoaderCircle className="rw-spin" size={15}/> : <Plug size={15}/>}Check capabilities</button>{isDesktop() && <button disabled={busy} onClick={() => void connect(true)}><Cpu size={15}/>Use desktop runtime</button>}{client && <button onClick={() => { setClient(null); setCapabilities(null); setToken(''); setError(null); }}><Unplug size={15}/>Disconnect</button>}</div>{client?.useNative && <p className="rw-note">Requests use native IPC and the configured runtime session. Runtime credentials are kept outside JavaScript; the bearer token field is not used.</p>}<ErrorNotice error={error}/>{capabilities && <><div className="rw-capabilities">{(['axislab', 'nf4', 'lora', 'rag', 'webrtc', 'gguf'] as const).map(key => <span key={key} className={`rw-chip ${capabilities[key].available ? 'good' : ''}`} title={capabilities[key].reason}>{key.toUpperCase()} · {capabilities[key].available ? 'available' : 'unavailable'}</span>)}</div><ResultJson value={capabilities} label="Backend capability evidence"/></>}</div>}
    <nav className="rw-tabs" aria-label="Research modules" role="tablist">{tabs.map(item => <button role="tab" aria-selected={tab === item.id} aria-controls={`research-${item.id}`} id={`tab-${item.id}`} key={item.id} className={tab === item.id ? 'selected' : ''} onClick={() => setTab(item.id)}><item.icon size={18}/><span><strong>{item.name}</strong><small>{item.description}</small></span></button>)}</nav>
    <div id={`research-${tab}`} aria-labelledby={`tab-${tab}`} role="tabpanel" className="rw-tab-content">{tab === 'axis' && <AxisPanel {...context}/>} {tab === 'training' && <TrainingPanel {...context}/>} {tab === 'arena' && <ArenaPanel {...context}/>} {tab === 'foundry' && <FoundryPanel/>} {tab === 'rag' && <RagPanel {...context}/>} {tab === 'export' && <ExportPanel {...context}/>}</div>
  </section>;
}
export default ResearchWorkbench;
