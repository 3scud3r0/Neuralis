import { useEffect, useState } from 'react';
import { desktop, isDesktop, type NativePreferences, type NativeStatus } from '../../../desktop/bridge';
import { Laptop, Play, Square, RefreshCw, Save, FolderOpen, Upload, Download } from 'lucide-react';
import { useStudio } from '../app/StudioContext';
import { Panel, Field, Select, Button, Badge, KeyValue, Callout } from '../components/ui';

export function DesktopRuntime() {
  const { engine, snapshot, importCheckpoint, notify } = useStudio();
  const [preferences, setPreferences] = useState<NativePreferences | null>(null);
  const [status, setStatus] = useState<NativeStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [diagnostics, setDiagnostics] = useState<string[]>([]);
  const available = isDesktop();
  const run = async (operation: () => Promise<void>) => {
    setBusy(true);
    try { await operation(); }
    catch (error) { notify(String(error), 'error'); }
    finally { setBusy(false); }
  };
  useEffect(() => {
    if (!available) return;
    void Promise.all([desktop.preferences(), desktop.status()]).then(([prefs, current]) => {
      setPreferences(prefs); setStatus(current);
    }).catch(error => notify(`Native runtime: ${String(error)}`, 'error'));
  }, [available, notify]);
  if (!available) return <Panel title="Desktop runtime" action={<Badge>Browser session</Badge>}><div className="settings-section"><Laptop size={27} className="cyan-text" /><p className="muted">Native process supervision and local filesystem operations are available in the installed Neuralis desktop application. Browser model training and JSON exports remain available here.</p></div></Panel>;
  return <Panel title="Native desktop runtime" action={<Badge tone={status?.phase === 'ready' ? 'green' : 'neutral'} dot>{status?.phase ?? 'Loading'}</Badge>}><div className="control-stack">
    {preferences && <><Field label="Runtime mode"><Select label="Native runtime mode" value={preferences.mode} onChange={mode => setPreferences(current => current ? { ...current, mode: mode as NativePreferences['mode'] } : current)} options={[{ value: 'browser_only', label: 'Browser demos only' }, { value: 'local', label: 'Supervised local Python backend' }, { value: 'remote', label: 'Connect to a remote backend' }]} /></Field>{preferences.mode === 'local' && <><Field label="Python executable" hint="Select the Python executable from your prepared backend environment"><input value={preferences.python_executable} onChange={event => setPreferences({ ...preferences, python_executable: event.target.value })} /></Field><Field label="Repository root"><input value={preferences.repository_root ?? ''} onChange={event => setPreferences({ ...preferences, repository_root: event.target.value || null })} placeholder="/path/to/Neuralis" /></Field></>}{preferences.mode === 'remote' && <Field label="Remote backend endpoint"><input type="url" value={preferences.remote_endpoint ?? ''} onChange={event => setPreferences({ ...preferences, remote_endpoint: event.target.value || null })} placeholder="https://your-backend.example" /></Field>}<Button disabled={busy} onClick={() => void run(async () => { await desktop.savePreferences(preferences); notify('Native preferences saved.', 'success'); })}><Save size={14} />Save native preferences</Button></>}
    {status && <><KeyValue label="Managed process" value={status.managed ? 'Yes' : 'No'} /><KeyValue label="Endpoint" value={status.endpoint ?? 'Disconnected'} /><KeyValue label="Process ID" value={status.process_id ?? '—'} /><p className="muted">{status.message}</p></>}
    <div className="inline-actions"><Button disabled={busy} variant="primary" onClick={() => void run(async () => { setStatus(await desktop.start()); })}><Play size={13} />Start backend</Button><Button disabled={busy} onClick={() => void run(async () => { await desktop.stop(); setStatus(await desktop.status()); })}><Square size={13} />Stop</Button><Button disabled={busy} variant="ghost" onClick={() => void run(async () => setStatus(await desktop.status()))} title="Refresh native status"><RefreshCw size={14} /></Button></div>
    <div className="inline-actions"><Button disabled={busy} onClick={() => void run(async () => { const directory = await desktop.selectExportFolder(); if (!directory) return; const receipt = await desktop.exportCheckpoint(directory, `neuralis-${snapshot.demoId}-${snapshot.step}.json`, engine.exportCheckpoint()); notify(`Exported ${receipt.bytes} bytes to ${receipt.path}`, 'success'); })}><Download size={13} />Native export</Button><Button disabled={busy} onClick={() => void run(async () => { const path = await desktop.selectCheckpoint(); if (!path) return; const [text] = await desktop.importCheckpoint(path); importCheckpoint(text); })}><Upload size={13} />Native import</Button></div>
    <Button disabled={busy} variant="ghost" onClick={() => void run(async () => setDiagnostics(await desktop.diagnostics()))}><FolderOpen size={14} />Read runtime diagnostics</Button>
    {diagnostics.length > 0 && <pre className="native-diagnostics">{diagnostics.join('\n')}</pre>}
    <Callout>The desktop supervisor launches only the Python process you configure. Review your executable and repository path before starting.</Callout>
  </div></Panel>;
}
