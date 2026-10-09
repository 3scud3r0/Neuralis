import { useEffect, useRef, useState } from 'react';
import { Download, GitBranch, RefreshCw, SkipBack, SkipForward } from 'lucide-react';
import { decodeStoredFrame, episodePath, policyBars, type ArenaClient, type EpisodeSummary, type StoredFrame } from './client';
import './arena.css';

export function TimeTravelDebugger({client}: {client: ArenaClient | null}) {
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [episode, setEpisode] = useState('');
  const [step, setStep] = useState(0);
  const [frame, setFrame] = useState<StoredFrame | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [action, setAction] = useState(1);
  const [branch, setBranch] = useState<unknown>(null);
  const cache = useRef(new Map<string, StoredFrame>());
  const sequence = useRef(0);
  const selected = episodes.find(item => item.episode_id === episode);
  useEffect(() => {setEpisodes([]);setEpisode('');setFrame(null);cache.current.clear();sequence.current++;}, [client]);
  useEffect(() => {
    const version = ++sequence.current;
    if (!client || !episode) return;
    const key = `${episode}:${step}`;
    const cached = cache.current.get(key);
    if (cached) {setFrame(cached);setBusy(false);return;}
    setBusy(true);setError('');
    void client.request<unknown>(`${episodePath(episode)}/records/${step}`).then(value => {
      const decoded = decodeStoredFrame(value);
      if (version !== sequence.current) return;
      if (cache.current.size >= 8) cache.current.delete(cache.current.keys().next().value!);
      cache.current.set(key, decoded);setFrame(decoded);
    }).catch(failure => {if (version === sequence.current) setError(failure instanceof Error ? failure.message : String(failure));})
      .finally(() => {if (version === sequence.current) setBusy(false);});
    return () => {sequence.current++;};
  }, [client, episode, step]);
  const refresh = async () => {
    if (!client) return;
    setError('');setBusy(true);
    try {
      const rows = await client.request<EpisodeSummary[]>('arena/episodes');
      if (!Array.isArray(rows)) throw new Error('Invalid recorded episode list.');
      setEpisodes(rows);
      if (!rows.some(item => item.episode_id === episode)) {setEpisode(rows[0]?.episode_id ?? '');setStep(rows[0]?.first_step ?? 0);}
    } catch (failure) {setError(failure instanceof Error ? failure.message : String(failure));}
    finally {setBusy(false);}
  };
  const replay = async () => {
    if (!client || !frame) return;
    setError('');setBusy(true);
    try {setBranch(await client.request(`${episodePath(episode)}/branch`, {step: frame.step, actions: [action]}));}
    catch (failure) {setError(failure instanceof Error ? failure.message : String(failure));}
    finally {setBusy(false);}
  };
  const exportRecord = () => {
    if (!frame) return;
    const link = document.createElement('a');
    const url = URL.createObjectURL(new Blob([JSON.stringify(frame, null, 2)], {type: 'application/json'}));
    link.href = url;link.download = `${episode}-frame-${frame.step}.json`;link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const canReplay = frame?.has_snapshot && frame.metadata.environment === 'deterministic-gridworld';
  return <section className="arena-runtime-card">
    <div className="arena-runtime-heading"><div><span className="eyebrow">PERSISTED FRAME EVIDENCE</span><h3>Temporal debugger</h3></div><span className="arena-runtime-badge">Exact stored observations</span></div>
    <p>Seek an observed frame and inspect the action values, reward and recorder metadata captured alongside its pixels.</p>
    <div className="arena-runtime-controls"><button disabled={!client || busy} onClick={() => void refresh()}><RefreshCw size={14}/>Load recorded episodes</button><select aria-label="Recorded episode" value={episode} onChange={event => {setEpisode(event.target.value);setStep(0);setFrame(null);setBranch(null);}}><option value="">Select an episode</option>{episodes.map(item => <option key={item.episode_id} value={item.episode_id}>{item.episode_id} · {item.steps} frames · return {item.return.toFixed(2)}</option>)}</select><button disabled={!frame} onClick={exportRecord}><Download size={14}/>Export frame evidence</button></div>
    {selected && <div className="arena-timeline"><button aria-label="Previous recorded frame" disabled={step <= selected.first_step} onClick={() => setStep(value => value - 1)}><SkipBack size={15}/></button><input aria-label="Stored frame timeline" type="range" min={selected.first_step} max={selected.last_step} value={step} onChange={event => setStep(Number(event.target.value))}/><button aria-label="Next recorded frame" disabled={step >= selected.last_step} onClick={() => setStep(value => value + 1)}><SkipForward size={15}/></button><span>Frame {step} / {selected.last_step}</span></div>}
    {busy && <p className="arena-runtime-note" role="status">Reading captured evidence…</p>}
    {frame && <div className="arena-evidence-grid"><div><img className="arena-recorded-image" src={`data:image/png;base64,${frame.frame_png}`} alt={`Actual stored frame ${frame.step} of ${frame.episode_id}`}/><div className="arena-evidence-readout"><span>Action <strong>{frame.action}</strong></span><span>Reward <strong>{frame.reward.toFixed(4)}</strong></span><span>Terminal <strong>{frame.terminated ? 'Yes' : 'No'}</strong></span><span>Snapshot <strong>{frame.has_snapshot ? 'Available' : 'Not recorded'}</strong></span></div></div><div><h4>Captured action values</h4>{policyBars(frame.policy).slice(0, 32).map((item, index) => <div className={`arena-policy-row ${index === frame.action ? 'selected' : ''}`} key={index}><span>{index}</span><div><i style={{width: `${item.fraction * 100}%`}}/></div><code>{item.value.toFixed(5)}</code></div>)}<p className="arena-runtime-note">Kind: {String(frame.metadata.policy_kind ?? frame.metadata.policy ?? 'Caller-supplied policy output')}. Values keep their original units.</p><details><summary>Recorder metadata</summary><pre>{JSON.stringify(frame.metadata, null, 2)}</pre></details></div></div>}
    {!frame && <p className="arena-runtime-empty">Load an episode from the local API. Actual Mario recordings are created by demos/super-mario/arena.py.</p>}
    <div className="arena-runtime-controls"><label>Branch action<select aria-label="Replay branch action" value={action} onChange={event => setAction(Number(event.target.value))}>{['Up', 'Right', 'Down', 'Left'].map((name, index) => <option value={index} key={name}>{name}</option>)}</select></label><button disabled={!canReplay || busy} onClick={() => void replay()}><GitBranch size={14}/>Replay from captured gridworld state</button></div>
    <p className="arena-runtime-note">State replay is available for the supplied gridworld adapter. Mario observations do not include emulator snapshots; altering its policy cannot reconstruct an unrecorded emulator state. Policy evidence and supplied public rationales do not reveal hidden thoughts.</p>
    {branch !== null && <details open><summary>Actual branch result</summary><pre>{JSON.stringify(branch, null, 2)}</pre></details>}
    {error && <p role="alert" className="arena-runtime-error">{error}</p>}
  </section>;
}
