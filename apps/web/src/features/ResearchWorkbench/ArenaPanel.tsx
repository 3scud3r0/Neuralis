import { useEffect, useRef, useState } from 'react';
import { Download, Flag, Gamepad2, Pause, Play, SkipBack, SkipForward, Upload } from 'lucide-react';
import { GRID, decodeEpisode, episodeReturn, trainArena, type ArenaEpisode } from './arenaEngine';
import { LiveStreamView } from '../RLArena/LiveStreamView';
import { TimeTravelDebugger } from '../RLArena/TimeTravelDebugger';
import { DownloadButton, ErrorNotice, HorizontalBars, PanelTitle, Stat, errorMessage, type ExtensionContext } from './shared';

export function ArenaPanel({ client, capabilities }: ExtensionContext) {
  const [episode, setEpisode] = useState<ArenaEpisode>(() => trainArena(42, 0));
  const [seed, setSeed] = useState(42);
  const [transitions, setTransitions] = useState(25000);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(300);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const frame = episode.frames[Math.min(cursor, episode.frames.length - 1)];
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => setCursor(value => { if (value >= episode.frames.length - 1) { setPlaying(false); return value; } return value + 1; }), speed);
    return () => clearInterval(timer);
  }, [playing, episode, speed]);
  const train = () => {
    setBusy(true); setError(null); setPlaying(false);
    // Yield once to paint the bounded computation status before CPU work.
    setTimeout(() => {
      try { setEpisode(trainArena(seed, transitions)); setCursor(0); }
      catch (failure) { setError(errorMessage(failure)); } finally { setBusy(false); }
    }, 0);
  };
  const importEpisode = async (file: File) => {
    setPlaying(false);
    try { if (file.size > 8_000_000) throw new Error('Trajectory file exceeds 8 MB.'); const next = decodeEpisode(await file.text()); setEpisode(next); setCursor(0); setError(null); }
    catch (failure) { setError(errorMessage(failure)); }
  };
  return <div className="rw-panel">
    <PanelTitle eyebrow="03 / REINFORCEMENT ARENA" title="Follow every action. Inspect every reward.">Train a real tabular Q-learning policy in a bounded gridworld, then scrub its greedy trajectory alongside action values.</PanelTitle>
    <div className="rw-banner"><Gamepad2 size={18}/><span>This browser arena runs gridworld. Super Mario requires the separate local emulator project and its dependencies. No emulator or live Mario stream is running here.</span></div>
    <ErrorNotice error={error}/>
    <div className="rw-stats"><Stat label="Training transitions" value={episode.training?.transitions.toLocaleString() ?? 'Imported'} note="Actual environment updates"/><Stat label="Greedy goal reach rate" value={episode.training ? `${(episode.training.successRate * 100).toFixed(1)}%` : '—'} note="All nonterminal open starting cells"/><Stat label="Trajectory return" value={episodeReturn(episode).toFixed(3)} note={`${episode.frames.filter(item => item.action >= 0).length} actions from start cell 0`}/><Stat label="Bellman residual" value={episode.training?.bellmanResidual.toExponential(2) ?? '—'} note="Mean squared error across open actions"/></div>
    <div className="rw-arena-grid"><div className="rw-card rw-stage"><div className="rw-card-heading"><h3>{episode.environment}</h3><span className="rw-chip">Recorded rollout · frame {cursor + 1}/{episode.frames.length}</span></div>{frame.image ? <img className="rw-imported-frame" src={frame.image} alt={`Imported trajectory frame ${frame.step}`}/> : <div className="rw-gridworld" role="img" aria-label={`Gridworld position ${frame.state}, goal 35, hazard 20`}>{Array.from({ length: 36 }, (_, state) => <div className={`rw-grid-cell ${GRID.walls.includes(state as never) ? 'wall' : ''} ${state === GRID.goal ? 'goal' : ''} ${state === GRID.hazard ? 'hazard' : ''} ${state === frame.state ? 'agent' : ''}`} key={state}>{state === frame.state ? <Gamepad2 size={24}/> : state === GRID.goal ? <Flag size={21}/> : state === GRID.hazard ? '×' : GRID.walls.includes(state as never) ? '' : <small>{state}</small>}</div>)}</div>}<div className="rw-scrubber"><input aria-label="Trajectory frame" type="range" min={0} max={episode.frames.length - 1} value={cursor} onChange={event => { setCursor(Number(event.target.value)); setPlaying(false); }}/><div className="rw-actions"><button aria-label="First frame" onClick={() => { setCursor(0); setPlaying(false); }}><SkipBack size={16}/></button><button className="rw-primary" onClick={() => { if (cursor === episode.frames.length - 1) setCursor(0); setPlaying(value => !value); }}>{playing ? <Pause size={16}/> : <Play size={16}/>} {playing ? 'Pause replay' : 'Play replay'}</button><button aria-label="Next frame" disabled={cursor === episode.frames.length - 1} onClick={() => setCursor(value => Math.min(value + 1, episode.frames.length - 1))}><SkipForward size={16}/></button><label>Frame interval<select value={speed} onChange={event => setSpeed(Number(event.target.value))}><option value={100}>100 ms</option><option value={300}>300 ms</option><option value={600}>600 ms</option></select></label></div></div><div className="rw-legend"><span className="steered">Agent</span><span className="target">Goal +1</span><span className="danger">Hazard −1</span><span>Step −0.025 · collision −0.05</span></div></div>
      <div className="rw-card"><h3>Synchronized policy evidence</h3><div className="rw-live-readout"><div><span>State</span><strong>{frame.state}</strong></div><div><span>Selected action</span><strong>{frame.action < 0 ? 'Terminal' : GRID.actions[frame.action]}</strong></div><div><span>Immediate reward</span><strong>{frame.reward.toFixed(3)}</strong></div><div><span>Cumulative return</span><strong>{episodeReturn(episode, cursor).toFixed(3)}</strong></div></div><h4>Action values at this state</h4><HorizontalBars values={frame.activations} labels={[...GRID.actions]}/><p className="rw-note">These four values are a tabular Q function, not hidden neural activations. Greedy replay selects the largest action value; training uses exploration.</p><div className="rw-fields"><label>Seed<input type="number" value={seed} onChange={event => setSeed(Number(event.target.value))}/></label><label>Training transitions<select value={transitions} onChange={event => setTransitions(Number(event.target.value))}>{[0, 5000, 25000, 50000].map(value => <option key={value} value={value}>{value.toLocaleString()}</option>)}</select></label></div><button className="rw-primary" disabled={busy} onClick={train}><Play size={15}/>{busy ? 'Training policy…' : 'Train and record rollout'}</button><div className="rw-actions"><button onClick={() => fileInput.current?.click()}><Upload size={15}/>Import trajectory</button><DownloadButton filename="neuralis-gridworld-episode.json" data={episode} label="Export episode"/><input hidden ref={fileInput} type="file" accept=".json,application/json" onChange={event => { const file = event.target.files?.[0]; if (file) void importEpisode(file); event.target.value = ''; }}/></div></div></div>
    <div className="rw-card"><div className="rw-card-heading"><h3>Trajectory log</h3><span className="rw-chip">Seed {episode.seed}</span></div><div className="rw-trajectory"><table><thead><tr><th>Frame</th><th>State</th><th>Action</th><th>Reward</th><th>Terminal</th></tr></thead><tbody>{episode.frames.slice(Math.max(0, cursor - 50), Math.max(0, cursor - 50) + 100).map((item) => <tr key={item.step} className={item.step === cursor ? 'selected' : ''} onClick={() => { setCursor(item.step); setPlaying(false); }}><td><button aria-label={`Select frame ${item.step + 1}`}>{item.step}</button></td><td>{item.state}</td><td>{item.action < 0 ? '—' : GRID.actions[item.action]}</td><td>{item.reward.toFixed(3)}</td><td>{item.done ? 'Yes' : 'No'}</td></tr>)}</tbody></table></div><p className="rw-note">The log renders at most 100 frames around the selected frame. Trajectory imports use neuralis.arena/v1 with grid states 0–35, action indices 0–3 (−1 for a terminal display), reward, done and four action values. Optional embedded PNG/JPEG/WebP frames are displayed alongside that evidence.</p></div>
    <div className="rw-banner"><Download size={18}/><span>Live WebRTC capability: {capabilities?.webrtc.available ? 'Advertised by backend. Use the captured video panel to negotiate a session explicitly.' : capabilities?.webrtc.reason ?? 'No connected backend advertises a live stream. Offline trajectory replay is available.'}</span></div>
    <LiveStreamView client={client}/>
    <TimeTravelDebugger client={client}/>
  </div>;
}
