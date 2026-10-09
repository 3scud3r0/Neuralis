import { useMemo, useState } from 'react';
import { ArrowUpRight, FlaskConical, Play, RotateCcw } from 'lucide-react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { decodePairs, evaluateAxis, examplePairs, fitAxis, steerActivation, type ActivationPair } from './axisEngine';
import type { AxisResponse } from './extensionClient';
import { BackendButton, DownloadButton, ErrorNotice, HorizontalBars, PanelTitle, ResultJson, Stat, errorMessage, type ExtensionContext } from './shared';

export function AxisPanel({ client, capabilities }: ExtensionContext) {
  const [source, setSource] = useState(JSON.stringify(examplePairs, null, 2));
  const [pairs, setPairs] = useState<ActivationPair[]>(examplePairs);
  const [strength, setStrength] = useState(0.8);
  const [sample, setSample] = useState(3);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [remote, setRemote] = useState<AxisResponse | null>(null);
  const [showSource, setShowSource] = useState(false);
  const axis = useMemo(() => fitAxis(pairs), [pairs]);
  const evaluation = useMemo(() => evaluateAxis(axis, pairs), [axis, pairs]);
  const selected = pairs[Math.min(sample, pairs.length - 1)];
  const steered = steerActivation(selected.negative, axis, strength);
  const plot = selected.negative.slice(0, 128).map((value, index) => ({ feature: index + 1, original: value, steered: steered[index], positive: selected.positive[index] }));
  const fit = () => {
    try { const next = decodePairs(source); const fitted = fitAxis(next); evaluateAxis(fitted, next); setPairs(next); setSample(0); setRemote(null); setError(null); }
    catch (failure) { setError(errorMessage(failure)); }
  };
  const runRemote = async () => {
    if (!client) return;
    setBusy(true); setError(null);
    try {
      const train = pairs.filter(pair => pair.split === 'train');
      const held = pairs.filter(pair => pair.split === 'evaluation');
      setRemote(await client.axis({ positive: train.map(pair => pair.positive), negative: train.map(pair => pair.negative), eval_positive: held.map(pair => pair.positive), eval_negative: held.map(pair => pair.negative) }));
    } catch (failure) { setError(errorMessage(failure)); } finally { setBusy(false); }
  };
  return <div className="rw-panel">
    <PanelTitle eyebrow="01 / REPRESENTATION ENGINEERING" title="Find a direction. Measure its effect.">Fit a unit concept axis from paired activation differences, hold out evaluation pairs, and inspect a numerical intervention.</PanelTitle>
    <div className="rw-banner"><FlaskConical size={18}/><span>The bundled six-dimensional captures are a constructed numerical example. Their labels illustrate the method; they are not activations from a language model. Import your own captures for a model experiment.</span></div>
    <ErrorNotice error={error}/>
    <div className="rw-stats"><Stat label="Held-out accuracy" value={`${(evaluation.accuracy * 100).toFixed(1)}%`} note={`${evaluation.scores.length * 2} activation samples`}/><Stat label="ROC AUC" value={evaluation.auc.toFixed(3)} note="All positive / negative comparisons"/><Stat label="Pair ordering" value={`${(evaluation.pairOrdering * 100).toFixed(0)}%`} note="Positive projection exceeds negative"/><Stat label="Average margin" value={evaluation.meanMargin.toFixed(3)} note="Held-out projection difference"/></div>
    <div className="rw-two-col">
      <div className="rw-card"><div className="rw-card-heading"><h3>Concept direction</h3><span className="rw-chip">{axis.dimension} dimensions</span></div><p>Mean positive minus negative activation, normalized to unit length. The classification threshold uses training midpoints only.</p><HorizontalBars values={axis.vector.slice(0, 64)}/>{axis.dimension > 64 && <p className="rw-note">Showing the first 64 of {axis.dimension} features. Fitting and exported artifacts use every dimension.</p>}<div className="rw-mini-stats"><span>{axis.trainPairs} training pairs</span><span>Difference norm {axis.meanDifferenceNorm.toFixed(3)}</span></div><div className="rw-actions"><button onClick={() => setShowSource(value => !value)}>{showSource ? 'Close capture editor' : 'Edit paired captures'}</button><DownloadButton filename="neuralis-concept-axis.json" data={{ schema: 'neuralis.axis/v1', axis, evaluation, pairs }}/></div></div>
      <div className="rw-card"><div className="rw-card-heading"><h3>Steering preview</h3><span className="rw-chip">Browser calculation</span></div><div className="rw-fields"><label>Capture pair<select value={sample} onChange={event => setSample(Number(event.target.value))}>{pairs.map((pair, index) => <option key={pair.id} value={index}>{pair.id} · {pair.split}</option>)}</select></label><label>Strength α <strong>{strength.toFixed(2)}</strong><input aria-label="Axis steering strength" type="range" min={-3} max={3} step={0.05} value={strength} onChange={event => setStrength(Number(event.target.value))}/></label></div><div className="rw-chart"><ResponsiveContainer width="100%" height={220}><LineChart data={plot}><CartesianGrid stroke="#213347" strokeDasharray="3 3"/><XAxis dataKey="feature" stroke="#778ba3"/><YAxis stroke="#778ba3" width={35}/><Tooltip contentStyle={{ background: '#0e1929', borderColor: '#31435b', color: '#dce8f5' }}/><Line dataKey="original" stroke="#7c8da7" dot={false} strokeWidth={2}/><Line dataKey="steered" stroke="#51d6cf" strokeWidth={2}/><Line dataKey="positive" stroke="#a190ff" strokeDasharray="4 4" dot={false}/></LineChart></ResponsiveContainer></div><div className="rw-legend"><span className="original">Original negative</span><span className="steered">Steered: h + αv</span><span className="target">Paired positive</span></div><p>A unit axis shifts its own projection by exactly α. This preview does not generate text or establish causal behavior. {axis.dimension > 128 ? `Showing the first 128 of ${axis.dimension} features; the intervention uses all dimensions.` : ''}</p><DownloadButton filename="neuralis-steering-preview.json" data={{ pair: selected.id, strength, original: selected.negative, steered, target: selected.positive }}/></div>
    </div>
    {showSource && <div className="rw-card"><h3>Paired activation editor</h3><p>Each object needs unique id, negative, positive, and an explicit train or evaluation split. Keep a held-out split to avoid reporting fit quality as generalization.</p><textarea aria-label="Paired activations JSON" rows={10} value={source} onChange={event => setSource(event.target.value)}/><div className="rw-actions"><button className="rw-primary" onClick={fit}><Play size={15}/>Fit and evaluate</button><button onClick={() => { setSource(JSON.stringify(examplePairs, null, 2)); setPairs(examplePairs); setSample(3); setRemote(null); setError(null); }}><RotateCcw size={15}/>Reset example</button></div></div>}
    <div className="rw-card"><div className="rw-card-heading"><h3>Held-out pair projections</h3><BackendButton available={Boolean(client && capabilities?.axislab.available)} busy={busy} onClick={runRemote}><ArrowUpRight size={15}/>Evaluate with AxisLab backend</BackendButton></div><div className="rw-table-scroll"><table><thead><tr><th>Pair</th><th>Negative score</th><th>Positive score</th><th>Margin</th><th>Ordering</th></tr></thead><tbody>{evaluation.scores.slice(0, 100).map(row => <tr key={row.id}><td>{row.id}</td><td>{row.negative.toFixed(4)}</td><td>{row.positive.toFixed(4)}</td><td>{row.margin.toFixed(4)}</td><td><span className={`rw-chip ${row.margin > 0 ? 'good' : 'warn'}`}>{row.margin > 0 ? 'Preserved' : 'Reversed'}</span></td></tr>)}</tbody></table></div>{evaluation.scores.length > 100 && <p className="rw-note">Showing the first 100 held-out pairs. Metrics and exports include all {evaluation.scores.length} evaluation pairs.</p>}{!capabilities?.axislab.available && <p className="rw-note">Connect a backend with AxisLab installed to evaluate through the actual package. The browser method above remains available offline.</p>}{remote && <><div className="rw-banner good">Backend method: {remote.method} · AUC {remote.auc.toFixed(3)} · {remote.training_samples} training / {remote.evaluation_samples} evaluation samples</div><ResultJson value={remote} label="AxisLab response and fitted direction"/></>}</div>
  </div>;
}
