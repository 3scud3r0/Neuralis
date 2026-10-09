import { useState } from 'react';
import { Link } from 'react-router-dom';
import { FlaskConical, Trash2, Download, Play, GitCompareArrows, Clock3, ChevronRight } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';
import { useStudio } from '../app/StudioContext';
import { DEMOS } from '../engine';
import { downloadText, type RunRecord } from '../app/storage';
import { Panel, PageHeader, Button, Badge, EmptyState, MetricsChart, KeyValue, Select, Callout, number, percentage } from '../components/ui';

export function Experiments() {
  const { history, clearHistory, selectDemo } = useStudio();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [replayIndex, setReplayIndex] = useState<number | null>(null);
  const selected = history.find(run => run.id === selectedId) ?? history[0];
  const replay = selected && replayIndex !== null ? selected.metrics[Math.min(replayIndex, selected.metrics.length - 1)] : undefined;
  return <>
    <PageHeader eyebrow="EXPERIMENT TRACKER" title="A record of your discoveries." description="Inspect real run histories, replay captured metrics, and reproduce a configuration." actions={<><Button disabled={!history.length} onClick={() => downloadText('neuralis-experiments.json', JSON.stringify(history, null, 2))}><Download size={14} />Export history</Button><Link to="/new-experiment" className="button button-primary"><FlaskConical size={14} />New experiment</Link></>} />
    {history.length ? <div className="experiments-layout"><Panel title="Local run history" subtitle={`${history.length} recorded runs · most recent first`} action={<Button variant="ghost" onClick={clearHistory} title="Clear local run history"><Trash2 size={14} /></Button>}><div className="experiment-list">{history.map(run => <button className={`experiment-item ${selected?.id === run.id ? 'selected' : ''}`} key={run.id} onClick={() => { setSelectedId(run.id); setReplayIndex(null); }}><span className="experiment-icon"><FlaskConical size={19} /></span><span><strong>{DEMOS.find(item => item.id === run.demoId)?.title ?? run.demoId}</strong><small>{new Date(run.createdAt).toLocaleString()}</small><span className="experiment-meta">{run.step.toLocaleString()} steps · seed {run.seed} · loss {number(run.loss, 4)}</span></span><Badge tone={run.state === 'completed' ? 'green' : run.state === 'failed' ? 'red' : 'neutral'}>{run.state}</Badge><ChevronRight size={13} /></button>)}</div></Panel>{selected && <div className="experiment-detail"><Panel title={DEMOS.find(item => item.id === selected.demoId)?.title ?? selected.demoId} subtitle={`Run ${selected.id}`} action={<Badge tone="violet">Seed {selected.seed}</Badge>}><div className="experiment-summary"><KeyValue label="Final loss" value={number(selected.loss, 6)} /><KeyValue label="Final task score" value={percentage(selected.accuracy)} /><KeyValue label="Optimizer steps" value={selected.step.toLocaleString()} /></div><MetricsChart metrics={selected.metrics} height={260} /><div className="panel-bottom"><Link to="/training" className="button button-primary" onClick={() => selectDemo(selected.demoId)}><Play size={14} />Open this demo</Link><Button onClick={() => downloadText(`run-${selected.id}.json`, JSON.stringify(selected, null, 2))}><Download size={14} />Export</Button></div></Panel><Panel title="Metric replay" subtitle="Scrub through the measurements recorded during this run" action={<Clock3 size={15} />}><div className="replay-controls"><input type="range" aria-label="Captured metric replay" min={0} max={Math.max(0, selected.metrics.length - 1)} value={replayIndex ?? selected.metrics.length - 1} onChange={event => setReplayIndex(Number(event.target.value))} /><span className="mono">Capture {(replayIndex ?? selected.metrics.length - 1) + 1} / {selected.metrics.length}</span></div><div className="replay-values"><KeyValue label="Step" value={replay?.step ?? selected.metrics.at(-1)?.step ?? 0} /><KeyValue label="Loss" value={number(replay?.loss ?? selected.loss, 6)} /><KeyValue label="Task score" value={percentage(replay?.accuracy ?? selected.accuracy)} /></div><Callout>This replay contains recorded metrics. Restore a saved checkpoint from the Training page to inspect model parameters at an earlier step.</Callout></Panel></div>}</div> : <Panel><EmptyState title="Your experiment history starts with a real run" description="Train a demo and complete or stop the run to archive its actual metrics here." icon={<FlaskConical size={34} />} action={<Link to="/training" className="button button-primary"><Play size={14} />Start an experiment</Link>} /></Panel>}
  </>;
}

export function CompareRuns() {
  const { history } = useStudio();
  const [leftId, setLeftId] = useState(history[0]?.id ?? '');
  const [rightId, setRightId] = useState(history[1]?.id ?? '');
  const [metric, setMetric] = useState('loss');
  const left = history.find(run => run.id === leftId);
  const right = history.find(run => run.id === rightId);
  const options = history.map(run => ({ value: run.id, label: `${DEMOS.find(item => item.id === run.demoId)?.title ?? run.demoId} · seed ${run.seed} · ${run.step} steps` }));
  const data = comparisonData(left, right, metric as 'loss' | 'accuracy');
  return <>
    <PageHeader eyebrow="RUN COMPARISON" title="What changed between experiments?" description="Compare recorded learning curves and final measurements from two runs." />
    {history.length < 2 ? <Panel><EmptyState title="Two runs unlock comparison" description="Complete two experiments, then return to compare their loss and task score." icon={<GitCompareArrows size={34} />} action={<Link to="/training" className="button button-primary">Train another model</Link>} /></Panel> : <>
      <div className="comparison-selectors"><Panel title="Run A" action={<span className="color-dot cyan" />}><Select label="First comparison run" value={leftId} onChange={setLeftId} options={options} /></Panel><Panel title="Run B" action={<span className="color-dot violet" />}><Select label="Second comparison run" value={rightId} onChange={setRightId} options={options} /></Panel></div>
      <Panel title="Learning curve comparison" subtitle="Measurements aligned by optimizer step" action={<Select label="Comparison metric" value={metric} onChange={setMetric} options={[{ value: 'loss', label: 'Training loss' }, { value: 'accuracy', label: 'Task score' }]} />}><div className="comparison-chart"><ResponsiveContainer width="100%" height={330}><LineChart data={data} margin={{ left: -10, right: 25, top: 20 }}><CartesianGrid stroke="#223147" strokeDasharray="3 5" /><XAxis dataKey="step" tick={{ fill: '#8192ac', fontSize: 11 }} /><YAxis tick={{ fill: '#8192ac', fontSize: 11 }} /><Tooltip contentStyle={{ background: '#101c2d', border: '1px solid #2b3e56', borderRadius: 8 }} /><Line name="Run A" dataKey="left" connectNulls={false} stroke="#51c8ff" dot={false} strokeWidth={2} isAnimationActive={false} /><Line name="Run B" dataKey="right" connectNulls={false} stroke="#a68bff" dot={false} strokeWidth={2} isAnimationActive={false} /></LineChart></ResponsiveContainer></div><table className="data-table comparison-table"><thead><tr><th>Measurement</th><th>Run A</th><th>Run B</th><th>Difference (B − A)</th></tr></thead><tbody>{[{ label: 'Final loss', a: left?.loss, b: right?.loss }, { label: 'Task score', a: left?.accuracy, b: right?.accuracy }, { label: 'Optimizer steps', a: left?.step, b: right?.step }, { label: 'Seed', a: left?.seed, b: right?.seed }].map(row => <tr key={row.label}><td>{row.label}</td><td className="mono cyan-text">{row.a === undefined ? '—' : number(row.a, 4)}</td><td className="mono violet-text">{row.b === undefined ? '—' : number(row.b, 4)}</td><td className="mono">{row.a === undefined || row.b === undefined ? '—' : number(row.b - row.a, 4)}</td></tr>)}</tbody></table></Panel>
      <Callout>Compare runs on the same task and evaluation protocol for a meaningful interpretation. A lower loss across different datasets does not by itself indicate a better model.</Callout>
    </>}
  </>;
}
function comparisonData(left: RunRecord | undefined, right: RunRecord | undefined, metric: 'loss' | 'accuracy') {
  const rows = new Map<number, { step: number; left?: number; right?: number }>();
  left?.metrics.forEach(point => rows.set(point.step, { ...rows.get(point.step), step: point.step, left: point[metric] }));
  right?.metrics.forEach(point => rows.set(point.step, { ...rows.get(point.step), step: point.step, right: point[metric] }));
  return Array.from(rows.values()).sort((a, b) => a.step - b.step);
}
