import { useMemo, useState } from 'react';
import { Table2, ScanEye, Download, Search, Database, ArrowUpRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useStudio } from '../app/StudioContext';
import { TensorInspector, AttentionView } from '../visualization';
import { downloadText } from '../app/storage';
import { Panel, PageHeader, Badge, Button, EmptyState, Callout, Select, number, MetricCard } from '../components/ui';

export function Tensors() {
  const { snapshot } = useStudio();
  const values = snapshot.layers.flatMap(layer => layer.weights.flat());
  const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  const variance = values.length ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length : 0;
  const sparsity = values.length ? values.filter(value => Math.abs(value) < 1e-8).length / values.length : 0;
  return <>
    <PageHeader eyebrow="TENSOR INSPECTOR" title="Every weight tells a story." description="Inspect captured matrices, activation distributions, and gradients at full precision." actions={<Button onClick={() => downloadText('neuralis-tensors.json', JSON.stringify(snapshot.layers, null, 2))}><Download size={15} />Export tensors</Button>} />
    <div className="metric-grid"><MetricCard label="Captured weights" value={values.length.toLocaleString()} detail="Across all captured matrices" color="cyan" /><MetricCard label="Mean weight" value={number(mean, 6)} detail="Arithmetic mean of captured weights" color="violet" /><MetricCard label="Standard deviation" value={number(Math.sqrt(variance), 6)} detail="Population standard deviation" color="green" /><MetricCard label="Near-zero sparsity" value={`${(sparsity * 100).toFixed(2)}%`} detail="Absolute value below 1e−8" color="amber" /></div>
    <Panel title="Model tensors" subtitle={`Step ${snapshot.step} · ${snapshot.layers.length} layers captured`} action={<Badge tone="cyan"><Table2 size={12} />Measured values</Badge>}><TensorInspector layers={snapshot.layers} /></Panel>
    <Callout>Tensor values are the captured state of your active model. Initial weights are visible before training; gradients become informative after optimizer steps. Export the exact values for further analysis.</Callout>
  </>;
}

function actualAttention(extras?: Record<string, unknown>): { matrix: number[][]; labels: string[] } | null {
  if (!extras) return null;
  const candidate = extras.attention ?? extras.attentionMatrix;
  if (!Array.isArray(candidate) || !candidate.every(row => Array.isArray(row) && row.every(item => typeof item === 'number' && Number.isFinite(item)))) return null;
  const labels = Array.isArray(extras.tokens) && extras.tokens.every(item => typeof item === 'string') ? extras.tokens as string[] : candidate.map((_, index) => `Token ${index}`);
  return { matrix: candidate as number[][], labels };
}
export function Attention() {
  const { snapshot } = useStudio();
  const attention = actualAttention(snapshot.extras);
  return <>
    <PageHeader eyebrow="ATTENTION ANALYSIS" title="Discover the relationships between tokens." description="Inspect attention computed by the active model, with no invented token connections." actions={attention && <Button onClick={() => downloadText('attention.json', JSON.stringify(attention, null, 2))}><Download size={14} />Export attention</Button>} />
    {attention ? <Panel title="Computed attention" subtitle="Query-to-key attention probabilities" action={<Badge tone="violet"><ScanEye size={12} />{attention.labels.length} tokens</Badge>}><AttentionView matrix={attention.matrix} labels={attention.labels} /></Panel> : <Panel><EmptyState title="This model has no captured attention matrix" description="Dense neural networks do not use token attention. Choose a language attention demo to inspect a real computed matrix." icon={<ScanEye size={32} />} action={<Link to="/demos" className="button button-primary">Explore language demos<ArrowUpRight size={14} /></Link>} /></Panel>}
    <Callout>Attention is an internal weighting mechanism, and attention scores alone do not establish a causal explanation. Use controlled interventions and paired outputs to test a hypothesis.</Callout>
  </>;
}

export function DataExplorer() {
  const { snapshot } = useStudio();
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(0);
  const [format, setFormat] = useState('decimal');
  const samples = useMemo(() => snapshot.samples.map((sample, index) => ({ ...sample, index })).filter(sample => `${sample.label ?? ''} ${sample.input.join(' ')} ${sample.target.join(' ')}`.toLowerCase().includes(query.toLowerCase())), [snapshot.samples, query]);
  const active = snapshot.samples[selected];
  const display = (value: number) => format === 'scientific' ? value.toExponential(4) : number(value, 4);
  return <>
    <PageHeader eyebrow="DATA EXPLORER" title="Know what your model sees." description="Browse real captured examples, their targets, and the model’s predictions." actions={<Button onClick={() => downloadText('captured-samples.json', JSON.stringify(snapshot.samples, null, 2))}><Download size={14} />Export samples</Button>} />
    <div className="data-toolbar"><div className="library-search"><Search size={15} /><input value={query} aria-label="Search captured examples" placeholder="Search sample values or labels…" onChange={event => setQuery(event.target.value)} /></div><Badge tone="cyan">{samples.length} captured samples</Badge><Select label="Numeric display format" value={format} onChange={setFormat} options={[{ value: 'decimal', label: 'Decimal values' }, { value: 'scientific', label: 'Scientific notation' }]} /></div>
    <div className="data-layout"><Panel title="Captured examples" subtitle="A bounded view of the active dataset"><div className="sample-grid">{samples.map(sample => <button key={sample.index} className={`sample-card ${selected === sample.index ? 'selected' : ''}`} onClick={() => setSelected(sample.index)}><div className="sample-values">{sample.input.slice(0, 24).map((value, index) => <span key={index} style={{ background: `rgba(${value >= 0 ? '83, 195, 249' : '168, 126, 255'}, ${Math.min(1, Math.abs(value)) * .65 + .1})` }} title={String(value)} />)}</div><strong>{sample.label ?? `Sample ${sample.index + 1}`}</strong><small>{sample.input.length} features · {sample.target.length} targets</small></button>)}</div>{samples.length === 0 && <EmptyState title="No matching examples" description="Change your search or choose another demo." icon={<Database size={23} />} />}</Panel><Panel title="Selected example" subtitle={active?.label ?? `Sample ${selected + 1}`}>{active ? <div className="sample-detail"><span className="section-label">MODEL INPUT</span><div className="numeric-grid">{active.input.map((value, index) => <div key={index}><span>x{index}</span><code>{display(value)}</code></div>)}</div><span className="section-label">TARGET VS. PREDICTION</span><table className="data-table"><thead><tr><th>Output</th><th>Target</th><th>Prediction</th><th>Absolute error</th></tr></thead><tbody>{active.target.map((target, index) => <tr key={index}><td>{index}</td><td className="mono">{display(target)}</td><td className="mono cyan-text">{display(active.prediction[index] ?? 0)}</td><td className="mono">{display(Math.abs(target - (active.prediction[index] ?? 0)))}</td></tr>)}</tbody></table></div> : <EmptyState title="No sample selected" description="Select an example to inspect its values." />}</Panel></div>
    <Callout>These examples are captured from the engine, rather than a complete dataset browser. Generated demo data is reproducible from the run seed. Predictions update as training progresses.</Callout>
  </>;
}
