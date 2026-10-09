import { useEffect, useMemo, useState } from 'react';
import { FlaskConical, Play, RotateCcw, Download, ArrowRight, Target } from 'lucide-react';
import { useStudio } from '../app/StudioContext';
import type { InterventionResult, InterventionSpec } from '../types';
import NetworkScene from '../visualization/NetworkScene';
import { downloadText } from '../app/storage';
import { Panel, PageHeader, Select, Field, Button, Callout, KeyValue, Badge, number, EmptyState } from '../components/ui';

export function Interventions() {
  const { snapshot, engine, notify } = useStudio();
  const [layerIndex, setLayerIndex] = useState(1);
  const [neuron, setNeuron] = useState(0);
  const [kind, setKind] = useState<InterventionSpec['kind']>('ablate');
  const [value, setValue] = useState(0);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [result, setResult] = useState<InterventionResult | null>(null);
  const [customInput, setCustomInput] = useState('');
  const layers = snapshot.layers;
  const layer = layers[layerIndex];
  const sample = snapshot.samples[sampleIndex];
  useEffect(() => { setResult(null); setSampleIndex(0); setLayerIndex(Math.min(1, layers.length - 1)); }, [snapshot.demoId, layers.length]);
  const maxDelta = useMemo(() => result ? Math.max(...result.delta.map(Math.abs), 0.00001) : 1, [result]);
  const run = () => {
    try {
      const input = customInput.trim() ? customInput.split(',').map(text => Number(text.trim())) : sample?.input;
      if (!input || input.some(item => !Number.isFinite(item))) throw new Error('Supply a valid comma-separated numeric input or select a captured sample.');
      if (!layer || neuron < 0 || neuron >= layer.size) throw new Error('Select a neuron within the current layer.');
      setResult(engine.intervene(input, { layer: layerIndex, neuron, kind, value }));
      notify('Intervention evaluated against the same input and model weights.', 'success');
    } catch (error) { notify(error instanceof Error ? error.message : 'This model does not support activation interventions.', 'error'); }
  };
  return <>
    <PageHeader eyebrow="CAUSAL EXPERIMENTS" title="Observe. Intervene. Understand." description="Change a single activation and measure how the prediction responds." actions={<Button disabled={!result} onClick={() => result && downloadText('intervention-result.json', JSON.stringify(result, null, 2))}><Download size={14} />Export result</Button>} />
    <div className="intervention-layout"><Panel title="Intervention setup" subtitle="Inference-only changes; model weights remain fixed"><div className="control-stack"><Field label="Intervention type"><Select label="Intervention type" value={kind} onChange={next => { setKind(next as InterventionSpec['kind']); setValue(next === 'ablate' ? 0 : 1); }} options={[{ value: 'ablate', label: 'Ablate · set activation to zero' }, { value: 'scale', label: 'Scale · multiply activation' }, { value: 'steer', label: 'Steer · add activation offset' }]} /></Field><Field label="Target layer"><Select label="Target layer" value={String(layerIndex)} onChange={next => { setLayerIndex(Number(next)); setNeuron(0); }} options={layers.map((item, index) => ({ value: String(index), label: `${index} · ${item.name} (${item.size} neurons)` }))} /></Field><Field label="Neuron index"><input type="number" min={0} max={(layer?.size ?? 1) - 1} step={1} value={neuron} onChange={event => setNeuron(Number(event.target.value))} /></Field>{kind !== 'ablate' && <Field label={kind === 'scale' ? 'Scale factor' : 'Activation offset'}><div className="range-with-value"><input type="range" min={-5} max={5} step={0.1} value={value} onChange={event => setValue(Number(event.target.value))} /><output>{value.toFixed(1)}</output></div></Field>}<Field label="Captured input"><Select label="Captured input" value={String(sampleIndex)} onChange={next => setSampleIndex(Number(next))} options={snapshot.samples.map((item, index) => ({ value: String(index), label: item.label ?? `Sample ${index + 1} · [${item.input.map(x => number(x, 2)).join(', ')}]` }))} /></Field><Field label="Or provide a custom input" hint="Comma-separated values matching the model input dimension"><input value={customInput} onChange={event => setCustomInput(event.target.value)} placeholder={sample?.input.join(', ') ?? '0, 1'} /></Field><Button variant="primary" onClick={run}><FlaskConical size={15} />Run intervention</Button><Button variant="ghost" onClick={() => { setResult(null); setCustomInput(''); }}><RotateCcw size={14} />Reset experiment</Button></div></Panel><div className="intervention-main"><Panel title="Target circuit" subtitle="Select a neuron directly from the model view" action={<Badge tone="violet">Layer {layerIndex} · Neuron {neuron}</Badge>}><NetworkScene layers={layers} height={280} selected={{ layer: layerIndex, neuron }} onSelect={(l, n) => { setLayerIndex(l); setNeuron(n); }} mode="activations" /></Panel><Panel title="Prediction response" subtitle="A paired evaluation with a controlled activation change" action={result ? <Badge tone="green" dot>Measured</Badge> : undefined}>{result ? <div className="intervention-results"><div className="intervention-input"><Target size={15} /><span>Input</span><code>[{result.input.map(x => number(x, 3)).join(', ')}]</code></div><div className="prediction-comparison"><div><span className="section-label">BASELINE</span>{result.before.map((item, index) => <div className="prediction-value" key={index}><span>Output {index}</span><strong>{number(item, 6)}</strong></div>)}</div><ArrowRight size={24} /><div><span className="section-label">AFTER INTERVENTION</span>{result.after.map((item, index) => <div className="prediction-value" key={index}><span>Output {index}</span><strong className="violet-text">{number(item, 6)}</strong></div>)}</div></div><div className="delta-chart">{result.delta.map((item, index) => <div className="delta-row" key={index}><span>Δ output {index}</span><div><span style={{ width: `${Math.abs(item) / maxDelta * 100}%`, background: item >= 0 ? 'var(--green)' : 'var(--rose)' }} /></div><code>{item > 0 ? '+' : ''}{number(item, 6)}</code></div>)}</div>{result.klDivergence !== undefined && <KeyValue label="KL divergence" value={number(result.klDivergence, 6)} mono />}</div> : <EmptyState title="What happens if this neuron changes?" description="Choose a captured input, set an intervention, and run a paired prediction to find out." icon={<FlaskConical size={24} />} action={<Button onClick={run}><Play size={13} />Evaluate hypothesis</Button>} />}</Panel></div></div>
    <Callout>Single-neuron changes measure sensitivity for this input. They do not establish a universal semantic role or prove an entire causal explanation. Interventions are evaluated through the active engine’s supported forward pass.</Callout>
  </>;
}
