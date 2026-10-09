import { useMemo, useState } from 'react';
import { Layers, Orbit, Workflow, GitBranch, Table2, Download } from 'lucide-react';
import { useStudio } from '../app/StudioContext';
import { ArchitectureEditor, sequentialArchitecture, EmbeddingView, CircuitView, TensorSliceInspector, denseTensor } from '../visualization';
import { Panel, PageHeader, Select, Badge, Button, Callout, EmptyState } from '../components/ui';
import { downloadText } from '../app/storage';

export function RepresentationExplorer() {
  const { snapshot } = useStudio();
  const [tab, setTab] = useState<'architecture' | 'projection' | 'circuits' | 'slices'>('architecture');
  const [vectorSource, setVectorSource] = useState('input');
  const [layerIndex, setLayerIndex] = useState('1');
  const graph = useMemo(() => sequentialArchitecture(snapshot.layers.map(layer => layer.size), snapshot.layers.map(layer => layer.name)), [snapshot.layers]);
  const vectors = snapshot.samples.map(sample => vectorSource === 'prediction' ? sample.prediction : vectorSource === 'target' ? sample.target : sample.input);
  const labels = snapshot.samples.map((sample, index) => sample.label ?? `Sample ${index + 1}`);
  const groups = snapshot.samples.map(sample => sample.target.length > 1 ? `Target ${sample.target.indexOf(Math.max(...sample.target))}` : sample.target[0] > .5 ? 'Target > 0.5' : 'Target ≤ 0.5');
  const layer = snapshot.layers[Number(layerIndex)] ?? snapshot.layers[0];
  const tensor = useMemo(() => {
    if (!layer?.weights.length || !layer.weights[0]?.length) return null;
    return denseTensor(layer.weights.flat(), [layer.weights.length, layer.weights[0].length]);
  }, [layer]);
  return <>
    <PageHeader eyebrow="REPRESENTATION EXPLORER" title="Explore the structure behind the prediction." description="Architecture diagrams, sample projections, captured circuits, and exact tensor slices." actions={<Button onClick={() => downloadText('representation-capture.json', JSON.stringify({ layers: snapshot.layers, samples: snapshot.samples }, null, 2))}><Download size={14} />Export capture</Button>} />
    <Panel title="Research views" action={<Badge tone="violet">Step {snapshot.step}</Badge>}><div className="tabs representation-tabs">{[{ id: 'architecture', title: 'Architecture', icon: Workflow }, { id: 'projection', title: 'Sample projection', icon: Orbit }, { id: 'circuits', title: 'Weight circuits', icon: GitBranch }, { id: 'slices', title: 'Tensor slices', icon: Table2 }].map(item => <button key={item.id} className={tab === item.id ? 'active' : ''} onClick={() => setTab(item.id as typeof tab)}><item.icon size={13} />{item.title}</button>)}</div>
      {tab === 'architecture' && <><ArchitectureEditor key={snapshot.demoId} graph={graph} readOnly /><div className="representation-note"><Layers size={14} /><p>This diagram describes the captured sequential layer widths. Use New experiment to change supported trainable dense architectures; specialist captured structures are inspected rather than compiled from this diagram.</p></div></>}
      {tab === 'projection' && <><div className="representation-toolbar"><Select label="Projection vector source" value={vectorSource} onChange={setVectorSource} options={[{ value: 'input', label: 'Captured input vectors' }, { value: 'prediction', label: 'Captured prediction vectors' }, { value: 'target', label: 'Captured target vectors' }]} /><span className="muted">{vectors.length} captured vectors · centered PCA</span></div><EmbeddingView vectors={vectors} labels={labels} groups={groups} title={`PCA of captured ${vectorSource} vectors`} /><div className="representation-note"><Orbit size={14} /><p>These are captured {vectorSource} vectors, projected by actual centered PCA. They are not inferred hidden embeddings. Capturing per-example hidden states requires explicit model instrumentation.</p></div></>}
      {tab === 'circuits' && <><CircuitView layers={snapshot.layers} maxEdges={400} /><div className="representation-note"><GitBranch size={14} /><p>This circuit view ranks captured connections by weight magnitude. Weight magnitude measures a parameter, rather than causal feature importance or integrated gradients.</p></div></>}
      {tab === 'slices' && <><div className="representation-toolbar"><Select label="Slice layer" value={layerIndex} onChange={setLayerIndex} options={snapshot.layers.map((item, index) => ({ value: String(index), label: `${item.name} · ${item.weights.length} × ${item.weights[0]?.length ?? 0}` }))} /></div>{tensor ? <TensorSliceInspector key={layer.id} tensor={tensor} axisNames={['Output neuron', 'Input neuron']} title={`${layer.name} weight matrix`} /> : <EmptyState title="No matrix captured for this layer" description="Choose a layer that contains a weight matrix." icon={<Table2 size={27} />} />}</>}
    </Panel>
    <Callout>Each research view states exactly what data it uses and how it is transformed. Interpretability claims should be tested against paired predictions and held-out examples.</Callout>
  </>;
}
