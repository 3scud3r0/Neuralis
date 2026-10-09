import { useEffect, useMemo, useState } from 'react';
import { useStudio } from '../../app/StudioContext';
import { Badge, Button, Callout, Panel } from '../../components/ui';
import { ArchitectureEditor } from '../../visualization/ArchitectureEditor';
import { analyzeArchitecture, sequentialArchitecture, type ArchitectureGraph } from '../../visualization/architecture';
import { compileBrowserArchitecture, FIXED_ARCHITECTURE_DEMOS, insertDenseBeforeOutput } from './graphModel';
import { BackendGraphSurgery } from './BackendGraphSurgery';
import '../AdvancedResearch/advanced.css';

export function GraphEditor() {
  const { snapshot,config,settings,setConfig,notify } = useStudio();
  const initial = useMemo(() => sequentialArchitecture(snapshot.layers.map(layer => layer.size),snapshot.layers.map(layer => layer.name)),[snapshot.runId]);
  const [draft,setDraft] = useState<ArchitectureGraph>(initial);
  const [revision,setRevision] = useState(0);
  const [error,setError] = useState('');
  const [removeEdge,setRemoveEdge] = useState('');
  const [node,setNode] = useState('');
  useEffect(() => { setDraft(initial); setRevision(value => value+1); }, [initial]);
  const analysis = useMemo(() => analyzeArchitecture(draft),[draft]);
  const fixed = FIXED_ARCHITECTURE_DEMOS.has(snapshot.demoId);
  let compiled: number[] | null = null, capability = '';
  try { compiled = compileBrowserArchitecture(draft,[snapshot.layers[0].size,snapshot.layers.at(-1)!.size]); } catch (caught) { capability = caught instanceof Error ? caught.message : 'Unsupported graph'; }
  const externalChange = (next: ArchitectureGraph) => { setDraft(next); setRevision(value => value+1); setError(''); };
  const selected = draft.nodes.find(item => item.id === node);
  return <Panel title="Architecture laboratory" subtitle="Edit a graph, infer tensor shapes, then apply supported dense chains." action={<Badge tone={analysis.issues.some(issue => issue.severity === 'error') ? 'red' : 'green'}>{analysis.issues.some(issue => issue.severity === 'error') ? 'Shape errors' : 'DAG checked'}</Badge>}>
    <div className="advanced-graph-toolbar"><Button onClick={() => { try { externalChange(insertDenseBeforeOutput(draft)); } catch (caught) { setError(String(caught)); } }}>Insert hidden layer</Button><Button variant="ghost" onClick={() => externalChange(initial)}>Reset from current capture</Button><label>Remove edge<select value={removeEdge} onChange={event => setRemoveEdge(event.target.value)}><option value="">Choose connection</option>{draft.edges.map((edge,index) => <option key={`${edge.source}:${edge.target}`} value={index}>{edge.source} → {edge.target}</option>)}</select></label><Button disabled={!removeEdge} onClick={() => { externalChange({...draft,edges:draft.edges.filter((_,index) => index !== Number(removeEdge))}); setRemoveEdge(''); }}>Disconnect</Button></div>
    <ArchitectureEditor key={`${snapshot.runId}-${revision}`} graph={draft} onChange={setDraft} />
    <div className="advanced-graph-toolbar"><label>Dense width<select value={node} onChange={event => setNode(event.target.value)}><option value="">Select dense node</option>{draft.nodes.filter(item => item.operation === 'dense').map(item => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label>{selected && <label>Units<input aria-label="Selected dense units" type="number" min={1} max={128} value={selected.units ?? 16} onChange={event => externalChange({...draft,nodes:draft.nodes.map(item => item.id === node ? {...item,units:Number(event.target.value)} : item)})} /></label>}<Button variant="primary" disabled={fixed || compiled === null || snapshot.state === 'running'} onClick={() => { if (!compiled || fixed) return; setConfig({ hiddenLayers:compiled }); notify('Validated dense architecture applied. Model parameters were reinitialized.','success'); }}>Apply to browser engine</Button></div>
    {error && <p role="alert" className="viz-error">{error}</p>}<Callout>{fixed ? 'The active specialist model has a fixed implemented architecture. Its initial diagram is a captured stage-width summary; inferred dense parameter counts do not describe its internal specialist operations. Select a dense demo to apply hidden layer changes.' : capability || `Ready to run ${compiled?.length ?? 0} hidden layers with the current demo's activation and objective. Applying resets model parameters.`}</Callout><p className="advanced-note">The canvas checks directed cycles and dimensions, including add and concat branches. The browser engine executes dense chains; the optional backend section below compiles supported explicit DAGs, including skip connections. Recurrent cycles require bounded unrolling and remain unsupported.</p>
    <BackendGraphSurgery canvasGraph={draft} inputSamples={snapshot.samples.map(sample => sample.input)} initialUrl={settings.backendUrl} seed={config.seed} />
  </Panel>;
}
export { compileBrowserArchitecture, insertDenseBeforeOutput } from './graphModel';
export { GraphEditor as ComputationalGraphCanvas };
export { BackendGraphSurgery } from './BackendGraphSurgery';
