import { useMemo, useState } from 'react';
import { Activity, GitBranch, Orbit, Waves } from 'lucide-react';
import { useStudio } from '../../app/StudioContext';
import { Badge, Button, Callout, EmptyState, PageHeader, Panel, Select } from '../../components/ui';
import { LatentSpaceExplorer } from '../../visualization/components/LatentSpaceExplorer';
import { LatentSpaceGalaxy } from '../../visualization/webgpu/LatentSpaceGalaxy';
import { attentionSplines } from '../../visualization/webgpu/AttentionSplines';
import { attentionEntropy } from '../../visualization/numerics';
import { GraphEditor } from '../GraphEditor';
import { InferenceStream, type InferenceStreamProps } from '../InferenceStream';
import { activationFlow, capturedAttentionHeads, finiteMatrix } from './captures';
import './advanced.css';

export function AdvancedResearch(props: InferenceStreamProps = {}) {
  const {snapshot,settings,selectDemo} = useStudio();
  const [tab,setTab] = useState('activation');
  const [vectorSource,setVectorSource] = useState('prediction');
  const [head,setHead] = useState('0');
  const [threshold,setThreshold] = useState(.02);
  const [query,setQuery] = useState<number | null>(null);
  const flow = useMemo(() => activationFlow(snapshot),[snapshot]);
  const heads = useMemo(() => capturedAttentionHeads(snapshot),[snapshot]);
  const matrix = heads[Math.min(Number(head),heads.length-1)] ?? [];
  const splines = useMemo(() => attentionSplines(matrix,threshold),[matrix,threshold]);
  const vocabulary = Array.isArray(snapshot.extras?.vocabulary) ? snapshot.extras.vocabulary.map(String) : [];
  const tokens = Array.isArray(snapshot.extras?.tokens) ? snapshot.extras.tokens.map(String) : [];
  const embeddings = finiteMatrix(snapshot.extras?.tokenEmbeddings);
  const residual = snapshot.extras?.residualStream as Record<string,unknown> | undefined;
  const representations = [
    {value:'prediction',label:'Captured prediction vectors',vectors:snapshot.samples.map(sample => sample.prediction),labels:snapshot.samples.map((sample,index) => sample.label ?? `Sample ${index+1}`)},
    {value:'input',label:'Captured input vectors',vectors:snapshot.samples.map(sample => sample.input),labels:snapshot.samples.map((sample,index) => sample.label ?? `Sample ${index+1}`)},
    ...(embeddings ? [{value:'embedding',label:'Learned vocabulary embeddings',vectors:embeddings,labels:vocabulary}] : []),
    ...['embedding','attention','feedforward'].flatMap(stage => { const vectors = finiteMatrix(residual?.[stage]); return vectors ? [{value:`residual-${stage}`,label:`Context token ${stage} residuals`,vectors,labels:tokens}] : []; }),
  ];
  const selectedSource = representations.find(item => item.value === vectorSource) ?? representations[0];
  const entropy = matrix.length ? matrix.reduce((sum,row) => sum+attentionEntropy(row),0)/matrix.length : 0;
  return <>
    <PageHeader eyebrow="ADVANCED RESEARCH" title="See computation take shape." description="Measured activations, learned token geometry, causal attention, and editable architecture in one research space." actions={<Badge tone="cyan">LIVE CAPTURE · STEP {snapshot.step}</Badge>} />
    <div className="advanced-overview"><div><Activity size={17} /><strong>{snapshot.layers.reduce((sum,layer) => sum+layer.activations.length,0)}</strong><span>captured activations</span></div><div><Orbit size={17} /><strong>{embeddings?.length ?? snapshot.samples.length}</strong><span>{embeddings ? 'learned token vectors' : 'captured examples'}</span></div><div><Waves size={17} /><strong>{heads.length}</strong><span>actual attention heads</span></div><div><GitBranch size={17} /><strong>{snapshot.parameters.toLocaleString()}</strong><span>model parameters</span></div></div>
    <Panel title="Representation observatory" action={<Badge tone="violet">BOUNDED RENDER BUDGET</Badge>}><div className="tabs">{[{id:'activation',label:'Activation volume',icon:Activity},{id:'galaxy',label:'Latent galaxy',icon:Orbit},{id:'attention',label:'Attention splines',icon:Waves}].map(item => <button key={item.id} className={tab === item.id ? 'active' : ''} onClick={() => setTab(item.id)}><item.icon size={14} />{item.label}</button>)}</div>
      {tab === 'activation' && <><LatentSpaceExplorer points={flow} title="Current layer activations" reducedMotion={settings.reducedMotion} /><p className="advanced-note">Layer grids show actual captured activations. Position is a schematic layer/neuron layout; color and brightness show the signed value normalized by this capture's maximum magnitude. At most 256 neurons per layer are uniformly sampled.</p><div className="advanced-layer-strip">{snapshot.layers.map(layer => <div key={layer.id}><strong>{layer.name}</strong><span>{layer.size} units</span><small>L2 {Math.sqrt(layer.activations.reduce((sum,value) => sum+value*value,0)).toFixed(3)}</small></div>)}</div></>}
      {tab === 'galaxy' && <><div className="advanced-graph-toolbar"><Select label="Representation source" value={selectedSource.value} onChange={setVectorSource} options={representations.map(item => ({value:item.value,label:item.label}))} /><Badge>{selectedSource.vectors[0]?.length ?? 0} dimensions</Badge></div><LatentSpaceGalaxy vectors={selectedSource.vectors} labels={selectedSource.labels} source={selectedSource.label.toLowerCase()} reducedMotion={settings.reducedMotion} /></>}
      {tab === 'attention' && (heads.length ? <><div className="advanced-attention-controls"><Select label="Attention head" value={String(Math.min(Number(head),heads.length-1))} onChange={value => {setHead(value);setQuery(null);}} options={heads.map((_,index) => ({value:String(index),label:`Learned head ${index}`}))} /><label>Minimum weight<input type="range" min={0} max={1} step={.01} value={threshold} onChange={event => setThreshold(Number(event.target.value))} /><span>{threshold.toFixed(2)}</span></label><Badge tone="cyan">{entropy.toFixed(3)} bits mean entropy</Badge><Button variant="ghost" onClick={() => setQuery(null)}>Show every query</Button></div><div className="advanced-attention-canvas"><svg viewBox="0 0 960 330" role="img" aria-label="Cubic attention curves from query tokens to key tokens"><defs><linearGradient id="advanced-attention-gradient" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#31ccff" /><stop offset="1" stopColor="#a87bff" /></linearGradient></defs>{splines.filter(edge => query === null || edge.query === query).map(edge => <path key={`${edge.query}-${edge.key}`} d={edge.path} stroke="url(#advanced-attention-gradient)" fill="none" strokeWidth={edge.width} opacity={.12+.72*Math.min(1,edge.weight)}><title>q{edge.query} → k{edge.key}: {edge.weight.toFixed(6)}</title></path>)}{matrix.map((_,index) => {const x = 40+index/Math.max(1,matrix.length-1)*880;return <g key={index} role="button" tabIndex={0} onClick={() => setQuery(index)} onKeyDown={event => {if(event.key === 'Enter') setQuery(index);}} aria-label={`Filter query ${tokens[index] ?? index}`}><circle cx={x} cy={48} r={query === index ? 10 : 6} fill="#31ccff" /><text x={x} y={25} textAnchor="middle" fill="#bcecff" fontSize={13}>{tokens[index] ?? `q${index}`}</text><circle cx={x} cy={282} r={6} fill="#a87bff" /><text x={x} y={313} textAnchor="middle" fill="#cab8ff" fontSize={13}>{tokens[index] ?? `k${index}`}</text></g>;})}</svg></div><p className="advanced-note">Top row queries attend to bottom row keys. Cubic curve width and opacity encode the real learned causal attention weight. Click a query to isolate its outgoing attention. {splines.length} curves above threshold; at most 512 are displayed.</p><div className="advanced-attention-matrix"><table><thead><tr><th>Query / key</th>{matrix[0]?.map((_,index) => <th key={index}>{tokens[index] ?? index}</th>)}</tr></thead><tbody>{matrix.map((row,index) => <tr key={index}><th>{tokens[index] ?? index}</th>{row.map((value,key) => <td key={key} style={{background:`rgba(97,109,245,${value*.65})`}}>{value.toFixed(4)}</td>)}</tr>)}</tbody></table></div></> : <EmptyState title="No attention tensor in this model" description="The causal Transformer captures actual trained query–key attention. Dense layers do not have an attention matrix." icon={<Waves size={26} />} action={<Button onClick={() => selectDemo('tiny-transformer')}>Initialize causal Transformer</Button>} />)}
    </Panel>
    <InferenceStream {...props} />
    <GraphEditor key={snapshot.demoId} />
    <Callout>Graphics acceleration is optional. The renderer requests a low-power adapter and falls back to a CPU canvas when WebGPU is unavailable. The model's optimization and measurements still run in the browser's numerical engine.</Callout>
  </>;
}
export { InferenceStream } from '../InferenceStream';
export { GraphEditor, ComputationalGraphCanvas } from '../GraphEditor';
export { LatentSpaceGalaxy } from '../../visualization/webgpu/LatentSpaceGalaxy';
