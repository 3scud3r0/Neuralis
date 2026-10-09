import { useEffect, useRef, useState } from 'react';
import { Braces, Cable, Download, Play, RotateCcw } from 'lucide-react';
import { readStored, downloadText, type WorkspaceSettings } from '../../app/storage';
import { Badge, Button, Callout } from '../../components/ui';
import { validateBackendUrl } from '../../integrations/backendClient';
import { analyzeArchitecture, type ArchitectureGraph } from '../../visualization/architecture';
import { finiteMatrix } from '../AdvancedResearch/captures';
import { desktop, isDesktop } from '../../../../desktop/bridge';

type BackendOperation = 'linear' | 'relu' | 'tanh' | 'gelu' | 'silu' | 'identity' | 'add' | 'concat' | 'layernorm';
interface BackendNode { id: string; operation: BackendOperation; inputs: string[]; width?: number }
interface BackendGraph { input_width: number; nodes: BackendNode[]; output: string; seed: number }
interface GraphEvaluation { outputs: number[][]; graph: { order: string[]; dimensions: Record<string,number>; parameters: number; input_width: number; output: string }; compiled: true }
interface BackendGraphSurgeryProps { canvasGraph?: ArchitectureGraph; inputSamples?: number[][]; initialUrl?: string; seed?: number }

const residualPreset: BackendGraph = {
  input_width:2,seed:42,
  nodes:[
    {id:'hidden',operation:'linear',inputs:['input'],width:8},
    {id:'relu',operation:'relu',inputs:['hidden']},
    {id:'skip',operation:'linear',inputs:['input'],width:8},
    {id:'residual',operation:'add',inputs:['relu','skip']},
    {id:'output',operation:'linear',inputs:['residual'],width:2},
  ],output:'output',
};
const defaultInputs = [[0,0],[0,1],[1,0],[1,1]];

export function validateGraphEvaluation(data: unknown, sampleCount: number): GraphEvaluation {
  if (!data || typeof data !== 'object') throw new Error('Backend response omitted compiled graph metadata.');
  const evaluation = data as GraphEvaluation;
  const outputs = finiteMatrix(evaluation.outputs,256);
  if (evaluation.compiled !== true || !evaluation.graph || !Array.isArray(evaluation.graph.order) || !outputs || outputs.length !== sampleCount || !Number.isFinite(evaluation.graph.parameters)) throw new Error('Backend response omitted valid compiled graph metadata or finite aligned outputs.');
  return {...evaluation,outputs};
}

/** Native authentication and network budgets belong to Rust, never to the page's token field. */
export async function evaluateNativeGraph(payload: { graph: BackendGraph; inputs: number[][] }): Promise<GraphEvaluation> {
  const status = await desktop.status();
  if (status.phase !== 'ready' || !status.endpoint) throw new Error('Desktop backend is not ready. Start or connect the runtime in Settings.');
  return validateGraphEvaluation(await desktop.request<unknown>('POST','/api/v1/extensions/graph/evaluate',payload),payload.inputs.length);
}

/** Maps explicit canvas operations; no browser model parameters or implicit activations are transferred. */
export function canvasToBackendGraph(graph: ArchitectureGraph, seed = 42): BackendGraph {
  const analysis = analyzeArchitecture(graph);
  const invalid = analysis.issues.find(issue => issue.severity === 'error');
  if (invalid) throw new Error(invalid.message);
  const inputs = graph.nodes.filter(node => node.operation === 'input');
  const outputs = graph.nodes.filter(node => node.operation === 'output');
  if (inputs.length !== 1 || inputs[0].shape?.length !== 1 || outputs.length !== 1) throw new Error('Backend import requires one vector input and one explicit output.');
  const input = inputs[0];
  const supported = new Set(['dense','relu','tanh','add','concat','layernorm','output']);
  const mapping = new Map(graph.nodes.map(node => [node.id,node.id === input.id ? 'input' : node.id]));
  if (graph.nodes.some(node => node.id === 'input' && node.id !== input.id)) throw new Error('The backend reserves “input” for the graph input.');
  const nodes = analysis.order.filter(id => id !== input.id).map(id => {
    const node = graph.nodes.find(node => node.id === id)!;
    if (!supported.has(node.operation)) throw new Error(`The optional backend does not support canvas operation “${node.operation}”.`);
    if (node.operation === 'dense' && node.bias === false) throw new Error('The backend linear operation currently includes a bias.');
    if (node.operation === 'concat' && (node.axis ?? 0) !== 0) throw new Error('Backend concatenation supports only the feature axis for vector nodes.');
    return {id,operation:(node.operation === 'dense' ? 'linear' : node.operation === 'output' ? 'identity' : node.operation) as BackendOperation,inputs:graph.edges.filter(edge => edge.target === id).map(edge => mapping.get(edge.source)!),...(node.operation === 'dense' ? {width:node.units} : {})};
  });
  return {input_width:input.shape![0],nodes,output:outputs[0].id,seed};
}

function OutputChart({ values }: { values: number[][] }) {
  if (!values.length) return null;
  const colors = ['#51c8ff','#b393ff','#5ae2b2','#ffc26f','#ff879e','#79a6ff','#d998ff','#b9df5f'];
  const channels = Math.min(8,values[0].length);
  const all = values.flatMap(row => row.slice(0,channels));
  const lo = Math.min(...all), hi = Math.max(...all), span = Math.max(1e-12,hi-lo);
  const x = (index:number) => 46+index/Math.max(1,values.length-1)*790;
  const y = (value:number) => 220-(value-lo)/span*175;
  return <div className="backend-graph-chart"><svg viewBox="0 0 880 270" aria-label="Actual backend graph output values by evaluation sample" role="img"><line x1={46} y1={220} x2={836} y2={220} stroke="#354965" /><text x={8} y={50} fill="#92a6bf" fontSize={11}>{hi.toFixed(3)}</text><text x={8} y={220} fill="#92a6bf" fontSize={11}>{lo.toFixed(3)}</text>{Array.from({length:channels},(_,channel) => <g key={channel}><polyline points={values.map((row,index) => `${x(index)},${y(row[channel])}`).join(' ')} fill="none" stroke={colors[channel]} strokeWidth={2} />{values.map((row,index) => <circle key={index} cx={x(index)} cy={y(row[channel])} r={3.5} fill={colors[channel]}><title>Sample {index+1} · output {channel}: {row[channel]}</title></circle>)}<text x={46+channel*99} y={258} fill={colors[channel]} fontSize={10}>Output {channel}</text></g>)}</svg><p className="advanced-note">Evaluation sample index on the horizontal axis; raw output value on the vertical axis. {channels} of {values[0].length} output dimensions displayed.</p></div>;
}

export function BackendGraphSurgery({ canvasGraph,inputSamples,initialUrl,seed = 42 }: BackendGraphSurgeryProps) {
  const [url,setUrl] = useState(initialUrl ?? readStored<Partial<WorkspaceSettings>>('settings',{}).backendUrl ?? 'http://127.0.0.1:8000');
  const [token,setToken] = useState('');
  const [graphText,setGraphText] = useState(JSON.stringify({...residualPreset,seed},null,2));
  const [inputText,setInputText] = useState(JSON.stringify(defaultInputs,null,2));
  const [result,setResult] = useState<GraphEvaluation | null>(null);
  const [submitted,setSubmitted] = useState<{graph:BackendGraph;inputs:number[][]} | null>(null);
  const [error,setError] = useState('');
  const [busy,setBusy] = useState(false);
  const [elapsed,setElapsed] = useState<number | null>(null);
  const [nativeMode,setNativeMode] = useState(false);
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(),[]);
  const clear = () => { setResult(null);setSubmitted(null);setElapsed(null);setError(''); };
  const useDesktopRuntime = async () => {
    clear();setBusy(true);
    try {
      const status = await desktop.status();
      if (status.phase !== 'ready' || !status.endpoint) throw new Error('Desktop backend is not ready. Start or connect the runtime in Settings.');
      setNativeMode(true);setUrl(status.endpoint);setToken('');
    } catch (caught) {setError(caught instanceof Error ? caught.message : String(caught));}
    finally {setBusy(false);}
  };
  const importCanvas = () => {
    if (!canvasGraph) return;
    try {
      const converted = canvasToBackendGraph(canvasGraph,seed);
      setGraphText(JSON.stringify(converted,null,2));
      const samples = inputSamples?.filter(row => row.length === converted.input_width && row.every(Number.isFinite)).slice(0,8);
      setInputText(JSON.stringify(samples?.length ? samples : [Array<number>(converted.input_width).fill(0)],null,2));
      clear();
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Canvas graph import failed.'); }
  };
  const evaluate = async () => {
    clear();
    try {
      const base = nativeMode ? null : validateBackendUrl(url);
      if (graphText.length > 500_000 || inputText.length > 500_000) throw new Error('Graph and input JSON each have a 500 KB request budget.');
      const graph = JSON.parse(graphText) as BackendGraph;
      const inputs = finiteMatrix(JSON.parse(inputText),256);
      if (!graph || typeof graph !== 'object' || !Array.isArray(graph.nodes)) throw new Error('Graph JSON must contain input_width, nodes, output, and seed.');
      if (!inputs || inputs[0].length !== graph.input_width) throw new Error('Input JSON must be a finite rectangular matrix matching graph input_width, with at most 256 samples.');
      const controller = nativeMode ? null : new AbortController();abort.current = controller;setBusy(true);
      const timeout = controller ? setTimeout(() => controller.abort(),30_000) : null;
      const began = performance.now();
      try {
        let evaluation: GraphEvaluation;
        if (nativeMode) evaluation = await evaluateNativeGraph({graph,inputs});
        else {
        const response = await fetch(`${base}/api/v1/extensions/graph/evaluate`,{method:'POST',headers:{Accept:'application/json','Content-Type':'application/json',...(token ? {Authorization:`Bearer ${token}`} : {})},body:JSON.stringify({graph,inputs}),signal:controller!.signal});
        if (Number(response.headers.get('content-length')) > 2_000_000) throw new Error('Graph response exceeds the 2 MB preview budget.');
        const text = await response.text();
        if (text.length > 2_000_000) throw new Error('Graph response exceeds the 2 MB preview budget.');
        let data: unknown;
        try { data = JSON.parse(text); } catch { throw new Error(`Backend returned non-JSON content (HTTP ${response.status}).`); }
        if (!response.ok) {
          const detail = (data as {detail?:unknown}).detail;
          throw new Error(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `Graph evaluation failed (HTTP ${response.status}).`);
        }
        evaluation = validateGraphEvaluation(data,inputs.length);
        }
        if (!controller?.signal.aborted) {setResult(evaluation);setSubmitted({graph,inputs});setElapsed(performance.now()-began);}
      } finally {if (timeout) clearTimeout(timeout);}
    } catch (caught) {setError(abort.current?.signal.aborted ? 'Graph request cancelled or exceeded the 30-second timeout.' : caught instanceof Error ? caught.message : String(caught));}
    finally {setBusy(false);abort.current = null;}
  };
  return <section className="backend-graph-surgery" aria-label="Optional backend graph surgery"><div className="backend-graph-heading"><div><h3><Cable size={16} />Backend graph surgery</h3><p>Compile and evaluate explicit residual and concatenation DAGs on your connected Python backend.</p></div><Badge tone={result ? 'green' : 'amber'}>{result ? 'BACKEND COMPILED' : 'OPTIONAL SERVER'}</Badge></div>
    <div className="backend-graph-connection"><label>{nativeMode ? 'Desktop runtime endpoint' : 'Graph backend URL'}<input aria-label="Graph backend URL" type="url" value={url} disabled={busy || nativeMode} onChange={event => {setUrl(event.target.value);clear();}} /></label>{!nativeMode && <label>Session bearer token<input aria-label="Graph session token" type="password" autoComplete="off" value={token} disabled={busy} onChange={event => setToken(event.target.value)} placeholder="Optional authentication" /></label>}<span>{nativeMode ? 'Native IPC · runtime authentication stays in Rust.' : 'Token stays in this page’s memory.'}</span>{isDesktop() && <Button disabled={busy} onClick={() => void useDesktopRuntime()}>Use desktop runtime</Button>}{nativeMode && <Button disabled={busy} onClick={() => {setNativeMode(false);clear();}}>Use browser HTTP</Button>}</div>
    {nativeMode && <p className="advanced-note">Requests use the native runtime bridge with its 20-second timeout and response limits. In-flight native IPC requests cannot be cancelled from this page.</p>}
    <div className="backend-graph-editors"><label><span><Braces size={14} />Executable graph JSON</span><textarea aria-label="Backend graph JSON" value={graphText} onChange={event => {setGraphText(event.target.value);clear();}} disabled={busy} rows={22} spellCheck={false} /></label><label><span>Evaluation input matrix</span><textarea aria-label="Graph evaluation inputs" value={inputText} onChange={event => {setInputText(event.target.value);clear();}} disabled={busy} rows={22} spellCheck={false} /></label></div>
    <div className="advanced-graph-toolbar"><Button variant="primary" disabled={busy} onClick={() => void evaluate()}><Play size={14} />{busy ? 'Compiling and evaluating…' : 'Compile and evaluate'}</Button>{busy && !nativeMode && abort.current && <Button onClick={() => abort.current?.abort()}>Cancel request</Button>}<Button disabled={busy} onClick={() => {setGraphText(JSON.stringify({...residualPreset,seed},null,2));setInputText(JSON.stringify(defaultInputs,null,2));clear();}}><RotateCcw size={14} />Residual preset</Button>{canvasGraph && <Button disabled={busy} onClick={importCanvas}>Import explicit canvas graph</Button>}{result && submitted && <Button onClick={() => downloadText('neuralis-backend-graph-evaluation.json',JSON.stringify({request:submitted,response:result},null,2))}><Download size={14} />Export measured result</Button>}</div>
    {error && <p className="viz-error" role="alert">{error} {nativeMode ? 'Check the native runtime status in Settings.' : 'Check the backend endpoint, allowed browser origin, and session authentication.'} Browser graph editing remains available.</p>}
    <p className="advanced-note">Supported operations: linear, relu, tanh, gelu, silu, identity, add, concat, and layernorm. Linear nodes require width; every node names its input dependencies. The backend checks cycles and dimensions before compilation. Each request initializes new seeded parameters; it does not transfer the browser model's weights or train this graph.</p>
    {result && submitted && <div className="backend-graph-result"><div className="advanced-projection-summary"><span>{result.graph.parameters.toLocaleString()} actual parameters</span><span>{result.graph.order.length} compiled nodes</span><span>{result.outputs.length} evaluated samples</span><span>{elapsed?.toFixed(1)} ms request round trip</span></div><OutputChart values={result.outputs} /><div className="backend-graph-output-table"><table><thead><tr><th>Sample</th><th>Input</th>{result.outputs[0].slice(0,16).map((_,index) => <th key={index}>Output {index}</th>)}</tr></thead><tbody>{result.outputs.slice(0,64).map((row,index) => <tr key={index}><th>{index+1}</th><td>{submitted.inputs[index].slice(0,16).map(value => value.toFixed(3)).join(', ')}{submitted.inputs[index].length > 16 ? ' …' : ''}</td>{row.slice(0,16).map((value,channel) => <td key={channel}>{value.toFixed(6)}</td>)}</tr>)}</tbody></table></div><p className="advanced-note">Exact values shown for up to 64 samples and 16 output dimensions. Export includes the entire measured response.</p><details><summary>Compiled graph evidence</summary><pre>{JSON.stringify(result.graph,null,2)}</pre></details></div>}
    <Callout>The optional endpoint performs real graph compilation and numerical inference. Without that server, static hosting still provides the canvas editor and supported sequential browser engine. Arbitrary recurrent cycles remain unsupported.</Callout>
  </section>;
}
