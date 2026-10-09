import { useEffect, useRef, useState } from 'react';
import { Play, Square, ScanLine } from 'lucide-react';
import { useStudio } from '../app/StudioContext';
import { Badge, Button, Callout, Panel } from '../components/ui';
import { TransformerModel, TRANSFORMER_VOCABULARY } from '../engine/transformer';
import { Random } from '../engine/random';
import { chooseToken, encodeContext, inspectTransformer, logitLens, type TransformerInferenceCapture } from './AdvancedResearch/inference';
import './AdvancedResearch/advanced.css';

export interface RemoteInferenceResult { tokens: string[]; probabilities?: number[][]; model: string }
export interface InferenceStreamProps { onRemoteInfer?: (prompt: string, signal: AbortSignal) => Promise<RemoteInferenceResult> }
interface GeneratedToken { token: number; probability: number; capture: TransformerInferenceCapture; duration: number }

export function InferenceStream({onRemoteInfer}: InferenceStreamProps = {}) {
  const {config,snapshot,engine} = useStudio();
  const [prompt,setPrompt] = useState('the robot');
  const [limit,setLimit] = useState(8);
  const [temperature,setTemperature] = useState(0);
  const [tokens,setTokens] = useState<GeneratedToken[]>([]);
  const [selected,setSelected] = useState(0);
  const [contextToken,setContextToken] = useState(3);
  const [running,setRunning] = useState(false);
  const [error,setError] = useState('');
  const [source,setSource] = useState('Capture a model to begin');
  const [remoteResult,setRemoteResult] = useState<RemoteInferenceResult | null>(null);
  const modelRef = useRef<TransformerModel | null>(null);
  const generation = useRef(0);
  const timeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const remoteAbort = useRef<AbortController | null>(null);
  const stop = () => { generation.current++; if (timeout.current) clearTimeout(timeout.current); remoteAbort.current?.abort(); setRunning(false); };
  useEffect(() => () => { generation.current++; if (timeout.current) clearTimeout(timeout.current); remoteAbort.current?.abort(); },[]);
  const start = () => {
    stop(); setError(''); setRemoteResult(null);
    try {
      let context = encodeContext(prompt);
      if (!Number.isInteger(limit) || limit < 1 || limit > 32) throw new Error('Generation is bounded to 1–32 tokens.');
      const localConfig = {...config,demoId:'tiny-transformer',customDataset:undefined};
      const model = new TransformerModel(localConfig);
      if (snapshot.demoId === 'tiny-transformer') {
        const checkpoint = JSON.parse(engine.exportCheckpoint()) as {model:unknown}; model.restore(checkpoint.model);
        setSource(`Frozen current Transformer · step ${snapshot.step} · seed ${config.seed}`);
      } else setSource(`Fresh untrained Transformer · seed ${config.seed}. Train the causal Transformer demo for learned grammar.`);
      modelRef.current = model;
      const random = new Random(config.seed);
      const id = ++generation.current;
      const result: GeneratedToken[] = [];
      setTokens([]); setSelected(0); setRunning(true);
      const next = () => {
        if (id !== generation.current) return;
        try {
          const begun = performance.now(); const capture = inspectTransformer(model,context);
          const chosen = chooseToken(capture.logits,temperature,random.uniform());
          result.push({token:chosen.token,probability:chosen.probabilities[chosen.token],capture,duration:performance.now()-begun});
          setTokens([...result]); setSelected(result.length-1);
          context = [...context.slice(1),chosen.token];
          if (result.length < limit && TRANSFORMER_VOCABULARY[chosen.token] !== '.') timeout.current = setTimeout(next,90);
          else setRunning(false);
        } catch (caught) { setError(caught instanceof Error ? caught.message : 'Inference failed.'); setRunning(false); }
      };
      timeout.current = setTimeout(next,0);
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Invalid prompt.'); }
  };
  const chosen = tokens[Math.min(selected,tokens.length-1)];
  const lens = chosen && modelRef.current ? logitLens(modelRef.current,chosen.capture,contextToken) : [];
  const remote = async () => {
    if (!onRemoteInfer) return; stop(); setError(''); setRunning(true); setTokens([]); setRemoteResult(null);
    const abort = new AbortController(); remoteAbort.current = abort;
    try { const result = await onRemoteInfer(prompt,abort.signal); if (!abort.signal.aborted) { setRemoteResult(result); setSource(`Remote response · ${result.model}`); } }
    catch (caught) { if (!abort.signal.aborted) setError(caught instanceof Error ? caught.message : 'Remote inference failed.'); }
    finally { if (!abort.signal.aborted) setRunning(false); }
  };
  return <Panel title="Inference stream" subtitle="Autoregressive generation from a real, four-context-token Transformer" action={<Badge tone="violet">LOCAL NUMERICAL MODEL</Badge>}>
    <div className="advanced-inference-controls"><label>Prompt<input value={prompt} onChange={event => setPrompt(event.target.value)} disabled={running} /></label><label>Token budget<input type="number" min={1} max={32} value={limit} onChange={event => setLimit(Number(event.target.value))} disabled={running} /></label><label>Temperature<input type="number" min={0} max={2} step={0.1} value={temperature} onChange={event => setTemperature(Number(event.target.value))} disabled={running} /></label><Button onClick={start} disabled={running} variant="primary"><Play size={14} />Generate</Button><Button onClick={stop} disabled={!running}><Square size={14} />Stop</Button>{onRemoteInfer && <Button onClick={() => void remote()} disabled={running}>Remote inference</Button>}</div>
    <p className="advanced-note">Vocabulary: {TRANSFORMER_VOCABULARY.join(' · ')}. Temperature 0 uses greedy selection; other values use a seeded probability draw.</p>
    <div className="advanced-token-stream"><span className="advanced-prompt">{prompt}</span>{tokens.map((item,index) => <button className={index === selected ? 'selected' : ''} key={index} onClick={() => setSelected(index)} title={`p=${item.probability.toFixed(5)}; ${item.duration.toFixed(2)} ms`}>{TRANSFORMER_VOCABULARY[item.token]}<small>{(item.probability*100).toFixed(1)}%</small></button>)}{remoteResult?.tokens.map((token,index) => <span key={index}>{token}</span>)}{running && <span className="advanced-cursor" aria-label="Generating">▍</span>}</div>
    <p className="advanced-note">{source}. {tokens.length > 0 && `${tokens.length} generated tokens · ${(tokens.reduce((sum,item) => sum+item.duration,0)).toFixed(1)} ms measured model compute.`}</p>
    {error && <p role="alert" className="viz-error">{error}</p>}
    {chosen && <div className="advanced-lens"><div className="advanced-lens-heading"><strong><ScanLine size={15} />Token logit lens</strong><label>Context token<select value={contextToken} onChange={event => setContextToken(Number(event.target.value))}>{chosen.capture.context.map((token,index) => <option key={index} value={index}>{index}: {TRANSFORMER_VOCABULARY[token]}</option>)}</select></label></div><table><thead><tr><th>Residual stage</th><th>Top decoded token</th><th>Probability</th><th>Raw logit</th></tr></thead><tbody>{lens.map(row => { const top = row.probabilities.indexOf(Math.max(...row.probabilities)); return <tr key={row.stage}><td>{row.stage}</td><td>{TRANSFORMER_VOCABULARY[top]}</td><td>{(row.probabilities[top]*100).toFixed(2)}%</td><td>{row.logits[top].toFixed(4)}</td></tr>; })}</tbody></table><p className="advanced-note">Each stage uses the model's final normalization and trained vocabulary readout on that token's actual residual vector. Intermediate decoding is a diagnostic projection; only the last context token's final stage is the next-token prediction.</p><div className="advanced-probability-grid">{chosen.capture.probabilities.map((probability,index) => <div key={index}><span>{TRANSFORMER_VOCABULARY[index]}</span><progress value={probability} max={1} /><small>{(probability*100).toFixed(2)}%</small></div>)}</div></div>}
    <Callout>This tiny model learns an explicit synthetic grammar. Its stream is real numerical inference, with a frozen parameter snapshot and bounded CPU execution. Remote token responses appear only when an explicit remote callback is configured; remote hidden states are not fabricated.</Callout>
  </Panel>;
}
