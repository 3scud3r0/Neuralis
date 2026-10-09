import { useMemo, useRef, useState } from 'react';
import { CheckCircle2, Database, Play, Upload } from 'lucide-react';
import { downloadText } from '../../app/storage';
import { generateArithmetic, splitRecords, toJsonl, validateJsonl, type ArithmeticRecord, type FoundryReport } from './foundryEngine';
import { DownloadButton, ErrorNotice, PanelTitle, Stat, errorMessage } from './shared';

const initialRecords = generateArithmetic(24, 42, 25, ['addition', 'subtraction', 'multiplication']);
export function FoundryPanel() {
  const [count, setCount] = useState(100);
  const [seed, setSeed] = useState(42);
  const [maximum, setMaximum] = useState(25);
  const [kinds, setKinds] = useState<ArithmeticRecord['metadata']['kind'][]>(['addition', 'subtraction', 'multiplication']);
  const [records, setRecords] = useState(initialRecords);
  const [source, setSource] = useState(toJsonl(initialRecords));
  const [report, setReport] = useState<FoundryReport>(() => validateJsonl(toJsonl(initialRecords)));
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [fraction, setFraction] = useState(0.2);
  const input = useRef<HTMLInputElement>(null);
  const split = useMemo(() => records.length >= 2 ? splitRecords(records, fraction, seed) : null, [records, fraction, seed]);
  const generate = () => {
    try { const next = generateArithmetic(count, seed, maximum, kinds); const jsonl = toJsonl(next); setRecords(next); setSource(jsonl); setReport(validateJsonl(jsonl)); setDirty(false); setError(null); }
    catch (failure) { setError(errorMessage(failure)); }
  };
  const validate = (text = source) => {
    try { const next = validateJsonl(text); setReport(next); setRecords(next.records); setDirty(false); setError(null); }
    catch (failure) { setError(errorMessage(failure)); }
  };
  return <div className="rw-panel">
    <PanelTitle eyebrow="04 / DATA FOUNDRY" title="Make data you can actually verify.">Generate deterministic arithmetic chat records, recompute every answer, detect duplicates and export a reproducible train / evaluation split.</PanelTitle>
    <ErrorNotice error={error}/>
    <div className="rw-stats"><Stat label="Accepted records" value={report.valid.toLocaleString()} note={`${report.total} nonempty input rows`}/><Stat label="Validation issues" value={report.issues.length} note="Syntax, schema, answer and duplicates"/><Stat label="Train / evaluation" value={split ? `${split.train.length} / ${split.evaluation.length}` : '—'} note="Unique expression split"/><Stat label="Generation method" value="Deterministic" note="Seeded PRNG + exact integer arithmetic"/></div>
    <div className="rw-two-col"><div className="rw-card"><div className="rw-card-heading"><h3>Dataset recipe</h3><span className="rw-chip">Runs in your browser</span></div><div className="rw-fields"><label>Number of records<input type="number" min={2} max={2000} value={count} onChange={event => setCount(Number(event.target.value))}/></label><label>Seed<input type="number" value={seed} onChange={event => setSeed(Number(event.target.value))}/></label><label>Maximum operand<input type="number" min={1} max={1000} value={maximum} onChange={event => setMaximum(Number(event.target.value))}/></label><label>Evaluation fraction<select value={fraction} onChange={event => setFraction(Number(event.target.value))}>{[0.1, 0.2, 0.3, 0.4].map(value => <option key={value} value={value}>{value * 100}%</option>)}</select></label></div><fieldset className="rw-check-list"><legend>Allowed operations</legend>{(['addition', 'subtraction', 'multiplication'] as const).map(kind => <label className="rw-checkbox" key={kind}><input type="checkbox" checked={kinds.includes(kind)} onChange={event => setKinds(current => event.target.checked ? [...current, kind] : current.filter(value => value !== kind))}/>{kind.charAt(0).toUpperCase() + kind.slice(1)}</label>)}</fieldset><button className="rw-primary" onClick={generate}><Play size={15}/>Generate and validate</button><div className="rw-banner"><Database size={17}/><span>This is a programmatic data generator. It calls no language model and claims no diversity beyond the selected arithmetic space.</span></div></div>
      <div className="rw-card"><div className="rw-card-heading"><h3>Independent validation</h3><span className={`rw-chip ${dirty ? 'warn' : report.issues.length ? 'warn' : 'good'}`}>{dirty ? 'Changes not validated' : report.issues.length ? 'Review required' : 'Checks passed'}</span></div><div className="rw-validation-check"><CheckCircle2 size={18}/><div><strong>Schema and conversation roles</strong><p>Each accepted row has a unique id, user / assistant messages and bounded integer metadata.</p></div></div><div className="rw-validation-check"><CheckCircle2 size={18}/><div><strong>Prompt / answer agreement</strong><p>Both the canonical prompt and the answer are checked against a freshly recomputed result.</p></div></div><div className="rw-validation-check"><CheckCircle2 size={18}/><div><strong>Expression deduplication</strong><p>Repeated operation / operand tuples are rejected before splitting. Equivalent commutative problems can still occur across splits.</p></div></div><p className="rw-note">A held-out expression split tests different exact problems. It does not guarantee held-out operands or test extrapolation to longer arithmetic.</p><div className="rw-actions"><button onClick={() => input.current?.click()}><Upload size={15}/>Import JSONL</button><input ref={input} type="file" accept=".jsonl,application/json,text/plain" hidden onChange={async event => {
        const file = event.target.files?.[0]; if (!file) return;
        try { if (file.size > 4_000_000) throw new Error('Import is limited to 4 MB.'); const text = await file.text(); setSource(text); validate(text); }
        catch (failure) { setError(errorMessage(failure)); }
        event.target.value = '';
      }}/><button onClick={() => validate()}>Validate editor</button></div></div></div>
    <div className="rw-card"><div className="rw-card-heading"><h3>Record preview</h3><span className="rw-chip">First {Math.min(8, records.length)} accepted records</span></div><div className="rw-table-scroll"><table><thead><tr><th>Identifier</th><th>User instruction</th><th>Verified answer</th><th>Operation</th></tr></thead><tbody>{records.slice(0, 8).map(record => <tr key={record.id}><td><code>{record.id}</code></td><td>{record.messages[0].content}</td><td><strong>{record.messages[1].content}</strong></td><td>{record.metadata.kind}</td></tr>)}</tbody></table></div><div className="rw-actions"><DownloadButton filename="neuralis-math.jsonl" data={toJsonl(records)} label="Export accepted JSONL" mime="application/x-ndjson"/>{split && <><button onClick={() => downloadText('neuralis-math-train.jsonl', toJsonl(split.train), 'application/x-ndjson')}>Export training split</button><button onClick={() => downloadText('neuralis-math-evaluation.jsonl', toJsonl(split.evaluation), 'application/x-ndjson')}>Export evaluation split</button><DownloadButton filename="neuralis-foundry-manifest.json" label="Export provenance" data={{ schema: 'neuralis.foundry-manifest/v1', seed, fraction, trainIds: split.train.map(record => record.id), evaluationIds: split.evaluation.map(record => record.id), validation: { total: report.total, accepted: report.valid, issues: report.issues } }}/></>}</div></div>
    <details className="rw-details"><summary>JSONL editor · edit a row to test validation</summary><textarea aria-label="Arithmetic JSONL editor" rows={10} value={source} onChange={event => { setSource(event.target.value); setDirty(true); }}/><button onClick={() => validate()}>Revalidate edited source</button></details>
    {report.issues.length > 0 && <div className="rw-card"><h3>Validation issues</h3><div className="rw-table-scroll"><table><thead><tr><th>Line</th><th>Category</th><th>Reason</th></tr></thead><tbody>{report.issues.slice(0, 100).map((issue, index) => <tr key={index}><td>{issue.line}</td><td>{issue.category}</td><td>{issue.message}</td></tr>)}</tbody></table></div>{report.issues.length > 100 && <p>Showing the first 100 of {report.issues.length} issues.</p>}</div>}
  </div>;
}
