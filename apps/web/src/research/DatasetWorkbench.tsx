import { useMemo, useRef, useState } from 'react';
import { CheckCircle2, Database, Download, FileSpreadsheet, Play, Upload } from 'lucide-react';
import { useStudio } from '../app/StudioContext';
import { Random } from '../engine/random';
import { downloadText } from '../app/storage';
import { duplicateRows, parseCsv, prepareDataset, profileColumns, type ColumnProfile, type CsvTable, type EncodedDataset, type FeatureSpec } from './datasetTools';
import './research.css';

function exampleCsv(): string {
  const random = new Random(42);
  const rows = ['horizontal,vertical,quadrant'];
  for (let index = 0; index < 240; index++) {
    const x = random.uniform() * 2 - 1;
    const y = random.uniform() * 2 - 1;
    rows.push(`${x.toFixed(6)},${y.toFixed(6)},${x * y > 0 ? 'matching' : 'opposite'}`);
  }
  return rows.join('\n');
}

export function DatasetWorkbench() {
  const { config, setConfig, notify } = useStudio();
  const [source, setSource] = useState('');
  const [name, setName] = useState('My CSV experiment');
  const [delimiter, setDelimiter] = useState(',');
  const [table, setTable] = useState<CsvTable | null>(null);
  const [profiles, setProfiles] = useState<ColumnProfile[]>([]);
  const [target, setTarget] = useState('');
  const [task, setTask] = useState<'classification' | 'regression'>('classification');
  const [selected, setSelected] = useState<string[]>([]);
  const [scaling, setScaling] = useState<NonNullable<FeatureSpec['scaling']>>('standard');
  const [validation, setValidation] = useState(0.2);
  const [prepared, setPrepared] = useState<EncodedDataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const duplicates = useMemo(() => table ? duplicateRows(table).duplicates : 0, [table]);

  const parse = (text = source) => {
    try {
      const parsed = parseCsv(text, { delimiter, maxRows: 10000 });
      const info = profileColumns(parsed);
      setTable(parsed);
      setProfiles(info);
      setTarget(parsed.columns[parsed.columns.length - 1]);
      setSelected(parsed.columns.slice(0, -1));
      setPrepared(null);
      setError(null);
      setLoaded(false);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'The dataset could not be parsed.');
      setTable(null);
      setPrepared(null);
    }
  };

  const build = (): EncodedDataset | null => {
    if (!table) return null;
    try {
      const features: FeatureSpec[] = selected.filter(column => column !== target).map(column => ({
        column,
        kind: profiles.find(profile => profile.name === column)?.type === 'numeric' ? 'numeric' : 'categorical',
        scaling,
        imputation: 'median',
      }));
      const dataset = prepareDataset(table, { features, target, task, validationFraction: validation, seed: config.seed });
      setPrepared(dataset);
      setError(null);
      return dataset;
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'The dataset could not be prepared.');
      setPrepared(null);
      return null;
    }
  };

  const load = () => {
    const dataset = build();
    if (!dataset) return;
    const take = (rows: number[][], indices: number[]) => indices.map(index => rows[index]);
    setConfig({
      demoId: 'custom-data',
      hiddenLayers: [24, 16],
      learningRate: 0.01,
      customDataset: {
        name,
        inputs: take(dataset.inputs, dataset.split.train),
        targets: take(dataset.targets, dataset.split.train),
        validationInputs: take(dataset.inputs, dataset.split.validation),
        validationTargets: take(dataset.targets, dataset.split.validation),
        task,
        labels: dataset.labels,
        featureNames: dataset.featureNames,
      },
    });
    setLoaded(true);
    notify('Custom dataset loaded. Only training rows were used to fit preprocessing; validation remains held out.', 'success');
  };

  return <section className="research-panel" aria-label="Custom CSV dataset preparation">
    <div className="research-heading"><FileSpreadsheet size={22}/><div><h2>Bring your own question</h2><p>Import a CSV, fit preprocessing on training rows and train a real dense model locally.</p></div><span className="research-chip">No upload to a server</span></div>
    <div className="research-actions"><button onClick={() => input.current?.click()}><Upload size={15}/>Choose CSV</button><button onClick={() => { const text = exampleCsv(); setSource(text); setName('Quadrant classification'); parse(text); }}>Use a reproducible example</button><input type="file" accept=".csv,.tsv,text/csv,text/tab-separated-values" hidden ref={input} onChange={async event => {
      const file = event.target.files?.[0];
      if (!file) return;
      try {
        if (file.size > 8 * 1024 * 1024) throw new Error('Import is limited to 8 MiB.');
        const text = await file.text();
        setSource(text);
        setName(file.name.replace(/\.[^.]+$/, ''));
        parse(text);
      } catch (failure) {
        setError(failure instanceof Error ? failure.message : 'The file could not be read.');
      }
      event.target.value = '';
    }}/></div>
    <details className="research-details" open={!table}><summary>CSV source and parsing</summary><textarea aria-label="CSV source" rows={7} value={source} onChange={event => { setSource(event.target.value); setLoaded(false); }} placeholder={'feature_one,feature_two,label\n0.1,0.2,positive\n-0.4,0.7,negative'}/><div className="research-actions"><label>Delimiter<select value={delimiter} onChange={event => setDelimiter(event.target.value)}><option value=",">Comma</option><option value=";">Semicolon</option><option value={'\t'}>Tab</option></select></label><button disabled={!source.trim()} onClick={() => parse()}><Database size={15}/>Inspect dataset</button></div></details>
    {error && <div className="research-error" role="alert">{error}</div>}
    {table && <>
      <div className="research-stat-row"><span><strong>{table.rows.length.toLocaleString()}</strong> observations</span><span><strong>{table.columns.length}</strong> columns</span><span><strong>{duplicates}</strong> duplicate observations</span></div>
      {duplicates > 0 && <p className="research-warning">Duplicate observations can inflate validation performance if identical rows occur in both splits. Remove duplicates or use group-aware splitting before claiming generalization.</p>}
      <div className="research-grid"><div className="research-controls">
        <label>Experiment name<input value={name} onChange={event => setName(event.target.value)}/></label>
        <label>Target column<select aria-label="CSV target column" value={target} onChange={event => { setTarget(event.target.value); setPrepared(null); setLoaded(false); }}>{table.columns.map(column => <option key={column}>{column}</option>)}</select></label>
        <label>Task<select value={task} onChange={event => { setTask(event.target.value as typeof task); setPrepared(null); setLoaded(false); }}><option value="classification">Classification</option><option value="regression">Regression</option></select></label>
        <label>Numeric scaling<select value={scaling} onChange={event => { setScaling(event.target.value as typeof scaling); setPrepared(null); }}><option value="standard">Standard score (training mean/std)</option><option value="minmax">Min/max (training extrema)</option><option value="robust">Robust (training median/IQR)</option><option value="none">None</option></select></label>
        <label>Validation fraction<input type="number" step={0.05} min={0.05} max={0.45} value={validation} onChange={event => { setValidation(Number(event.target.value)); setPrepared(null); }}/></label>
        <p>Numeric missing values use the training median. Categorical features use training-only one-hot vocabularies; unseen validation categories are identified.</p>
      </div><div className="research-table-wrap"><table className="research-table"><thead><tr><th>Input</th><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr></thead><tbody>{profiles.map(profile => <tr key={profile.name}><td><input aria-label={`Use ${profile.name} as an input`} type="checkbox" disabled={profile.name === target} checked={selected.includes(profile.name) && profile.name !== target} onChange={event => { setSelected(current => event.target.checked ? [...current, profile.name] : current.filter(column => column !== profile.name)); setPrepared(null); }}/></td><td>{profile.name}{profile.name === target && <span className="research-chip">target</span>}</td><td>{profile.type}</td><td>{profile.missing}</td><td>{profile.unique}</td></tr>)}</tbody></table></div></div>
      <div className="research-actions"><button onClick={() => build()}>Validate and preview encoding</button><button className="research-primary" onClick={load}><Play size={15}/>Load into training workspace</button>{loaded && <span className="research-success"><CheckCircle2 size={15}/>Ready to train from Training</span>}</div>
    </>}
    {prepared && <div className="research-result"><h3>Preprocessing provenance</h3><p>{prepared.split.train.length} training rows · {prepared.split.validation.length} held-out rows · {prepared.featureNames.length} encoded features · seed {prepared.split.seed}</p><p>{prepared.split.method}. All imputation values, scaling parameters and category vocabularies were fitted exclusively on training rows.</p>{prepared.warnings.map(warning => <p key={warning} className="research-warning">{warning}</p>)}<div className="research-actions"><button onClick={() => downloadText('neuralis-preprocessing.json', JSON.stringify({ schema: 'neuralis.preprocessing/v1', encoder: prepared.encoder, split: prepared.split, labels: prepared.labels, task: prepared.task }, null, 2))}><Download size={15}/>Export encoder and split</button></div><pre>{JSON.stringify({ featureNames: prepared.featureNames, firstTrainingInput: prepared.inputs[prepared.split.train[0]], firstTrainingTarget: prepared.targets[prepared.split.train[0]] }, null, 2)}</pre></div>}
  </section>;
}
