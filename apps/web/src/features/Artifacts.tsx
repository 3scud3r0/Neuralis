import { useMemo, useRef, useState } from 'react';
import { Save, Upload, Download, Search, FolderArchive, Trash2, RotateCcw, FileJson, GitCompareArrows, Copy, Check } from 'lucide-react';
import { useStudio } from '../app/StudioContext';
import { DEMOS } from '../engine';
import type { RunConfig } from '../types';
import type { SavedCheckpoint } from '../app/storage';
import { downloadText } from '../app/storage';
import { Panel, PageHeader, Select, Button, Badge, EmptyState, KeyValue, Field, Callout } from '../components/ui';

interface CheckpointMetadata {
  schema: string;
  version: number;
  runId: string;
  step: number;
  config: RunConfig;
}
function metadata(checkpoint: SavedCheckpoint): CheckpointMetadata | null {
  try {
    const value = JSON.parse(checkpoint.data) as CheckpointMetadata;
    if (!value.config || typeof value.config.demoId !== 'string') return null;
    return value;
  } catch { return null; }
}
export function Artifacts() {
  const { checkpoints, saveCheckpoint, loadCheckpoint, deleteCheckpoint, importCheckpoint, snapshot, notify } = useStudio();
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [sort, setSort] = useState('newest');
  const [selectedId, setSelectedId] = useState(checkpoints[0]?.id ?? '');
  const [compareId, setCompareId] = useState('');
  const [name, setName] = useState('');
  const file = useRef<HTMLInputElement>(null);
  const filtered = useMemo(() => checkpoints.filter(item => (filter === 'all' || item.demoId === filter) && `${item.name} ${item.demoId}`.toLowerCase().includes(query.toLowerCase())).sort((a, b) => sort === 'step' ? b.step - a.step : sort === 'size' ? b.data.length - a.data.length : Date.parse(b.createdAt) - Date.parse(a.createdAt)), [checkpoints, query, filter, sort]);
  const selected = checkpoints.find(item => item.id === selectedId) ?? filtered[0];
  const comparison = checkpoints.find(item => item.id === compareId);
  const selectedMeta = selected ? metadata(selected) : null;
  const comparisonMeta = comparison ? metadata(comparison) : null;
  const totalBytes = checkpoints.reduce((sum, item) => sum + new Blob([item.data]).size, 0);
  const importFile = async (input: File) => {
    if (input.size > 10 * 1024 * 1024) { notify('Checkpoint exceeds the 10 MB browser import limit. Use the backend for larger models.', 'error'); return; }
    importCheckpoint(await input.text());
  };
  return <>
    <PageHeader eyebrow="CHECKPOINTS & ARTIFACTS" title="Keep every meaningful model state." description="Name, inspect, compare, and export your saved browser checkpoints." actions={<><Button onClick={() => file.current?.click()} disabled={snapshot.state === 'running'}><Upload size={14} />Import checkpoint</Button><Button variant="primary" onClick={() => saveCheckpoint()}><Save size={14} />Save current state</Button></>} />
    <input type="file" ref={file} hidden accept="application/json,.json" onChange={async event => { const selectedFile = event.target.files?.[0]; if (selectedFile) await importFile(selectedFile); event.target.value = ''; }} />
    <div className="artifact-summary"><div><FolderArchive size={22} /><span><strong>{checkpoints.length}</strong><small>Saved checkpoints</small></span></div><div><FileJson size={22} /><span><strong>{(totalBytes / 1024).toFixed(1)} KB</strong><small>Local artifact data</small></span></div><div><Save size={22} /><span><strong>Step {snapshot.step.toLocaleString()}</strong><small>Current save point</small></span></div></div>
    <div className="artifact-toolbar"><div className="library-search"><Search size={15} /><input value={query} aria-label="Search checkpoints" onChange={event => setQuery(event.target.value)} placeholder="Search checkpoint name…" /></div><Select label="Filter artifact model" value={filter} onChange={setFilter} options={[{ value: 'all', label: 'All models' }, ...Array.from(new Set(checkpoints.map(item => item.demoId))).map(id => ({ value: id, label: DEMOS.find(demo => demo.id === id)?.title ?? id }))]} /><Select label="Sort artifacts" value={sort} onChange={setSort} options={[{ value: 'newest', label: 'Newest first' }, { value: 'step', label: 'Highest step' }, { value: 'size', label: 'Largest file' }]} /></div>
    <div className="artifacts-layout"><Panel title="Saved model states" subtitle="Stored in this browser; export important artifacts">{filtered.length ? <div className="artifact-list">{filtered.map(item => <button key={item.id} className={`artifact-list-item ${selected?.id === item.id ? 'selected' : ''}`} onClick={() => setSelectedId(item.id)}><span className="artifact-file-icon"><FileJson size={20} /></span><span><strong>{item.name}</strong><small>{DEMOS.find(demo => demo.id === item.demoId)?.title ?? item.demoId}</small><span className="artifact-item-meta">Step {item.step.toLocaleString()} · {new Date(item.createdAt).toLocaleDateString()}</span></span><Badge tone="cyan">{(new Blob([item.data]).size / 1024).toFixed(1)} KB</Badge></button>)}</div> : <EmptyState title="No checkpoints match this view" description="Save a model state or adjust the search and model filter." icon={<FolderArchive size={27} />} />}</Panel><div className="artifact-detail-stack"><Panel title="Checkpoint details" action={selected && <Badge tone="green"><Check size={10} />Local artifact</Badge>}>{selected ? <><div className="artifact-detail"><h3>{selected.name}</h3><KeyValue label="Demo" value={selected.demoId} /><KeyValue label="Step" value={selected.step.toLocaleString()} /><KeyValue label="Created" value={new Date(selected.createdAt).toLocaleString()} /><KeyValue label="File size" value={`${new Blob([selected.data]).size.toLocaleString()} bytes`} /><KeyValue label="Schema" value={selectedMeta?.schema ?? 'Unknown'} /><KeyValue label="Version" value={selectedMeta?.version ?? 'Unknown'} /><KeyValue label="Seed" value={selectedMeta?.config.seed ?? 'Unknown'} /><KeyValue label="Optimizer" value={selectedMeta?.config.optimizer ?? 'adam'} /><KeyValue label="Architecture override" value={selectedMeta?.config.hiddenLayers?.join(' → ') ?? 'Demo default'} /></div><div className="panel-bottom"><Button variant="primary" disabled={snapshot.state === 'running'} onClick={() => loadCheckpoint(selected)}><RotateCcw size={14} />Restore model</Button><Button onClick={() => downloadText(`neuralis-${selected.demoId}-${selected.step}.json`, selected.data)}><Download size={14} />Export</Button><button className="icon-button" aria-label="Delete selected checkpoint" onClick={() => { deleteCheckpoint(selected.id); setSelectedId(''); }}><Trash2 size={15} /></button></div></> : <EmptyState title="Select an artifact" description="Checkpoint metadata and actions appear here." />}</Panel><Panel title="Save a named checkpoint" subtitle="Capture the exact current model and optimizer"><div className="control-stack"><Field label="Checkpoint name"><input value={name} onChange={event => setName(event.target.value)} placeholder="Before activation experiment" maxLength={80} /></Field><Button onClick={() => { saveCheckpoint(name); setName(''); }}><Save size={14} />Save current state</Button></div></Panel><Panel title="Compare checkpoint configuration" action={<GitCompareArrows size={16} />}><div className="control-stack"><Select label="Comparison checkpoint" value={compareId} onChange={setCompareId} options={[{ value: '', label: 'Choose another checkpoint' }, ...checkpoints.filter(item => item.id !== selected?.id).map(item => ({ value: item.id, label: `${item.name} · step ${item.step}` }))]} />{selectedMeta && comparisonMeta && <table className="data-table metadata-comparison"><thead><tr><th>Setting</th><th>Selected</th><th>Comparison</th></tr></thead><tbody>{(['seed', 'learningRate', 'batchSize', 'maxSteps', 'optimizer', 'captureEvery'] as const).map(key => <tr key={key} className={selectedMeta.config[key] !== comparisonMeta.config[key] ? 'changed' : ''}><td>{key}</td><td>{String(selectedMeta.config[key] ?? 'default')}</td><td>{String(comparisonMeta.config[key] ?? 'default')}</td></tr>)}</tbody></table>}<Button disabled={!selectedMeta} variant="ghost" onClick={async () => { if (!selectedMeta) return; try { await navigator.clipboard.writeText(JSON.stringify(selectedMeta.config, null, 2)); notify('Checkpoint configuration copied.', 'success'); } catch { notify('Clipboard is unavailable. Export the checkpoint instead.'); } }}><Copy size={14} />Copy selected configuration</Button></div></Panel></div></div>
    <Callout>Browser storage is limited and can be cleared by your browser. This workspace keeps up to 15 local checkpoints. Export files for durable storage, or use native filesystem export in the desktop application.</Callout>
  </>;
}
