import { useMemo, useRef, useState } from 'react';
import { BookOpen, Database, LockKeyhole, Search, ShieldAlert, Trash2, Upload } from 'lucide-react';
import { readStored, removeStored, writeStored } from '../../app/storage';
import { buildRagIndex, ragExample, restoreRagIndex, searchRag, type RagDocument, type RagIndex, type SearchHit } from './ragEngine';
import type { RagBackendResult } from './extensionClient';
import { BackendButton, DownloadButton, ErrorNotice, PanelTitle, ResultJson, Stat, errorMessage, type ExtensionContext } from './shared';

export function RagPanel({ client, capabilities }: ExtensionContext) {
  const [index, setIndex] = useState<RagIndex>(() => {
    const saved = readStored<unknown>('browser-rag', null);
    try { return saved ? restoreRagIndex(JSON.stringify(saved)) : buildRagIndex(ragExample); } catch { return buildRagIndex(ragExample); }
  });
  const [query, setQuery] = useState('How do low rank adapters preserve frozen weights?');
  const [hits, setHits] = useState<SearchHit[]>(() => searchRag(index, 'How do low rank adapters preserve frozen weights?', 3));
  const [source, setSource] = useState('My document');
  const [text, setText] = useState('');
  const [chunkTokens, setChunkTokens] = useState(80);
  const [overlap, setOverlap] = useState(16);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [backendResult, setBackendResult] = useState<RagBackendResult | null>(null);
  const [collection, setCollection] = useState('local');
  const [remoteIndex, setRemoteIndex] = useState<unknown>(null);
  const input = useRef<HTMLInputElement>(null);
  const importedIndex = useRef<HTMLInputElement>(null);
  const totalCharacters = useMemo(() => index.documents.reduce((sum, document) => sum + document.text.length, 0), [index]);
  const build = (documents: RagDocument[]) => {
    const next = buildRagIndex(documents, chunkTokens, overlap);
    setIndex(next); setHits(searchRag(next, query, 5)); setError(null); setBackendResult(null);
    return next;
  };
  const addDocument = () => {
    try {
      if (!text.trim() || !source.trim()) throw new Error('Enter document text and a source name.');
      const identifier = `doc-${Date.now()}-${index.documents.length}`;
      build([...index.documents, { id: identifier, source: source.trim(), text }]);
      setText(''); setNotice('Document indexed in memory. Use Save in browser to persist this collection.');
    } catch (failure) { setError(errorMessage(failure)); }
  };
  const search = () => {
    try { setHits(searchRag(index, query, 5)); setError(null); setBackendResult(null); }
    catch (failure) { setError(errorMessage(failure)); }
  };
  const remote = async (kind: 'index' | 'search') => {
    if (!client) return;
    setBusy(true); setError(null); setNotice(null);
    try {
      if (kind === 'index') {
        if (!text.trim()) throw new Error('Enter document text to send to the encrypted backend collection.');
        setRemoteIndex(await client.index({ text, source, collection }));
      } else setBackendResult(await client.search({ query, collection, limit: 5 }));
    } catch (failure) { setError(errorMessage(failure)); } finally { setBusy(false); }
  };
  return <div className="rw-panel">
    <PanelTitle eyebrow="05 / LOCAL RETRIEVAL" title="Find a passage. Keep its source.">Chunk your own documents and rank passages with deterministic signed feature hashing and cosine similarity. No language model or embedding service is called.</PanelTitle>
    <div className="rw-banner warn"><ShieldAlert size={18}/><span>Browser storage is not encrypted. Text stays in this browser unless you explicitly send it to the connected backend. Saved collections can be read by this site on the same device.</span></div>
    <ErrorNotice error={error}/>{notice && <div className="rw-banner good" role="status">{notice}</div>}
    <div className="rw-stats"><Stat label="Indexed documents" value={index.documents.length}/><Stat label="Passage chunks" value={index.chunks.length} note={`${index.chunkTokens} tokens / ${index.overlapTokens} overlap`}/><Stat label="Index dimensions" value={index.dimensions} note="Signed unigrams + weighted bigrams"/><Stat label="Corpus size" value={`${(totalCharacters / 1000).toFixed(1)}k`} note="Characters · 2 MB text budget"/></div>
    <div className="rw-search-bar"><Search size={21}/><input aria-label="Retrieval query" value={query} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') search(); }} placeholder="Ask a retrieval question…"/><button className="rw-primary" onClick={search}>Search passages</button></div>
    <div className="rw-two-col"><div className="rw-card"><div className="rw-card-heading"><h3>Retrieved evidence</h3><span className="rw-chip">{backendResult ? 'Encrypted backend results' : 'Browser cosine ranking'}</span></div>{backendResult ? backendResult.results.map((hit, index) => <article className="rw-hit" key={index}><div><span className="rw-hit-rank">{index + 1}</span><strong>{hit.source}</strong><code>{hit.score.toFixed(4)}</code></div><p>{hit.text}</p></article>) : hits.length ? hits.map((hit, index) => <article className="rw-hit" key={hit.chunk.id}><div><span className="rw-hit-rank">{index + 1}</span><strong>{hit.chunk.source}</strong><code>{hit.score.toFixed(4)}</code></div><p>{hit.chunk.text}</p><footer><span>Characters {hit.chunk.start}–{hit.chunk.end}</span><span>{hit.matchedTerms.slice(0, 8).join(' · ') || 'Hash similarity only'}</span></footer></article>) : <div className="rw-empty"><Search size={26}/><h4>No positively similar passages</h4><p>Try terms that appear in your corpus. Hash collisions can affect results; scores are similarity values, not probabilities.</p></div>}<p className="rw-note">These results are retrieved passages. The workbench does not invent an answer. Hash embeddings provide lexical retrieval and can collide; they do not replace a semantic embedding model.</p></div>
      <div className="rw-card"><h3>Add a document</h3><label>Source / filename<input value={source} onChange={event => setSource(event.target.value)} maxLength={200}/></label><textarea aria-label="RAG document text" rows={8} value={text} onChange={event => setText(event.target.value)} placeholder="Paste plain text to index locally…"/><div className="rw-fields"><label>Chunk tokens<select value={chunkTokens} onChange={event => setChunkTokens(Number(event.target.value))}>{[32, 64, 80, 128, 256].map(value => <option key={value}>{value}</option>)}</select></label><label>Overlap tokens<input type="number" min={0} max={chunkTokens - 1} value={overlap} onChange={event => setOverlap(Number(event.target.value))}/></label></div><div className="rw-actions"><button className="rw-primary" onClick={addDocument}><Database size={15}/>Add to browser index</button><button onClick={() => input.current?.click()}><Upload size={15}/>Read text file</button><input ref={input} hidden type="file" accept=".txt,.md,.csv,text/plain,text/markdown,text/csv" onChange={async event => {
        const file = event.target.files?.[0]; if (!file) return;
        try { if (file.size > 200000) throw new Error('A document is limited to 200,000 bytes.'); setText(await file.text()); setSource(file.name); setNotice('File read into the editor. Add it to the browser index when ready.'); }
        catch (failure) { setError(errorMessage(failure)); } event.target.value = '';
      }}/></div><button onClick={() => { try { build(index.documents); setNotice('Current documents rechunked with the selected settings.'); } catch (failure) { setError(errorMessage(failure)); } }}>Rebuild current corpus</button></div></div>
    <div className="rw-card"><div className="rw-card-heading"><h3><BookOpen size={18}/>Collection management</h3><span className="rw-chip">In-memory until saved</span></div><div className="rw-document-list">{index.documents.map(document => <div key={document.id}><span><strong>{document.source}</strong><small>{document.text.length.toLocaleString()} characters · {index.chunks.filter(chunk => chunk.documentId === document.id).length} chunks</small></span><button aria-label={`Remove ${document.source}`} disabled={index.documents.length === 1} onClick={() => { try { build(index.documents.filter(item => item.id !== document.id)); } catch (failure) { setError(errorMessage(failure)); } }}><Trash2 size={15}/></button></div>)}</div><div className="rw-actions"><button onClick={() => setNotice(writeStored('browser-rag', index) ? 'Collection saved in unencrypted browser storage.' : 'Browser storage is unavailable or full; export the collection instead.')}>Save in browser</button><button onClick={() => { removeStored('browser-rag'); setNotice('Saved collection cleared from this browser. The current in-memory index remains available.'); }}>Clear saved collection</button><DownloadButton filename="neuralis-browser-rag.json" data={index} label="Export collection"/><button onClick={() => importedIndex.current?.click()}>Import collection</button><button onClick={() => { setIndex(buildRagIndex(ragExample)); setHits(searchRag(buildRagIndex(ragExample), query, 5)); setNotice('Example corpus restored in memory.'); }}>Restore example</button><input ref={importedIndex} hidden type="file" accept=".json,application/json" onChange={async event => {
      const file = event.target.files?.[0]; if (!file) return;
      try { if (file.size > 5_000_000) throw new Error('Collection import exceeds 5 MB.'); const next = restoreRagIndex(await file.text()); setIndex(next); setHits(searchRag(next, query, 5)); setError(null); setNotice('Collection imported; embeddings rebuilt from original documents.'); }
      catch (failure) { setError(errorMessage(failure)); } event.target.value = '';
    }}/></div></div>
    <div className="rw-card"><div className="rw-card-heading"><h3><LockKeyhole size={18}/>Encrypted backend collection</h3><span className={`rw-chip ${capabilities?.rag.available && capabilities.rag.encrypted ? 'good' : ''}`}>{capabilities?.rag.available && capabilities.rag.encrypted ? 'Advertised by connected backend' : 'Optional integration'}</span></div><p>These actions explicitly send your editor text or query to your configured backend. Backend collection operations require an instance-admin session. Browser storage above remains unencrypted.</p><div className="rw-actions"><label>Backend collection<input value={collection} onChange={event => setCollection(event.target.value)} maxLength={64}/></label><BackendButton available={Boolean(client && capabilities?.rag.available && capabilities.rag.encrypted)} busy={busy} onClick={() => void remote('index')}>Send editor text to backend</BackendButton><BackendButton available={Boolean(client && capabilities?.rag.available && capabilities.rag.encrypted)} busy={busy} onClick={() => void remote('search')}>Search backend collection</BackendButton></div>{!capabilities?.rag.available && <p className="rw-note">{capabilities?.rag.reason ?? 'Connect a backend with encrypted local retrieval enabled to use this integration.'}</p>}{Boolean(remoteIndex) && <ResultJson value={remoteIndex} label="Backend indexing response"/>}{backendResult && <ResultJson value={backendResult} label="Backend search response"/>}</div>
  </div>;
}
