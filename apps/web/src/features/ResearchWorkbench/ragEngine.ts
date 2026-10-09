export interface RagDocument { id: string; source: string; text: string; }
export interface RagChunk { id: string; documentId: string; source: string; start: number; end: number; text: string; embedding: number[]; }
export interface RagIndex { schema: 'neuralis.browser-rag/v1'; dimensions: number; chunkTokens: number; overlapTokens: number; documents: RagDocument[]; chunks: RagChunk[]; }
export interface SearchHit { chunk: RagChunk; score: number; matchedTerms: string[]; }
export const ragExample = [
  { id: 'retrieval', source: 'Retrieval notes', text: 'Retrieval augmented generation separates document search from response generation. A retriever selects relevant passages, then a language model can ground its answer in those sources. Retrieval quality should be measured independently from fluent output. Preserve document identifiers and offsets to cite passages.' },
  { id: 'privacy', source: 'Privacy notes', text: 'The browser research index stays on this device unless you explicitly send text to a connected backend. Browser localStorage is not encrypted. Clear the saved collection after working with sensitive material. The optional local backend can encrypt a collection on disk when its encryption capability is available.' },
  { id: 'quantization', source: 'Quantization notes', text: 'NF4 is a four bit quantization format with a codebook designed for normally distributed weights. Low rank adapters train a small matrix update while a frozen base preserves its original weights. A CPU demonstration can verify loss, adapter parameter count and the maximum change to the frozen base. Small demos do not establish language model quality.' },
];
const stopwords = new Set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can', 'for', 'from', 'in', 'is', 'it', 'of', 'on', 'or', 'the', 'this', 'to', 'with']);
export function terms(text: string): string[] { return (text.normalize('NFKC').toLocaleLowerCase('en').match(/[\p{L}\p{N}]+/gu) ?? []).filter(word => !stopwords.has(word)); }
export function hashTerm(term: string): number {
  let hash = 2166136261;
  for (const character of term) { hash ^= character.codePointAt(0)!; hash = Math.imul(hash, 16777619); }
  return hash >>> 0;
}
export function embedText(text: string, dimensions = 256): number[] {
  if (!Number.isInteger(dimensions) || dimensions < 32 || dimensions > 1024) throw new Error('Hash dimensions must be between 32 and 1,024.');
  if (text.length > 200000) throw new Error('Embedding input exceeds 200,000 characters.');
  const tokens = terms(text);
  const frequencies = new Map<string, number>();
  for (const token of tokens) frequencies.set(token, (frequencies.get(token) ?? 0) + 1);
  for (let index = 1; index < tokens.length; index++) {
    const bigram = `${tokens[index - 1]}_${tokens[index]}`;
    frequencies.set(bigram, (frequencies.get(bigram) ?? 0) + 0.5);
  }
  const vector = new Array<number>(dimensions).fill(0);
  for (const [token, count] of frequencies) {
    const hash = hashTerm(token);
    vector[hash % dimensions] += (hash & 0x80000000 ? -1 : 1) * Math.log1p(count);
  }
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return vector.map(value => norm ? value / norm : 0);
}
export function buildRagIndex(documents: RagDocument[], chunkTokens = 80, overlapTokens = 16, dimensions = 256): RagIndex {
  if (!documents.length || documents.length > 100 || documents.reduce((sum, document) => sum + document.text.length, 0) > 2_000_000) throw new Error('Index 1–100 documents with at most 2 MB of text.');
  if (!Number.isInteger(chunkTokens) || chunkTokens < 16 || chunkTokens > 256 || !Number.isInteger(overlapTokens) || overlapTokens < 0 || overlapTokens >= chunkTokens) throw new Error('Chunk size must be 16–256 tokens with a smaller nonnegative overlap.');
  const ids = new Set<string>();
  const chunks: RagChunk[] = [];
  for (const document of documents) {
    if (typeof document.id !== 'string' || !document.id || ids.has(document.id) || typeof document.source !== 'string' || typeof document.text !== 'string' || document.text.length > 200000) throw new Error('Documents need unique identifiers, sources and text of at most 200,000 characters.');
    ids.add(document.id);
    const matches = [...document.text.matchAll(/\S+/g)];
    for (let offset = 0; offset < matches.length; offset += chunkTokens - overlapTokens) {
      const first = matches[offset];
      const last = matches[Math.min(matches.length - 1, offset + chunkTokens - 1)];
      const start = first.index!;
      const end = last.index! + last[0].length;
      const text = document.text.slice(start, end);
      chunks.push({ id: `${document.id}:${start}`, documentId: document.id, source: document.source, start, end, text, embedding: embedText(text, dimensions) });
      if (chunks.length > 5000) throw new Error('The chunk budget is 5,000; increase chunk size or reduce input.');
      if (offset + chunkTokens >= matches.length) break;
    }
  }
  if (!chunks.length) throw new Error('Documents contain no searchable text.');
  return { schema: 'neuralis.browser-rag/v1', dimensions, chunkTokens, overlapTokens, documents: documents.map(document => ({ ...document })), chunks };
}
export function cosine(a: number[], b: number[]): number {
  if (!a.length || a.length !== b.length || [...a, ...b].some(value => !Number.isFinite(value))) throw new Error('Cosine vectors must have equal finite dimensions.');
  let dot = 0; let an = 0; let bn = 0;
  a.forEach((value, index) => { dot += value * b[index]; an += value * value; bn += b[index] * b[index]; });
  return an && bn ? Math.max(-1, Math.min(1, dot / Math.sqrt(an * bn))) : 0;
}
export function searchRag(index: RagIndex, query: string, limit = 5): SearchHit[] {
  if (!query.trim() || query.length > 4000 || !Number.isInteger(limit) || limit < 1 || limit > 20) throw new Error('Supply a query of 1–4,000 characters and a limit of 1–20.');
  const vector = embedText(query, index.dimensions);
  const queryTerms = new Set(terms(query));
  return index.chunks.map(chunk => ({ chunk, score: cosine(vector, chunk.embedding), matchedTerms: [...new Set(terms(chunk.text))].filter(term => queryTerms.has(term)) }))
    .filter(hit => hit.score > 0).sort((a, b) => b.score - a.score || a.chunk.id.localeCompare(b.chunk.id)).slice(0, limit);
}
export function restoreRagIndex(text: string): RagIndex {
  if (text.length > 5_000_000) throw new Error('Saved RAG import exceeds 5 MB.');
  const input = JSON.parse(text) as RagIndex;
  if (input?.schema !== 'neuralis.browser-rag/v1' || !Array.isArray(input.documents)) throw new Error('Unsupported browser RAG index.');
  // Rebuild vectors and offsets rather than trusting imported embeddings.
  return buildRagIndex(input.documents, input.chunkTokens, input.overlapTokens, input.dimensions);
}
