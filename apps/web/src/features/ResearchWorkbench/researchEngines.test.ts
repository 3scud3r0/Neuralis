import { describe, expect, it } from 'vitest';
import { decodePairs, dot, evaluateAxis, fitAxis, steerActivation, type ActivationPair } from './axisEngine';
import { decodeEpisode, episodeReturn, trainArena } from './arenaEngine';
import { adapterBudget, dequantizeNF4, quantizationDemo, quantizeNF4 } from './efficiencyEngine';
import { conversionRecipe, makeRecipe } from './exportEngine';
import { generateArithmetic, splitRecords, toJsonl, validateJsonl } from './foundryEngine';
import { buildRagIndex, cosine, embedText, restoreRagIndex, searchRag } from './ragEngine';

describe('paired concept direction evaluation', () => {
  const pairs: ActivationPair[] = [
    { id: 'train-a', negative: [-1, 100], positive: [1, 100], split: 'train' },
    { id: 'train-b', negative: [-2, -100], positive: [2, -100], split: 'train' },
    { id: 'held', negative: [-0.5, 1000], positive: [0.5, -1000], split: 'evaluation' },
    { id: 'reversed', negative: [0.25, 0], positive: [-0.25, 0], split: 'evaluation' },
  ];
  it('fits a unit direction using training pairs and reports held-out errors', () => {
    const axis = fitAxis(pairs);
    expect(axis.vector).toEqual([1, 0]);
    expect(axis.threshold).toBe(0);
    const evaluation = evaluateAxis(axis, pairs);
    expect(evaluation.accuracy).toBe(0.5);
    expect(evaluation.pairOrdering).toBe(0.5);
    expect(evaluation.auc).toBe(0.75);
    const changed = pairs.map(pair => pair.split === 'evaluation' ? { ...pair, positive: [999, 999] } : pair);
    expect(fitAxis(changed)).toEqual(axis);
  });
  it('shifts projection by alpha and preserves the perpendicular component', () => {
    const axis = fitAxis(pairs);
    const before = [2, 7];
    const after = steerActivation(before, axis, -0.7);
    expect(dot(after, axis.vector) - dot(before, axis.vector)).toBeCloseTo(-0.7);
    expect(after[1]).toBe(before[1]);
    expect(before).toEqual([2, 7]);
  });
  it('rejects zero differences, split leakage ambiguity and malformed captures', () => {
    expect(() => fitAxis(pairs.map(pair => ({ ...pair, positive: pair.negative })))).toThrow('zero');
    expect(() => decodePairs(JSON.stringify(pairs.map(pair => ({ ...pair, split: 'test' }))))).toThrow('split');
    expect(() => decodePairs(JSON.stringify([...pairs, pairs[0]]))).toThrow('unique');
    expect(() => fitAxis(pairs.map(pair => ({ ...pair, negative: [1] })))).toThrow('vectors');
  });
});
describe('verifiable arithmetic foundry', () => {
  it('creates exact deterministic records and disjoint seeded splits', () => {
    const records = generateArithmetic(90, 8, 5, ['addition', 'multiplication', 'subtraction']);
    expect(records).toEqual(generateArithmetic(90, 8, 5, ['addition', 'multiplication', 'subtraction']));
    const report = validateJsonl(toJsonl(records));
    expect(report.issues).toEqual([]);
    expect(report.valid).toBe(90);
    const split = splitRecords(records, 0.2, 99);
    const trainIds = new Set(split.train.map(record => record.id));
    expect(split.evaluation).toHaveLength(18);
    expect(split.evaluation.some(record => trainIds.has(record.id))).toBe(false);
    expect([...split.train, ...split.evaluation]).toHaveLength(records.length);
    expect(split).toEqual(splitRecords(records, 0.2, 99));
  });
  it('rejects answer tampering, prompt tampering, duplicates and invalid JSON', () => {
    const [record] = generateArithmetic(1, 3, 10, ['addition']);
    const wrong = structuredClone(record); wrong.messages[1].content = '100000';
    const prompt = structuredClone(record); prompt.messages[0].content = 'Compute something else';
    const text = [JSON.stringify(record), JSON.stringify(record), JSON.stringify(wrong), JSON.stringify(prompt), '{bad'].join('\n');
    const report = validateJsonl(text);
    expect(report.valid).toBe(1);
    expect(report.issues.map(issue => issue.category)).toEqual(['duplicate', 'answer', 'answer', 'syntax']);
  });
  it('exhaustively completes small high-occupancy spaces without duplicates', () => {
    expect(validateJsonl(toJsonl(generateArithmetic(4, 42, 1, ['subtraction']))).valid).toBe(4);
    expect(() => generateArithmetic(5, 42, 1, ['subtraction'])).toThrow('space');
    expect(() => generateArithmetic(1, 42, 10, [])).toThrow('Choose');
  });
});
describe('bounded browser retrieval', () => {
  const documents = [
    { id: 'a', source: 'Astronomy', text: 'Galaxies contain stars planets nebulae gravity telescopes.' },
    { id: 'b', source: 'Baking', text: 'Bread dough flour yeast oven baking recipes.' },
  ];
  it('ranks an exact lexical passage above unrelated evidence', () => {
    const index = buildRagIndex(documents);
    const hits = searchRag(index, 'bread dough yeast', 2);
    expect(hits[0].chunk.documentId).toBe('b');
    expect(hits[0].matchedTerms).toEqual(expect.arrayContaining(['bread', 'dough', 'yeast']));
    expect(cosine(embedText('bread dough'), embedText('bread dough'))).toBeCloseTo(1);
    expect(embedText('the and of')).toEqual(new Array(256).fill(0));
  });
  it('maintains original character offsets and overlapping token coverage', () => {
    const text = Array.from({ length: 40 }, (_, i) => `token${i}`).join('  ');
    const index = buildRagIndex([{ id: 'offsets', source: 'test', text }], 16, 4);
    expect(index.chunks).toHaveLength(3);
    for (const chunk of index.chunks) expect(text.slice(chunk.start, chunk.end)).toBe(chunk.text);
    expect(index.chunks[1].text.startsWith('token12')).toBe(true);
    expect(index.chunks.at(-1)?.text.endsWith('token39')).toBe(true);
  });
  it('rebuilds forged imported vectors and enforces chunk budgets', () => {
    const index = buildRagIndex(documents);
    const forged = structuredClone(index); forged.chunks[0].embedding.fill(100);
    expect(restoreRagIndex(JSON.stringify(forged))).toEqual(index);
    expect(() => buildRagIndex(documents, 16, 16)).toThrow('overlap');
    expect(() => buildRagIndex([documents[0], documents[0]])).toThrow('unique');
    expect(() => searchRag(index, '')).toThrow('query');
  });
});
describe('four bit reference packing', () => {
  it('packs paired codes and recovers exact endpoints including odd sizes', () => {
    const weights = [-1, 0, 1, 0.5, -0.5, 0, 1, -1, 0];
    const quantized = quantizeNF4(weights, 8);
    expect(quantized.packed).toHaveLength(5);
    const decoded = dequantizeNF4(quantized);
    expect(decoded[0]).toBe(-1);
    expect(decoded[1]).toBe(0);
    expect(decoded[2]).toBe(1);
    expect(decoded.at(-1)).toBe(0);
    expect(dequantizeNF4(quantizeNF4(new Array(9).fill(0), 8))).toEqual(new Array(9).fill(0));
  });
  it('measures reconstruction error and includes scale overhead in storage', () => {
    const demo = quantizationDemo(42, 256, 64);
    expect(demo.packedBytes).toBe(144);
    expect(demo.originalBytes).toBe(1024);
    expect(demo.mse).toBeGreaterThan(0);
    expect(demo.mse).toBeLessThan(0.01);
    const budget = adapterBudget(1024, 1024, 4);
    expect(budget.adapterParameters).toBe(8192);
    expect(budget.adapterTrainingBytes).toBeLessThan(budget.fullTrainingBytes);
  });
});
describe('arena policy evidence', () => {
  it('learns a goal-reaching policy and records aligned action values and rewards', () => {
    const before = trainArena(42, 0);
    const after = trainArena(42, 25000);
    expect(after.training?.successRate).toBeGreaterThan(before.training?.successRate ?? 0);
    expect(after.training?.successRate).toBeGreaterThanOrEqual(0.9);
    expect(after.frames.at(-1)?.state).toBe(35);
    expect(episodeReturn(after)).toBeGreaterThan(0);
    for (const frame of after.frames.filter(frame => frame.action >= 0)) expect(frame.action).toBe(frame.activations.indexOf(Math.max(...frame.activations)));
    const restored = decodeEpisode(JSON.stringify(after));
    expect(restored.frames).toEqual(after.frames);
  });
  it('rejects malformed frame ordering and external image URLs', () => {
    const episode = trainArena(42, 0);
    episode.frames[0].step = 5;
    expect(() => decodeEpisode(JSON.stringify(episode))).toThrow('frame');
    episode.frames[0].step = 0;
    episode.frames[0].image = 'https://remote.invalid/frame.png';
    expect(() => decodeEpisode(JSON.stringify(episode))).toThrow('data URLs');
  });
});
describe('export recipe inputs', () => {
  it('quotes shell metacharacters and rejects Modelfile injection', () => {
    expect(makeRecipe('local-demo', './weird$(touch file).gguf', 0.5).modelfile).toContain('FROM "./weird$(touch file).gguf"');
    expect(conversionRecipe("./Bob's model", './file.gguf')[0]).toContain("'./Bob'\\''s model'");
    expect(() => makeRecipe('test;rm', './file.gguf', 0.5)).toThrow('name');
    expect(() => makeRecipe('test', './file.gguf\nSYSTEM injected', 0.5)).toThrow('path');
  });
});
