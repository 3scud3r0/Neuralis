import { describe, expect, it } from 'vitest';
import { exportCsv, fitEncoder, parseCsv, prepareDataset, splitRows, transformRows } from './datasetTools';

describe('CSV and leakage-safe preprocessing', () => {
  it('handles quotes, escaped quotes, delimiters and embedded newlines', () => {
    const table = parseCsv('name,value\r\n"a,b",1\r\n"say ""hi""\nnext",2\r\n');
    expect(table.rows).toEqual([['a,b', '1'], ['say "hi"\nnext', '2']]);
    expect(parseCsv(exportCsv(table.columns, table.rows))).toEqual(table);
  });
  it('rejects malformed headers, quotes and ragged rows', () => {
    expect(() => parseCsv('x,x\n1,2')).toThrow(/unique/);
    expect(() => parseCsv('x,y\n1')).toThrow(/fields/);
    expect(() => parseCsv('x\n"open')).toThrow(/closed/);
  });
  it('fits numeric transforms only to selected training rows', () => {
    const table = parseCsv('x,label\n0,a\n2,a\n10000,b');
    const encoder = fitEncoder(table, [{ column: 'x', kind: 'numeric', scaling: 'standard' }], [0, 1]);
    expect(encoder.transforms[0]).toMatchObject({ offset: 1 });
    expect(transformRows(table, encoder).inputs[2][0]).toBeGreaterThan(1000);
  });
  it('stratifies deterministic splits while preserving disjoint coverage', () => {
    const split = splitRows(10, 0.2, 42, ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']);
    expect(split.validation).toHaveLength(2);
    expect(new Set([...split.train, ...split.validation]).size).toBe(10);
    expect(splitRows(10, 0.2, 42, ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'])).toEqual(split);
  });
  it('prevents target leakage and prepares one-hot class targets', () => {
    const table = parseCsv('x,label\n0,a\n1,a\n2,b\n3,b\n4,b');
    expect(() => prepareDataset(table, { features: [{ column: 'label', kind: 'categorical' }], target: 'label', task: 'classification' })).toThrow(/target/);
    const result = prepareDataset(table, { features: [{ column: 'x', kind: 'numeric' }], target: 'label', task: 'classification' });
    expect(result.labels).toEqual(['a', 'b']);
    expect(result.targets[0]).toEqual([1, 0]);
  });
  it('reports unseen categories without fitting them from validation', () => {
    const table = parseCsv('x\nred\nred\nblue');
    const encoder = fitEncoder(table, [{ column: 'x', kind: 'categorical' }], [0, 1]);
    const result = transformRows(table, encoder);
    expect(result.inputs[2]).toEqual([0]);
    expect(result.warnings).toHaveLength(1);
  });
});
