import { Random } from '../engine/random';
import { quantile, summarize } from './statistics';

export interface CsvTable { columns: string[]; rows: string[][]; delimiter: string; }
export interface ColumnProfile {
  name: string; type: 'numeric' | 'categorical'; missing: number; unique: number;
  minimum?: number; maximum?: number; mean?: number; standardDeviation?: number;
  categories?: { value: string; count: number }[];
}
export interface FeatureSpec {
  column: string; kind: 'numeric' | 'categorical';
  scaling?: 'none' | 'standard' | 'minmax' | 'robust';
  imputation?: 'mean' | 'median' | 'zero' | 'reject';
}
export interface NumericTransform {
  column: string; kind: 'numeric'; offset: number; divisor: number; imputed: number;
  scaling: NonNullable<FeatureSpec['scaling']>; rejectMissing: boolean;
}
export interface CategoryTransform {
  column: string; kind: 'categorical'; categories: string[]; unknownPolicy: 'zero' | 'reject';
}
export interface FittedEncoder {
  version: 1; transforms: (NumericTransform | CategoryTransform)[];
  featureNames: string[]; fittedRows: number;
}
export interface SplitIndices { train: number[]; validation: number[]; seed: number; method: string; }
export interface EncodedDataset {
  inputs: number[][]; targets: number[][]; labels?: string[];
  task: 'classification' | 'regression'; featureNames: string[];
  encoder: FittedEncoder; split: SplitIndices; warnings: string[];
}

const MISSING = new Set(['', 'na', 'n/a', 'null', 'nan', '?']);
const MAX_TEXT = 8 * 1024 * 1024;
const MAX_ROWS = 20000;
const MAX_COLUMNS = 256;

export function isMissing(value: string): boolean {
  return MISSING.has(value.trim().toLowerCase());
}

function numberOrNull(value: string): number | null {
  if (isMissing(value)) return null;
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) ? parsed : null;
}

export function parseCsv(text: string, options: { delimiter?: string; header?: boolean; strict?: boolean; maxRows?: number } = {}): CsvTable {
  if (text.length > MAX_TEXT) throw new Error('The CSV exceeds the 8 MiB browser import budget.');
  const delimiter = options.delimiter ?? ',';
  if (delimiter.length !== 1 || ['"', '\r', '\n'].includes(delimiter)) throw new Error('The delimiter must be one non-quote, non-newline character.');
  const source = text.replace(/^\uFEFF/, '');
  const records: string[][] = [];
  let row: string[] = [];
  let field = '';
  let quoted = false;
  let closedQuote = false;
  const pushField = () => {
    row.push(field);
    field = '';
    closedQuote = false;
    if (row.length > MAX_COLUMNS) throw new Error('The CSV exceeds the 256-column budget.');
  };
  const pushRow = () => {
    pushField();
    if (row.some(value => value.trim() !== '') || row.length > 1) records.push(row);
    row = [];
    if (records.length > (options.maxRows ?? MAX_ROWS) + 1) throw new Error('The CSV exceeds the configured row budget.');
  };
  for (let index = 0; index < source.length; index++) {
    const character = source[index];
    if (quoted) {
      if (character === '"' && source[index + 1] === '"') {
        field += '"';
        index++;
      } else if (character === '"') {
        quoted = false;
        closedQuote = true;
      } else field += character;
      continue;
    }
    if (character === delimiter) pushField();
    else if (character === '\n' || character === '\r') {
      if (character === '\r' && source[index + 1] === '\n') index++;
      pushRow();
    } else if (character === '"') {
      if (field.length || closedQuote) throw new Error(`Unexpected quote near character ${index}.`);
      quoted = true;
    } else if (closedQuote && !/\s/.test(character)) {
      throw new Error(`Unexpected text after a closing quote near character ${index}.`);
    } else if (!closedQuote) field += character;
  }
  if (quoted) throw new Error('A quoted CSV field was not closed.');
  if (field || row.length || closedQuote) pushRow();
  if (!records.length) throw new Error('The CSV contains no data.');
  const hasHeader = options.header ?? true;
  const width = records[0].length;
  const columns = hasHeader ? records.shift()!.map(value => value.trim()) : Array.from({ length: width }, (_, index) => `feature_${index + 1}`);
  if (columns.some(value => !value) || new Set(columns).size !== columns.length) throw new Error('CSV headers must be nonempty and unique.');
  const rows = records.map((record, index) => {
    if (record.length !== width && (options.strict ?? true)) throw new Error(`CSV row ${index + (hasHeader ? 2 : 1)} has ${record.length} fields; expected ${width}.`);
    return Array.from({ length: width }, (_, column) => record[column] ?? '');
  });
  if (!rows.length) throw new Error('The CSV has a header but no observations.');
  return { columns, rows, delimiter };
}

export function exportCsv(columns: string[], rows: (string | number | boolean | null)[][]): string {
  const escape = (value: string | number | boolean | null) => {
    const text = value === null ? '' : String(value);
    return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  if (rows.some(row => row.length !== columns.length)) throw new Error('CSV rows must align with the output headers.');
  return [columns.map(escape).join(','), ...rows.map(row => row.map(escape).join(','))].join('\r\n') + '\r\n';
}

export function profileColumns(table: CsvTable): ColumnProfile[] {
  return table.columns.map((name, column) => {
    const values = table.rows.map(row => row[column]);
    const present = values.filter(value => !isMissing(value));
    const numeric = present.map(numberOrNull);
    const counts = new Map<string, number>();
    present.forEach(value => counts.set(value, (counts.get(value) ?? 0) + 1));
    const base = { name, missing: values.length - present.length, unique: counts.size };
    if (present.length && numeric.every(value => value !== null)) {
      const stats = summarize(numeric as number[]);
      return { ...base, type: 'numeric', minimum: stats.minimum, maximum: stats.maximum, mean: stats.mean, standardDeviation: stats.standardDeviation };
    }
    return { ...base, type: 'categorical', categories: [...counts].map(([value, count]) => ({ value, count })).sort((a, b) => b.count - a.count || a.value.localeCompare(b.value)).slice(0, 100) };
  });
}

export function splitRows(count: number, validationFraction = 0.2, seed = 42, labels?: string[]): SplitIndices {
  if (!Number.isInteger(count) || count < 3 || count > MAX_ROWS || !(validationFraction > 0 && validationFraction < 0.5)) {
    throw new Error('A split requires 3–20000 rows and a validation fraction between zero and 0.5.');
  }
  if (labels && labels.length !== count) throw new Error('Stratification labels must align with rows.');
  const random = new Random(seed);
  let train: number[] = [];
  let validation: number[] = [];
  if (labels) {
    const groups = new Map<string, number[]>();
    labels.forEach((label, index) => { if (!groups.has(label)) groups.set(label, []); groups.get(label)!.push(index); });
    for (const indices of groups.values()) {
      random.shuffle(indices);
      const size = indices.length > 1 ? Math.max(1, Math.min(indices.length - 1, Math.round(indices.length * validationFraction))) : 0;
      validation.push(...indices.slice(0, size));
      train.push(...indices.slice(size));
    }
    if (!validation.length) throw new Error('Every class occurs only once; a stratified validation split is unavailable.');
    random.shuffle(train);
    random.shuffle(validation);
  } else {
    const indices = random.shuffle(Array.from({ length: count }, (_, index) => index));
    const size = Math.max(1, Math.round(count * validationFraction));
    validation = indices.slice(0, size);
    train = indices.slice(size);
  }
  return { train, validation, seed, method: labels ? 'Seeded stratified holdout (rare singleton classes stay in training)' : 'Seeded random holdout' };
}

export function fitEncoder(table: CsvTable, features: FeatureSpec[], trainingIndices: number[], unknownPolicy: 'zero' | 'reject' = 'zero'): FittedEncoder {
  if (!features.length || features.length > 128 || !trainingIndices.length || new Set(trainingIndices).size !== trainingIndices.length) {
    throw new Error('Feature encoding requires selected features and unique training-row indices.');
  }
  if (trainingIndices.some(index => !Number.isInteger(index) || index < 0 || index >= table.rows.length)) throw new Error('A training index is outside the dataset.');
  if (new Set(features.map(feature => feature.column)).size !== features.length) throw new Error('A feature column may be selected only once.');
  const featureNames: string[] = [];
  const transforms = features.map(feature => {
    const column = table.columns.indexOf(feature.column);
    if (column < 0) throw new Error(`Feature column ${feature.column} does not exist.`);
    const values = trainingIndices.map(index => table.rows[index][column]);
    if (feature.kind === 'categorical') {
      const categories = [...new Set(values.map(value => isMissing(value) ? '<missing>' : value))].sort();
      if (categories.length > 64) throw new Error(`Column ${feature.column} has too many categories for one-hot browser encoding.`);
      categories.forEach(value => featureNames.push(`${feature.column}=${value}`));
      return { column: feature.column, kind: 'categorical', categories, unknownPolicy } as CategoryTransform;
    }
    const parsed = values.map(numberOrNull);
    const present = parsed.filter((value): value is number => value !== null);
    if (!present.length) throw new Error(`Column ${feature.column} has no numeric training values.`);
    if (values.some((value, index) => !isMissing(value) && parsed[index] === null)) throw new Error(`Column ${feature.column} contains nonnumeric text.`);
    const rejectMissing = feature.imputation === 'reject';
    if (rejectMissing && parsed.some(value => value === null)) throw new Error(`Column ${feature.column} has missing training values.`);
    const stats = summarize(present);
    const imputed = feature.imputation === 'zero' ? 0 : feature.imputation === 'median' ? stats.median : stats.mean;
    const complete = parsed.map(value => value ?? imputed);
    const fitted = summarize(complete);
    const scaling = feature.scaling ?? 'standard';
    let offset = 0;
    let divisor = 1;
    if (scaling === 'standard') {
      offset = fitted.mean;
      divisor = fitted.standardDeviation || 1;
    } else if (scaling === 'minmax') {
      offset = fitted.minimum;
      divisor = fitted.maximum - fitted.minimum || 1;
    } else if (scaling === 'robust') {
      const sorted = complete.sort((a, b) => a - b);
      offset = quantile(sorted, 0.5);
      divisor = quantile(sorted, 0.75) - quantile(sorted, 0.25) || 1;
    }
    featureNames.push(feature.column);
    return { column: feature.column, kind: 'numeric', offset, divisor, imputed, scaling, rejectMissing } as NumericTransform;
  });
  if (featureNames.length > 128) throw new Error('Expanded one-hot encoding exceeds the 128-feature browser budget.');
  return { version: 1, transforms, featureNames, fittedRows: trainingIndices.length };
}

export function transformRows(table: CsvTable, encoder: FittedEncoder, indices?: number[]): { inputs: number[][]; warnings: string[] } {
  if (encoder.version !== 1) throw new Error('Unsupported feature-encoder version.');
  const warnings = new Set<string>();
  const selected = indices ?? Array.from({ length: table.rows.length }, (_, index) => index);
  const inputs = selected.map(index => {
    const row = table.rows[index];
    if (!row) throw new Error('A selected row is outside the dataset.');
    return encoder.transforms.flatMap(transform => {
      const column = table.columns.indexOf(transform.column);
      if (column < 0) throw new Error(`Required column ${transform.column} is missing from this table.`);
      const raw = row[column];
      if (transform.kind === 'categorical') {
        const value = isMissing(raw) ? '<missing>' : raw;
        if (!transform.categories.includes(value)) {
          if (transform.unknownPolicy === 'reject') throw new Error(`Unknown category in ${transform.column}.`);
          warnings.add(`Unseen categories in ${transform.column} were encoded as an all-zero vector.`);
        }
        return transform.categories.map(category => Number(category === value));
      }
      const value = numberOrNull(raw);
      if (value === null && (!isMissing(raw) || transform.rejectMissing)) throw new Error(`Invalid numeric value in ${transform.column}, row ${index + 1}.`);
      return [((value ?? transform.imputed) - transform.offset) / transform.divisor];
    });
  });
  return { inputs, warnings: [...warnings] };
}

export function prepareDataset(table: CsvTable, options: {
  features: FeatureSpec[]; target: string; task: 'classification' | 'regression';
  validationFraction?: number; seed?: number;
}): EncodedDataset {
  const targetColumn = table.columns.indexOf(options.target);
  if (targetColumn < 0) throw new Error('Select a target column present in the dataset.');
  if (options.features.some(feature => feature.column === options.target)) throw new Error('The target column must not be included among input features.');
  const rawTargets = table.rows.map(row => row[targetColumn]);
  if (rawTargets.some(isMissing)) throw new Error('Target values cannot be missing. Remove or label those rows before importing.');
  const split = splitRows(table.rows.length, options.validationFraction ?? 0.2, options.seed ?? 42, options.task === 'classification' ? rawTargets : undefined);
  const encoder = fitEncoder(table, options.features, split.train);
  const transformed = transformRows(table, encoder);
  let labels: string[] | undefined;
  let targets: number[][];
  if (options.task === 'classification') {
    labels = [...new Set(split.train.map(index => rawTargets[index]))].sort();
    if (labels.length < 2 || labels.length > 64) throw new Error('Classification requires 2–64 training classes.');
    targets = rawTargets.map(value => {
      if (!labels!.includes(value)) throw new Error('A validation class is absent from training.');
      return labels!.map(label => Number(label === value));
    });
  } else {
    targets = rawTargets.map(value => {
      const number = numberOrNull(value);
      if (number === null) throw new Error('Regression targets must be finite numbers.');
      return [number];
    });
  }
  return { inputs: transformed.inputs, targets, labels, task: options.task, featureNames: encoder.featureNames, encoder, split, warnings: transformed.warnings };
}

export function gaussianAugmentation(inputs: number[][], options: { copies?: number; deviation?: number; seed?: number } = {}): number[][] {
  const copies = options.copies ?? 1;
  const deviation = options.deviation ?? 0.01;
  if (!Number.isInteger(copies) || copies < 1 || copies > 20 || !Number.isFinite(deviation) || deviation < 0 || inputs.length * copies > MAX_ROWS) {
    throw new Error('The requested augmentation exceeds its row or noise budget.');
  }
  const random = new Random(options.seed ?? 42);
  return Array.from({ length: copies }, () => inputs.map(row => row.map(value => {
    if (!Number.isFinite(value)) throw new Error('Augmentation requires finite numeric features.');
    return value + random.normal(0, deviation);
  }))).flat();
}

export function duplicateRows(table: CsvTable): { groups: number[][]; duplicates: number } {
  const groups = new Map<string, number[]>();
  table.rows.forEach((row, index) => {
    const key = JSON.stringify(row);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(index);
  });
  const repeated = [...groups.values()].filter(group => group.length > 1);
  return { groups: repeated, duplicates: repeated.reduce((sum, group) => sum + group.length - 1, 0) };
}
