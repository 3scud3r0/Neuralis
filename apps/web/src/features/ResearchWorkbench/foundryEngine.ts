import { Random } from '../../engine/random';

export interface ArithmeticRecord {
  id: string;
  messages: [{ role: 'user'; content: string }, { role: 'assistant'; content: string }];
  metadata: { kind: 'addition' | 'subtraction' | 'multiplication'; a: number; b: number; seed: number };
}
export interface ValidationIssue { line: number; category: 'syntax' | 'schema' | 'answer' | 'duplicate'; message: string; }
export interface FoundryReport { valid: number; total: number; duplicates: number; issues: ValidationIssue[]; records: ArithmeticRecord[]; }
const operators = { addition: '+', subtraction: '-', multiplication: '×' } as const;
export function arithmeticAnswer(kind: ArithmeticRecord['metadata']['kind'], a: number, b: number): number {
  if (!Number.isSafeInteger(a) || !Number.isSafeInteger(b) || Math.abs(a) > 10000 || Math.abs(b) > 10000) throw new Error('Operands must be bounded safe integers.');
  if (kind === 'addition') return a + b;
  if (kind === 'subtraction') return a - b;
  if (kind === 'multiplication') return a * b;
  throw new Error('Unknown arithmetic operation.');
}
export function generateArithmetic(count: number, seed: number, maxOperand: number, kinds: ArithmeticRecord['metadata']['kind'][]): ArithmeticRecord[] {
  if (!Number.isInteger(count) || count < 1 || count > 2000) throw new Error('Generate between 1 and 2,000 records.');
  if (!Number.isInteger(maxOperand) || maxOperand < 1 || maxOperand > 1000) throw new Error('Maximum operand must be between 1 and 1,000.');
  if (!Number.isSafeInteger(seed)) throw new Error('Seed must be a safe integer.');
  if (!kinds.length || new Set(kinds).size !== kinds.length || kinds.some(kind => !Object.hasOwn(operators, kind))) throw new Error('Choose one or more distinct supported operations.');
  if (count > kinds.length * (maxOperand + 1) ** 2) throw new Error('The requested unique dataset exceeds this operand space.');
  const random = new Random(seed);
  const used = new Set<string>();
  const records: ArithmeticRecord[] = [];
  const add = (kind: ArithmeticRecord['metadata']['kind'], a: number, b: number) => {
    const key = `${kind}:${a}:${b}`;
    if (used.has(key)) return;
    used.add(key);
    records.push({ id: `math-${seed}-${records.length.toString().padStart(4, '0')}`,
      messages: [{ role: 'user', content: `Compute ${a} ${operators[kind]} ${b}. Return only the integer answer.` },
        { role: 'assistant', content: String(arithmeticAnswer(kind, a, b)) }], metadata: { kind, a, b, seed } });
  };
  for (let attempts = 0; records.length < count && attempts < count * 30; attempts++) add(random.choice(kinds), random.integer(maxOperand + 1), random.integer(maxOperand + 1));
  // Deterministic exhaustive completion keeps high-occupancy requests bounded.
  for (const kind of kinds) for (let a = 0; a <= maxOperand && records.length < count; a++) for (let b = 0; b <= maxOperand && records.length < count; b++) add(kind, a, b);
  return records;
}
export function toJsonl(records: ArithmeticRecord[]): string { return records.map(record => JSON.stringify(record)).join('\n') + '\n'; }
export function validateJsonl(text: string): FoundryReport {
  if (text.length > 4_000_000) throw new Error('JSONL validation is limited to 4 MB.');
  const lines = text.split(/\r?\n/);
  if (lines.length > 10001) throw new Error('JSONL validation is limited to 10,000 rows.');
  const issues: ValidationIssue[] = [];
  const records: ArithmeticRecord[] = [];
  const used = new Set<string>();
  let total = 0;
  let duplicates = 0;
  lines.forEach((line, index) => {
    if (!line.trim()) return;
    total++;
    let item: ArithmeticRecord;
    try { item = JSON.parse(line) as ArithmeticRecord; } catch { issues.push({ line: index + 1, category: 'syntax', message: 'Invalid JSON.' }); return; }
    const metadata = item?.metadata;
    if (!item || typeof item.id !== 'string' || !item.id || !Array.isArray(item.messages) || item.messages.length !== 2 ||
      item.messages[0]?.role !== 'user' || typeof item.messages[0]?.content !== 'string' || item.messages[1]?.role !== 'assistant' || typeof item.messages[1]?.content !== 'string' ||
      !metadata || !Object.hasOwn(operators, metadata.kind) || !Number.isSafeInteger(metadata.a) || !Number.isSafeInteger(metadata.b) ||
      Math.abs(metadata.a) > 10000 || Math.abs(metadata.b) > 10000 || !Number.isSafeInteger(metadata.seed)) {
      issues.push({ line: index + 1, category: 'schema', message: 'Expected id, two role messages and bounded arithmetic metadata.' }); return;
    }
    const expectedPrompt = `Compute ${metadata.a} ${operators[metadata.kind]} ${metadata.b}. Return only the integer answer.`;
    if (item.messages[0].content !== expectedPrompt || item.messages[1].content.trim() !== String(arithmeticAnswer(metadata.kind, metadata.a, metadata.b))) {
      issues.push({ line: index + 1, category: 'answer', message: 'Prompt or answer does not agree with independently recomputed metadata.' }); return;
    }
    const key = `${metadata.kind}:${metadata.a}:${metadata.b}`;
    if (used.has(key) || records.some(record => record.id === item.id)) {
      duplicates++;
      issues.push({ line: index + 1, category: 'duplicate', message: 'Duplicate expression or identifier.' }); return;
    }
    used.add(key);
    records.push(item);
  });
  return { valid: records.length, total, duplicates, issues, records };
}
export function splitRecords(records: ArithmeticRecord[], fraction: number, seed: number): { train: ArithmeticRecord[]; evaluation: ArithmeticRecord[] } {
  if (!Number.isFinite(fraction) || fraction <= 0 || fraction >= 1 || records.length < 2) throw new Error('A split needs at least two records and a fraction between zero and one.');
  const shuffled = new Random(seed).shuffle([...records]);
  const size = Math.max(1, Math.min(records.length - 1, Math.round(records.length * fraction)));
  return { evaluation: shuffled.slice(0, size), train: shuffled.slice(size) };
}
