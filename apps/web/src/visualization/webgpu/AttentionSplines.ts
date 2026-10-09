import { validateAttention } from '../attention';

export interface AttentionSpline { query: number; key: number; weight: number; path: string; width: number }
export function attentionSplines(matrix: number[][], threshold = 0.02, width = 960, height = 330, budget = 512): AttentionSpline[] {
  const shape = validateAttention(matrix);
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('Attention threshold must lie in [0, 1].');
  if (!Number.isInteger(budget) || budget < 0 || budget > 2048) throw new Error('Spline budget must lie in [0, 2048].');
  const edges: AttentionSpline[] = [];
  for (let query = 0; query < shape.rows; query++) for (let key = 0; key < shape.columns; key++) {
    const weight = matrix[query][key]; if (!weight || weight < threshold) continue;
    const x1 = 40 + query / Math.max(1,shape.rows-1) * (width-80);
    const x2 = 40 + key / Math.max(1,shape.columns-1) * (width-80);
    const y1 = 48, y2 = height-48;
    const bend = (x2-x1)*0.16;
    edges.push({ query, key, weight, width: 0.6+5*Math.sqrt(weight), path: `M ${x1} ${y1} C ${x1+bend} ${height*0.45}, ${x2-bend} ${height*0.55}, ${x2} ${y2}` });
  }
  return edges.sort((a,b) => b.weight-a.weight || a.query-b.query || a.key-b.key).slice(0,budget);
}
