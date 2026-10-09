import { attentionEntropy, clamp, validateMatrix } from "./numerics";

export type AttentionMatrix = number[][];
export type MultiHeadAttention = AttentionMatrix[];
export type LayerAttentionCapture = MultiHeadAttention[];
export interface RolloutOptions {
  headFusion?: "mean" | "max" | "min";
  residualWeight?: number;
  discardFraction?: number;
  startLayer?: number;
  endLayer?: number;
}
export interface HeadSummary {
  head: number;
  meanEntropy: number;
  meanSelfAttention: number;
  meanMaxAttention: number;
  nonzeroFraction: number;
}

export function validateAttention(
  matrix: AttentionMatrix,
  square = false,
): { rows: number; columns: number } {
  const shape = validateMatrix(matrix);
  if (shape.rows > 512 || shape.columns > 512)
    throw new Error(
      "Attention preview supports at most 512 query and key tokens.",
    );
  if (square && shape.rows !== shape.columns)
    throw new Error("Rollout requires square self-attention matrices.");
  if (
    matrix.some((row) =>
      row.some((value) => !Number.isFinite(value) || value < 0),
    )
  )
    throw new Error("Attention weights must be finite and non-negative.");
  return shape;
}

export function normalizeAttention(matrix: AttentionMatrix): AttentionMatrix {
  validateAttention(matrix);
  return matrix.map((row) => {
    const sum = row.reduce((total, value) => total + value, 0);
    return row.map((value) => (sum > 0 ? value / sum : 0));
  });
}

export function fuseAttentionHeads(
  heads: MultiHeadAttention,
  mode: "mean" | "max" | "min" = "mean",
): AttentionMatrix {
  if (!heads.length) return [];
  if (heads.length > 128)
    throw new Error("Attention preview supports at most 128 heads per layer.");
  const shape = validateAttention(heads[0]);
  if (heads.length * shape.rows * shape.columns > 4000000)
    throw new Error(
      "Head fusion exceeds the four-million-value preview budget.",
    );
  for (const head of heads) {
    const current = validateAttention(head);
    if (current.rows !== shape.rows || current.columns !== shape.columns)
      throw new Error("All attention heads must have matching shapes.");
  }
  return Array.from({ length: shape.rows }, (_, row) =>
    Array.from({ length: shape.columns }, (_, column) => {
      let value = mode === "min" ? Infinity : 0;
      for (const head of heads)
        value =
          mode === "max"
            ? Math.max(value, head[row][column])
            : mode === "min"
              ? Math.min(value, head[row][column])
              : value + head[row][column];
      return mode === "mean" ? value / heads.length : value;
    }),
  );
}

function multiplyAttention(
  left: AttentionMatrix,
  right: AttentionMatrix,
): AttentionMatrix {
  const size = left.length;
  const result = Array.from({ length: size }, () =>
    Array<number>(size).fill(0),
  );
  for (let row = 0; row < size; row++) {
    for (let shared = 0; shared < size; shared++) {
      const multiplier = left[row][shared];
      if (!multiplier) continue;
      for (let column = 0; column < size; column++)
        result[row][column] += multiplier * right[shared][column];
    }
  }
  return result;
}

/** Left-multiply successive query-to-key transition matrices; identity residuals model an explicit skip path. */
export function attentionRollout(
  capture: LayerAttentionCapture,
  options: RolloutOptions = {},
): AttentionMatrix {
  if (!capture.length) return [];
  if (capture.length > 128)
    throw new Error("Attention rollout supports at most 128 layers.");
  const start = clamp(
    Math.floor(options.startLayer ?? 0),
    0,
    capture.length - 1,
  );
  const end = clamp(
    Math.floor(options.endLayer ?? capture.length - 1),
    start,
    capture.length - 1,
  );
  const residual = clamp(options.residualWeight ?? 1, 0, 100);
  const discard = clamp(options.discardFraction ?? 0, 0, 0.99);
  const first = fuseAttentionHeads(capture[start], options.headFusion);
  const shape = validateAttention(first, true);
  if (!shape.rows)
    throw new Error(
      "Attention rollout requires captured heads in each selected layer.",
    );
  if ((end - start + 1) * shape.rows ** 3 > 40000000)
    throw new Error(
      "Rollout exceeds the 40-million-operation preview budget. Reduce token count or layer range.",
    );
  let result: AttentionMatrix = Array.from({ length: shape.rows }, (_, row) =>
    Array.from({ length: shape.rows }, (_, column) => (row === column ? 1 : 0)),
  );
  for (let layer = start; layer <= end; layer++) {
    const fused = fuseAttentionHeads(capture[layer], options.headFusion);
    const current = validateAttention(fused, true);
    if (current.rows !== shape.rows)
      throw new Error(
        "Rollout layers must use the same self-attention token dimensions.",
      );
    const masked = fused.map((row, rowIndex) => {
      const output = [...row];
      if (discard > 0) {
        const ranked = row
          .map((value, index) => ({ value, index }))
          .filter((item) => item.index !== rowIndex)
          .sort((a, b) => a.value - b.value || a.index - b.index);
        const count = Math.floor(ranked.length * discard);
        for (let i = 0; i < count; i++) output[ranked[i].index] = 0;
      }
      output[rowIndex] += residual;
      return output;
    });
    result = multiplyAttention(normalizeAttention(masked), result);
  }
  return result;
}

export function summarizeHeads(heads: MultiHeadAttention): HeadSummary[] {
  return heads.map((head, index) => {
    const normalized = normalizeAttention(head);
    const rows = normalized.length;
    let entropy = 0;
    let self = 0;
    let maximum = 0;
    let nonzero = 0;
    let total = 0;
    for (let row = 0; row < rows; row++) {
      entropy += attentionEntropy(normalized[row]);
      self += normalized[row][row] || 0;
      maximum += normalized[row].reduce(
        (max, value) => Math.max(max, value),
        0,
      );
      for (const value of normalized[row]) {
        if (value > 0) nonzero++;
        total++;
      }
    }
    return {
      head: index,
      meanEntropy: rows ? entropy / rows : 0,
      meanSelfAttention: rows ? self / rows : 0,
      meanMaxAttention: rows ? maximum / rows : 0,
      nonzeroFraction: total ? nonzero / total : 0,
    };
  });
}

/** Symmetric Jensen-Shannon divergence in bits, averaged over query rows. */
export function attentionHeadDivergence(
  left: AttentionMatrix,
  right: AttentionMatrix,
): number {
  const a = normalizeAttention(left);
  const b = normalizeAttention(right);
  if (
    a.length !== b.length ||
    a.some((row, index) => row.length !== b[index].length)
  )
    throw new Error(
      "Attention head comparison requires matching query and key dimensions.",
    );
  let divergence = 0;
  let included = 0;
  for (let row = 0; row < a.length; row++) {
    if (
      !a[row].some((value) => value > 0) ||
      !b[row].some((value) => value > 0)
    )
      continue;
    let rowDivergence = 0;
    for (let column = 0; column < a[row].length; column++) {
      const p = a[row][column];
      const q = b[row][column];
      const mean = (p + q) / 2;
      if (p) rowDivergence += (p * Math.log2(p / mean)) / 2;
      if (q) rowDivergence += (q * Math.log2(q / mean)) / 2;
    }
    divergence += rowDivergence;
    included++;
  }
  return included ? divergence / included : 0;
}

export function headDivergenceMatrix(
  heads: MultiHeadAttention,
): AttentionMatrix {
  if (heads.length > 32)
    throw new Error("Pairwise head comparison supports at most 32 heads.");
  const shape = heads.length
    ? validateAttention(heads[0])
    : { rows: 0, columns: 0 };
  if (heads.length ** 2 * shape.rows * shape.columns > 20000000)
    throw new Error(
      "Pairwise head comparison exceeds the preview operation budget. Reduce head or token count.",
    );
  const result = Array.from({ length: heads.length }, () =>
    Array<number>(heads.length).fill(0),
  );
  for (let i = 0; i < heads.length; i++)
    for (let j = i + 1; j < heads.length; j++) {
      const divergence = attentionHeadDivergence(heads[i], heads[j]);
      result[i][j] = divergence;
      result[j][i] = divergence;
    }
  return result;
}

export function checkCausalMask(
  matrix: AttentionMatrix,
  tolerance = 1e-8,
): { valid: boolean; leakage: number; violations: number } {
  validateAttention(matrix, true);
  let leakage = 0;
  let violations = 0;
  for (let row = 0; row < matrix.length; row++)
    for (let column = row + 1; column < matrix.length; column++) {
      leakage += matrix[row][column];
      if (matrix[row][column] > tolerance) violations++;
    }
  return { valid: violations === 0, leakage, violations };
}

export function attentionFlowToToken(
  matrix: AttentionMatrix,
  target: number,
): number[] {
  const { columns } = validateAttention(matrix);
  if (!Number.isInteger(target) || target < 0 || target >= columns)
    throw new Error("Target token is out of bounds.");
  return normalizeAttention(matrix).map((row) => row[target]);
}
