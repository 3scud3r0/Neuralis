import {
  cosineSimilarity,
  pearsonCorrelation,
  projectPCA,
  summarize,
  type Projection,
  type Vector3Tuple,
} from "./numerics";

export interface TrajectoryStep {
  layer: number;
  norm: number;
  displacement: number;
  relativeDisplacement: number;
  cosineToPrevious: number;
  cosineToInitial: number;
  accumulatedDistance: number;
}
export interface RepresentationSimilarity {
  cka: number;
  pairwiseDistanceCorrelation: number;
  samples: number;
  leftDimensions: number;
  rightDimensions: number;
  leftMeanNorm: number;
  rightMeanNorm: number;
}

function validateRepresentations(vectors: number[][]): {
  samples: number;
  dimensions: number;
} {
  if (!vectors.length || !vectors[0].length)
    throw new Error(
      "Representation analysis requires at least one non-empty vector.",
    );
  const dimension = vectors[0].length;
  if (
    vectors.some(
      (vector) =>
        vector.length !== dimension ||
        vector.some((value) => !Number.isFinite(value)),
    )
  )
    throw new Error(
      "Representation vectors must have matching finite dimensions.",
    );
  if (vectors.length * dimension > 2000000)
    throw new Error("Representation capture exceeds the preview budget.");
  return { samples: vectors.length, dimensions: dimension };
}

export function representationNorms(vectors: number[][]): number[] {
  validateRepresentations(vectors);
  return vectors.map((vector) =>
    Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)),
  );
}

export function analyzeTrajectory(trajectory: number[][]): TrajectoryStep[] {
  if (!trajectory.length) return [];
  validateRepresentations(trajectory);
  const norms = representationNorms(trajectory);
  let accumulatedDistance = 0;
  return trajectory.map((vector, layer) => {
    const previous = trajectory[Math.max(0, layer - 1)];
    const displacement = Math.sqrt(
      vector.reduce(
        (sum, value, axis) => sum + (value - previous[axis]) ** 2,
        0,
      ),
    );
    accumulatedDistance += displacement;
    return {
      layer,
      norm: norms[layer],
      displacement,
      relativeDisplacement:
        norms[Math.max(0, layer - 1)] > 0
          ? displacement / norms[Math.max(0, layer - 1)]
          : 0,
      cosineToPrevious: cosineSimilarity(vector, previous),
      cosineToInitial: cosineSimilarity(vector, trajectory[0]),
      accumulatedDistance,
    };
  });
}

export function sharedTrajectoryProjection(trajectories: number[][][]): {
  projection: Projection;
  points: Vector3Tuple[][];
} {
  const projection = projectPCA(trajectories.flat());
  let offset = 0;
  const points = trajectories.map((trajectory) => {
    const result = projection.points.slice(offset, offset + trajectory.length);
    offset += trajectory.length;
    return result;
  });
  return { projection, points };
}

export function cosineSimilarityMatrix(vectors: number[][]): number[][] {
  const shape = validateRepresentations(vectors);
  if (shape.samples ** 2 * shape.dimensions > 20000000)
    throw new Error(
      "Pairwise representation similarity exceeds the preview operation budget.",
    );
  const norms = representationNorms(vectors);
  const result = Array.from({ length: shape.samples }, () =>
    Array<number>(shape.samples).fill(0),
  );
  for (let i = 0; i < shape.samples; i++) {
    result[i][i] = norms[i] > 0 ? 1 : 0;
    for (let j = i + 1; j < shape.samples; j++) {
      const similarity = cosineSimilarity(vectors[i], vectors[j]);
      result[i][j] = similarity;
      result[j][i] = similarity;
    }
  }
  return result;
}

function centeredLinearGram(vectors: number[][]): number[][] {
  const { samples, dimensions } = validateRepresentations(vectors);
  if (samples ** 2 * dimensions > 20000000)
    throw new Error(
      "CKA exceeds the preview operation budget. Use fewer captured samples.",
    );
  const mean = Array.from(
    { length: dimensions },
    (_, axis) =>
      vectors.reduce((sum, vector) => sum + vector[axis], 0) / samples,
  );
  const centered = vectors.map((vector) =>
    vector.map((value, axis) => value - mean[axis]),
  );
  const gram = Array.from({ length: samples }, () =>
    Array<number>(samples).fill(0),
  );
  for (let i = 0; i < samples; i++)
    for (let j = i; j < samples; j++) {
      const product = centered[i].reduce(
        (sum, value, axis) => sum + value * centered[j][axis],
        0,
      );
      gram[i][j] = product;
      gram[j][i] = product;
    }
  return gram;
}

/** Biased linear centered-kernel alignment across aligned samples; invariant to translation and isotropic scale. */
export function linearCKA(left: number[][], right: number[][]): number {
  const leftShape = validateRepresentations(left);
  const rightShape = validateRepresentations(right);
  if (leftShape.samples !== rightShape.samples)
    throw new Error("CKA requires the same aligned samples in both captures.");
  const a = centeredLinearGram(left);
  const b = centeredLinearGram(right);
  let cross = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < a.length; j++) {
      cross += a[i][j] * b[i][j];
      normA += a[i][j] ** 2;
      normB += b[i][j] ** 2;
    }
  return normA > 0 && normB > 0
    ? Math.max(0, Math.min(1, cross / Math.sqrt(normA * normB)))
    : 0;
}

export function pairwiseSquaredDistances(vectors: number[][]): number[] {
  const { samples, dimensions } = validateRepresentations(vectors);
  if (samples ** 2 * dimensions > 20000000)
    throw new Error(
      "Pairwise distance computation exceeds the preview operation budget.",
    );
  const distances: number[] = [];
  for (let i = 0; i < samples; i++)
    for (let j = i + 1; j < samples; j++)
      distances.push(
        vectors[i].reduce(
          (sum, value, axis) => sum + (value - vectors[j][axis]) ** 2,
          0,
        ),
      );
  return distances;
}

export function compareRepresentations(
  left: number[][],
  right: number[][],
): RepresentationSimilarity {
  const leftShape = validateRepresentations(left);
  const rightShape = validateRepresentations(right);
  if (leftShape.samples !== rightShape.samples)
    throw new Error(
      "Representation comparison requires aligned sample identities and counts.",
    );
  const cka = linearCKA(left, right);
  const pairwiseDistanceCorrelation = pearsonCorrelation(
    pairwiseSquaredDistances(left),
    pairwiseSquaredDistances(right),
  );
  return {
    cka,
    pairwiseDistanceCorrelation,
    samples: leftShape.samples,
    leftDimensions: leftShape.dimensions,
    rightDimensions: rightShape.dimensions,
    leftMeanNorm: summarize(representationNorms(left)).mean,
    rightMeanNorm: summarize(representationNorms(right)).mean,
  };
}

export function representationDrift(
  reference: number[][],
  current: number[][],
): {
  absolute: number[];
  relative: number[];
  cosine: number[];
  meanDisplacement: number;
  meanCosine: number;
} {
  const a = validateRepresentations(reference);
  const b = validateRepresentations(current);
  if (a.samples !== b.samples || a.dimensions !== b.dimensions)
    throw new Error(
      "Elementwise drift requires identical capture shapes and aligned samples.",
    );
  const norms = representationNorms(reference);
  const absolute = current.map((vector, i) =>
    Math.sqrt(
      vector.reduce(
        (sum, value, axis) => sum + (value - reference[i][axis]) ** 2,
        0,
      ),
    ),
  );
  const relative = absolute.map((value, i) =>
    norms[i] > 0 ? value / norms[i] : 0,
  );
  const cosine = reference.map((vector, i) =>
    cosineSimilarity(vector, current[i]),
  );
  return {
    absolute,
    relative,
    cosine,
    meanDisplacement: summarize(absolute).mean,
    meanCosine: summarize(cosine).mean,
  };
}

export function trajectoryTurns(trajectory: number[][]): number[] {
  if (trajectory.length < 3) return [];
  validateRepresentations(trajectory);
  const steps = trajectory
    .slice(1)
    .map((vector, layer) =>
      vector.map((value, axis) => value - trajectory[layer][axis]),
    );
  return steps.slice(1).map((step, index) => {
    const normA = summarize(steps[index]).l2;
    const normB = summarize(step).l2;
    return normA > 0 && normB > 0
      ? Math.acos(cosineSimilarity(steps[index], step))
      : 0;
  });
}
