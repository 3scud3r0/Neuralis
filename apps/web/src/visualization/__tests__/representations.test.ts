import { describe, expect, it } from "vitest";
import {
  analyzeTrajectory,
  compareRepresentations,
  cosineSimilarityMatrix,
  linearCKA,
  pairwiseSquaredDistances,
  representationDrift,
  sharedTrajectoryProjection,
  trajectoryTurns,
} from "../representations";

describe("original-space representation inspection", () => {
  it("makes linear CKA invariant to translation, isotropic scaling and orthogonal rotation", () => {
    const source = [
      [1, 3],
      [2, -1],
      [-4, 2],
      [0, 4],
    ];
    const transformed = source.map(([x, y]) => [-3 * y + 12, 3 * x - 7]);
    expect(linearCKA(source, transformed)).toBeCloseTo(1, 12);
    expect(
      compareRepresentations(source, transformed).pairwiseDistanceCorrelation,
    ).toBeCloseTo(1);
  });
  it("supports different dimensionalities with aligned sample identities", () => {
    const source = [
      [1, 0],
      [0, 1],
      [-1, 0],
      [0, -1],
    ];
    const extended = source.map(([x, y]) => [x, y, 0]);
    expect(linearCKA(source, extended)).toBeCloseTo(1);
    expect(compareRepresentations(source, extended)).toMatchObject({
      samples: 4,
      leftDimensions: 2,
      rightDimensions: 3,
    });
    expect(() => linearCKA(source, extended.slice(1))).toThrow(/aligned/);
  });
  it("returns a defined zero result for constant representations", () => {
    expect(
      linearCKA(
        [
          [1, 1],
          [1, 1],
        ],
        [
          [2, 2],
          [2, 2],
        ],
      ),
    ).toBe(0);
    expect(
      cosineSimilarityMatrix([
        [0, 0],
        [1, 0],
      ]),
    ).toEqual([
      [0, 0],
      [0, 1],
    ]);
  });
  it("computes original-space distance and exact drift", () => {
    expect(
      pairwiseSquaredDistances([
        [0, 0],
        [3, 4],
        [3, 0],
      ]),
    ).toEqual([25, 9, 16]);
    const drift = representationDrift(
      [
        [1, 0],
        [0, 1],
      ],
      [
        [2, 0],
        [1, 1],
      ],
    );
    expect(drift.absolute).toEqual([1, 1]);
    expect(drift.relative).toEqual([1, 1]);
    expect(drift.cosine[0]).toBe(1);
    expect(drift.cosine[1]).toBeCloseTo(Math.SQRT1_2);
    expect(() => representationDrift([[1]], [[1, 2]])).toThrow();
  });
});

describe("residual trajectory geometry", () => {
  it("distinguishes Euclidean movement, angular change and accumulated path length", () => {
    const trajectory = [
      [1, 0],
      [2, 0],
      [2, 2],
      [3, 2],
    ];
    const steps = analyzeTrajectory(trajectory);
    expect(steps.map((step) => step.displacement)).toEqual([0, 1, 2, 1]);
    expect(steps.map((step) => step.accumulatedDistance)).toEqual([0, 1, 3, 4]);
    expect(steps[2].relativeDisplacement).toBe(1);
    expect(trajectoryTurns(trajectory)).toEqual([Math.PI / 2, Math.PI / 2]);
  });
  it("uses one PCA basis across trajectories and preserves trajectory boundaries", () => {
    const trajectories = [
      [
        [0, 0],
        [1, 0],
      ],
      [
        [0, 1],
        [1, 1],
        [2, 1],
      ],
    ];
    const result = sharedTrajectoryProjection(trajectories);
    expect(result.points.map((points) => points.length)).toEqual([2, 3]);
    expect(result.projection.points).toEqual(result.points.flat());
    expect(() => analyzeTrajectory([[0], [0, 1]])).toThrow();
  });
});
