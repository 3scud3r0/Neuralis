import { useMemo } from "react";
import { EmbeddingView } from "./EmbeddingView";
import { ResidualView } from "./ResearchViews";
import { Histogram, TensorHeatmap } from "./TensorInspector";
import {
  analyzeTrajectory,
  compareRepresentations,
  cosineSimilarityMatrix,
  representationDrift,
  trajectoryTurns,
} from "./representations";
import { exportTensorCSV } from "./exports";
import "./visualization.css";

export interface RepresentationInspectorProps {
  vectors: number[][];
  labels?: string[];
  groups?: string[];
  reference?: number[][];
  referenceLabel?: string;
  title?: string;
}

export function RepresentationInspector({
  vectors,
  labels,
  groups,
  reference,
  referenceLabel = "Reference capture",
  title = "Representation analysis",
}: RepresentationInspectorProps) {
  const result = useMemo(() => {
    try {
      const matrix = vectors.length ? cosineSimilarityMatrix(vectors) : [];
      const comparison =
        reference?.length && vectors.length
          ? compareRepresentations(reference, vectors)
          : null;
      const drift =
        reference?.length &&
        vectors.length &&
        reference[0].length === vectors[0].length
          ? representationDrift(reference, vectors)
          : null;
      return { matrix, comparison, drift, error: "" };
    } catch (error) {
      return {
        matrix: [],
        comparison: null,
        drift: null,
        error:
          error instanceof Error
            ? error.message
            : "Representation analysis failed.",
      };
    }
  }, [vectors, reference]);
  return (
    <section className="viz-representation-inspector">
      <EmbeddingView
        vectors={vectors}
        labels={labels}
        groups={groups}
        title={title}
      />
      {result.error && (
        <p className="viz-error" role="alert">
          {result.error}
        </p>
      )}
      <div className="viz-slice-reduction">
        <TensorHeatmap
          matrix={result.matrix}
          title="Original-space cosine similarity"
          domain={1}
          height={300}
        />
      </div>
      {result.comparison && (
        <>
          <div className="viz-inspector-toolbar">
            <h3>Aligned sample comparison</h3>
            <span className="viz-axis-description">
              {referenceLabel} · {result.comparison.samples} samples
            </span>
          </div>
          <div className="viz-comparison-metrics">
            <div>
              <span>Linear CKA</span>
              <strong>{result.comparison.cka.toFixed(5)}</strong>
            </div>
            <div>
              <span>Distance correlation</span>
              <strong>
                {result.comparison.pairwiseDistanceCorrelation.toFixed(5)}
              </strong>
            </div>
            <div>
              <span>Reference dimensions</span>
              <strong>{result.comparison.leftDimensions}</strong>
            </div>
            <div>
              <span>Current dimensions</span>
              <strong>{result.comparison.rightDimensions}</strong>
            </div>
            <div>
              <span>Mean captured norm</span>
              <strong>{result.comparison.rightMeanNorm.toPrecision(5)}</strong>
            </div>
          </div>
        </>
      )}
      {result.drift && (
        <>
          <Histogram
            values={result.drift.absolute}
            title="Elementwise representation drift"
          />
          <Histogram
            values={result.drift.cosine}
            title="Similarity to reference sample"
          />
        </>
      )}
      <p className="viz-note">
        Cosine and CKA operate on original representations. Comparison requires
        the same samples in the same order. Linear CKA uses centered Gram
        matrices and returns zero for a constant representation.
      </p>
    </section>
  );
}

export interface TrajectoryInspectorProps {
  trajectories: number[][][];
  labels?: string[];
  layerLabels?: string[];
}
export function TrajectoryInspector({
  trajectories,
  labels = [],
  layerLabels = [],
}: TrajectoryInspectorProps) {
  const result = useMemo(() => {
    try {
      return {
        steps: trajectories.map((trajectory) => analyzeTrajectory(trajectory)),
        turns: trajectories.map((trajectory) => trajectoryTurns(trajectory)),
        error: "",
      };
    } catch (error) {
      return {
        steps: [],
        turns: [],
        error:
          error instanceof Error
            ? error.message
            : "Trajectory analysis failed.",
      };
    }
  }, [trajectories]);
  return (
    <section className="viz-representation-inspector">
      <ResidualView
        trajectories={trajectories}
        labels={labels}
        layerLabels={layerLabels}
      />
      {result.error && (
        <p className="viz-error" role="alert">
          {result.error}
        </p>
      )}
      <div className="viz-inspector-toolbar">
        <h3>Layer-wise representation movement</h3>
        <button
          onClick={() =>
            exportTensorCSV(
              result.steps.flatMap((steps, trajectory) =>
                steps.map((step) => [
                  trajectory,
                  step.layer,
                  step.norm,
                  step.displacement,
                  step.relativeDisplacement,
                  step.cosineToPrevious,
                  step.cosineToInitial,
                  step.accumulatedDistance,
                ]),
              ),
              "residual-trajectory-statistics.csv",
            )
          }
        >
          Export metrics
        </button>
      </div>
      <div className="viz-table-scroll">
        <table>
          <thead>
            <tr>
              <th>Trajectory</th>
              <th>Layer</th>
              <th>Norm</th>
              <th>Step distance</th>
              <th>Relative step</th>
              <th>Previous cosine</th>
              <th>Initial cosine</th>
              <th>Path length</th>
              <th>Turn</th>
            </tr>
          </thead>
          <tbody>
            {result.steps.flatMap((steps, trajectory) =>
              steps.map((step) => (
                <tr key={`${trajectory}-${step.layer}`}>
                  <td>{labels[trajectory] || trajectory}</td>
                  <td>{layerLabels[step.layer] || step.layer}</td>
                  <td>{step.norm.toPrecision(4)}</td>
                  <td>{step.displacement.toPrecision(4)}</td>
                  <td>{step.relativeDisplacement.toPrecision(4)}</td>
                  <td>{step.cosineToPrevious.toFixed(4)}</td>
                  <td>{step.cosineToInitial.toFixed(4)}</td>
                  <td>{step.accumulatedDistance.toPrecision(4)}</td>
                  <td>
                    {step.layer >= 2
                      ? `${((result.turns[trajectory][step.layer - 2] * 180) / Math.PI).toFixed(1)}°`
                      : "—"}
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      <p className="viz-note">
        Movement and turn angles use original-dimensional captured vectors; all
        trajectories must retain a consistent representation dimension. Zero
        reference norms yield relative displacement zero.
      </p>
    </section>
  );
}
