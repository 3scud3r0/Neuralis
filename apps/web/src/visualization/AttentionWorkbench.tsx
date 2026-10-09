import { useMemo, useState } from "react";
import { AttentionView } from "./AttentionView";
import { TensorHeatmap } from "./TensorInspector";
import {
  attentionHeadDivergence,
  attentionRollout,
  checkCausalMask,
  fuseAttentionHeads,
  headDivergenceMatrix,
  summarizeHeads,
  type LayerAttentionCapture,
  type RolloutOptions,
} from "./attention";
import { clamp } from "./numerics";
import "./visualization.css";

export interface AttentionWorkbenchProps {
  capture: LayerAttentionCapture;
  labels?: string[];
  layerNames?: string[];
  sourceDescription?: string;
}

export function AttentionWorkbench({
  capture,
  labels = [],
  layerNames = [],
  sourceDescription = "Supplied attention capture",
}: AttentionWorkbenchProps) {
  const [tab, setTab] = useState<"heads" | "rollout" | "comparison">("heads");
  const [layerIndex, setLayerIndex] = useState(0);
  const [headIndex, setHeadIndex] = useState(0);
  const [compareHead, setCompareHead] = useState(1);
  const [fusion, setFusion] =
    useState<NonNullable<RolloutOptions["headFusion"]>>("mean");
  const [residual, setResidual] = useState(1);
  const [discard, setDiscard] = useState(0);
  const [startLayer, setStartLayer] = useState(0);
  const layer = clamp(layerIndex, 0, Math.max(0, capture.length - 1));
  const heads = capture[layer] || [];
  const head = clamp(headIndex, 0, Math.max(0, heads.length - 1));
  const other = clamp(compareHead, 0, Math.max(0, heads.length - 1));
  const result = useMemo(() => {
    try {
      const summaries = summarizeHeads(heads);
      const matrix =
        tab === "rollout"
          ? attentionRollout(capture, {
              startLayer: Math.min(startLayer, layer),
              endLayer: layer,
              residualWeight: residual,
              discardFraction: discard,
              headFusion: fusion,
            })
          : heads[head] || [];
      const differences =
        tab === "comparison" && heads[other]
          ? matrix.map((row, query) =>
              row.map((value, key) => value - heads[other][query][key]),
            )
          : [];
      const divergence =
        tab === "comparison" && heads[other]
          ? attentionHeadDivergence(matrix, heads[other])
          : 0;
      const divergences =
        tab === "comparison" ? headDivergenceMatrix(heads) : [];
      const causal =
        matrix.length && matrix.length === matrix[0].length
          ? checkCausalMask(matrix)
          : null;
      return {
        matrix,
        differences,
        divergence,
        divergences,
        summaries,
        causal,
        error: "",
      };
    } catch (error) {
      return {
        matrix: [],
        differences: [],
        divergence: 0,
        divergences: [],
        summaries: [],
        causal: null,
        error:
          error instanceof Error
            ? error.message
            : "Attention computation failed.",
      };
    }
  }, [
    capture,
    heads,
    head,
    other,
    tab,
    layer,
    startLayer,
    residual,
    discard,
    fusion,
  ]);
  if (!capture.length || !heads.length)
    return (
      <div className="viz-empty">
        <h3>No multi-head capture</h3>
        <p>
          This tool requires actual layer × head × query × key attention arrays.
        </p>
      </div>
    );
  return (
    <section className="viz-attention-workbench">
      <div className="viz-inspector-toolbar">
        <div className="viz-tab-list">
          {(["heads", "rollout", "comparison"] as const).map((value) => (
            <button
              key={value}
              aria-pressed={tab === value}
              onClick={() => setTab(value)}
            >
              {value}
            </button>
          ))}
        </div>
        <label>
          Layer{" "}
          <select
            value={layer}
            onChange={(event) => setLayerIndex(Number(event.target.value))}
          >
            {capture.map((_, index) => (
              <option key={index} value={index}>
                {layerNames[index] || `Layer ${index}`}
              </option>
            ))}
          </select>
        </label>
        {tab !== "rollout" && (
          <label>
            Head{" "}
            <select
              value={head}
              onChange={(event) => setHeadIndex(Number(event.target.value))}
            >
              {heads.map((_, index) => (
                <option key={index} value={index}>
                  Head {index}
                </option>
              ))}
            </select>
          </label>
        )}
        {tab === "comparison" && (
          <label>
            Compare{" "}
            <select
              value={other}
              onChange={(event) => setCompareHead(Number(event.target.value))}
            >
              {heads.map((_, index) => (
                <option key={index} value={index}>
                  Head {index}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {tab === "rollout" && (
        <div className="viz-inspector-toolbar">
          <label>
            From layer{" "}
            <input
              type="number"
              min="0"
              max={layer}
              value={Math.min(startLayer, layer)}
              onChange={(event) =>
                setStartLayer(clamp(Number(event.target.value), 0, layer))
              }
            />
          </label>
          <label>
            Head fusion{" "}
            <select
              value={fusion}
              onChange={(event) =>
                setFusion(
                  event.target.value as NonNullable<
                    RolloutOptions["headFusion"]
                  >,
                )
              }
            >
              <option>mean</option>
              <option>max</option>
              <option>min</option>
            </select>
          </label>
          <label>
            Residual weight{" "}
            <input
              type="number"
              min="0"
              max="100"
              step="0.1"
              value={residual}
              onChange={(event) =>
                setResidual(clamp(Number(event.target.value), 0, 100))
              }
            />
          </label>
          <label>
            Discard weakest{" "}
            <input
              type="number"
              min="0"
              max="0.99"
              step="0.05"
              value={discard}
              onChange={(event) =>
                setDiscard(clamp(Number(event.target.value), 0, 0.99))
              }
            />
          </label>
        </div>
      )}
      {result.error ? (
        <p className="viz-error" role="alert">
          {result.error}
        </p>
      ) : (
        <>
          <AttentionView
            matrix={result.matrix}
            labels={labels}
            title={
              tab === "rollout" ? "Attention rollout" : `Attention head ${head}`
            }
            causal={result.causal?.valid}
          />
          {tab === "comparison" && (
            <>
              <div className="viz-comparison-metrics">
                <div>
                  <span>Head Jensen–Shannon divergence</span>
                  <strong>{result.divergence.toFixed(5)} bits</strong>
                </div>
              </div>
              <div className="viz-inspector-grid">
                <TensorHeatmap
                  matrix={result.differences}
                  title={`Head ${head} minus head ${other}`}
                />
                <TensorHeatmap
                  matrix={result.divergences}
                  sequential
                  domain={1}
                  title="Pairwise head divergence (bits)"
                />
              </div>
            </>
          )}
          <div className="viz-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Head</th>
                  <th>Mean entropy</th>
                  <th>Mean self attention</th>
                  <th>Mean maximum</th>
                  <th>Nonzero weights</th>
                </tr>
              </thead>
              <tbody>
                {result.summaries.map((summary) => (
                  <tr key={summary.head}>
                    <td>{summary.head}</td>
                    <td>{summary.meanEntropy.toFixed(4)} bits</td>
                    <td>{(summary.meanSelfAttention * 100).toFixed(2)}%</td>
                    <td>{(summary.meanMaxAttention * 100).toFixed(2)}%</td>
                    <td>{(summary.nonzeroFraction * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {result.causal && (
            <p className="viz-note">
              Upper-triangle mask check:{" "}
              {result.causal.valid
                ? "no future-token leakage above 10⁻⁸"
                : `${result.causal.violations} violations, total leakage ${result.causal.leakage.toPrecision(4)}`}
              .
            </p>
          )}
        </>
      )}
      <p className="viz-note">
        {sourceDescription}.{" "}
        {tab === "rollout"
          ? "Rollout fuses measured heads, discards selected weak non-diagonal entries, adds an explicit identity residual and left-multiplies normalized transitions. This is an attention-path heuristic, not a causal attribution."
          : "Head summaries and comparisons are computed from normalized source rows; zero-mass rows remain zero."}
      </p>
    </section>
  );
}

export default AttentionWorkbench;
