import { useEffect, useMemo, useRef, useState } from "react";
import type { LayerData } from "../types";
import {
  clamp,
  divergingColor,
  finite,
  histogram,
  rgbCss,
  summarize,
  symmetricDomain,
  validateMatrix,
  type NumericSummary,
} from "./numerics";
import {
  exportCanvasImage,
  exportLayersJSON,
  exportTensorCSV,
} from "./exports";
import "./visualization.css";

export interface TensorHeatmapProps {
  matrix: number[][];
  title?: string;
  domain?: number;
  height?: number;
  onSelect?: (row: number, column: number, value: number) => void;
  sequential?: boolean;
}
interface Cell {
  row: number;
  column: number;
  value: number;
}

/** Canvas previews are sampled by exact index; the source tensor remains untouched for CSV and statistics. */
export function TensorHeatmap({
  matrix,
  title = "Tensor heatmap",
  domain: suppliedDomain,
  height = 290,
  onSelect,
  sequential = false,
}: TensorHeatmapProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(500);
  const [cell, setCell] = useState<Cell | null>(null);
  const [error, setError] = useState("");
  const shape = useMemo(() => {
    try {
      return { ...validateMatrix(matrix), error: "" };
    } catch (caught) {
      return {
        rows: 0,
        columns: 0,
        error: caught instanceof Error ? caught.message : "Invalid tensor.",
      };
    }
  }, [matrix]);
  const values = useMemo(() => matrix.flat(), [matrix]);
  const domain =
    suppliedDomain && suppliedDomain > 0
      ? suppliedDomain
      : symmetricDomain(values);
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) =>
      setWidth(Math.max(1, entries[0].contentRect.width)),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const element = canvas.current;
    if (!element || !shape.rows || !shape.columns) return;
    const context = element.getContext("2d");
    if (!context) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    element.width = Math.ceil(width * ratio);
    element.height = Math.ceil(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = "#07101f";
    context.fillRect(0, 0, width, height);
    const columns = Math.min(
      shape.columns,
      192,
      Math.max(1, Math.floor(width / 2)),
    );
    const rows = Math.min(shape.rows, 192, Math.max(1, Math.floor(height / 2)));
    const cellWidth = width / columns;
    const cellHeight = height / rows;
    for (let row = 0; row < rows; row++) {
      const sourceRow = Math.floor((row * shape.rows) / rows);
      for (let column = 0; column < columns; column++) {
        const sourceColumn = Math.floor((column * shape.columns) / columns);
        const value = matrix[sourceRow][sourceColumn];
        if (!Number.isFinite(value)) context.fillStyle = "#fbbf24";
        else if (sequential) {
          const intensity = clamp(value / domain, 0, 1);
          context.fillStyle = rgbCss([
            0.06 + intensity * 0.27,
            0.1 + intensity * 0.7,
            0.21 + intensity * 0.72,
          ]);
        } else context.fillStyle = rgbCss(divergingColor(value, domain));
        context.fillRect(
          column * cellWidth,
          row * cellHeight,
          Math.ceil(cellWidth + 0.1),
          Math.ceil(cellHeight + 0.1),
        );
      }
    }
    if (cell && cell.row < shape.rows && cell.column < shape.columns) {
      context.strokeStyle = "#ffffff";
      context.lineWidth = 1.5;
      context.strokeRect(
        (cell.column / shape.columns) * width,
        (cell.row / shape.rows) * height,
        Math.max(cellWidth, width / shape.columns),
        Math.max(cellHeight, height / shape.rows),
      );
    }
  }, [matrix, shape, width, height, domain, cell, sequential]);
  const choose = (row: number, column: number, notify = false) => {
    const boundedRow = clamp(row, 0, shape.rows - 1);
    const boundedColumn = clamp(column, 0, shape.columns - 1);
    const value = matrix[boundedRow]?.[boundedColumn];
    if (value === undefined) return;
    setCell({ row: boundedRow, column: boundedColumn, value });
    if (notify) onSelect?.(boundedRow, boundedColumn, value);
  };
  const save = async () => {
    if (!canvas.current) return;
    try {
      await exportCanvasImage(canvas.current, "neuralis-tensor.png");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Export failed.");
    }
  };
  return (
    <div className="viz-heatmap-wrap" ref={container}>
      <div className="viz-chart-heading">
        <strong>{title}</strong>
        <span>
          {shape.rows} × {shape.columns}
        </span>
        <button onClick={save} disabled={!shape.rows}>
          PNG
        </button>
      </div>
      {shape.error ? (
        <p className="viz-error" role="alert">
          {shape.error}
        </p>
      ) : !shape.rows || !shape.columns ? (
        <div className="viz-empty" style={{ height }}>
          No tensor values captured.
        </div>
      ) : (
        <canvas
          ref={canvas}
          style={{ width: "100%", height }}
          tabIndex={0}
          role="img"
          aria-label={`${title}, ${shape.rows} rows and ${shape.columns} columns. Use arrow keys to inspect values.`}
          onMouseMove={(event) => {
            const rect = event.currentTarget.getBoundingClientRect();
            choose(
              Math.floor(
                ((event.clientY - rect.top) / rect.height) * shape.rows,
              ),
              Math.floor(
                ((event.clientX - rect.left) / rect.width) * shape.columns,
              ),
            );
          }}
          onClick={() => {
            if (cell) onSelect?.(cell.row, cell.column, cell.value);
          }}
          onKeyDown={(event) => {
            const row = cell?.row || 0;
            const column = cell?.column || 0;
            if (event.key === "ArrowUp") {
              event.preventDefault();
              choose(row - 1, column);
            }
            if (event.key === "ArrowDown") {
              event.preventDefault();
              choose(row + 1, column);
            }
            if (event.key === "ArrowLeft") {
              event.preventDefault();
              choose(row, column - 1);
            }
            if (event.key === "ArrowRight") {
              event.preventDefault();
              choose(row, column + 1);
            }
            if (event.key === "Enter" && cell)
              onSelect?.(cell.row, cell.column, cell.value);
          }}
        />
      )}
      <div className="viz-chart-caption">
        <span aria-live="polite">
          {cell
            ? `[${cell.row}, ${cell.column}] = ${Number.isFinite(cell.value) ? cell.value.toPrecision(6) : "non-finite"}`
            : "Hover or use arrow keys to inspect a value"}
        </span>
        <span>
          {shape.rows > 192 || shape.columns > 192
            ? "Sampled preview · CSV preserves all values"
            : "Exact cells"}{" "}
          · {sequential ? "Sequential" : "Diverging"} scale
        </span>
      </div>
      {error && (
        <p className="viz-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

export function Histogram({
  values,
  title = "Value distribution",
  bins = 36,
  height = 180,
}: {
  values: number[];
  title?: string;
  bins?: number;
  height?: number;
}) {
  const distribution = useMemo(() => histogram(values, bins), [values, bins]);
  const max = Math.max(1, ...distribution.map((bin) => bin.count));
  const left = 38;
  const top = 12;
  const plotWidth = 562;
  const plotHeight = 125;
  return (
    <div className="viz-histogram">
      <div className="viz-chart-heading">
        <strong>{title}</strong>
        <span>
          {values.filter(Number.isFinite).length.toLocaleString()} values
        </span>
      </div>
      {distribution.length ? (
        <svg
          viewBox="0 0 620 175"
          height={height}
          width="100%"
          role="img"
          aria-label={`${title}: ${distribution.length} histogram bins`}
        >
          {[0, 0.5, 1].map((fraction) => (
            <g key={fraction}>
              <line
                x1={left}
                x2={left + plotWidth}
                y1={top + plotHeight * (1 - fraction)}
                y2={top + plotHeight * (1 - fraction)}
                stroke="#20334b"
                strokeDasharray="3 4"
              />
              <text
                x={left - 5}
                y={top + plotHeight * (1 - fraction) + 4}
                fill="#7f94ad"
                fontSize="10"
                textAnchor="end"
              >
                {Math.round(max * fraction)}
              </text>
            </g>
          ))}
          {distribution.map((bin, index) => {
            const barHeight = (bin.count / max) * plotHeight;
            const width = plotWidth / distribution.length;
            return (
              <rect
                key={index}
                x={left + width * index + 0.5}
                y={top + plotHeight - barHeight}
                width={Math.max(0.2, width - 1)}
                height={barHeight}
                rx="1"
                fill={index < distribution.length / 2 ? "#22b8dc" : "#9471ee"}
              >
                <title>
                  {bin.lower.toPrecision(4)} to {bin.upper.toPrecision(4)}:{" "}
                  {bin.count} values
                </title>
              </rect>
            );
          })}
          <text x={left} y="158" fill="#a1b1c6" fontSize="11">
            {distribution[0].lower.toPrecision(3)}
          </text>
          <text
            x={left + plotWidth}
            y="158"
            textAnchor="end"
            fill="#a1b1c6"
            fontSize="11"
          >
            {distribution[distribution.length - 1].upper.toPrecision(3)}
          </text>
        </svg>
      ) : (
        <div className="viz-empty" style={{ height }}>
          No finite values.
        </div>
      )}
    </div>
  );
}

export function StatisticsTable({
  values,
  title = "Tensor statistics",
}: {
  values: number[];
  title?: string;
}) {
  const summary = useMemo(() => summarize(values), [values]);
  const entries: [string, keyof NumericSummary][] = [
    ["Count", "count"],
    ["Non-finite", "invalid"],
    ["Mean", "mean"],
    ["Median", "median"],
    ["Std. deviation", "standardDeviation"],
    ["Minimum", "min"],
    ["Maximum", "max"],
    ["L1 norm", "l1"],
    ["L2 norm", "l2"],
    ["Sparsity", "sparsity"],
  ];
  return (
    <div className="viz-statistics">
      <h4>{title}</h4>
      <dl>
        {entries.map(([label, key]) => (
          <div key={key}>
            <dt>{label}</dt>
            <dd>
              {key === "count" || key === "invalid"
                ? summary[key].toLocaleString()
                : key === "sparsity"
                  ? `${(summary[key] * 100).toFixed(2)}%`
                  : summary[key].toPrecision(5)}
            </dd>
          </div>
        ))}
      </dl>
      <p>Sparsity: |value| ≤ 10⁻⁸. Statistics use all finite source values.</p>
    </div>
  );
}

export interface TensorInspectorProps {
  layers: LayerData[];
  initialLayer?: number;
  initialMode?: "weights" | "activations" | "gradients" | "biases";
  onSelect?: (layer: number, neuron: number) => void;
}

export function TensorInspector({
  layers,
  initialLayer = 1,
  initialMode = "weights",
  onSelect,
}: TensorInspectorProps) {
  const [layerIndex, setLayerIndex] = useState(initialLayer);
  const [mode, setMode] = useState(initialMode);
  const index = clamp(layerIndex, 0, Math.max(0, layers.length - 1));
  const layer = layers[index];
  const matrix = useMemo(
    () =>
      !layer
        ? []
        : mode === "weights"
          ? layer.weights
          : [
              mode === "activations"
                ? layer.activations
                : mode === "gradients"
                  ? layer.gradients
                  : layer.biases,
            ],
    [layer, mode],
  );
  const values = useMemo(() => matrix.flat(), [matrix]);
  return (
    <section className="viz-tensor-inspector" aria-label="Tensor inspector">
      <div className="viz-inspector-toolbar">
        <label>
          Layer{" "}
          <select
            value={index}
            onChange={(event) => setLayerIndex(Number(event.target.value))}
            disabled={!layers.length}
          >
            {layers.map((item, i) => (
              <option value={i} key={`${item.id}-${i}`}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <div className="viz-tab-list" aria-label="Tensor type">
          {(["weights", "activations", "gradients", "biases"] as const).map(
            (item) => (
              <button
                key={item}
                onClick={() => setMode(item)}
                aria-pressed={mode === item}
              >
                {item}
              </button>
            ),
          )}
        </div>
        <button
          onClick={() =>
            exportTensorCSV(matrix, `${layer?.id || "tensor"}-${mode}.csv`)
          }
          disabled={!values.length}
        >
          Export CSV
        </button>
        <button
          onClick={() => exportLayersJSON(layers)}
          disabled={!layers.length}
        >
          Full capture
        </button>
      </div>
      {!layer ? (
        <div className="viz-empty">Start an experiment to capture tensors.</div>
      ) : (
        <>
          <div className="viz-inspector-grid">
            <TensorHeatmap
              matrix={matrix}
              title={`${layer.name} · ${mode}`}
              onSelect={(row, column) =>
                onSelect?.(index, mode === "weights" ? row : column)
              }
            />
            <StatisticsTable values={values} />
          </div>
          <Histogram values={values} />
          <p className="viz-note">
            {mode === "weights"
              ? "Rows are output units; columns are input units. The full matrix is available in CSV."
              : mode === "gradients"
                ? "Captured gradient per unit. Consult the experiment metadata for parameter-level gradient coverage."
                : "This view displays actual values from the latest captured forward pass."}
          </p>
        </>
      )}
    </section>
  );
}

export default TensorInspector;
