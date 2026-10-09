import { useMemo, useState } from "react";
import { attentionEntropy, clamp, finite, validateMatrix } from "./numerics";
import { TensorHeatmap } from "./TensorInspector";
import { exportTensorCSV } from "./exports";
import "./visualization.css";

export interface AttentionViewProps {
  matrix: number[][];
  labels?: string[];
  title?: string;
  causal?: boolean;
  maxTokens?: number;
}

export function AttentionView({
  matrix,
  labels = [],
  title = "Attention explorer",
  causal = false,
  maxTokens = 64,
}: AttentionViewProps) {
  const [selected, setSelected] = useState(0);
  const [topK, setTopK] = useState(8);
  const validation = useMemo(() => {
    try {
      const shape = validateMatrix(matrix);
      if (
        matrix.some((row) =>
          row.some((value) => !Number.isFinite(value) || value < 0),
        )
      )
        throw new Error(
          "Attention matrices must contain finite, non-negative weights.",
        );
      return { ...shape, error: "" };
    } catch (error) {
      return {
        rows: 0,
        columns: 0,
        error:
          error instanceof Error ? error.message : "Invalid attention matrix.",
      };
    }
  }, [matrix]);
  const query = clamp(selected, 0, Math.max(0, validation.rows - 1));
  const row = matrix[query] || [];
  const total = row.reduce((sum, value) => sum + finite(value), 0);
  const ranked = useMemo(
    () =>
      row
        .map((value, index) => ({ value, index }))
        .sort((a, b) => b.value - a.value)
        .slice(0, topK),
    [row, topK],
  );
  const label = (index: number) => labels[index] || `Token ${index}`;
  const graphTokens = Math.min(validation.columns, clamp(maxTokens, 2, 96));
  const centerX = 240;
  const centerY = 190;
  const radius = 132;
  const positions = Array.from({ length: graphTokens }, (_, index) => ({
    x:
      centerX +
      Math.cos((index / graphTokens) * Math.PI * 2 - Math.PI / 2) * radius,
    y:
      centerY +
      Math.sin((index / graphTokens) * Math.PI * 2 - Math.PI / 2) * radius,
  }));
  const queryPosition = positions[query];
  if (validation.error)
    return (
      <div className="viz-error" role="alert">
        {validation.error}
      </div>
    );
  if (!validation.rows || !validation.columns)
    return (
      <div className="viz-empty">
        <h3>No attention capture</h3>
        <p>
          This experiment does not expose an attention matrix. Dense connections
          are available in Network geometry.
        </p>
      </div>
    );
  return (
    <section className="viz-attention" aria-label={title}>
      <div className="viz-inspector-toolbar">
        <h3>{title}</h3>
        <label>
          Query{" "}
          <select
            value={query}
            onChange={(event) => setSelected(Number(event.target.value))}
          >
            {matrix.map((_, index) => (
              <option key={index} value={index}>
                {index} · {label(index)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Top keys{" "}
          <input
            type="number"
            min="1"
            max="32"
            value={topK}
            onChange={(event) =>
              setTopK(clamp(Number(event.target.value), 1, 32))
            }
          />
        </label>
        <button onClick={() => exportTensorCSV(matrix, "attention.csv")}>
          Export matrix
        </button>
      </div>
      <div className="viz-attention-grid">
        <div className="viz-attention-graph">
          <svg
            viewBox="0 0 480 380"
            width="100%"
            role="img"
            aria-label={`Top attention connections for ${label(query)}`}
          >
            <defs>
              <radialGradient id="attention-node">
                <stop stopColor="#6ee7ff" />
                <stop offset="1" stopColor="#3b82f6" />
              </radialGradient>
            </defs>
            <circle
              cx={centerX}
              cy={centerY}
              r={radius}
              fill="none"
              stroke="#224264"
              strokeDasharray="3 5"
            />
            {queryPosition &&
              ranked
                .filter(
                  (item) => item.index < graphTokens && item.index !== query,
                )
                .map((item) => {
                  const target = positions[item.index];
                  const strength = total ? item.value / total : 0;
                  return (
                    <path
                      key={item.index}
                      d={`M${queryPosition.x},${queryPosition.y} Q${centerX},${centerY} ${target.x},${target.y}`}
                      fill="none"
                      stroke={strength > 0.25 ? "#ce80ff" : "#38bdf8"}
                      opacity={0.2 + strength * 0.8}
                      strokeWidth={1 + strength * 7}
                    >
                      <title>
                        {label(query)} → {label(item.index)}:{" "}
                        {(strength * 100).toFixed(2)}%
                      </title>
                    </path>
                  );
                })}
            {positions.map((position, index) => (
              <g
                key={index}
                role="button"
                tabIndex={0}
                aria-label={`Select query ${label(index)}`}
                onClick={() => {
                  if (index < validation.rows) setSelected(index);
                }}
                onKeyDown={(event) => {
                  if (
                    (event.key === "Enter" || event.key === " ") &&
                    index < validation.rows
                  ) {
                    event.preventDefault();
                    setSelected(index);
                  }
                }}
              >
                {index === query && (
                  <circle
                    cx={position.x}
                    cy={position.y}
                    r="16"
                    fill="#a855f7"
                    opacity="0.2"
                  />
                )}
                <circle
                  cx={position.x}
                  cy={position.y}
                  r={index === query ? 8 : 5}
                  fill={index === query ? "#d8a9ff" : "url(#attention-node)"}
                  stroke="#addefe"
                  strokeWidth="0.6"
                />
                <text
                  x={position.x}
                  y={position.y + (position.y > centerY ? 20 : -14)}
                  textAnchor="middle"
                  fill={index === query ? "#e9d5ff" : "#94aec9"}
                  fontSize="10"
                >
                  {label(index).slice(0, 16)}
                </text>
              </g>
            ))}
          </svg>
          <div className="viz-chart-caption">
            <span>
              {causal
                ? "Causal mask expected from source capture"
                : "Connections from supplied attention weights"}
            </span>
            <span>
              {validation.columns > graphTokens
                ? `Showing first ${graphTokens} tokens`
                : `${validation.columns} keys`}
            </span>
          </div>
        </div>
        <div className="viz-attention-rank">
          <h4>{label(query)}</h4>
          <dl>
            <div>
              <dt>Row sum</dt>
              <dd>{total.toFixed(5)}</dd>
            </div>
            <div>
              <dt>Entropy</dt>
              <dd>{attentionEntropy(row).toFixed(3)} bits</dd>
            </div>
          </dl>
          <h4>Most attended keys</h4>
          {ranked.map((item) => (
            <button
              key={item.index}
              onClick={() => {
                if (item.index < validation.rows) setSelected(item.index);
              }}
              title={`Key ${item.index}`}
            >
              <span>{label(item.index)}</span>
              <div>
                <i
                  style={{
                    width: `${total ? (item.value / total) * 100 : 0}%`,
                  }}
                />
              </div>
              <strong>
                {total ? (item.value / total).toFixed(4) : "0.0000"}
              </strong>
            </button>
          ))}
        </div>
      </div>
      <TensorHeatmap
        matrix={matrix}
        title="Query × key attention"
        sequential
        domain={matrix.reduce(
          (max, current) =>
            current.reduce((peak, value) => Math.max(peak, value), max),
          1e-12,
        )}
        onSelect={(rowIndex) => setSelected(rowIndex)}
      />
      <p className="viz-note">
        Display uses the supplied attention capture. Rows are queries, columns
        are keys; ranking probabilities are normalized per row for inspection.
      </p>
    </section>
  );
}

export default AttentionView;
