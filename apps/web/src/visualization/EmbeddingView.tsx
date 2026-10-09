import { useMemo, useState } from "react";
import {
  clamp,
  projectPCA,
  rgbCss,
  divergingColor,
  type Projection,
} from "./numerics";
import { exportTensorCSV } from "./exports";
import "./visualization.css";

export interface EmbeddingViewProps {
  vectors: number[][];
  labels?: string[];
  groups?: string[];
  title?: string;
  onSelect?: (index: number) => void;
}
const GROUP_COLORS = [
  "#38bdf8",
  "#a78bfa",
  "#2dd4bf",
  "#fb7185",
  "#fbbf24",
  "#c084fc",
  "#f97316",
  "#93c5fd",
];

export function EmbeddingView({
  vectors,
  labels = [],
  groups = [],
  title = "Embedding projection",
  onSelect,
}: EmbeddingViewProps) {
  const [selected, setSelected] = useState<number | null>(null);
  const [axisX, setAxisX] = useState(0);
  const [axisY, setAxisY] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [showLabels, setShowLabels] = useState(false);
  const projection = useMemo<{
    value: Projection | null;
    error: string;
  }>(() => {
    try {
      return { value: projectPCA(vectors), error: "" };
    } catch (error) {
      return {
        value: null,
        error: error instanceof Error ? error.message : "Projection failed.",
      };
    }
  }, [vectors]);
  const points = projection.value?.points || [];
  const groupNames = Array.from(new Set(groups));
  const scale =
    Math.max(
      1e-9,
      ...points.flatMap((point) => [
        Math.abs(point[axisX]),
        Math.abs(point[axisY]),
      ]),
    ) / zoom;
  const location = (index: number) => ({
    x: 360 + (points[index][axisX] / scale) * 310,
    y: 220 - (points[index][axisY] / scale) * 175,
  });
  const choose = (index: number) => {
    setSelected(index);
    onSelect?.(index);
  };
  if (projection.error)
    return (
      <div className="viz-error" role="alert">
        {projection.error}
      </div>
    );
  if (!vectors.length)
    return (
      <div className="viz-empty">
        <h3>No embedding vectors captured</h3>
        <p>
          Supply captured representations to compute a centered PCA projection.
        </p>
      </div>
    );
  return (
    <section className="viz-embeddings" aria-label={title}>
      <div className="viz-inspector-toolbar">
        <h3>{title}</h3>
        <label>
          Horizontal{" "}
          <select
            value={axisX}
            onChange={(event) => setAxisX(Number(event.target.value))}
          >
            {[0, 1, 2].map((axis) => (
              <option key={axis} value={axis}>
                PC {axis + 1}
              </option>
            ))}
          </select>
        </label>
        <label>
          Vertical{" "}
          <select
            value={axisY}
            onChange={(event) => setAxisY(Number(event.target.value))}
          >
            {[0, 1, 2].map((axis) => (
              <option key={axis} value={axis}>
                PC {axis + 1}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={() => setShowLabels((value) => !value)}
          aria-pressed={showLabels}
        >
          Labels
        </button>
        <button onClick={() => exportTensorCSV(points, "pca-projection.csv")}>
          Export PCA
        </button>
      </div>
      <div className="viz-embedding-layout">
        <div>
          <svg
            viewBox="0 0 720 440"
            width="100%"
            role="img"
            aria-label={`Centered PCA projection of ${vectors.length} captured vectors`}
            onWheel={(event) =>
              setZoom((value) =>
                clamp(value * (event.deltaY > 0 ? 0.9 : 1.1), 0.2, 8),
              )
            }
          >
            <defs>
              <clipPath id="embedding-clip">
                <rect x="35" y="30" width="650" height="375" />
              </clipPath>
            </defs>
            <rect
              x="35"
              y="30"
              width="650"
              height="375"
              fill="#07101f"
              stroke="#1c3048"
            />
            {[-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75].map((fraction) => (
              <g key={fraction}>
                <line
                  x1={360 + fraction * 310}
                  x2={360 + fraction * 310}
                  y1="30"
                  y2="405"
                  stroke={fraction ? "#15273d" : "#2d4e6f"}
                />
                <line
                  x1="35"
                  x2="685"
                  y1={220 + fraction * 175}
                  y2={220 + fraction * 175}
                  stroke={fraction ? "#15273d" : "#2d4e6f"}
                />
              </g>
            ))}
            <g clipPath="url(#embedding-clip)">
              {points.map((point, index) => {
                const position = location(index);
                const color = groups[index]
                  ? GROUP_COLORS[
                      groupNames.indexOf(groups[index]) % GROUP_COLORS.length
                    ]
                  : rgbCss(divergingColor(point[2], scale));
                return (
                  <g
                    key={index}
                    tabIndex={0}
                    role="button"
                    aria-label={`Select vector ${labels[index] || index}`}
                    onClick={() => choose(index)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") choose(index);
                    }}
                  >
                    <circle
                      cx={position.x}
                      cy={position.y}
                      r={selected === index ? 6 : 3.5}
                      fill={color}
                      fillOpacity={selected === index ? 1 : 0.72}
                      stroke={selected === index ? "#ffffff" : "none"}
                    >
                      <title>
                        {labels[index] || `Vector ${index}`}: [
                        {point.map((value) => value.toPrecision(3)).join(", ")}]
                      </title>
                    </circle>
                    {showLabels && points.length <= 120 && (
                      <text
                        x={position.x + 7}
                        y={position.y - 5}
                        fontSize="9"
                        fill="#b5c7de"
                      >
                        {labels[index] || index}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
            <text
              x="360"
              y="429"
              textAnchor="middle"
              fill="#8aa2be"
              fontSize="11"
            >
              Principal component {axisX + 1}
            </text>
            <text
              x="12"
              y="220"
              transform="rotate(-90 12 220)"
              textAnchor="middle"
              fill="#8aa2be"
              fontSize="11"
            >
              Principal component {axisY + 1}
            </text>
          </svg>
          <div className="viz-chart-caption">
            <span>
              {vectors.length.toLocaleString()} vectors · {vectors[0].length}{" "}
              dimensions
            </span>
            <label>
              Zoom{" "}
              <input
                aria-label="Embedding zoom"
                type="range"
                min="0.2"
                max="8"
                step="0.1"
                value={zoom}
                onChange={(event) => setZoom(Number(event.target.value))}
              />
            </label>
          </div>
        </div>
        <aside className="viz-embedding-summary">
          <h4>Explained variance</h4>
          {[0, 1, 2].map((index) => (
            <div className="viz-variance-row" key={index}>
              <span>PC {index + 1}</span>
              <strong>
                {(
                  (projection.value?.explainedVariance[index] || 0) * 100
                ).toFixed(2)}
                %
              </strong>
            </div>
          ))}
          <p>
            Centered PCA, computed from these vectors. Missing rank dimensions
            are zero.
          </p>
          {groupNames.length > 0 && (
            <>
              <h4>Source groups</h4>
              {groupNames.map((group, index) => (
                <div className="viz-variance-row" key={group}>
                  <span>
                    <i
                      style={{
                        background: GROUP_COLORS[index % GROUP_COLORS.length],
                      }}
                    />
                    {group}
                  </span>
                  <strong>
                    {groups.filter((value) => value === group).length}
                  </strong>
                </div>
              ))}
            </>
          )}
          {selected !== null && (
            <>
              <h4>Selected vector</h4>
              <strong>{labels[selected] || `Vector ${selected}`}</strong>
              <dl>
                {points[selected].map((value, axis) => (
                  <div key={axis}>
                    <dt>PC {axis + 1}</dt>
                    <dd>{value.toPrecision(5)}</dd>
                  </div>
                ))}
              </dl>
            </>
          )}
        </aside>
      </div>
      <p className="viz-note">
        PCA preserves directions of greatest variance; visual proximity is a
        projection and does not establish semantic equivalence.
      </p>
    </section>
  );
}

export default EmbeddingView;
