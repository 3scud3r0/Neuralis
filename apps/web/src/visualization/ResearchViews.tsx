import { useMemo, useState } from "react";
import type { LayerData, SceneProps } from "../types";
import NetworkScene from "./NetworkScene";
import {
  clamp,
  compareVectors,
  cosineSimilarity,
  projectPCA,
  selectEdges,
  summarize,
  type Projection,
  type Vector3Tuple,
} from "./numerics";
import { Histogram, TensorHeatmap } from "./TensorInspector";
import "./visualization.css";

export interface ResidualViewProps {
  trajectories: number[][][];
  labels?: string[];
  layerLabels?: string[];
  title?: string;
}

export function ResidualView({
  trajectories,
  labels = [],
  layerLabels = [],
  title = "Residual trajectories",
}: ResidualViewProps) {
  const [selected, setSelected] = useState(0);
  const [showAll, setShowAll] = useState(true);
  const result = useMemo<{
    projection: Projection | null;
    points: Vector3Tuple[][];
    error: string;
  }>(() => {
    try {
      const projection = projectPCA(trajectories.flat());
      let offset = 0;
      const points = trajectories.map((trajectory) => {
        const slice = projection.points.slice(
          offset,
          offset + trajectory.length,
        );
        offset += trajectory.length;
        return slice;
      });
      return { projection, points, error: "" };
    } catch (error) {
      return {
        projection: null,
        points: [],
        error: error instanceof Error ? error.message : "Projection failed.",
      };
    }
  }, [trajectories]);
  const index = clamp(selected, 0, Math.max(0, trajectories.length - 1));
  const magnitude = result.points
    .flat()
    .reduce(
      (max, point) => Math.max(max, Math.abs(point[0]), Math.abs(point[1])),
      1e-9,
    );
  const x = (point: Vector3Tuple) => 320 + (point[0] / magnitude) * 270;
  const y = (point: Vector3Tuple) => 185 - (point[1] / magnitude) * 145;
  const colors = [
    "#38bdf8",
    "#a78bfa",
    "#2dd4bf",
    "#fb7185",
    "#fbbf24",
    "#818cf8",
  ];
  const similarities = (trajectories[index] || []).map((vector) =>
    cosineSimilarity(vector, trajectories[index][0]),
  );
  if (result.error)
    return (
      <div className="viz-error" role="alert">
        {result.error}
      </div>
    );
  if (!trajectories.length)
    return (
      <div className="viz-empty">
        <h3>No residual capture</h3>
        <p>
          Capture equal-dimensional representations across layers to compare
          trajectories.
        </p>
      </div>
    );
  return (
    <section className="viz-residual">
      <div className="viz-inspector-toolbar">
        <h3>{title}</h3>
        <label>
          Trajectory{" "}
          <select
            value={index}
            onChange={(event) => setSelected(Number(event.target.value))}
          >
            {trajectories.map((_, i) => (
              <option value={i} key={i}>
                {labels[i] || `Vector ${i}`}
              </option>
            ))}
          </select>
        </label>
        <button
          aria-pressed={showAll}
          onClick={() => setShowAll((value) => !value)}
        >
          All trajectories
        </button>
      </div>
      <div className="viz-residual-grid">
        <svg
          viewBox="0 0 640 370"
          width="100%"
          role="img"
          aria-label="Residual representations in a shared PCA basis"
        >
          <rect
            x="25"
            y="20"
            width="590"
            height="325"
            fill="#07101f"
            stroke="#1d304b"
          />
          {[-0.5, 0, 0.5].map((fraction) => (
            <g key={fraction}>
              <line
                x1={320 + fraction * 270}
                x2={320 + fraction * 270}
                y1="20"
                y2="345"
                stroke="#213249"
              />
              <line
                x1="25"
                x2="615"
                y1={185 + fraction * 145}
                y2={185 + fraction * 145}
                stroke="#213249"
              />
            </g>
          ))}
          {result.points.map(
            (points, trajectory) =>
              (showAll || index === trajectory) && (
                <g key={trajectory} opacity={trajectory === index ? 1 : 0.3}>
                  <polyline
                    points={points
                      .map((point) => `${x(point)},${y(point)}`)
                      .join(" ")}
                    fill="none"
                    stroke={colors[trajectory % colors.length]}
                    strokeWidth={trajectory === index ? 2.5 : 1}
                  />
                  {points.map((point, layer) => (
                    <g key={layer}>
                      <circle
                        cx={x(point)}
                        cy={y(point)}
                        r={trajectory === index ? 4 : 2.5}
                        fill={colors[trajectory % colors.length]}
                      >
                        <title>
                          {labels[trajectory] || `Vector ${trajectory}`} ·{" "}
                          {layerLabels[layer] || `Layer ${layer}`}
                        </title>
                      </circle>
                      {trajectory === index && (
                        <text
                          x={x(point) + 7}
                          y={y(point) - 7}
                          fill="#c3d4e9"
                          fontSize="10"
                        >
                          {layerLabels[layer] || layer}
                        </text>
                      )}
                    </g>
                  ))}
                </g>
              ),
          )}
          <text
            x="320"
            y="362"
            fill="#8aa2be"
            textAnchor="middle"
            fontSize="11"
          >
            Shared principal component 1
          </text>
        </svg>
        <div>
          <h4>Similarity to initial layer</h4>
          <svg
            viewBox="0 0 290 220"
            width="100%"
            role="img"
            aria-label="Cosine similarity by layer"
          >
            {[-1, 0, 1].map((value) => (
              <g key={value}>
                <line
                  x1="30"
                  x2="275"
                  y1={110 - value * 85}
                  y2={110 - value * 85}
                  stroke="#213249"
                />
                <text
                  x="22"
                  y={114 - value * 85}
                  textAnchor="end"
                  fill="#8aa2be"
                  fontSize="10"
                >
                  {value}
                </text>
              </g>
            ))}
            <polyline
              points={similarities
                .map(
                  (similarity, layer) =>
                    `${30 + (layer / Math.max(1, similarities.length - 1)) * 245},${110 - similarity * 85}`,
                )
                .join(" ")}
              fill="none"
              stroke="#a78bfa"
              strokeWidth="2"
            />
            {similarities.map((similarity, layer) => (
              <circle
                key={layer}
                cx={30 + (layer / Math.max(1, similarities.length - 1)) * 245}
                cy={110 - similarity * 85}
                r="3"
                fill="#c4b5fd"
              >
                <title>
                  {layerLabels[layer] || `Layer ${layer}`}:{" "}
                  {similarity.toFixed(5)}
                </title>
              </circle>
            ))}
          </svg>
          <p className="viz-note">
            All trajectories share one centered PCA basis. Layer-wise cosine
            similarity uses the original representations.
          </p>
        </div>
      </div>
    </section>
  );
}

export interface CircuitViewProps {
  layers: LayerData[];
  maxEdges?: number;
  onSelect?: SceneProps["onSelect"];
  title?: string;
}

export function CircuitView({
  layers,
  maxEdges = 240,
  onSelect,
  title = "Weight circuits",
}: CircuitViewProps) {
  const [threshold, setThreshold] = useState(0);
  const [budget, setBudget] = useState(maxEdges);
  const edges = useMemo(
    () => selectEdges(layers, budget, threshold),
    [layers, budget, threshold],
  );
  const nodeMap = useMemo(() => {
    const map = new Map<number, Set<number>>();
    for (const edge of edges) {
      if (!map.has(edge.sourceLayer)) map.set(edge.sourceLayer, new Set());
      if (!map.has(edge.targetLayer)) map.set(edge.targetLayer, new Set());
      map.get(edge.sourceLayer)!.add(edge.source);
      map.get(edge.targetLayer)!.add(edge.target);
    }
    return new Map(
      Array.from(map, ([layer, nodes]) => [
        layer,
        Array.from(nodes).sort((a, b) => a - b),
      ]),
    );
  }, [edges]);
  const location = (layer: number, neuron: number) => {
    const nodes = nodeMap.get(layer) || [];
    return {
      x: 60 + (layer / Math.max(1, layers.length - 1)) * 680,
      y: 45 + (nodes.indexOf(neuron) / Math.max(1, nodes.length - 1)) * 305,
    };
  };
  const highest = Math.max(1e-9, edges[0]?.score || 0);
  return (
    <section className="viz-circuit">
      <div className="viz-inspector-toolbar">
        <h3>{title}</h3>
        <label>
          Top edges{" "}
          <input
            type="number"
            value={budget}
            min="0"
            max="1200"
            step="20"
            onChange={(event) =>
              setBudget(clamp(Number(event.target.value), 0, 1200))
            }
          />
        </label>
        <label>
          Min |weight|{" "}
          <input
            type="number"
            value={threshold}
            min="0"
            step="0.01"
            onChange={(event) =>
              setThreshold(Math.max(0, Number(event.target.value)))
            }
          />
        </label>
      </div>
      {edges.length ? (
        <svg
          viewBox="0 0 800 400"
          width="100%"
          role="img"
          aria-label={`${edges.length} strongest parameter connections`}
        >
          {layers.map((layer, index) => (
            <g key={`${layer.id}-${index}`}>
              <line
                x1={60 + (index / Math.max(1, layers.length - 1)) * 680}
                x2={60 + (index / Math.max(1, layers.length - 1)) * 680}
                y1="35"
                y2="355"
                stroke="#132940"
              />
              <text
                x={60 + (index / Math.max(1, layers.length - 1)) * 680}
                y="380"
                textAnchor="middle"
                fill="#8ca9c8"
                fontSize="10"
              >
                {layer.name}
              </text>
            </g>
          ))}
          {edges.map((edge, index) => {
            const a = location(edge.sourceLayer, edge.source);
            const b = location(edge.targetLayer, edge.target);
            return (
              <path
                key={index}
                d={`M${a.x},${a.y} C${(a.x + b.x) / 2},${a.y} ${(a.x + b.x) / 2},${b.y} ${b.x},${b.y}`}
                fill="none"
                stroke={edge.weight >= 0 ? "#a78bfa" : "#22d3ee"}
                opacity={0.15 + (edge.score / highest) * 0.7}
                strokeWidth={0.4 + (edge.score / highest) * 2}
              >
                <title>
                  {layers[edge.sourceLayer].name}[{edge.source}] →{" "}
                  {layers[edge.targetLayer].name}[{edge.target}]:{" "}
                  {edge.weight.toPrecision(6)}
                </title>
              </path>
            );
          })}
          {Array.from(nodeMap, ([layer, neurons]) =>
            neurons.map((neuron) => {
              const point = location(layer, neuron);
              return (
                <g
                  key={`${layer}-${neuron}`}
                  role="button"
                  tabIndex={0}
                  aria-label={`${layers[layer].name} neuron ${neuron}`}
                  onClick={() => onSelect?.(layer, neuron)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") onSelect?.(layer, neuron);
                  }}
                >
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r="3.5"
                    fill="#87d8ff"
                    stroke="#07101f"
                  />
                  <title>
                    {layers[layer].name} · Neuron {neuron}
                  </title>
                </g>
              );
            }),
          )}
        </svg>
      ) : (
        <div className="viz-empty">
          No parameter connections meet this threshold.
        </div>
      )}
      <div className="viz-chart-caption">
        <span>
          {nodeMap.size} layers ·{" "}
          {Array.from(nodeMap.values()).reduce(
            (sum, nodes) => sum + nodes.length,
            0,
          )}{" "}
          connected units · {edges.length} edges
        </span>
        <span>
          <i className="viz-legend-cyan" /> Negative weight{" "}
          <i className="viz-legend-purple" /> Positive weight
        </span>
      </div>
      <p className="viz-note">
        This is a parameter graph ranked by |weight|. Causal importance requires
        a measured intervention; weight magnitude alone does not establish a
        causal circuit.
      </p>
    </section>
  );
}

export interface ComparisonViewProps {
  left: LayerData[];
  right: LayerData[];
  leftLabel?: string;
  rightLabel?: string;
}

export function ComparisonView({
  left,
  right,
  leftLabel = "Reference capture",
  rightLabel = "Comparison capture",
}: ComparisonViewProps) {
  const [mode, setMode] =
    useState<NonNullable<SceneProps["mode"]>>("activations");
  const [selected, setSelected] = useState<SceneProps["selected"]>(null);
  const [layer, setLayer] = useState(1);
  const [reset, setReset] = useState(0);
  const index = clamp(
    layer,
    0,
    Math.max(0, Math.min(left.length, right.length) - 1),
  );
  const comparison = useMemo(() => {
    const a = left[index];
    const b = right[index];
    if (!a || !b)
      return {
        result: null,
        differences: [] as number[],
        error: "Select two captures to compare.",
      };
    const valuesA =
      mode === "activations"
        ? a.activations
        : mode === "gradients"
          ? a.gradients
          : a.weights.flat();
    const valuesB =
      mode === "activations"
        ? b.activations
        : mode === "gradients"
          ? b.gradients
          : b.weights.flat();
    if (
      mode === "weights" &&
      (a.weights.length !== b.weights.length ||
        a.weights.some((row, i) => row.length !== b.weights[i]?.length))
    )
      return {
        result: null,
        differences: [],
        error:
          "Parameter matrix shapes differ. Compare captures with matching architectures.",
      };
    try {
      return {
        result: compareVectors(valuesA, valuesB),
        differences: valuesA.map((value, i) => value - valuesB[i]),
        error: "",
      };
    } catch (error) {
      return {
        result: null,
        differences: [],
        error:
          error instanceof Error ? error.message : "Tensors are incompatible.",
      };
    }
  }, [left, right, index, mode]);
  return (
    <section className="viz-comparison">
      <div className="viz-inspector-toolbar">
        <h3>Capture comparison</h3>
        <label>
          Color{" "}
          <select
            value={mode}
            onChange={(event) =>
              setMode(event.target.value as NonNullable<SceneProps["mode"]>)
            }
          >
            <option value="activations">Activations</option>
            <option value="weights">Weights</option>
            <option value="gradients">Gradients</option>
          </select>
        </label>
        <label>
          Layer{" "}
          <select
            value={index}
            onChange={(event) => setLayer(Number(event.target.value))}
          >
            {left.slice(0, right.length).map((item, i) => (
              <option key={`${item.id}-${i}`} value={i}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <button onClick={() => setReset((value) => value + 1)}>
          Reset both views
        </button>
      </div>
      <div className="viz-comparison-scenes">
        <div>
          <h4>{leftLabel}</h4>
          <NetworkScene
            key={`left-${reset}`}
            layers={left}
            mode={mode}
            selected={selected}
            onSelect={(layer, neuron) => setSelected({ layer, neuron })}
            height={330}
          />
        </div>
        <div>
          <h4>{rightLabel}</h4>
          <NetworkScene
            key={`right-${reset}`}
            layers={right}
            mode={mode}
            selected={selected}
            onSelect={(layer, neuron) => setSelected({ layer, neuron })}
            height={330}
          />
        </div>
      </div>
      {comparison.error ? (
        <p className="viz-note">{comparison.error}</p>
      ) : (
        comparison.result && (
          <>
            <div className="viz-comparison-metrics">
              {Object.entries(comparison.result).map(([key, value]) => (
                <div key={key}>
                  <span>{key.replace(/([A-Z])/g, " $1")}</span>
                  <strong>
                    {key === "compared"
                      ? value.toLocaleString()
                      : value.toPrecision(5)}
                  </strong>
                </div>
              ))}
            </div>
            <Histogram
              values={comparison.differences}
              title={`${left[index]?.name} · reference minus comparison`}
            />
            <TensorHeatmap
              matrix={[comparison.differences]}
              title="Element-wise difference"
              height={90}
            />
          </>
        )
      )}
      <p className="viz-note">
        Color mode, selected unit and view reset are linked. Each camera can be
        navigated independently. Numerical comparison uses all compatible tensor
        elements.
      </p>
    </section>
  );
}

export function NeuronInspector({
  layers,
  selected,
}: {
  layers: LayerData[];
  selected: SceneProps["selected"];
}) {
  if (!selected)
    return (
      <aside className="viz-neuron-inspector">
        <h4>Unit inspection</h4>
        <p>
          Click a unit in the network to inspect its captured values and
          incoming parameters.
        </p>
      </aside>
    );
  const layer = layers[selected.layer];
  if (!layer || selected.neuron >= layer.size || selected.neuron < 0)
    return (
      <aside className="viz-neuron-inspector">
        <p>The selected unit is unavailable in this capture.</p>
      </aside>
    );
  const weights = layer.weights[selected.neuron] || [];
  const stats = summarize(weights);
  return (
    <aside className="viz-neuron-inspector">
      <h4>
        {layer.name} · unit {selected.neuron}
      </h4>
      <dl>
        <div>
          <dt>Activation</dt>
          <dd>
            {layer.activations[selected.neuron]?.toPrecision(6) ??
              "Not captured"}
          </dd>
        </div>
        <div>
          <dt>Gradient</dt>
          <dd>
            {layer.gradients[selected.neuron]?.toPrecision(6) ?? "Not captured"}
          </dd>
        </div>
        <div>
          <dt>Bias</dt>
          <dd>
            {layer.biases[selected.neuron]?.toPrecision(6) ?? "Not applicable"}
          </dd>
        </div>
        <div>
          <dt>Incoming weights</dt>
          <dd>{weights.length}</dd>
        </div>
        <div>
          <dt>Weight L2</dt>
          <dd>{stats.l2.toPrecision(5)}</dd>
        </div>
      </dl>
      {weights.length > 0 && (
        <Histogram
          values={weights}
          title="Incoming weights"
          bins={24}
          height={145}
        />
      )}
    </aside>
  );
}
