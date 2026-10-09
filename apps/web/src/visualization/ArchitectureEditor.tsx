import { useMemo, useState } from "react";
import {
  analyzeArchitecture,
  connectArchitecture,
  layoutArchitecture,
  parseArchitecture,
  removeArchitectureNode,
  renameArchitectureNode,
  type ArchitectureGraph,
  type ArchitectureOperation,
} from "./architecture";
import { downloadBlob } from "./exports";
import "./visualization.css";

export interface ArchitectureEditorProps {
  graph: ArchitectureGraph;
  onChange?: (graph: ArchitectureGraph) => void;
  readOnly?: boolean;
}
const OPERATION_COLORS: Record<ArchitectureOperation, string> = {
  input: "#22d3ee",
  dense: "#818cf8",
  relu: "#a78bfa",
  tanh: "#a78bfa",
  sigmoid: "#a78bfa",
  softmax: "#fb7185",
  layernorm: "#2dd4bf",
  add: "#fbbf24",
  concat: "#f97316",
  output: "#fb7185",
};

export function ArchitectureEditor({
  graph: initialGraph,
  onChange,
  readOnly = false,
}: ArchitectureEditorProps) {
  const [edited, setEdited] = useState<ArchitectureGraph | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [operation, setOperation] = useState<ArchitectureOperation>("dense");
  const [error, setError] = useState("");
  const graph = edited || initialGraph;
  const analysis = useMemo(() => analyzeArchitecture(graph), [graph]);
  const positions = useMemo(() => layoutArchitecture(graph), [graph]);
  const positionMap = new Map(
    positions.map((position) => [position.id, position]),
  );
  const width = Math.max(740, ...positions.map((position) => position.x + 190));
  const height = Math.max(
    240,
    ...positions.map((position) => position.y + 115),
  );
  const chosen = graph.nodes.find((node) => node.id === selected);
  const save = (next: ArchitectureGraph) => {
    setEdited(next);
    onChange?.(next);
    setError("");
  };
  const connect = () => {
    try {
      save(connectArchitecture(graph, source, target));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Connection failed.");
    }
  };
  const add = () => {
    if (graph.nodes.length >= 256) {
      setError("Architecture node budget reached.");
      return;
    }
    let counter = graph.nodes.length + 1;
    let id = `${operation}-${counter}`;
    while (graph.nodes.some((node) => node.id === id))
      id = `${operation}-${++counter}`;
    const node = {
      id,
      name: `${operation[0].toUpperCase()}${operation.slice(1)} ${counter}`,
      operation,
      ...(operation === "dense" ? { units: 16 } : {}),
      ...(operation === "input" ? { shape: [4] } : {}),
    };
    save({ ...graph, nodes: [...graph.nodes, node] });
    setSelected(id);
  };
  const load = async (file: File) => {
    try {
      if (file.size > 2000000)
        throw new Error("Architecture file is too large.");
      save(parseArchitecture(await file.text()));
      setSelected(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Import failed.");
    }
  };
  return (
    <section
      className="viz-architecture"
      aria-label="Architecture graph editor"
    >
      <div className="viz-inspector-toolbar">
        <h3>Architecture graph</h3>
        <span className="viz-axis-description">
          {graph.nodes.length} operations ·{" "}
          {analysis.totalParameters.toLocaleString()} inferred parameters
        </span>
        <button
          onClick={() =>
            downloadBlob(
              new Blob([JSON.stringify(graph, null, 2)], {
                type: "application/json",
              }),
              "neuralis-architecture.json",
            )
          }
        >
          Export graph
        </button>
        {!readOnly && (
          <label>
            Import{" "}
            <input
              type="file"
              accept=".json,application/json"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void load(file);
                event.target.value = "";
              }}
            />
          </label>
        )}
      </div>
      {!readOnly && (
        <div className="viz-inspector-toolbar">
          <label>
            Operation{" "}
            <select
              value={operation}
              onChange={(event) =>
                setOperation(event.target.value as ArchitectureOperation)
              }
            >
              {Object.keys(OPERATION_COLORS).map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <button onClick={add}>Add operation</button>
          <label>
            From{" "}
            <select
              value={source}
              onChange={(event) => setSource(event.target.value)}
            >
              <option value="">Select node</option>
              {graph.nodes.map((node) => (
                <option key={node.id} value={node.id}>
                  {node.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            To{" "}
            <select
              value={target}
              onChange={(event) => setTarget(event.target.value)}
            >
              <option value="">Select node</option>
              {graph.nodes.map((node) => (
                <option key={node.id} value={node.id}>
                  {node.name}
                </option>
              ))}
            </select>
          </label>
          <button onClick={connect} disabled={!source || !target}>
            Connect
          </button>
          {edited && (
            <button
              onClick={() => {
                setEdited(null);
                setSelected(null);
                onChange?.(initialGraph);
              }}
            >
              Restore captured architecture
            </button>
          )}
        </div>
      )}
      {error && (
        <p className="viz-error" role="alert">
          {error}
        </p>
      )}
      <div className="viz-architecture-layout">
        <div className="viz-architecture-scroll">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            width={width}
            height={height}
            role="img"
            aria-label="Directed model architecture"
          >
            <defs>
              <marker
                id="architecture-arrow"
                markerWidth="8"
                markerHeight="8"
                refX="7"
                refY="4"
                orient="auto"
              >
                <path d="M0,1 L7,4 L0,7" fill="none" stroke="#6588ae" />
              </marker>
            </defs>
            {graph.edges.map((edge, index) => {
              const a = positionMap.get(edge.source);
              const b = positionMap.get(edge.target);
              if (!a || !b) return null;
              return (
                <path
                  key={`${edge.source}-${edge.target}-${index}`}
                  d={`M${a.x + 132},${a.y + 31} C${(a.x + b.x + 132) / 2},${a.y + 31} ${(a.x + b.x + 132) / 2},${b.y + 31} ${b.x - 5},${b.y + 31}`}
                  fill="none"
                  stroke="#406187"
                  strokeWidth="1.4"
                  markerEnd="url(#architecture-arrow)"
                >
                  <title>
                    {edge.source} → {edge.target}
                  </title>
                </path>
              );
            })}
            {graph.nodes.map((node) => {
              const position = positionMap.get(node.id);
              if (!position) return null;
              const shape = analysis.shapes.get(node.id);
              const invalid = analysis.issues.some(
                (issue) => issue.node === node.id && issue.severity === "error",
              );
              return (
                <g
                  key={node.id}
                  transform={`translate(${position.x}, ${position.y})`}
                  tabIndex={0}
                  role="button"
                  aria-label={`${node.name}, ${node.operation}, shape ${shape?.join(" by ") || "unknown"}`}
                  onClick={() => setSelected(node.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") setSelected(node.id);
                  }}
                >
                  <rect
                    width="132"
                    height="64"
                    rx="7"
                    fill={selected === node.id ? "#173252" : "#0c1c30"}
                    stroke={
                      invalid
                        ? "#fb7185"
                        : selected === node.id
                          ? "#60c6f2"
                          : "#294565"
                    }
                    strokeWidth={selected === node.id ? 1.8 : 1}
                  />
                  <rect
                    x="0"
                    y="10"
                    width="3"
                    height="44"
                    rx="2"
                    fill={OPERATION_COLORS[node.operation]}
                  />
                  <text x="12" y="19" fill="#d0e1f4" fontSize="10">
                    {node.name.slice(0, 19)}
                  </text>
                  <text
                    x="12"
                    y="36"
                    fill={OPERATION_COLORS[node.operation]}
                    fontSize="9"
                  >
                    {node.operation}
                  </text>
                  <text x="12" y="52" fill="#819bb9" fontSize="9">
                    {shape ? `[${shape.join(" × ")}]` : "Shape pending"}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
        <aside className="viz-architecture-properties">
          <h4>{chosen ? chosen.name : "Select an operation"}</h4>
          {chosen && (
            <>
              <dl>
                <div>
                  <dt>Operation</dt>
                  <dd>{chosen.operation}</dd>
                </div>
                <div>
                  <dt>Depth</dt>
                  <dd>{analysis.depth.get(chosen.id) ?? "—"}</dd>
                </div>
                <div>
                  <dt>Parameters</dt>
                  <dd>
                    {(analysis.parameters.get(chosen.id) || 0).toLocaleString()}
                  </dd>
                </div>
                <div>
                  <dt>Output shape</dt>
                  <dd>[{analysis.shapes.get(chosen.id)?.join(", ") || "?"}]</dd>
                </div>
              </dl>
              {!readOnly && (
                <div className="viz-architecture-form">
                  <label>
                    Name
                    <input
                      value={chosen.name}
                      maxLength={200}
                      onChange={(event) => {
                        try {
                          save(
                            renameArchitectureNode(
                              graph,
                              chosen.id,
                              event.target.value,
                            ),
                          );
                        } catch {
                          /* Keep the last valid name during an empty edit. */
                        }
                      }}
                    />
                  </label>
                  {chosen.operation === "dense" && (
                    <label>
                      Units
                      <input
                        type="number"
                        min="1"
                        max="1000000"
                        value={chosen.units || 1}
                        onChange={(event) =>
                          save({
                            ...graph,
                            nodes: graph.nodes.map((node) =>
                              node.id === chosen.id
                                ? { ...node, units: Number(event.target.value) }
                                : node,
                            ),
                          })
                        }
                      />
                    </label>
                  )}
                  {chosen.operation === "input" && (
                    <label>
                      Input dimensions
                      <input
                        value={(chosen.shape || []).join(", ")}
                        onChange={(event) =>
                          save({
                            ...graph,
                            nodes: graph.nodes.map((node) =>
                              node.id === chosen.id
                                ? {
                                    ...node,
                                    shape: event.target.value
                                      .split(",")
                                      .map((value) => Number(value.trim())),
                                  }
                                : node,
                            ),
                          })
                        }
                      />
                    </label>
                  )}
                  {chosen.operation === "concat" && (
                    <label>
                      Axis
                      <input
                        type="number"
                        min="0"
                        value={chosen.axis ?? 0}
                        onChange={(event) =>
                          save({
                            ...graph,
                            nodes: graph.nodes.map((node) =>
                              node.id === chosen.id
                                ? { ...node, axis: Number(event.target.value) }
                                : node,
                            ),
                          })
                        }
                      />
                    </label>
                  )}
                  <button
                    onClick={() => {
                      save(removeArchitectureNode(graph, chosen.id));
                      setSelected(null);
                    }}
                  >
                    Remove operation
                  </button>
                </div>
              )}
            </>
          )}
          <h4>Validation</h4>
          {analysis.issues.length ? (
            <ul className="viz-architecture-issues">
              {analysis.issues.map((issue, index) => (
                <li key={index} className={`is-${issue.severity}`}>
                  <strong>{issue.node || issue.severity}</strong>
                  {issue.message}
                </li>
              ))}
            </ul>
          ) : (
            <p className="viz-valid">Acyclic graph · All shapes resolve</p>
          )}
        </aside>
      </div>
      {!readOnly && graph.edges.length > 0 && (
        <details className="viz-edge-list">
          <summary>Connections ({graph.edges.length})</summary>
          {graph.edges.map((edge, index) => (
            <div key={`${edge.source}-${edge.target}-${index}`}>
              <span>
                {graph.nodes.find((node) => node.id === edge.source)?.name} →{" "}
                {graph.nodes.find((node) => node.id === edge.target)?.name}
              </span>
              <button
                onClick={() =>
                  save({
                    ...graph,
                    edges: graph.edges.filter((_, i) => i !== index),
                  })
                }
              >
                Remove
              </button>
            </div>
          ))}
        </details>
      )}
      <p className="viz-note">
        This editor validates and exports architecture specifications. It does
        not reconfigure a running experiment; use a compatible execution adapter
        to train a custom graph.
      </p>
    </section>
  );
}

export default ArchitectureEditor;
