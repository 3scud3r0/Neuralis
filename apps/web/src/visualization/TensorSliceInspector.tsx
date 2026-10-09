import { useMemo, useState } from "react";
import { TensorHeatmap, StatisticsTable, Histogram } from "./TensorInspector";
import {
  denseTensor,
  largestTensorElements,
  parseTensor,
  reduceTensor,
  serializeTensor,
  sliceTensor,
  tensorChannelStatistics,
  validateTensor,
  type DenseTensor,
  type TensorReduction,
} from "./tensors";
import { downloadBlob, exportTensorCSV } from "./exports";
import "./visualization.css";

export interface TensorSliceInspectorProps {
  tensor: DenseTensor;
  axisNames?: string[];
  title?: string;
}

export function TensorSliceInspector({
  tensor,
  axisNames = [],
  title = "Multidimensional tensor slices",
}: TensorSliceInspectorProps) {
  const [imported, setImported] = useState<DenseTensor | null>(null);
  const [rowAxis, setRowAxis] = useState(0);
  const [columnAxis, setColumnAxis] = useState(1);
  const [fixed, setFixed] = useState<number[]>([]);
  const [channelAxis, setChannelAxis] = useState(0);
  const [reduction, setReduction] = useState<TensorReduction>("mean");
  const [showReduction, setShowReduction] = useState(false);
  const [error, setError] = useState("");
  const source = imported || tensor;
  const activeRow = Math.min(rowAxis, Math.max(0, source.shape.length - 1));
  const activeColumn =
    columnAxis === activeRow || columnAxis >= source.shape.length
      ? (activeRow + 1) % Math.max(1, source.shape.length)
      : columnAxis;
  const activeChannel = Math.min(
    channelAxis,
    Math.max(0, source.shape.length - 1),
  );
  const result = useMemo(() => {
    try {
      validateTensor(source);
      const fixedIndices = source.shape.map((size, axis) =>
        Math.min(Math.max(0, fixed[axis] || 0), Math.max(0, size - 1)),
      );
      const matrix =
        source.shape.length < 2
          ? [Array.from(source.data)]
          : sliceTensor(source, activeRow, activeColumn, fixedIndices).matrix;
      const reduced = source.shape.length
        ? reduceTensor(source, activeChannel, reduction)
        : source;
      const reductionMatrix =
        reduced.shape.length < 2
          ? [Array.from(reduced.data)]
          : sliceTensor(reduced, 0, 1).matrix;
      const channels = source.shape.length
        ? tensorChannelStatistics(source, activeChannel)
        : [];
      const top = largestTensorElements(source, 10);
      return { matrix, reductionMatrix, reduced, channels, top, error: "" };
    } catch (caught) {
      return {
        matrix: [],
        reductionMatrix: [],
        reduced: denseTensor([], [0]),
        channels: [],
        top: [],
        error: caught instanceof Error ? caught.message : "Invalid tensor.",
      };
    }
  }, [source, activeRow, activeColumn, fixed, activeChannel, reduction]);
  const axisLabel = (axis: number) => axisNames[axis] || `Axis ${axis}`;
  const updateIndex = (axis: number, value: number) =>
    setFixed((previous) => {
      const next = [...previous];
      next[axis] = value;
      return next;
    });
  const save = () => {
    try {
      downloadBlob(
        new Blob([serializeTensor(source)], { type: "application/json" }),
        "neuralis-tensor.json",
      );
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Export failed.");
    }
  };
  return (
    <section className="viz-tensor-inspector" aria-label={title}>
      <div className="viz-inspector-toolbar">
        <h3>{title}</h3>
        <label>
          Load capture{" "}
          <input
            type="file"
            accept="application/json,.json"
            onChange={async (event) => {
              const file = event.target.files?.[0];
              if (!file) return;
              try {
                if (file.size > 64000000)
                  throw new Error("Capture file is too large.");
                setImported(parseTensor(await file.text()));
                setFixed([]);
                setError("");
              } catch (caught) {
                setError(
                  caught instanceof Error ? caught.message : "Import failed.",
                );
              }
              event.target.value = "";
            }}
          />
        </label>
        <button onClick={save}>Export capture</button>
        {imported && (
          <button onClick={() => setImported(null)}>
            Return to live capture
          </button>
        )}
      </div>
      <div className="viz-inspector-toolbar">
        <span className="viz-axis-description">
          {imported ? "Imported" : "Captured"} tensor · [
          {source.shape.join(" × ")}]
        </span>
        {source.shape.length > 1 && (
          <>
            <label>
              Rows{" "}
              <select
                value={activeRow}
                onChange={(event) => setRowAxis(Number(event.target.value))}
              >
                {source.shape.map((size, axis) => (
                  <option key={axis} value={axis}>
                    {axisLabel(axis)} · {size}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Columns{" "}
              <select
                value={activeColumn}
                onChange={(event) => setColumnAxis(Number(event.target.value))}
              >
                {source.shape.map((size, axis) => (
                  <option key={axis} value={axis} disabled={axis === activeRow}>
                    {axisLabel(axis)} · {size}
                  </option>
                ))}
              </select>
            </label>
          </>
        )}
        {source.shape.map(
          (size, axis) =>
            axis !== activeRow &&
            axis !== activeColumn && (
              <label key={axis}>
                {axisLabel(axis)}{" "}
                <input
                  type="number"
                  min="0"
                  max={Math.max(0, size - 1)}
                  value={Math.min(fixed[axis] || 0, Math.max(0, size - 1))}
                  onChange={(event) =>
                    updateIndex(axis, Number(event.target.value))
                  }
                />
              </label>
            ),
        )}
        <button
          onClick={() => exportTensorCSV(result.matrix, "tensor-slice.csv")}
          disabled={!result.matrix.length}
        >
          Slice CSV
        </button>
      </div>
      {(error || result.error) && (
        <p className="viz-error" role="alert">
          {error || result.error}
        </p>
      )}
      <div className="viz-inspector-grid">
        <TensorHeatmap
          matrix={result.matrix}
          title={`${axisLabel(activeRow)} × ${axisLabel(activeColumn)} slice`}
        />
        <StatisticsTable
          values={Array.from(source.data)}
          title="Full tensor statistics"
        />
      </div>
      <div className="viz-inspector-toolbar">
        <label>
          Channel axis{" "}
          <select
            value={activeChannel}
            onChange={(event) => setChannelAxis(Number(event.target.value))}
          >
            {source.shape.map((size, axis) => (
              <option key={axis} value={axis}>
                {axisLabel(axis)} · {size}
              </option>
            ))}
          </select>
        </label>
        <label>
          Reduction{" "}
          <select
            value={reduction}
            onChange={(event) =>
              setReduction(event.target.value as TensorReduction)
            }
          >
            {(["mean", "sum", "min", "max", "l2"] as const).map((operation) => (
              <option key={operation}>{operation}</option>
            ))}
          </select>
        </label>
        <button
          onClick={() => setShowReduction((value) => !value)}
          aria-pressed={showReduction}
        >
          Reduced tensor
        </button>
      </div>
      {showReduction && (
        <div className="viz-slice-reduction">
          <TensorHeatmap
            matrix={result.reductionMatrix}
            title={`${reduction} over ${axisLabel(activeChannel)} · shape [${result.reduced.shape.join(", ")}]`}
          />
        </div>
      )}
      <div className="viz-tensor-details">
        <div>
          <h4>Channel statistics</h4>
          <div className="viz-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Channel</th>
                  <th>Mean</th>
                  <th>Std. deviation</th>
                  <th>L2 norm</th>
                  <th>Sparsity</th>
                </tr>
              </thead>
              <tbody>
                {result.channels.slice(0, 128).map((stats, index) => (
                  <tr key={index}>
                    <td>{index}</td>
                    <td>{stats.mean.toPrecision(4)}</td>
                    <td>{stats.standardDeviation.toPrecision(4)}</td>
                    <td>{stats.l2.toPrecision(4)}</td>
                    <td>{(stats.sparsity * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {result.channels.length > 128 && (
            <p className="viz-note">Showing the first 128 channels.</p>
          )}
        </div>
        <div>
          <h4>Largest absolute elements</h4>
          <div className="viz-table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Coordinate</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {result.top.map((element) => (
                  <tr key={element.offset}>
                    <td>[{element.index.join(", ")}]</td>
                    <td>{element.value.toPrecision(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <Histogram
        values={Array.from(source.data)}
        title="Full tensor distribution"
      />
      <p className="viz-note">
        Slices use exact row-major source coordinates. Reductions aggregate
        every element on the selected axis; displayed reduced axes are the first
        two remaining axes.
      </p>
    </section>
  );
}

export default TensorSliceInspector;
