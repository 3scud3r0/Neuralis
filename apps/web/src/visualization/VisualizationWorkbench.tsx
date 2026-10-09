import { useState } from "react";
import type { LayerData, SceneProps } from "../types";
import NetworkScene from "./NetworkScene";
import { TensorInspector } from "./TensorInspector";
import { CircuitView, NeuronInspector } from "./ResearchViews";
import "./visualization.css";

export function VisualizationWorkbench({ layers }: { layers: LayerData[] }) {
  const [tab, setTab] = useState<"network" | "tensors" | "circuits">("network");
  const [mode, setMode] =
    useState<NonNullable<SceneProps["mode"]>>("activations");
  const [selected, setSelected] = useState<SceneProps["selected"]>(null);
  const choose = (layer: number, neuron: number) =>
    setSelected({ layer, neuron });
  return (
    <div className="viz-workbench">
      <div className="viz-inspector-toolbar">
        <div className="viz-tab-list">
          {(["network", "tensors", "circuits"] as const).map((item) => (
            <button
              key={item}
              aria-pressed={tab === item}
              onClick={() => setTab(item)}
            >
              {item}
            </button>
          ))}
        </div>
        {tab === "network" && (
          <label>
            Color by{" "}
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
        )}
      </div>
      {tab === "network" ? (
        <NetworkScene
          layers={layers}
          selected={selected}
          onSelect={choose}
          mode={mode}
        />
      ) : tab === "tensors" ? (
        <TensorInspector layers={layers} onSelect={choose} />
      ) : (
        <CircuitView layers={layers} onSelect={choose} />
      )}
      <NeuronInspector layers={layers} selected={selected} />
    </div>
  );
}

export default VisualizationWorkbench;
