import { useRef, useState } from "react";
import {
  cameraPreset,
  defaultVisualizationSettings,
  parseViewSettings,
  type CameraBookmark,
  type CameraPreset,
  type VisualizationSettings,
} from "./viewState";
import { downloadBlob } from "./exports";
import type { SceneProps } from "../types";

export interface SceneViewControlsProps {
  settings: VisualizationSettings;
  onChange: (settings: VisualizationSettings) => void;
  onCapture: () => Pick<CameraBookmark, "position" | "target" | "zoom"> | null;
  onNavigate: (
    camera: Pick<CameraBookmark, "position" | "target" | "zoom">,
  ) => void;
  selection: SceneProps["selected"];
  layerCount: number;
  networkId: string;
}

export function SceneViewControls({
  settings,
  onChange,
  onCapture,
  onNavigate,
  selection,
  layerCount,
  networkId,
}: SceneViewControlsProps) {
  const [open, setOpen] = useState(false);
  const [bookmarkName, setBookmarkName] = useState("");
  const [annotation, setAnnotation] = useState("");
  const [error, setError] = useState("");
  const sequence = useRef(0);
  const bookmarks = settings.bookmarks;
  const annotations = settings.annotations.filter((note) =>
    note.id.startsWith(`${networkId}:`),
  );
  const bookmark = () => {
    const camera = onCapture();
    if (!camera) {
      setError("The 3D camera is unavailable.");
      return;
    }
    if (bookmarks.length >= 32) {
      setError(
        "Delete a bookmark before adding another; this view supports 32.",
      );
      return;
    }
    const name = bookmarkName.trim() || `View ${bookmarks.length + 1}`;
    const value = {
      ...camera,
      id: `camera-${Date.now()}-${sequence.current++}`,
      name,
      createdAt: new Date().toISOString(),
    };
    onChange({ ...settings, bookmarks: [...bookmarks, value] });
    setBookmarkName("");
    setError("");
  };
  const annotate = () => {
    if (!selection || !annotation.trim()) return;
    if (settings.annotations.length >= 256) {
      setError("Annotation budget reached. Delete an existing annotation.");
      return;
    }
    const value = {
      id: `${networkId}:${Date.now()}-${sequence.current++}`,
      ...selection,
      text: annotation.trim(),
      createdAt: new Date().toISOString(),
    };
    onChange({ ...settings, annotations: [...settings.annotations, value] });
    setAnnotation("");
    setError("");
  };
  return (
    <div className="viz-view-controls">
      <div className="viz-inspector-toolbar">
        <button onClick={() => setOpen((value) => !value)} aria-expanded={open}>
          View settings & bookmarks
        </button>
        <label>
          Camera{" "}
          <select
            defaultValue="perspective"
            onChange={(event) =>
              onNavigate(
                cameraPreset(event.target.value as CameraPreset, layerCount),
              )
            }
          >
            <option value="perspective">Perspective</option>
            <option value="front">Front</option>
            <option value="top">Top</option>
            <option value="side">Side</option>
          </select>
        </label>
        <button
          onClick={() =>
            downloadBlob(
              new Blob([JSON.stringify(settings, null, 2)], {
                type: "application/json",
              }),
              "neuralis-view.json",
            )
          }
        >
          Export view
        </button>
      </div>
      {open && (
        <div className="viz-view-controls-body">
          <div className="viz-view-filter-grid">
            <label>
              Node budget
              <input
                type="number"
                value={settings.nodeBudget}
                min="32"
                max="10000"
                onChange={(event) =>
                  onChange({
                    ...settings,
                    nodeBudget: Math.min(
                      10000,
                      Math.max(32, Number(event.target.value)),
                    ),
                  })
                }
              />
            </label>
            <label>
              Weight threshold
              <input
                type="number"
                value={settings.weightThreshold}
                min="0"
                step="0.01"
                onChange={(event) =>
                  onChange({
                    ...settings,
                    weightThreshold: Math.max(0, Number(event.target.value)),
                  })
                }
              />
            </label>
            <label>
              Import view
              <input
                type="file"
                accept="application/json,.json"
                onChange={async (event) => {
                  const file = event.target.files?.[0];
                  if (!file) return;
                  try {
                    if (file.size > 500000)
                      throw new Error("View file is too large.");
                    onChange(parseViewSettings(await file.text()));
                    setError("");
                  } catch (caught) {
                    setError(
                      caught instanceof Error
                        ? caught.message
                        : "View import failed.",
                    );
                  }
                  event.target.value = "";
                }}
              />
            </label>
            <button onClick={() => onChange(defaultVisualizationSettings())}>
              Reset filters & saved views
            </button>
          </div>
          <div className="viz-bookmark-editor">
            <input
              aria-label="Bookmark name"
              value={bookmarkName}
              maxLength={120}
              placeholder="Name this camera view"
              onChange={(event) => setBookmarkName(event.target.value)}
            />
            <button onClick={bookmark}>Save camera</button>
          </div>
          <div className="viz-bookmark-list">
            {bookmarks.map((item) => (
              <div key={item.id}>
                <button onClick={() => onNavigate(item)}>{item.name}</button>
                <button
                  onClick={() =>
                    onChange({
                      ...settings,
                      bookmarks: bookmarks.filter(
                        (value) => value.id !== item.id,
                      ),
                    })
                  }
                  aria-label={`Delete bookmark ${item.name}`}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          <div className="viz-annotation-editor">
            <textarea
              aria-label="Selected unit annotation"
              value={annotation}
              onChange={(event) => setAnnotation(event.target.value)}
              maxLength={2000}
              placeholder={
                selection
                  ? `Observation for layer ${selection.layer}, unit ${selection.neuron}`
                  : "Select a unit to attach an observation"
              }
              disabled={!selection}
            />
            <button
              onClick={annotate}
              disabled={!selection || !annotation.trim()}
            >
              Annotate unit
            </button>
          </div>
          <div className="viz-annotation-list">
            {annotations.map((note) => (
              <article key={note.id}>
                <strong>
                  Layer {note.layer} · Unit {note.neuron}
                </strong>
                <p>{note.text}</p>
                <button
                  onClick={() =>
                    onChange({
                      ...settings,
                      annotations: settings.annotations.filter(
                        (value) => value.id !== note.id,
                      ),
                    })
                  }
                  aria-label="Delete annotation"
                >
                  Delete
                </button>
              </article>
            ))}
          </div>
          <p className="viz-note">
            View settings stay in this browser. Exports contain camera and
            observation data, without model weights or credentials.
          </p>
        </div>
      )}
      {error && (
        <p className="viz-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
