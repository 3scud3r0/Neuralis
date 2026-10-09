import { clamp, type Vector3Tuple } from "./numerics";

export interface CameraBookmark {
  id: string;
  name: string;
  position: Vector3Tuple;
  target: Vector3Tuple;
  zoom: number;
  createdAt: string;
}
export interface NeuronAnnotation {
  id: string;
  layer: number;
  neuron: number;
  text: string;
  createdAt: string;
}
export interface VisualizationSettings {
  format: "neuralis.view";
  version: 1;
  showEdges: boolean;
  edgeBudget: number;
  edgeOpacity: number;
  nodeBudget: number;
  weightThreshold: number;
  bookmarks: CameraBookmark[];
  annotations: NeuronAnnotation[];
}
export type CameraPreset = "perspective" | "front" | "top" | "side";

export function defaultVisualizationSettings(): VisualizationSettings {
  return {
    format: "neuralis.view",
    version: 1,
    showEdges: true,
    edgeBudget: 1200,
    edgeOpacity: 0.16,
    nodeBudget: 1600,
    weightThreshold: 0,
    bookmarks: [],
    annotations: [],
  };
}
function finiteTuple(value: unknown): value is Vector3Tuple {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every(
      (component) =>
        typeof component === "number" &&
        Number.isFinite(component) &&
        Math.abs(component) <= 1000000,
    )
  );
}

export function parseViewSettings(text: string): VisualizationSettings {
  if (text.length > 500000)
    throw new Error("View settings exceed the import budget.");
  const parsed: unknown = JSON.parse(text);
  if (!parsed || typeof parsed !== "object")
    throw new Error("Expected a view settings document.");
  const value = parsed as Record<string, unknown>;
  if (value.format !== "neuralis.view" || value.version !== 1)
    throw new Error("Expected neuralis.view version 1.");
  const defaults = defaultVisualizationSettings();
  const number = (
    key: keyof VisualizationSettings,
    min: number,
    max: number,
  ): number => {
    const candidate = value[key];
    if (candidate === undefined) return defaults[key] as number;
    if (typeof candidate !== "number" || !Number.isFinite(candidate))
      throw new Error(`View setting ${key} must be finite.`);
    return clamp(candidate, min, max);
  };
  if (value.showEdges !== undefined && typeof value.showEdges !== "boolean")
    throw new Error("showEdges must be a boolean.");
  const bookmarks = value.bookmarks ?? [];
  const annotations = value.annotations ?? [];
  if (
    !Array.isArray(bookmarks) ||
    bookmarks.length > 32 ||
    !Array.isArray(annotations) ||
    annotations.length > 256
  )
    throw new Error("View supports at most 32 bookmarks and 256 annotations.");
  const safeBookmarks = bookmarks.map((bookmark: unknown) => {
    if (!bookmark || typeof bookmark !== "object")
      throw new Error("Invalid camera bookmark.");
    const item = bookmark as Record<string, unknown>;
    if (
      typeof item.id !== "string" ||
      item.id.length > 120 ||
      typeof item.name !== "string" ||
      item.name.length > 120 ||
      !finiteTuple(item.position) ||
      !finiteTuple(item.target) ||
      typeof item.zoom !== "number" ||
      !Number.isFinite(item.zoom) ||
      item.zoom <= 0 ||
      typeof item.createdAt !== "string"
    )
      throw new Error("Invalid camera bookmark fields.");
    return {
      id: item.id,
      name: item.name,
      position: [...item.position],
      target: [...item.target],
      zoom: clamp(item.zoom, 0.01, 100),
      createdAt: item.createdAt,
    } as CameraBookmark;
  });
  const safeAnnotations = annotations.map((annotation: unknown) => {
    if (!annotation || typeof annotation !== "object")
      throw new Error("Invalid unit annotation.");
    const item = annotation as Record<string, unknown>;
    if (
      typeof item.id !== "string" ||
      item.id.length > 120 ||
      !Number.isInteger(item.layer) ||
      !Number.isInteger(item.neuron) ||
      Number(item.layer) < 0 ||
      Number(item.neuron) < 0 ||
      typeof item.text !== "string" ||
      item.text.length > 2000 ||
      typeof item.createdAt !== "string"
    )
      throw new Error("Invalid unit annotation fields.");
    return {
      id: item.id,
      layer: Number(item.layer),
      neuron: Number(item.neuron),
      text: item.text,
      createdAt: item.createdAt,
    };
  });
  return {
    format: "neuralis.view",
    version: 1,
    showEdges: (value.showEdges as boolean) ?? defaults.showEdges,
    edgeBudget: Math.floor(number("edgeBudget", 0, 20000)),
    edgeOpacity: number("edgeOpacity", 0.01, 0.9),
    nodeBudget: Math.floor(number("nodeBudget", 32, 10000)),
    weightThreshold: number("weightThreshold", 0, 1000000),
    bookmarks: safeBookmarks,
    annotations: safeAnnotations,
  };
}

export function cameraPreset(
  preset: CameraPreset,
  layers: number,
): Pick<CameraBookmark, "position" | "target" | "zoom"> {
  const distance = Math.max(13, layers * 3);
  const target: Vector3Tuple = [0, 0, 0];
  if (preset === "front")
    return { position: [0, 0.1, distance], target, zoom: 1 };
  if (preset === "top")
    return { position: [0, distance, 0.1], target, zoom: 1 };
  if (preset === "side")
    return { position: [distance, 1, 0.1], target, zoom: 1 };
  return {
    position: [Math.max(8, layers * 2.6), 5, distance],
    target,
    zoom: 1,
  };
}

export function loadViewSettings(storageKey: string): VisualizationSettings {
  try {
    const text = localStorage.getItem(storageKey);
    return text ? parseViewSettings(text) : defaultVisualizationSettings();
  } catch {
    return defaultVisualizationSettings();
  }
}

export function saveViewSettings(
  storageKey: string,
  settings: VisualizationSettings,
): boolean {
  try {
    localStorage.setItem(storageKey, JSON.stringify(settings));
    return true;
  } catch {
    return false;
  }
}
