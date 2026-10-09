import {
  Component,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { Canvas, useThree, type ThreeEvent } from "@react-three/fiber";
import { Grid, Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { LayerData, SceneProps } from "../types";
import {
  divergingColor,
  finite,
  layerValues,
  neuronPosition,
  sampledIndices,
  selectEdges,
  symmetricDomain,
  type EdgeRecord,
} from "./numerics";
import { exportCanvasImage } from "./exports";
import { SceneViewControls } from "./SceneViewControls";
import {
  loadViewSettings,
  saveViewSettings,
  type CameraBookmark,
} from "./viewState";
import "./visualization.css";

interface BoundaryProps {
  children: ReactNode;
  fallback: ReactNode;
}
class SceneBoundary extends Component<BoundaryProps, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

function supportsWebGL(): boolean {
  if (typeof document === "undefined") return false;
  try {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!context) return false;
    context.getExtension("WEBGL_lose_context")?.loseContext();
    return true;
  } catch {
    return false;
  }
}

interface NeuronInstancesProps {
  layer: LayerData;
  layerIndex: number;
  layerCount: number;
  indices: number[];
  values: number[];
  domain: number;
  selected: SceneProps["selected"];
  onSelect: SceneProps["onSelect"];
}

function NeuronInstances({
  layer,
  layerIndex,
  layerCount,
  indices,
  values,
  domain,
  selected,
  onSelect,
}: NeuronInstancesProps) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);
  const [hovered, setHovered] = useState<number | null>(null);
  useEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    for (let i = 0; i < indices.length; i++) {
      const neuron = indices[i];
      const value = finite(values[neuron]);
      const chosen =
        selected?.layer === layerIndex && selected.neuron === neuron;
      const radius = chosen
        ? 0.21
        : 0.11 + Math.min(0.065, (Math.abs(value) / domain) * 0.045);
      dummy.position.set(
        ...neuronPosition(layerIndex, neuron, layer.size, layerCount),
      );
      dummy.scale.setScalar(radius);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      const rgb = chosen ? [1, 0.95, 0.7] : divergingColor(value, domain);
      color.setRGB(rgb[0], rgb[1], rgb[2]);
      mesh.setColorAt(i, color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [
    indices,
    values,
    domain,
    selected,
    layerIndex,
    layerCount,
    layer.size,
    dummy,
    color,
  ]);
  const pick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    if (event.instanceId !== undefined)
      onSelect?.(layerIndex, indices[event.instanceId]);
  };
  return (
    <group>
      <instancedMesh
        ref={ref}
        args={[undefined, undefined, indices.length]}
        onClick={pick}
        onPointerOver={(event) => {
          event.stopPropagation();
          if (event.instanceId !== undefined)
            setHovered(indices[event.instanceId]);
        }}
        onPointerOut={() => setHovered(null)}
      >
        <sphereGeometry args={[1, 14, 10]} />
        <meshStandardMaterial
          roughness={0.3}
          metalness={0.35}
          emissive="#0a1e37"
          emissiveIntensity={0.65}
        />
      </instancedMesh>
      {hovered !== null && (
        <Html
          position={neuronPosition(layerIndex, hovered, layer.size, layerCount)}
          distanceFactor={13}
          center
          style={{ pointerEvents: "none" }}
        >
          <div className="viz-node-tooltip">
            <strong>{layer.name}</strong>
            <span>
              Neuron {hovered} · {finite(values[hovered]).toPrecision(4)}
            </span>
          </div>
        </Html>
      )}
    </group>
  );
}

function Connections({
  layers,
  edges,
  opacity,
}: {
  layers: LayerData[];
  edges: EdgeRecord[];
  opacity: number;
}) {
  const geometry = useMemo(() => {
    const positions = new Float32Array(edges.length * 6);
    const colors = new Float32Array(edges.length * 6);
    const domain = symmetricDomain(edges.map((edge) => edge.weight));
    edges.forEach((edge, index) => {
      const source = neuronPosition(
        edge.sourceLayer,
        edge.source,
        layers[edge.sourceLayer].size,
        layers.length,
      );
      const target = neuronPosition(
        edge.targetLayer,
        edge.target,
        layers[edge.targetLayer].size,
        layers.length,
      );
      positions.set(source, index * 6);
      positions.set(target, index * 6 + 3);
      const color = divergingColor(edge.weight, domain);
      colors.set(color, index * 6);
      colors.set(color, index * 6 + 3);
    });
    const buffer = new THREE.BufferGeometry();
    buffer.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    buffer.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    buffer.computeBoundingSphere();
    return buffer;
  }, [edges, layers]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={opacity}
        depthWrite={false}
      />
    </lineSegments>
  );
}

function LayerPlanes({ layers }: { layers: LayerData[] }) {
  return (
    <group>
      {layers.map((layer, index) => {
        const x = neuronPosition(index, 0, layer.size, layers.length)[0];
        return (
          <group key={`${layer.id}-${index}`} position={[x, 0, 0]}>
            <mesh rotation={[0, Math.PI / 2, 0]}>
              <planeGeometry args={[6.2, 6.2]} />
              <meshBasicMaterial
                color={
                  index === 0
                    ? "#22d3ee"
                    : index === layers.length - 1
                      ? "#f472b6"
                      : "#8b5cf6"
                }
                transparent
                opacity={0.028}
                side={THREE.DoubleSide}
                depthWrite={false}
              />
            </mesh>
            <Html
              position={[0, -4, 0]}
              center
              distanceFactor={18}
              style={{ pointerEvents: "none" }}
            >
              <div className="viz-layer-label">
                <span>
                  {index === 0
                    ? "INPUT"
                    : index === layers.length - 1
                      ? "OUTPUT"
                      : `LAYER ${index}`}
                </span>
                <strong>{layer.name}</strong>
                <small>{layer.size.toLocaleString()} units</small>
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}

type CameraLocation = Pick<CameraBookmark, "position" | "target" | "zoom">;
function SceneNavigation({
  command,
  onReady,
  maxDistance,
}: {
  command: CameraLocation | null;
  onReady: (capture: () => CameraLocation) => void;
  maxDistance: number;
}) {
  const { camera, invalidate } = useThree();
  const controls = useRef<OrbitControlsImpl>(null);
  useEffect(() => {
    onReady(() => ({
      position: camera.position.toArray() as [number, number, number],
      target: (controls.current?.target.toArray() as [
        number,
        number,
        number,
      ]) || [0, 0, 0],
      zoom: camera.zoom,
    }));
  }, [camera, onReady]);
  useEffect(() => {
    if (!command) return;
    camera.position.set(...command.position);
    camera.zoom = command.zoom;
    camera.updateProjectionMatrix();
    if (controls.current) {
      controls.current.target.set(...command.target);
      controls.current.update();
    }
    camera.lookAt(...command.target);
    invalidate();
  }, [command, camera, invalidate]);
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      minDistance={3}
      maxDistance={maxDistance}
    />
  );
}

function NetworkFallback({ layers, mode, onSelect }: SceneProps) {
  return (
    <div
      className="viz-fallback"
      role="region"
      aria-label="Neural network data"
    >
      <p>
        WebGL is unavailable. Explore the captured network through the
        accessible data view.
      </p>
      <div className="viz-fallback-layers">
        {layers.map((layer, layerIndex) => (
          <div key={`${layer.id}-${layerIndex}`}>
            <h4>{layer.name}</h4>
            <small>{layer.size} units</small>
            <div>
              {sampledIndices(layer.size, 24).map((neuron) => (
                <button
                  key={neuron}
                  onClick={() => onSelect?.(layerIndex, neuron)}
                  title={`Select ${layer.name}, neuron ${neuron}`}
                >
                  <span>{neuron}</span>
                  <strong>
                    {finite(
                      layerValues(layer, mode || "activations")[neuron],
                    ).toPrecision(3)}
                  </strong>
                </button>
              ))}
            </div>
            {layer.size > 24 && <small>Showing 24 evenly sampled units.</small>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function NetworkScene({
  layers,
  mode = "activations",
  selected,
  onSelect,
  paused,
  height = 480,
  className = "",
}: SceneProps) {
  const root = useRef<HTMLDivElement>(null);
  const [webgl, setWebgl] = useState<boolean | null>(null);
  const [reset, setReset] = useState(0);
  const [settings, setSettings] = useState(() =>
    loadViewSettings("neuralis.visualization.view"),
  );
  const { showEdges, edgeBudget, edgeOpacity: opacity } = settings;
  const [cameraCommand, setCameraCommand] = useState<CameraLocation | null>(
    null,
  );
  const captureCamera = useRef<(() => CameraLocation) | null>(null);
  const registerCapture = useCallback((capture: () => CameraLocation) => {
    captureCamera.current = capture;
  }, []);
  const [internalSelected, setInternalSelected] =
    useState<SceneProps["selected"]>(null);
  const activeSelected = selected !== undefined ? selected : internalSelected;
  const selectUnit = (layer: number, neuron: number) => {
    setInternalSelected({ layer, neuron });
    onSelect?.(layer, neuron);
  };
  const [exportError, setExportError] = useState("");
  useEffect(() => {
    setWebgl(supportsWebGL());
  }, []);
  useEffect(() => {
    saveViewSettings("neuralis.visualization.view", settings);
  }, [settings]);
  const networkId = useMemo(() => {
    const identity = layers
      .map((layer) => `${layer.id}:${layer.size}`)
      .join("|");
    let hash = 2166136261;
    for (let i = 0; i < identity.length; i++)
      hash = Math.imul(hash ^ identity.charCodeAt(i), 16777619);
    return (hash >>> 0).toString(16);
  }, [layers]);
  const indices = useMemo(
    () =>
      layers.map((layer) =>
        sampledIndices(
          layer.size,
          Math.min(
            512,
            Math.max(
              1,
              Math.floor(settings.nodeBudget / Math.max(layers.length, 1)),
            ),
          ),
        ),
      ),
    [layers, settings.nodeBudget],
  );
  const values = useMemo(
    () => layers.map((layer) => layerValues(layer, mode)),
    [layers, mode],
  );
  const domain = useMemo(() => symmetricDomain(values.flat()), [values]);
  const visible = useMemo(
    () => new Map(indices.map((neurons, index) => [index, new Set(neurons)])),
    [indices],
  );
  const edges = useMemo(
    () =>
      showEdges
        ? selectEdges(layers, edgeBudget, settings.weightThreshold, visible)
        : [],
    [layers, showEdges, edgeBudget, visible, settings.weightThreshold],
  );
  const unitCount = layers.reduce((sum, layer) => sum + layer.size, 0);
  const renderedCount = indices.reduce((sum, index) => sum + index.length, 0);
  const download = async () => {
    const canvas = root.current?.querySelector("canvas");
    if (!canvas) return;
    try {
      await exportCanvasImage(canvas, "neuralis-network.png");
      setExportError("");
    } catch (error) {
      setExportError(
        error instanceof Error ? error.message : "Image export failed.",
      );
    }
  };
  const fallback = (
    <NetworkFallback layers={layers} mode={mode} onSelect={selectUnit} />
  );
  if (!layers.length)
    return (
      <div className={`viz-network viz-empty ${className}`} style={{ height }}>
        <div className="viz-empty-symbol">◇</div>
        <h3>No captured network yet</h3>
        <p>
          Start an experiment to inspect real activations, weights and
          gradients.
        </p>
      </div>
    );
  return (
    <div
      ref={root}
      className={`viz-network ${className}`}
      style={{ minHeight: height }}
    >
      <div className="viz-scene-toolbar">
        <div>
          <span className={`viz-status-dot ${paused ? "is-paused" : ""}`} />
          <strong>Network geometry</strong>
          <small>
            {renderedCount.toLocaleString()} / {unitCount.toLocaleString()}{" "}
            units
          </small>
        </div>
        <div>
          <button
            onClick={() =>
              setSettings((value) => ({
                ...value,
                showEdges: !value.showEdges,
              }))
            }
            aria-pressed={showEdges}
          >
            Connections
          </button>
          <button
            onClick={() => {
              setCameraCommand(null);
              setReset((value) => value + 1);
            }}
          >
            Reset view
          </button>
          <button onClick={download} disabled={!webgl}>
            Export PNG
          </button>
        </div>
      </div>
      <div className="viz-scene-canvas" style={{ height }}>
        {webgl === null ? (
          <div className="viz-empty">Preparing renderer…</div>
        ) : !webgl ? (
          fallback
        ) : (
          <SceneBoundary fallback={fallback}>
            <Canvas
              key={reset}
              camera={{
                position: [
                  Math.max(8, layers.length * 2.6),
                  5,
                  Math.max(13, layers.length * 3),
                ],
                fov: 42,
                near: 0.1,
                far: Math.max(200, layers.length * 20),
              }}
              gl={{
                antialias: true,
                alpha: true,
                preserveDrawingBuffer: true,
                powerPreference: "high-performance",
              }}
              dpr={[1, 1.75]}
              onPointerMissed={() => {
                /* Selection intentionally persists while navigating. */
              }}
            >
              <color attach="background" args={["#07101f"]} />
              <ambientLight intensity={1.1} />
              <directionalLight position={[5, 10, 12]} intensity={2.2} />
              <pointLight
                position={[-12, 2, 5]}
                color="#22d3ee"
                intensity={15}
                distance={30}
              />
              <pointLight
                position={[12, 2, 3]}
                color="#a855f7"
                intensity={18}
                distance={30}
              />
              <Grid
                position={[0, -3.7, 0]}
                args={[Math.max(40, layers.length * 6), 35]}
                cellSize={1}
                cellThickness={0.35}
                cellColor="#153345"
                sectionSize={5}
                sectionThickness={0.6}
                sectionColor="#1c5776"
                fadeDistance={60}
                infiniteGrid
              />
              <LayerPlanes layers={layers} />
              <Connections layers={layers} edges={edges} opacity={opacity} />
              {layers.map((layer, index) => (
                <NeuronInstances
                  key={`${layer.id}-${index}`}
                  layer={layer}
                  layerIndex={index}
                  layerCount={layers.length}
                  indices={indices[index]}
                  values={values[index]}
                  domain={domain}
                  selected={activeSelected}
                  onSelect={selectUnit}
                />
              ))}
              <SceneNavigation
                command={cameraCommand}
                onReady={registerCapture}
                maxDistance={Math.max(80, layers.length * 15)}
              />
            </Canvas>
          </SceneBoundary>
        )}
        <div className="viz-scene-corner">
          <span>● {mode.toUpperCase()}</span>
          <small>
            {mode === "weights"
              ? "Node color: mean incoming weight"
              : mode === "gradients"
                ? "Node color: captured unit gradient"
                : "Node color: captured activation"}
          </small>
          <small>
            Drag to orbit · Scroll to zoom · Click a unit to inspect
          </small>
        </div>
        <div className="viz-color-legend">
          <span>−{domain.toPrecision(2)}</span>
          <div />
          <span>+{domain.toPrecision(2)}</span>
        </div>
      </div>
      <div className="viz-scene-footer">
        <label>
          Top |weight| edges{" "}
          <input
            aria-label="Connection budget"
            type="range"
            min="0"
            max="4000"
            step="100"
            value={edgeBudget}
            onChange={(event) =>
              setSettings((value) => ({
                ...value,
                edgeBudget: Number(event.target.value),
              }))
            }
          />
          <output>{edges.length.toLocaleString()}</output>
        </label>
        <label>
          Opacity{" "}
          <input
            aria-label="Connection opacity"
            type="range"
            min="0.03"
            max="0.7"
            step="0.01"
            value={opacity}
            onChange={(event) =>
              setSettings((value) => ({
                ...value,
                edgeOpacity: Number(event.target.value),
              }))
            }
          />
        </label>
        <span>Bounded geometry · Actual captured tensors</span>
      </div>
      <SceneViewControls
        settings={settings}
        onChange={setSettings}
        onCapture={() => captureCamera.current?.() || null}
        onNavigate={setCameraCommand}
        selection={activeSelected}
        layerCount={layers.length}
        networkId={networkId}
      />
      {exportError && (
        <p className="viz-error" role="alert">
          {exportError}
        </p>
      )}
    </div>
  );
}
