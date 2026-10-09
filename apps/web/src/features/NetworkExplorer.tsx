import { useMemo, useState } from 'react';
import { Maximize2, Camera, RotateCcw, Layers, MousePointer2, Download, SlidersHorizontal } from 'lucide-react';
import { useStudio } from '../app/StudioContext';
import { DEMOS } from '../engine';
import NetworkScene from '../visualization/NetworkScene';
import { Panel, PageHeader, Select, Toggle, Badge, Button, KeyValue, number, Callout } from '../components/ui';
import { RunControls } from '../components/RunControls';
import { downloadText } from '../app/storage';

export function NetworkExplorer() {
  const { snapshot, settings, exportRun, notify } = useStudio();
  const [mode, setMode] = useState<'activations' | 'weights' | 'gradients'>('activations');
  const [selected, setSelected] = useState<{ layer: number; neuron: number } | null>(null);
  const [motion, setMotion] = useState(!settings.reducedMotion);
  const [sceneKey, setSceneKey] = useState(0);
  const demo = DEMOS.find(item => item.id === snapshot.demoId)!;
  const selectedLayer = selected ? snapshot.layers[selected.layer] : undefined;
  const actualNodes = useMemo(() => snapshot.layers.reduce((sum, layer) => sum + layer.size, 0), [snapshot.layers]);
  const requestFullscreen = () => {
    const container = document.getElementById('network-scene-panel');
    if (container?.requestFullscreen) container.requestFullscreen().catch(() => notify('Fullscreen is unavailable in this browser.'));
  };
  const capture = () => {
    const canvas = document.querySelector('#network-scene-panel canvas') as HTMLCanvasElement | null;
    if (!canvas) { notify('The renderer does not expose a capture canvas. Export the run data instead.'); return; }
    try {
      const anchor = document.createElement('a');
      anchor.download = `neuralis-network-step-${snapshot.step}.png`;
      anchor.href = canvas.toDataURL('image/png');
      anchor.click();
      notify('Network view captured.', 'success');
    } catch { notify('This renderer cannot capture an image. Use your browser screenshot tool.', 'error'); }
  };
  return <>
    <PageHeader eyebrow="NETWORK EXPLORER" title="See inside the network." description="A spatial view of actual model architecture, activations, and connections." actions={<><Button onClick={capture}><Camera size={15} />Capture view</Button><Button onClick={exportRun}><Download size={15} />Export</Button></>} />
    <div className="explorer-layout"><div className="explorer-main"><Panel id="network-scene-panel" className="scene-panel" title={<><BoxIcon />{demo.title}</>} action={<div className="scene-toolbar"><Badge tone="cyan">{actualNodes} neurons</Badge><Button variant="ghost" onClick={requestFullscreen} title="Fullscreen"><Maximize2 size={15} /></Button></div>}><div className="scene-tabs">{(['activations', 'weights', 'gradients'] as const).map(value => <button className={mode === value ? 'active' : ''} key={value} onClick={() => setMode(value)}>{value[0].toUpperCase() + value.slice(1)}</button>)}<div className="scene-tabs-right"><Badge tone="violet">Step {snapshot.step}</Badge></div></div><div className="scene-stage"><NetworkScene key={sceneKey} layers={snapshot.layers} mode={mode} selected={selected} onSelect={(layer, neuron) => setSelected({ layer, neuron })} paused={!motion} height={490} /><div className="scene-help"><MousePointer2 size={12} />Drag to orbit · Scroll to zoom · Click a neuron to inspect</div></div><div className="scene-layer-strip">{snapshot.layers.map((layer, index) => <button key={layer.id} className={selected?.layer === index ? 'selected' : ''} onClick={() => setSelected({ layer: index, neuron: 0 })}><span className="layer-stripe" style={{ background: index === 0 ? 'var(--cyan)' : index === snapshot.layers.length - 1 ? 'var(--green)' : 'var(--violet)' }} /><strong>{layer.name}</strong><small>{layer.size} neurons</small></button>)}</div><div className="panel-bottom"><div className="network-legend"><span><i className="cyan" />Negative</span><span><i className="violet" />Neutral</span><span><i className="green" />Positive</span></div><RunControls compact /></div></Panel><Callout>The scene renders captured values from the active browser engine. Connections are bounded to keep exploration responsive; the tensor inspector retains the full captured matrix.</Callout></div><aside className="explorer-controls"><Panel title="View controls" action={<SlidersHorizontal size={15} />}><div className="control-stack"><FieldHeading text="COLOR BY" /><Select label="Color mode" value={mode} onChange={value => setMode(value as typeof mode)} options={[{ value: 'activations', label: 'Activation magnitude' }, { value: 'weights', label: 'Connection weights' }, { value: 'gradients', label: 'Gradient magnitude' }]} /><Toggle checked={motion} onChange={setMotion} label="Animate the scene" detail="Visual motion only; training is independent" /><Button onClick={() => setSceneKey(value => value + 1)}><RotateCcw size={14} />Reset camera</Button></div></Panel><Panel title="Selected neuron" subtitle="Click a node in the scene">{selectedLayer && selected ? <div className="info-stack"><Badge tone="violet">Layer {selected.layer} / Neuron {selected.neuron}</Badge><KeyValue label="Layer" value={selectedLayer.name} /><KeyValue label="Activation" value={number(selectedLayer.activations[selected.neuron] ?? 0, 6)} mono /><KeyValue label="Gradient" value={number(selectedLayer.gradients[selected.neuron] ?? 0, 6)} mono /><KeyValue label="Bias" value={number(selectedLayer.biases[selected.neuron] ?? 0, 6)} mono /><Button onClick={() => downloadText('neuron.json', JSON.stringify({ layer: selectedLayer.name, neuron: selected.neuron, activation: selectedLayer.activations[selected.neuron], weights: selectedLayer.weights[selected.neuron], bias: selectedLayer.biases[selected.neuron] }, null, 2))}><Download size={13} />Export neuron</Button></div> : <div className="selection-placeholder"><MousePointer2 size={24} /><p>Select a neuron to inspect its measured values.</p></div>}</Panel><Panel title="Architecture"><div className="architecture-stack">{snapshot.layers.map((layer, i) => <div className="architecture-block" key={layer.id}><span>{i.toString().padStart(2, '0')}</span><div><strong>{layer.name}</strong><small>{layer.shape.join(' × ')} shape</small></div><Layers size={14} /></div>)}</div></Panel></aside></div>
  </>;
}
function BoxIcon() { return <Layers size={15} />; }
function FieldHeading({ text }: { text: string }) { return <span className="section-label">{text}</span>; }
