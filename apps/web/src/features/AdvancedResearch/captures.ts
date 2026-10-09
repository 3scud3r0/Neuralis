import type { RunSnapshot } from '../../types';
import { sampledIndices } from '../../visualization/numerics';
import type { FlowPoint } from '../../visualization/webgpu/VolumetricFlowRenderer';

export function finiteMatrix(value: unknown, maximum = 8192): number[][] | null {
  if (!Array.isArray(value) || !value.length || value.length > maximum || !Array.isArray(value[0]) || !value[0].length) return null;
  const width = value[0].length;
  if (width > 8192 || width*value.length > 2_000_000 || value.some(row => !Array.isArray(row) || row.length !== width || row.some(item => typeof item !== 'number' || !Number.isFinite(item)))) return null;
  return value as number[][];
}
export function capturedAttentionHeads(snapshot: RunSnapshot): number[][][] {
  const heads = snapshot.extras?.attentionHeads;
  if (Array.isArray(heads) && heads.length <= 128) {
    const result = heads.map(head => finiteMatrix(head,512));
    if (result.every(head => head && head.every(row => row.every(value => value >= 0)))) return result as number[][][];
  }
  const matrix = finiteMatrix(snapshot.extras?.attention,512);
  return matrix && matrix.every(row => row.every(value => value >= 0)) ? [matrix] : [];
}
export function activationFlow(snapshot: RunSnapshot, perLayer = 256): FlowPoint[] {
  const values = snapshot.layers.flatMap(layer => layer.activations).filter(Number.isFinite);
  const scale = Math.max(1e-12,...values.map(Math.abs));
  return snapshot.layers.flatMap((layer,layerIndex) => {
    const indices = sampledIndices(layer.activations.length,perLayer);
    return indices.map((index,slot) => {
    const count = indices.length;
    const columns = Math.ceil(Math.sqrt(count)); const row = Math.floor(slot/columns),column = slot%columns;
    const rows = Math.ceil(count/columns);
    return {position:[(layerIndex/Math.max(1,snapshot.layers.length-1)-0.5)*3.2,(column-(columns-1)/2)*Math.min(.18,2.2/columns),(row-(rows-1)/2)*Math.min(.18,2.2/rows)] as [number,number,number],value:(layer.activations[index] ?? 0)/scale,label:`${layer.name} · neuron ${index}`};
  }); });
}
