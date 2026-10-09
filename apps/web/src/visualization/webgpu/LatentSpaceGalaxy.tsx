import { useMemo, useState } from 'react';
import { projectPCA } from '../numerics';
import { LatentSpaceExplorer } from '../components/LatentSpaceExplorer';
import type { FlowPoint } from './VolumetricFlowRenderer';

export interface LatentSpaceGalaxyProps { vectors: number[][]; labels?: string[]; source: string; reducedMotion?: boolean }
export function projectMeasuredPoints(vectors: number[][], labels: string[] = []): { points: FlowPoint[]; variance: number[] } {
  const bounded = vectors.slice(0,8192);
  const projection = projectPCA(bounded);
  const scale = Math.max(1e-12,...projection.points.flat().map(Math.abs));
  const norms = bounded.map(vector => Math.sqrt(vector.reduce((total,value) => total+value*value,0)));
  const maxNorm = Math.max(1e-12,...norms);
  return { points: projection.points.map((point,index) => ({ position: point.map(value => value/scale*1.6) as [number,number,number], value: norms[index]/maxNorm, label: labels[index] ?? `Vector ${index+1}` })), variance: projection.explainedVariance };
}
export function LatentSpaceGalaxy({ vectors, labels = [], source, reducedMotion }: LatentSpaceGalaxyProps) {
  const [selected,setSelected] = useState(0);
  const result = useMemo(() => projectMeasuredPoints(vectors,labels),[vectors,labels]);
  return <><LatentSpaceExplorer points={result.points} title={`PCA · ${source}`} reducedMotion={reducedMotion} /><div className="advanced-projection-summary"><span>PC1 {(100*(result.variance[0] ?? 0)).toFixed(1)}%</span><span>PC2 {(100*(result.variance[1] ?? 0)).toFixed(1)}%</span><span>PC3 {(100*(result.variance[2] ?? 0)).toFixed(1)}%</span><span>Centered deterministic power-iteration PCA</span></div>{vectors.length > 0 && <div className="advanced-vector-inspector"><label>Inspect vector<select value={Math.min(selected,vectors.length-1)} onChange={event => setSelected(Number(event.target.value))}>{vectors.slice(0,8192).map((_,index) => <option key={index} value={index}>{labels[index] ?? `Vector ${index+1}`}</option>)}</select></label><div className="advanced-vector-values">{vectors[Math.min(selected,vectors.length-1)]?.map((value,axis) => <span key={axis} title={`Dimension ${axis}`}><small>d{axis}</small>{value.toFixed(4)}</span>)}</div></div>}<p className="advanced-note">Points use actual {source}. Brightness encodes vector norm. Distances are projected and can lose information; this view does not claim UMAP or semantic clusters.</p></>;
}
