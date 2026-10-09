import { analyzeArchitecture, graphAdjacency, type ArchitectureGraph } from '../../visualization/architecture';

export const FIXED_ARCHITECTURE_DEMOS = new Set(['pca','k-means','gridworld','bandit','tiny-cnn','tiny-transformer','tiny-gru','super-mario']);

/** Only the exact sequential dense representation accepted by hiddenLayers is compiled. */
export function compileBrowserArchitecture(graph: ArchitectureGraph, endpoints: [number,number]): number[] {
  const analysis = analyzeArchitecture(graph);
  const invalid = analysis.issues.find(issue => issue.severity === 'error');
  if (invalid) throw new Error(invalid.message);
  const { incoming,outgoing } = graphAdjacency(graph);
  const ordered = analysis.order.map(id => graph.nodes.find(node => node.id === id)!);
  if (ordered.length < 3 || ordered[0].operation !== 'input' || ordered.at(-1)?.operation !== 'output') throw new Error('Browser execution requires one input, dense layers, and one output in a single chain.');
  if (graph.edges.length !== ordered.length-1 || ordered.some((node,index) => incoming.get(node.id)!.length !== (index ? 1 : 0) || outgoing.get(node.id)!.length !== (index === ordered.length-1 ? 0 : 1))) throw new Error('Branching DAGs are validated designs. The browser engine currently executes sequential dense chains only.');
  if (ordered.slice(1,-1).some(node => node.operation !== 'dense' || node.bias === false)) throw new Error('Only biased dense operations can be compiled; hidden activation and output objective stay defined by the selected demo.');
  if (ordered[0].shape?.length !== 1 || ordered[0].shape[0] !== endpoints[0] || ordered.at(-2)?.units !== endpoints[1]) throw new Error('Input and output widths must match the current dataset.');
  const hidden = ordered.slice(1,-2).map(node => node.units!);
  if (hidden.length > 8 || hidden.some(size => size < 1 || size > 128)) throw new Error('Browser hidden layers allow at most eight widths in [1, 128].');
  return hidden;
}

export function insertDenseBeforeOutput(graph: ArchitectureGraph, width = 16): ArchitectureGraph {
  const output = graph.nodes.find(node => node.operation === 'output');
  const edge = output && graph.edges.find(edge => edge.target === output.id);
  if (!output || !edge) throw new Error('Connect an output before inserting a hidden layer.');
  const finalDense = graph.nodes.find(node => node.id === edge.source);
  const preceding = finalDense && graph.edges.find(edge => edge.target === finalDense.id);
  if (!finalDense || finalDense.operation !== 'dense' || !preceding) throw new Error('Insert hidden layers before the final dataset readout.');
  let suffix = graph.nodes.length; while (graph.nodes.some(node => node.id === `dense-new-${suffix}`)) suffix++;
  const id = `dense-new-${suffix}`;
  return { ...graph, nodes: [...graph.nodes,{id,name:'Hidden dense',operation:'dense',units:width}], edges: [...graph.edges.filter(edge => edge !== preceding),{source:preceding.source,target:id},{source:id,target:finalDense.id}] };
}
