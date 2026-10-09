import { describe,expect,it } from 'vitest';
import { TransformerModel, TRANSFORMER_VOCABULARY } from '../../engine/transformer';
import { chooseToken,encodeContext,inspectTransformer,logitLens } from './inference';
import { projectMeasuredPoints } from '../../visualization/webgpu/LatentSpaceGalaxy';
import { attentionSplines } from '../../visualization/webgpu/AttentionSplines';
import { packFlowPoints,VolumetricFlowRenderer } from '../../visualization/webgpu/VolumetricFlowRenderer';
import { boundedDispatch } from '../../visualization/webgpu/ComputeShaderRouter';
import { sequentialArchitecture, type ArchitectureGraph } from '../../visualization/architecture';
import { compileBrowserArchitecture,insertDenseBeforeOutput } from '../GraphEditor/graphModel';
import { canvasToBackendGraph } from '../GraphEditor/BackendGraphSurgery';

const config = {demoId:'tiny-transformer',learningRate:.01,maxSteps:100,batchSize:8,seed:42,captureEvery:5};

describe('real token inference instrumentation',() => {
  it('matches the public Transformer output before and after actual optimizer updates',() => {
    const model = new TransformerModel(config);
    const context = encodeContext('the robot');
    for (let training = 0; training < 2; training++) {
      const capture = inspectTransformer(model,context);
      expect(capture.probabilities).toEqual(model.predict(context));
      expect(logitLens(model,capture,3).at(-1)?.probabilities).toEqual(model.predict(context));
      capture.attention.forEach(head => head.forEach((row,query) => {
        expect(row.reduce((sum,value) => sum+value,0)).toBeCloseTo(1,12);
        expect(row.slice(query+1).every(value => value === 0)).toBe(true);
      }));
      model.train(.01,8);
    }
  });
  it('decodes only vocabulary tokens and preserves a four-token causal context',() => {
    expect(encodeContext('the robot')).toEqual([TRANSFORMER_VOCABULARY.length-1,TRANSFORMER_VOCABULARY.length-1,TRANSFORMER_VOCABULARY.indexOf('the'),TRANSFORMER_VOCABULARY.indexOf('robot')]);
    expect(encodeContext('the robot learns a model')).toEqual(['robot','learns','a','model'].map(token => TRANSFORMER_VOCABULARY.indexOf(token)));
    expect(() => encodeContext('unknown-token')).toThrow('Unknown token');
  });
  it('uses stable softmax sampling and exact greedy selection at zero temperature',() => {
    expect(chooseToken([1000,1002,1001],0).token).toBe(1);
    expect(chooseToken([1000,1002,1001],1,0).token).toBe(0);
    expect(chooseToken([1000,1002,1001],1,.999).token).toBe(2);
    expect(chooseToken([1000,1002,1001],1,.4).probabilities.reduce((sum,p) => sum+p,0)).toBeCloseTo(1,14);
    expect(() => chooseToken([0,1],-1)).toThrow();
  });
});

describe('bounded measured visualization',() => {
  it('projects actual rank-one data deterministically and preserves zero variance honestly',() => {
    const vectors = [[1,2],[2,4],[3,6],[4,8]];
    const result = projectMeasuredPoints(vectors);
    expect(result).toEqual(projectMeasuredPoints(vectors));
    expect(result.variance[0]).toBeCloseTo(1,8);
    expect(result.points[0].position[0]).toBeCloseTo(-result.points[3].position[0],8);
    expect(projectMeasuredPoints([[3,3],[3,3]]).variance).toEqual([]);
  });
  it('renders the strongest real attention weights with monotonic widths and bounded edge count',() => {
    const edges = attentionSplines([[1,0,0],[.25,.75,0],[.1,.2,.7]],.15,960,330,3);
    expect(edges.map(edge => edge.weight)).toEqual([1,.75,.7]);
    expect(edges[0].width).toBeGreaterThan(edges[1].width);
    expect(edges.every(edge => edge.key <= edge.query)).toBe(true);
    expect(edges[0].path).toContain(' C ');
    expect(() => attentionSplines([[NaN]])).toThrow();
  });
  it('bounds storage and compute dispatch, retaining exact captured positions and activation values',() => {
    expect(Array.from(packFlowPoints([{position:[1,2,3],value:-.5}]))).toEqual([1,2,3,-.5]);
    expect(boundedDispatch(129)).toEqual({count:129,groups:3});
    expect(boundedDispatch(0).groups).toBe(0);
    expect(() => boundedDispatch(8193)).toThrow();
    expect(() => packFlowPoints([{position:[Infinity,0,0],value:1}])).toThrow();
  });
  it('initializes a usable canvas fallback when no GPU adapter exists',async () => {
    let fills = 0;
    const context = {fillStyle:'',fillRect:() => fills++,createRadialGradient:() => ({addColorStop:() => undefined})};
    const canvas = {width:600,height:400,getContext:(id:string) => id === '2d' ? context : null} as unknown as HTMLCanvasElement;
    const renderer = new VolumetricFlowRenderer(canvas);
    renderer.setPoints([{position:[0,0,0],value:1}]);
    expect((await renderer.initialize()).mode).toBe('canvas');
    renderer.render(); expect(fills).toBe(2);
    renderer.dispose(); renderer.render(); expect(fills).toBe(2);
  });
});

describe('architecture execution capability',() => {
  it('applies only hidden widths while preserving dataset endpoints and output readout',() => {
    const graph = sequentialArchitecture([2,8,8,1]);
    expect(compileBrowserArchitecture(graph,[2,1])).toEqual([8,8]);
    expect(compileBrowserArchitecture(insertDenseBeforeOutput(graph,12),[2,1])).toEqual([8,8,12]);
    expect(() => compileBrowserArchitecture(graph,[3,1])).toThrow('dataset');
  });
  it('rejects valid dimension-compatible skip DAGs as unsupported browser execution',() => {
    const graph: ArchitectureGraph = {format:'neuralis.architecture',version:1,nodes:[{id:'i',name:'Input',operation:'input',shape:[2]},{id:'d',name:'Dense',operation:'dense',units:2},{id:'a',name:'Skip add',operation:'add'},{id:'r',name:'Readout',operation:'dense',units:1},{id:'o',name:'Output',operation:'output'}],edges:[{source:'i',target:'d'},{source:'i',target:'a'},{source:'d',target:'a'},{source:'a',target:'r'},{source:'r',target:'o'}]};
    expect(() => compileBrowserArchitecture(graph,[2,1])).toThrow('sequential');
  });
  it('rejects directed cycles before producing an executable architecture',() => {
    const graph = sequentialArchitecture([2,8,1]);
    graph.edges.push({source:'dense-2',target:'dense-1'});
    expect(() => compileBrowserArchitecture(graph,[2,1])).toThrow('cycle');
  });
  it('preserves explicit residual branches when translating a validated canvas to the backend',() => {
    const graph: ArchitectureGraph = {format:'neuralis.architecture',version:1,nodes:[{id:'source',name:'Input',operation:'input',shape:[2]},{id:'branch',name:'Linear',operation:'dense',units:2},{id:'skip',name:'Residual add',operation:'add'},{id:'out',name:'Output',operation:'output'}],edges:[{source:'source',target:'branch'},{source:'source',target:'skip'},{source:'branch',target:'skip'},{source:'skip',target:'out'}]};
    const translated = canvasToBackendGraph(graph,73);
    expect(translated.input_width).toBe(2);
    expect(translated.seed).toBe(73);
    expect(translated.nodes.find(node => node.id === 'branch')).toEqual({id:'branch',operation:'linear',width:2,inputs:['input']});
    expect(translated.nodes.find(node => node.id === 'skip')?.inputs).toEqual(['input','branch']);
    expect(translated.nodes.at(-1)).toEqual({id:'out',operation:'identity',inputs:['skip']});
  });
  it('does not silently replace an unsupported canvas sigmoid with a backend operation',() => {
    const graph = sequentialArchitecture([2,2,1]);
    graph.nodes[1].operation = 'sigmoid';
    expect(() => canvasToBackendGraph(graph)).toThrow('does not support');
  });
});
