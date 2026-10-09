import { afterEach, describe, expect, it, vi } from 'vitest';
import { canvasToBackendGraph, evaluateNativeGraph, validateGraphEvaluation } from './BackendGraphSurgery';
import { sequentialArchitecture } from '../../visualization/architecture';

afterEach(() => { vi.unstubAllGlobals(); });
const payload = { graph:canvasToBackendGraph(sequentialArchitecture([2,4,1]),42),inputs:[[0,0],[1,0]] };
const response = {compiled:true,outputs:[[.25],[.75]],graph:{order:['dense-1','dense-2','output'],dimensions:{input:2,'dense-1':4,'dense-2':1,output:1},parameters:17,input_width:2,output:'output'}};

describe('native graph transport',() => {
  it('uses native IPC for evaluation without browser HTTP or JavaScript authentication headers',async () => {
    const fetcher = vi.fn(() => { throw new Error('Native graph evaluation must not use browser HTTP'); });
    const invoke = vi.fn(async (command:string,args?:Record<string,unknown>) => {
      if (command === 'backend_status') return {phase:'ready',endpoint:'http://127.0.0.1:8421',managed:true,process_id:123,message:'Ready'};
      if (command === 'backend_request') { expect(args).toEqual({method:'POST',path:'/api/v1/extensions/graph/evaluate',body:payload});return response; }
      throw new Error('Unexpected IPC command');
    });
    vi.stubGlobal('__TAURI__',{core:{invoke}});vi.stubGlobal('fetch',fetcher);
    await expect(evaluateNativeGraph(payload)).resolves.toEqual(response);
    expect(invoke.mock.calls.map(call => call[0])).toEqual(['backend_status','backend_request']);
    expect(fetcher).not.toHaveBeenCalled();
  });
  it('does not submit work to an unavailable desktop runtime',async () => {
    const invoke = vi.fn(async () => ({phase:'stopped',endpoint:null,managed:false,process_id:null,message:'Stopped'}));
    vi.stubGlobal('__TAURI__',{core:{invoke}});
    await expect(evaluateNativeGraph(payload)).rejects.toThrow('not ready');
    expect(invoke).toHaveBeenCalledTimes(1);
  });
  it('uses the same finite and sample alignment checks for native responses',async () => {
    const invoke = vi.fn(async (command:string) => command === 'backend_status' ? {phase:'ready',endpoint:'http://127.0.0.1:8421'} : {...response,outputs:[[NaN],[.75]]});
    vi.stubGlobal('__TAURI__',{core:{invoke}});
    await expect(evaluateNativeGraph(payload)).rejects.toThrow('finite aligned outputs');
    expect(() => validateGraphEvaluation(response,3)).toThrow('finite aligned outputs');
    expect(() => validateGraphEvaluation({...response,compiled:false},2)).toThrow('compiled graph');
  });
});
