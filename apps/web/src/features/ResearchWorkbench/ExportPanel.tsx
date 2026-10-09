import { useMemo, useState } from 'react';
import { ArrowUpRight, FileCode2, PackageCheck, Terminal } from 'lucide-react';
import { conversionRecipe, makeRecipe } from './exportEngine';
import type { ExportRecipe } from './extensionClient';
import { BackendButton, DownloadButton, ErrorNotice, PanelTitle, ResultJson, errorMessage, type ExtensionContext } from './shared';

export function ExportPanel({ client, capabilities }: ExtensionContext) {
  const [modelName, setModelName] = useState('neuralis-local');
  const [path, setPath] = useState('./model-f16.gguf');
  const [directory, setDirectory] = useState('./huggingface-model');
  const [temperature, setTemperature] = useState(0.7);
  const [remote, setRemote] = useState<ExportRecipe | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const local = useMemo(() => { try { return { recipe: makeRecipe(modelName, path, temperature), conversion: conversionRecipe(directory, path), error: null }; } catch (failure) { return { recipe: null, conversion: [], error: errorMessage(failure) }; } }, [modelName, path, temperature, directory]);
  const request = async () => {
    if (!client) return;
    setBusy(true); setError(null);
    try { setRemote(await client.recipe({ model_name: modelName, gguf_path: path })); }
    catch (failure) { setError(errorMessage(failure)); } finally { setBusy(false); }
  };
  return <div className="rw-panel">
    <PanelTitle eyebrow="06 / PORTABLE ARTIFACTS" title="Move a real model into your local stack.">Prepare a reviewed GGUF conversion recipe and an Ollama Modelfile. Every command is a recipe for your machine; nothing is executed by this page.</PanelTitle>
    <div className="rw-banner"><PackageCheck size={18}/><span>Numerical browser checkpoints and low-rank matrix demos are not complete GGUF language models. Use supported model weights, their matching tokenizer and a compatible converter.</span></div>
    <ErrorNotice error={error ?? local.error}/>
    <div className="rw-two-col"><div className="rw-card"><div className="rw-card-heading"><h3>Artifact recipe</h3><span className="rw-chip">No download or execution</span></div><label>Ollama model name<input value={modelName} onChange={event => { setModelName(event.target.value); setRemote(null); }}/></label><label>Existing or output GGUF path<input value={path} onChange={event => { setPath(event.target.value); setRemote(null); }}/></label><label>Hugging Face model directory<input value={directory} onChange={event => setDirectory(event.target.value)}/></label><label>Temperature<input type="number" min={0} max={2} step={0.1} value={temperature} onChange={event => { setTemperature(Number(event.target.value)); setRemote(null); }}/></label><p className="rw-note">Select a local directory containing the full model and tokenizer. An adapter directory requires a compatible merge step with its exact base model first.</p><BackendButton available={Boolean(client && !local.error)} busy={busy} onClick={request}><ArrowUpRight size={15}/>Get backend recipe</BackendButton><p className="rw-note">GGUF converter capability: {capabilities?.gguf.available ? 'Available on the backend; this request still prepares a recipe only.' : capabilities?.gguf.reason ?? 'Not connected. Downloadable recipes below work independently of backend availability.'}</p></div>
      <div className="rw-card"><div className="rw-card-heading"><h3><FileCode2 size={18}/>Modelfile</h3><span className="rw-chip">Generated from your fields</span></div><pre className="rw-code">{local.recipe?.modelfile ?? 'Correct the recipe fields to preview a Modelfile.'}</pre>{local.recipe && <DownloadButton filename="Modelfile" data={local.recipe.modelfile} label="Download Modelfile" mime="text/plain"/>}<h3><Terminal size={18}/>Import and run with Ollama</h3><pre className="rw-code">{local.recipe?.commands.join('\n') ?? ''}</pre><p>The FROM path is resolved from the Modelfile location. Move the Modelfile beside the GGUF file or use an absolute path supported by your local setup.</p></div></div>
    <div className="rw-card"><h3>Conversion and quantization workflow</h3><div className="rw-export-steps"><div><span>01</span><h4>Verify the source</h4><p>Confirm the model license, full weight files, tokenizer and supported architecture. Preserve the model revision and hashes with your experiment.</p></div><div><span>02</span><h4>Convert to full precision GGUF</h4><p>Use your checked-out llama.cpp converter and its documented Python requirements. F16 conversion can consume substantial disk and RAM.</p></div><div><span>03</span><h4>Quantize and evaluate</h4><p>Run the llama.cpp quantizer locally, then compare output quality and memory against the full precision model before selecting a quantization format.</p></div></div><pre className="rw-code">{local.conversion.join('\n')}</pre><div className="rw-actions">{local.recipe && <DownloadButton filename="neuralis-export-recipe.json" data={{ schema: 'neuralis.export-recipe/v1', modelName, modelDirectory: directory, ggufPath: path, ...local.recipe, conversionCommands: local.conversion }} label="Download full recipe"/>}{local.conversion.length > 0 && <DownloadButton filename="neuralis-conversion-commands.txt" data={local.conversion.join('\n') + '\n'} label="Download command reference" mime="text/plain"/>}</div>{local.recipe && <ul className="rw-requirements">{local.recipe.requirements.map(requirement => <li key={requirement}>{requirement}</li>)}</ul>}</div>
    {remote && <div className="rw-card"><h3>Backend recipe evidence</h3><ResultJson value={remote} label="Backend requirements, commands and Modelfile"/><DownloadButton filename="neuralis-backend-recipe.json" data={remote}/></div>}
  </div>;
}
