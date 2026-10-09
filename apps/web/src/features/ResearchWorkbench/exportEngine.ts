import type { ExportRecipe } from './extensionClient';
function quoteShell(value: string): string { return `'${value.replaceAll("'", "'\\''")}'`; }
export function makeRecipe(modelName: string, ggufPath: string, temperature: number): ExportRecipe {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$/.test(modelName)) throw new Error('Model name must use 1–64 letters, numbers, dots, underscores or hyphens.');
  if (!ggufPath.trim() || ggufPath.length > 1000 || /[\x00-\x1f]/.test(ggufPath) || !ggufPath.toLowerCase().endsWith('.gguf')) throw new Error('Choose a GGUF path without control characters.');
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) throw new Error('Temperature must be between 0 and 2.');
  const modelfile = `FROM ${JSON.stringify(ggufPath)}\nPARAMETER temperature ${temperature}\n`;
  return { modelfile, commands: [`ollama create ${quoteShell(modelName)} -f Modelfile`, `ollama run ${quoteShell(modelName)}`],
    requirements: ['An existing GGUF file whose architecture and tokenizer are supported by your installed Ollama version.', 'Ollama installed and running on your computer. This recipe downloads no weights and executes no commands.', 'Browser toy checkpoints and adapter matrices are not GGUF language models. Convert a supported full model or merge adapters with their exact base before conversion.'] };
}
export function conversionRecipe(modelDirectory: string, ggufPath: string): string[] {
  if (!modelDirectory.trim() || modelDirectory.length > 1000 || /[\x00-\x1f]/.test(modelDirectory)) throw new Error('Choose a local model directory without control characters.');
  makeRecipe('model', ggufPath, 0.7);
  return [`python llama.cpp/convert_hf_to_gguf.py ${quoteShell(modelDirectory)} --outfile ${quoteShell(ggufPath)} --outtype f16`,
    `llama-quantize ${quoteShell(ggufPath)} ${quoteShell(ggufPath.replace(/\.gguf$/i, '-Q4_K_M.gguf'))} Q4_K_M`];
}
