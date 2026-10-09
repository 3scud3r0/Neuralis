import { TransformerModel, TRANSFORMER_VOCABULARY } from '../../engine/transformer';
import { causalAttention } from '../../engine/layers';
import { softmax } from '../../engine/math';

export interface TransformerInferenceCapture {
  context: number[];
  probabilities: number[];
  logits: number[];
  attention: number[][][];
  stages: { name: string; vectors: number[][] }[];
}

export function encodeContext(text: string): number[] {
  const tokens = text.trim().split(/\s+/).filter(Boolean);
  const ids = tokens.map(token => {
    const index = TRANSFORMER_VOCABULARY.indexOf(token.toLowerCase());
    if (index < 0) throw new Error(`Unknown token “${token}”. Choose words from the displayed tiny vocabulary.`);
    return index;
  });
  const context = ids.slice(-4);
  while (context.length < 4) context.unshift(TRANSFORMER_VOCABULARY.length-1);
  return context;
}

/** Exact eval forward pass using the same public trained parameters and numerical operators as TransformerModel. */
export function inspectTransformer(model: TransformerModel, context: number[]): TransformerInferenceCapture {
  // Public predict owns the model's validation contract before we inspect its parameterized operations.
  model.predict(context);
  const embedded = context.map((token,position) => Array.from({length:8},(_,axis) => model.tokenEmbedding.values[token*8+axis]+model.positionEmbedding.values[position*8+axis]));
  const normalized = embedded.map(vector => model.norm1.forward(vector).output);
  const attention = causalAttention(normalized.map(vector => model.query.forward(vector)),normalized.map(vector => model.key.forward(vector)),normalized.map(vector => model.value.forward(vector)),2);
  const attended = embedded.map((vector,token) => { const contribution = model.attentionOutput.forward(attention.context[token]); return vector.map((value,axis) => value+contribution[axis]); });
  const hidden = attended.map(vector => model.ff1.forward(model.norm2.forward(vector).output).map(Math.tanh));
  const residual = attended.map((vector,token) => { const contribution = model.ff2.forward(hidden[token]); return vector.map((value,axis) => value+contribution[axis]); });
  const logits = model.readout.forward(model.finalNorm.forward(residual[3]).output);
  return { context:[...context],probabilities:softmax(logits),logits,attention:attention.probabilities.map(head => head.map(row => [...row])),stages:[{name:'Embedding',vectors:embedded},{name:'Attention residual',vectors:attended},{name:'Feed-forward residual',vectors:residual}] };
}

export function logitLens(model: TransformerModel, capture: TransformerInferenceCapture, contextToken: number): { stage: string; logits: number[]; probabilities: number[] }[] {
  if (!Number.isInteger(contextToken) || contextToken < 0 || contextToken > 3) throw new Error('Logit lens context token must be an integer in [0, 3].');
  return capture.stages.map(stage => { const logits = model.readout.forward(model.finalNorm.forward(stage.vectors[contextToken]).output); return {stage:stage.name,logits,probabilities:softmax(logits)}; });
}

export function chooseToken(logits: number[], temperature: number, randomValue = 0): { token: number; probabilities: number[] } {
  if (!logits.length || logits.some(value => !Number.isFinite(value))) throw new Error('Sampling needs finite vocabulary logits.');
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2 || !Number.isFinite(randomValue) || randomValue < 0 || randomValue >= 1) throw new Error('Temperature must be in [0, 2]; sample draw in [0, 1).');
  const probabilities = temperature === 0 ? softmax(logits) : softmax(logits.map(value => value/temperature));
  if (!temperature) return {token:logits.indexOf(Math.max(...logits)),probabilities};
  let sum = 0; for (let token = 0; token < probabilities.length; token++) { sum += probabilities[token]; if (randomValue < sum) return {token,probabilities}; }
  return {token:probabilities.length-1,probabilities};
}
