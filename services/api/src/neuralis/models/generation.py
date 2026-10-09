from pydantic import BaseModel, ConfigDict, Field
import torch

from neuralis.datasets.synthetic import TOKEN_TO_ID, VOCABULARY
from neuralis.runtime.devices import module_device


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(default="the ", min_length=1, max_length=512)
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0, le=5)
    top_k: int = Field(default=0, ge=0, le=128)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int = Field(default=42, ge=0, le=2147483647)


def filter_distribution(logits, top_k=0, top_p=1):
    filtered = logits.clone()
    if top_k:
        cutoff = filtered.topk(min(top_k, len(filtered))).values[-1]
        filtered[filtered < cutoff] = -float("inf")
    if top_p < 1:
        sorted_logits, indices = filtered.sort(descending=True)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[indices[remove]] = -float("inf")
    return filtered.softmax(-1)


def generate(model, request: GenerationRequest):
    if not hasattr(model, "embedding") or not hasattr(model, "position"):
        raise ValueError("Text generation requires the registered tiny causal transformer")
    unknown = sorted(set(request.prompt).difference(TOKEN_TO_ID))
    if unknown:
        raise ValueError(f"Prompt characters outside the local demonstrator vocabulary: {unknown}")
    tokens = [TOKEN_TO_ID[character] for character in request.prompt]
    generator = torch.Generator().manual_seed(request.seed)
    generated, telemetry = [], []
    context = model.position.num_embeddings
    with torch.no_grad():
        for step in range(request.max_new_tokens):
            prefix = torch.tensor([tokens[-context:]], dtype=torch.long, device=module_device(model))
            logits = model(prefix)[0, -1].float().cpu()
            if request.temperature == 0:
                distribution = logits.softmax(-1)
                index = int(logits.argmax())
            else:
                distribution = filter_distribution(logits / request.temperature, request.top_k, request.top_p)
                index = int(torch.multinomial(distribution, 1, generator=generator))
            tokens.append(index)
            generated.append(VOCABULARY[index])
            entropy = float(-(distribution * distribution.clamp_min(1e-12).log()).sum())
            telemetry.append({"step": step, "token_id": index, "character": VOCABULARY[index],
                              "sample_probability": float(distribution[index]), "entropy": entropy})
    continuation = "".join(generated)
    return {"prompt": request.prompt, "continuation": continuation, "text": request.prompt + continuation,
            "tokens": tokens, "telemetry": telemetry, "settings": request.model_dump(),
            "interpretation": "Character-level generation from the actual model snapshot, not an external language model."}
