"""Example: inspect GPT-2 locally with Transformers, downloaded only when run.

pip install transformers torch
python examples/hf_local_lens.py
This shows token predictions under a logit projection, NOT internal thoughts.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from axislab_poc import capture_hf_hidden_states, decode_states, export_trace_html


def main():
    model_id = "openai-community/gpt2"  # publicly hosted, small, and inspectable
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    inputs = tok("The result of adding two and three is", return_tensors="pt")
    _output, states = capture_hf_hidden_states(model, inputs["input_ids"],
                                              attention_mask=inputs["attention_mask"])
    # The trained output head is tied to embeddings in GPT-2.
    # IMPORTANT: intermediate states are not already in final-head coordinates.
    lm_weight = model.get_output_embeddings().weight.detach()
    final_norm = model.transformer.ln_f
    reads = decode_states(states, lm_weight, norm=final_norm, top_k=5)
    for layer, item in enumerate(reads):
        top_ids = item.token_ids[0].tolist()
        labels = tok.convert_ids_to_tokens(top_ids)
        print(f"layer {layer:02d}: {labels}")
    export_trace_html("gpt2_readout.html", reads,
                      stage_labels=[f"layer_{i}" for i in range(len(reads))],
                      metadata={"model": model_id, "scope": "local_activations"})
    print("Saved gpt2_readout.html")


if __name__ == '__main__':
    main()
