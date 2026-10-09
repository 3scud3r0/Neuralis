"""AxisLab research toolkit. Standard scientific methods independently implemented."""
from .core import Axis, capture_activations, fit_axis, temporary_steering
from .representations import (AxisBank, evaluate_axis, fit_paired_axis,
                              fit_pca_axis, fit_ridge_probe, pca_subspace, roc_auc)
from .interventions import (activation_patch, multi_axis_steering,
                            projection_ablation, transform_activation)
from .sparse import TopKSAE, SAETrainMetrics, train_topk_sae, sae_features, sae_delta_intervention
from .memory import (rank_one_update, batched_ridge_update, weight_edit,
                     preservation_drift, EditStats)
from .streaming import ActivationMoments, forward_moments
from .adapters import (transformer_lens_capture, nnsight_capture, saelens_encode_decode,
                       repe_registry, upstream_model_edit, transformer_lens_steer, nnsight_steer)

__version__ = "0.3.0"
__all__ = ["Axis", "capture_activations", "fit_axis", "temporary_steering",
           "AxisBank", "evaluate_axis", "fit_paired_axis", "fit_pca_axis",
           "fit_ridge_probe", "pca_subspace", "roc_auc", "activation_patch", "multi_axis_steering",
           "projection_ablation", "transform_activation", "TopKSAE",
           "SAETrainMetrics", "train_topk_sae", "sae_features", "sae_delta_intervention",
           "rank_one_update", "batched_ridge_update", "weight_edit", "preservation_drift",
           "EditStats", "ActivationMoments", "forward_moments",
           "transformer_lens_capture", "nnsight_capture", "saelens_encode_decode",
           "repe_registry", "upstream_model_edit", "transformer_lens_steer", "nnsight_steer"]

# v0.3: local/model-owned observability. These are statistical readouts, not hidden CoT transcripts.
from .lenses import (LensReadout, logit_lens, TunedLens, fit_tuned_lens, decode_states, trace_token_ranks)
from .traces import (LocalActivationTrace, TraceEntry, capture_hf_hidden_states)
from .causal_reasoning import (PatchingResult, causal_patch_effect, sae_feature_patch, component_logit_contributions)
from .interpretation import (FeatureHypothesis, inspect_sae_features, inspect_axis, vocabulary_partition)
from .reporting import export_trace_html, export_trace_json
__all__ += [
    "LensReadout", "logit_lens", "TunedLens", "fit_tuned_lens", "decode_states",
    "trace_token_ranks", "LocalActivationTrace", "TraceEntry", "capture_hf_hidden_states",
    "PatchingResult", "causal_patch_effect", "sae_feature_patch", "component_logit_contributions",
    "FeatureHypothesis", "inspect_sae_features", "inspect_axis", "vocabulary_partition",
    "export_trace_html", "export_trace_json",
]

from .lenses import coda_lens
from .circuits import CircuitNode, CircuitEdge, recurrent_attention_summary, export_circuit_html
__all__ += ["coda_lens", "CircuitNode", "CircuitEdge", "recurrent_attention_summary", "export_circuit_html"]
