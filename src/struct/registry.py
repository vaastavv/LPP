"""
Model registry for the CodeLLM structural-profiling study.

Static metadata (family, parameter count, language coverage) that cannot be
read from a config is listed here; dynamic fields (layers, hidden size,
context length) are read from the model config at run time by `model_info`.

The CODE_MODELS set is the recommended evaluation panel for the real sweep
(small enough for CPU/modest GPU, broad enough across families). Edit freely.
"""
from __future__ import annotations

# Recommended panel of CodeLLMs for the real run (Hugging Face repo ids).
CODE_MODELS = {
    "qwen25c-0.5b": "Qwen/Qwen2.5-Coder-0.5B",
    "qwen25c-1.5b": "Qwen/Qwen2.5-Coder-1.5B",
    "qwen25c-3b":   "Qwen/Qwen2.5-Coder-3B",
    "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
    "starcoder2-3b": "bigcode/starcoder2-3b",
}

# Static metadata keyed by model id OR short key.
_META = {
    "Qwen/Qwen2.5-Coder-0.5B":  {"family": "Qwen2.5-Coder", "params": "0.5B", "langs": "40+"},
    "Qwen/Qwen2.5-Coder-1.5B":  {"family": "Qwen2.5-Coder", "params": "1.5B", "langs": "40+"},
    "Qwen/Qwen2.5-Coder-3B":    {"family": "Qwen2.5-Coder", "params": "3B",   "langs": "40+"},
    "deepseek-ai/deepseek-coder-1.3b-base": {"family": "DeepSeek-Coder", "params": "1.3B", "langs": "80+"},
    "bigcode/starcoder2-3b":    {"family": "StarCoder2", "params": "3B", "langs": "17"},
    "Salesforce/codegen-350M-mono": {"family": "CodeGen", "params": "350M", "langs": "1"},
    "tiny-random-llama":        {"family": "TinyLlama(random)", "params": "<0.01B", "langs": "n/a"},
}


def static_meta(model_id: str) -> dict:
    return _META.get(model_id, {"family": "?", "params": "?", "langs": "?"})


def model_info(runner) -> dict:
    """Combine static metadata with config-derived fields from a loaded model."""
    cfg = getattr(runner.model, "config", None)
    layers = getattr(cfg, "num_hidden_layers", None)
    hidden = getattr(cfg, "hidden_size", None)
    ctx = getattr(cfg, "max_position_embeddings", None)
    meta = dict(static_meta(runner.model_id))
    meta.update({"layers": layers, "hidden": hidden, "context": ctx,
                 "model_id": runner.model_id})
    return meta
