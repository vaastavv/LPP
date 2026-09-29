"""
Central configuration for the LLM understanding-vs-pattern-prediction study.

Everything a run needs is here so experiments stay reproducible: which models,
how they are decoded, and where inputs/outputs live. As each model finishes
downloading, add one line to MODELS.
"""
import os

# ---------------------------------------------------------------------------
# Models under test.
# key   = short label used in result files and tables
# value = a Hugging Face repo id OR a local folder path to the weights.
#
# Start with the one you already have. Uncomment the rest as they download.
# Keep the three families (Llama / Qwen / Mistral) so results stay comparable
# to the Latent Performance Profiling paper you are building on.
# ---------------------------------------------------------------------------
MODELS = {
    # --- Recommended ~3B trio (three families, all fp16-safe on a Colab T4) ---
    # Two are fully open; llama-3.2-3b is gated (one `huggingface-cli login`).
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",            # Qwen family, open
    "falcon3-3b": "tiiuae/Falcon3-3B-Instruct",          # Falcon family, open
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",  # Llama family, GATED

    # --- Small models (fast plumbing / the paper's 0.5-1.5B points) ---
    "llama-1b": os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # open, Microsoft/HF

    # Gemma-2-2B is GATED and needs bf16/fp32 (fp16 can NaN on T4); the runner
    # defaults to fp16 on GPU, so use with care.
    "gemma2-2b": "google/gemma-2-2b-it",
    # "qwen-7b":   "Qwen/Qwen2.5-7B-Instruct",     # ~14GB fp16: needs A100, OOMs a T4
    # "mistral-7b":"mistralai/Mistral-7B-Instruct-v0.3",
}

# The default model a script uses when --model is not passed.
DEFAULT_MODEL = "llama-1b"

# ---------------------------------------------------------------------------
# Generation settings.
# Greedy (deterministic) decoding is used so that any variation we observe
# comes from the prompt change, not from sampling randomness.
# ---------------------------------------------------------------------------
GEN = {
    "do_sample": False,      # greedy => temperature is effectively 0
    "max_new_tokens": 256,   # room for short chain-of-thought before the answer
    "repetition_penalty": 1.0,
}

# Whether to wrap latent-profiling text in the chat template.
# The LPP paper profiles raw task-agnostic text, so default False.
LATENT_USE_CHAT_TEMPLATE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT, "data", "problems.jsonl")
RESULTS_DIR = os.path.join(ROOT, "results")

# Fixed seed for any place randomness could creep in (e.g. auto-variant sampling).
SEED = 42
