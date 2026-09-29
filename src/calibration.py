"""
Calibration text loaders for intrinsic (latent) metrics — LPP paper items #6
and #10.

Paper: Materials & Methods, "Datasets and prompts for intrinsic metrics"
(page 12): "Intrinsic metrics are computed using 100 randomly-sampled
task-agnostic texts from the Alpaca dataset. [...] All experiments set the
random seed to 42." Supplement §2.1, Figure 3 (page 21) additionally reports
Dolly and WikiText for the dataset-sensitivity check.

Why this matters: the study's own five problem families are arithmetic/QA
prompts, which push the model toward a confident numeric or single-word answer
and so artificially deflate the entropy floor. The paper's latent profile is
measured on GENERIC task-agnostic text — that is what these loaders provide.

Each loader samples deterministically (Random(seed).sample over the full split)
and caches the sampled strings to data/calibration/<name>_<n>_seed<seed>.jsonl.
Later calls read the cache directly, so once the sample is built the pipeline is
fully offline and reproducible.
"""
from __future__ import annotations

import json
import os
import random

import config

CACHE_DIR = os.path.join(config.ROOT, "data", "calibration")

# HuggingFace dataset ids for each supported calibration source.
DATASET_IDS = {
    "alpaca": ("tatsu-lab/alpaca", None),
    "dolly": ("databricks/databricks-dolly-15k", None),
    "wikitext": ("wikitext", "wikitext-103-raw-v1"),
}


def _cache_path(name: str, n: int, seed: int) -> str:
    return os.path.join(CACHE_DIR, f"{name}_{n}_seed{seed}.jsonl")


def _read_cache(path: str) -> list[str] | None:
    if not os.path.exists(path):
        return None
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts or None


def _write_cache(path: str, texts: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"prompt_id": i, "text": t}, ensure_ascii=False) + "\n")


def _row_to_text(name: str, row: dict) -> str:
    """Turn one dataset row into a single task-agnostic text string."""
    if name == "alpaca":
        # instruction + optional input (Alpaca schema).
        inp = row.get("input") or ""
        return row["instruction"] + (("\n" + inp) if inp else "")
    if name == "dolly":
        # instruction + optional context (databricks-dolly-15k schema).
        ctx = row.get("context") or ""
        return row["instruction"] + (("\n" + ctx) if ctx else "")
    if name == "wikitext":
        return row["text"]
    raise ValueError(f"unknown dataset {name!r}")


def load_dataset_sample(name: str, n: int = 100, seed: int = 42) -> list[str]:
    """
    Return `n` task-agnostic text strings from the named dataset, cached to
    data/calibration/<name>_<n>_seed<seed>.jsonl.

    name in {"alpaca", "dolly", "wikitext"}. Paper §"Datasets and prompts for
    intrinsic metrics" (page 12) and Supplement §2.1 (page 21).
    """
    if name not in DATASET_IDS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(DATASET_IDS)}")

    cache = _cache_path(name, n, seed)
    cached = _read_cache(cache)
    if cached is not None:
        return cached

    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - only hit without the dep
        raise ImportError(
            "The 'datasets' package is required to build the calibration cache. "
            "Install it (pip install datasets) or provide the cache file "
            f"{cache} directly."
        ) from e

    hf_id, hf_config = DATASET_IDS[name]
    ds = load_dataset(hf_id, hf_config) if hf_config else load_dataset(hf_id)
    split = ds["train"]

    rng = random.Random(seed)
    if name == "wikitext":
        # WikiText has many blank / heading-only lines; keep substantive ones.
        candidates = [i for i, t in enumerate(split["text"])
                      if len(t.strip()) >= 64 and not t.strip().startswith("=")]
    else:
        candidates = list(range(len(split)))

    idxs = rng.sample(candidates, min(n, len(candidates)))
    texts = [_row_to_text(name, split[i]) for i in idxs]

    _write_cache(cache, texts)
    return texts


def load_alpaca(n: int = 100, seed: int = 42) -> list[str]:
    """Return `n` Alpaca (instruction + input) strings. Paper §Datasets, p.12."""
    return load_dataset_sample("alpaca", n=n, seed=seed)
