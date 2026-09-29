#!/usr/bin/env python3
"""
Rolling-context latent profiling (LPP paper item #5).

For each variant prompt, compute next-token entropy at every prefix length in
the paper's schedule {10, 20, ..., 100} via a separate forward pass per length,
plus the max ER/PR of each partial context. This is the paper's actual
definition of the uncertainty floor (min entropy over the rolling schedule),
which differs from run_latent.py's single-pass min-over-positions.

Small models (<=1.5B) run on a laptop CPU in a few minutes; push 7B+ to a free
Colab/Kaggle GPU.

Usage:
    python run_latent_rolling.py --model qwen-0.5b [--data data/problems.jsonl]
"""
import argparse
import json
import os

import config
from src.experiment import load_problems, expand_family


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODELS))
    ap.add_argument("--data", default=config.DATA_FILE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_id = config.MODELS[args.model]
    print(f"Loading {args.model} ({model_id}) ...")
    from src.model_runner import ModelRunner
    from src.latent_rolling import signals_rolling, DEFAULT_PREFIX_LENGTHS
    runner = ModelRunner(model_id)
    print(f"Loaded on device: {runner.device}")

    families = load_problems(args.data)
    records = []
    for fam in families:
        for v in expand_family(fam):
            sig = signals_rolling(
                runner, v["prompt"],
                prefix_lengths=DEFAULT_PREFIX_LENGTHS,
                use_chat_template=config.LATENT_USE_CHAT_TEMPLATE,
            )
            records.append({
                "model": args.model,
                "family_id": fam["id"],
                "kind": v.get("kind", "variant"),
                # keys are stringified in JSON; readers should int() them.
                "entropy_by_prefix": sig["entropy_by_prefix"],
                "max_ER_by_prefix": sig["max_ER_by_prefix"],
                "max_PR_by_prefix": sig["max_PR_by_prefix"],
                "min_entropy_rolling": sig["min_entropy_rolling"],
                "mean_entropy_rolling": sig["mean_entropy_rolling"],
                "prefix_lengths_used": sig["prefix_lengths_used"],
            })

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = args.out or os.path.join(config.RESULTS_DIR, f"latent_rolling_{args.model}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rolling-latent records to {out}")

    if records:
        floor = min(r["min_entropy_rolling"] for r in records)
        print(f"\n=== {args.model} rolling uncertainty floor (LPP-style) ===")
        print(f"  min entropy over rolling schedule: {floor:.4f} nats")


if __name__ == "__main__":
    main()
