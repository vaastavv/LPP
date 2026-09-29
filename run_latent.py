#!/usr/bin/env python3
"""
Latent (intrinsic) profiling track (Track 2).

Computes the uncertainty floor (min next-token entropy) and the ER/PR profile
of the hidden states for each variant prompt. Requires a model whose weights
load in transformers (open-weight). Small models (<=3B) run on a laptop CPU;
push 7B+ to a free Colab/Kaggle GPU.

Usage:
    python run_latent.py --model llama-1b
"""
import argparse
import os
import config
from src.experiment import load_problems, run_latent, save_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODELS))
    ap.add_argument("--data", default=config.DATA_FILE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_id = config.MODELS[args.model]
    print(f"Loading {args.model} ({model_id}) ...")
    from src.model_runner import ModelRunner
    runner = ModelRunner(model_id)
    print(f"Loaded on device: {runner.device}")

    families = load_problems(args.data)
    records = run_latent(runner, families, config.LATENT_USE_CHAT_TEMPLATE, args.model)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = args.out or os.path.join(config.RESULTS_DIR, f"latent_{args.model}.jsonl")
    save_jsonl(records, out)
    print(f"Wrote {len(records)} latent records to {out}")

    # compact per-model summary using the LPP extremal statistics
    if records:
        min_ent = min(r["min_entropy"] for r in records)
        max_er = max(r["max_ER"] for r in records)
        max_pr = max(r["max_PR"] for r in records)
        print(f"\n=== {args.model} latent profile (extremal, LPP-style) ===")
        print(f"  uncertainty floor (min entropy): {min_ent:.4f} nats")
        print(f"  max effective rank (ER)        : {max_er:.2f}")
        print(f"  max participation ratio (PR)   : {max_pr:.2f}")


if __name__ == "__main__":
    main()
