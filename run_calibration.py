#!/usr/bin/env python3
"""
Calibration-set latent profiling (LPP paper items #6, #8, #10).

Runs the intrinsic metrics on task-agnostic calibration text (Alpaca by
default; Dolly / WikiText for the dataset-sensitivity check) rather than on the
study's own problem prompts. Paper: Materials & Methods "Datasets and prompts
for intrinsic metrics" (page 12): "maximum context length of 200 tokens, with a
designated prefix length of 100"; "random seed 42".

For each prompt it computes:
  - the single-pass profile (ModelRunner.signals): min/mean entropy, max ER/PR,
    per-layer ER/PR;
  - the rolling-context schedule (signals_rolling): entropy / ER / PR per prefix
    length (used by the prefix-sensitivity plot, item #7).

Output: results/calibration_<dataset>_<model>[_ctx<L>].jsonl. The _ctx<L>
suffix is added only when --context-length is passed explicitly, so a plain run
writes the canonical results/calibration_<dataset>_<model>.jsonl while the
context sweep (item #8) writes one file per length.

Usage:
    python run_calibration.py --model qwen-0.5b --n 10          # smoke test
    python run_calibration.py --model qwen-0.5b                 # 100 Alpaca prompts
    python run_calibration.py --model qwen-0.5b --dataset dolly
    for L in 50 100 200 500; do \
        python run_calibration.py --model qwen-0.5b --context-length $L; done
"""
import argparse
import json
import os

import config

DEFAULT_CONTEXT_LENGTH = 200   # paper: max context length of 200 tokens (page 12)


def _truncate(runner, text: str, max_tokens: int) -> str:
    """Truncate `text` to the first `max_tokens` tokens, then decode back."""
    ids = runner.tokenizer(text)["input_ids"][:max_tokens]
    return runner.tokenizer.decode(ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODELS))
    ap.add_argument("--dataset", default="alpaca", choices=["alpaca", "dolly", "wikitext"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=config.SEED)
    # default None so we can tell "run with paper default" from "sweep value".
    ap.add_argument("--context-length", type=int, default=None,
                    help="truncate each prompt to this many tokens (default 200; "
                         "passing it explicitly adds a _ctx<L> suffix to the output)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ctx = args.context_length if args.context_length is not None else DEFAULT_CONTEXT_LENGTH

    print(f"Loading calibration set: {args.dataset} (n={args.n}, seed={args.seed}) ...")
    from src.calibration import load_dataset_sample
    texts = load_dataset_sample(args.dataset, n=args.n, seed=args.seed)
    print(f"  {len(texts)} prompts")

    model_id = config.MODELS[args.model]
    print(f"Loading {args.model} ({model_id}) ...")
    from src.model_runner import ModelRunner
    from src.latent_rolling import signals_rolling, DEFAULT_PREFIX_LENGTHS
    runner = ModelRunner(model_id)
    print(f"Loaded on device: {runner.device}")

    records = []
    for i, raw_text in enumerate(texts):
        text = _truncate(runner, raw_text, ctx)
        sig = runner.signals(text, use_chat_template=config.LATENT_USE_CHAT_TEMPLATE)
        roll = signals_rolling(
            runner, text,
            prefix_lengths=DEFAULT_PREFIX_LENGTHS,
            use_chat_template=config.LATENT_USE_CHAT_TEMPLATE,
        )
        records.append({
            "model": args.model,
            "dataset": args.dataset,
            "prompt_id": i,
            "context_length": ctx,
            "min_entropy": sig["min_entropy"],
            "mean_entropy": sig["mean_entropy"],
            "max_ER": sig["max_ER"],
            "max_PR": sig["max_PR"],
            "last_layer_ER": sig["last_layer_ER"],
            "last_layer_PR": sig["last_layer_PR"],
            "ER_by_layer": sig["ER_by_layer"],
            "PR_by_layer": sig["PR_by_layer"],
            "num_tokens": sig["num_tokens"],
            "num_layers": sig["num_layers"],
            # rolling schedule (item #7 reads these)
            "entropy_by_prefix": roll["entropy_by_prefix"],
            "max_ER_by_prefix": roll["max_ER_by_prefix"],
            "max_PR_by_prefix": roll["max_PR_by_prefix"],
            "min_entropy_rolling": roll["min_entropy_rolling"],
            "mean_entropy_rolling": roll["mean_entropy_rolling"],
            "prefix_lengths_used": roll["prefix_lengths_used"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(texts)} done")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    suffix = f"_ctx{ctx}" if args.context_length is not None else ""
    out = args.out or os.path.join(
        config.RESULTS_DIR, f"calibration_{args.dataset}_{args.model}{suffix}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} calibration records to {out}")

    if records:
        floor = min(r["min_entropy"] for r in records)
        roll_floor = min(r["min_entropy_rolling"] for r in records)
        print(f"\n=== {args.model} on {args.dataset} (ctx={ctx}) ===")
        print(f"  uncertainty floor (single-pass min entropy): {floor:.4f} nats")
        print(f"  uncertainty floor (rolling schedule)       : {roll_floor:.4f} nats")


if __name__ == "__main__":
    main()
