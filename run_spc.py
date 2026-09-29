#!/usr/bin/env python3
"""
Symbolic Pattern Completion (SPC) evaluation — LPP paper item #11.

Generates n symbolic-sequence items, builds a 10-shot prompt for each, decodes
greedily (max_new_tokens=8), and scores character-level F1 of the predicted
final symbols against gold. Paper: Results §"LPP-informed tasks" (page 6),
Supplement §1.1.2, Table 2, Materials & Methods §"Evaluating LLMs" (page 13).

Usage:
    python run_spc.py --model qwen-0.5b --n 20    # smoke test
    python run_spc.py --model qwen-0.5b           # full 100-item eval
"""
import argparse
import json
import os

import config

N_SHOTS = 10          # paper: 10 in-context examples
MAX_NEW_TOKENS = 8    # paper: max_new_tokens=8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODELS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.spc_task import (
        generate_spc, prompt_template, char_f1, parse_prediction, grade_spc,
    )

    items = generate_spc(args.n, seed=args.seed)
    # A fixed 10-shot context (seed-reproducible, disjoint from the test seed)
    # shared across all test items, per paper's 10 in-context examples.
    shot_items = generate_spc(N_SHOTS, seed=args.seed + 1)
    shots = [(s["prompt_prefix"], s["gold"]) for s in shot_items]

    model_id = config.MODELS[args.model]
    print(f"Loading {args.model} ({model_id}) ...")
    from src.model_runner import ModelRunner
    runner = ModelRunner(model_id)
    print(f"Loaded on device: {runner.device}")

    gen = {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "repetition_penalty": 1.0}
    records = []
    for i, item in enumerate(items):
        prompt = prompt_template(item["prompt_prefix"], shots)
        raw = runner.chat(prompt, gen)
        pred = parse_prediction(raw)
        f1 = char_f1(pred, item["gold"])
        records.append({
            "model": args.model,
            "rule": item["rule"],
            "prompt": prompt,
            "prompt_prefix": item["prompt_prefix"],
            "gold": item["gold"],
            "raw_output": raw,
            "pred": pred,
            "char_f1": f1,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)} done")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = args.out or os.path.join(config.RESULTS_DIR, f"spc_{args.model}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} SPC records to {out}")

    summary = grade_spc(records)
    print(f"\n=== {args.model} SPC ===")
    print(f"  n            : {summary['n']}")
    print(f"  mean char-F1 : {summary['mean_char_f1']:.4f}")
    # per-rule breakdown is cheap and informative
    by_rule: dict[str, list[float]] = {}
    for r in records:
        by_rule.setdefault(r["rule"], []).append(r["char_f1"])
    for rule in sorted(by_rule):
        vals = by_rule[rule]
        print(f"    {rule:<18} {sum(vals) / len(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
