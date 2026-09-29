#!/usr/bin/env python3
"""
Ambiguous Reasoning (AR) evaluation — LPP paper item #12.

Generates n AR items, builds a 10-shot prompt for each, decodes greedily
(max_new_tokens=16), parses the status/answer response, and scores
ambiguity-flag accuracy, answer-choice accuracy, and their mean. Paper: Results
§"LPP-informed tasks" (page 6), Supplement §1.1.1, Table 2.

Usage:
    python run_ar.py --model qwen-0.5b --n 20    # smoke test
    python run_ar.py --model qwen-0.5b           # full 100-item eval
"""
import argparse
import json
import os

import config

N_SHOTS = 10           # paper: 10 in-context examples
MAX_NEW_TOKENS = 16    # paper: max_new_tokens=16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODELS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.ar_task import generate_ar, prompt_template, parse_ar_response, grade_ar

    items = generate_ar(args.n, seed=args.seed)
    # Fixed 10-shot context, seed-reproducible and disjoint from the test seed.
    shots = generate_ar(N_SHOTS, seed=args.seed + 1)

    model_id = config.MODELS[args.model]
    print(f"Loading {args.model} ({model_id}) ...")
    from src.model_runner import ModelRunner
    runner = ModelRunner(model_id)
    print(f"Loaded on device: {runner.device}")

    gen = {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "repetition_penalty": 1.0}
    records = []
    for i, item in enumerate(items):
        prompt = prompt_template(item, shots)
        raw = runner.chat(prompt, gen)
        parsed = parse_ar_response(raw)
        status_correct = int(parsed["status"] == item["gold_status"])
        answer_correct = int(parsed["answer"] == item["gold_answer"])
        records.append({
            "model": args.model,
            "source": item["source"],
            "prefix": item["prefix"],
            "prompt": prompt,
            "gold_status": item["gold_status"],
            "gold_answer": item["gold_answer"],
            "raw_output": raw,
            "pred_status": parsed["status"],
            "pred_answer": parsed["answer"],
            "status_correct": status_correct,
            "answer_correct": answer_correct,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)} done")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = args.out or os.path.join(config.RESULTS_DIR, f"ar_{args.model}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} AR records to {out}")

    summary = grade_ar(records)
    print(f"\n=== {args.model} AR ===")
    print(f"  n               : {summary['n']}")
    print(f"  status accuracy : {summary['status_accuracy']:.4f}")
    print(f"  answer accuracy : {summary['answer_accuracy']:.4f}")
    print(f"  mean accuracy   : {summary['mean_accuracy']:.4f}")


if __name__ == "__main__":
    main()
