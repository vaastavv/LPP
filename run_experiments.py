#!/usr/bin/env python3
"""
Accuracy / robustness track (Track 1).

Usage:
    python run_experiments.py                      # default model from config
    python run_experiments.py --model llama-1b
    python run_experiments.py --model llama-1b --data data/problems.jsonl

Outputs:
    results/accuracy_<model>.jsonl   (one row per variant, with raw output)
    prints a summary table to the console
"""
import argparse
import os
import config
from src.experiment import load_problems, run_accuracy, save_jsonl
from src.metrics import summarize, print_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODELS))
    ap.add_argument("--data", default=config.DATA_FILE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_id = config.MODELS[args.model]
    print(f"Loading {args.model} ({model_id}) ... this can take a minute on CPU.")
    from src.model_runner import ModelRunner
    runner = ModelRunner(model_id)
    print(f"Loaded on device: {runner.device}")

    families = load_problems(args.data)
    print(f"Running {sum(len(f.get('variants', [])) for f in families)}+ variants "
          f"across {len(families)} families...")
    records = run_accuracy(runner, families, config.GEN, args.model)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = args.out or os.path.join(config.RESULTS_DIR, f"accuracy_{args.model}.jsonl")
    save_jsonl(records, out)
    print(f"Wrote {len(records)} records to {out}")

    print_summary(summarize(records))


if __name__ == "__main__":
    main()
