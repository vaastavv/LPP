#!/usr/bin/env python3
"""
Expand data/templates.jsonl (GSM-Symbolic style) into a full dataset and merge
it with the hand-written data/problems.jsonl, writing data/dataset.jsonl.

Then run experiments against the combined file, e.g.:
    python expand_templates.py
    python run_experiments.py --model llama-1b --data data/dataset.jsonl
    python run_latent.py      --model llama-1b --data data/dataset.jsonl

Deterministic: the same --seed always produces the same dataset.
"""
import argparse
import json
import os
import random
import config
from src.templates import generate_family

TEMPLATES_FILE = os.path.join(config.ROOT, "data", "templates.jsonl")
OUT_FILE = os.path.join(config.ROOT, "data", "dataset.jsonl")


def load_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("//"):
                    out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default=TEMPLATES_FILE)
    ap.add_argument("--out", default=OUT_FILE)
    ap.add_argument("--n-numeric", type=int, default=5,
                    help="numeric variants per template")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-explicit", action="store_true",
                    help="exclude the hand-written problems.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    templates = load_jsonl(args.templates)
    generated = [generate_family(t, rng, n_numeric=args.n_numeric) for t in templates]

    explicit = [] if args.no_explicit else load_jsonl(config.DATA_FILE)
    combined = generated + explicit

    with open(args.out, "w", encoding="utf-8") as f:
        for fam in combined:
            f.write(json.dumps(fam, ensure_ascii=False) + "\n")

    nvar = sum(len(f["variants"]) for f in combined)
    print(f"Wrote {len(combined)} families ({nvar} variants) to {args.out}")
    print(f"  from templates : {len(generated)} families")
    print(f"  hand-written   : {len(explicit)} families")
    print(f"  seed           : {args.seed}   numeric/template: {args.n_numeric}")


if __name__ == "__main__":
    main()
