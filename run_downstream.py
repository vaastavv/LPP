#!/usr/bin/env python3
"""
Run the five downstream SE task proxies for one model (RQ3).

  python run_downstream.py --model tiny --tag A --seed 1 --hidden 96 --layers 4
  python run_downstream.py --model qwen25c-0.5b

Writes results/struct/downstream_<model>.json
"""
import argparse
import json
import os

import config
from run_struct import load_runner, OUT_DIR
from src.struct.downstream import run_downstream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    label = f"{args.model}{('-' + args.tag) if args.tag else ''}"
    print(f"Loading model '{label}' ...", flush=True)
    runner = load_runner(args.model, seed=args.seed, hidden=args.hidden,
                         layers=args.layers, heads=args.heads)
    gen = {"do_sample": False, "max_new_tokens": args.max_new_tokens}
    out = run_downstream(runner, label, gen=gen, limit=args.limit)

    tag = (f"_{args.tag}" if args.tag else "")
    path = os.path.join(OUT_DIR, f"downstream_{args.model}{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")
    for k, v in out["scores"].items():
        print(f"  {k:22s} {v:.3f}" if v == v else f"  {k:22s} n/a")


if __name__ == "__main__":
    main()
