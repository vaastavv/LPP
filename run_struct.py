#!/usr/bin/env python3
"""
Run structural profiling (SRS / CFS / DFBS) for one model over the corpus.

  python run_struct.py --model tiny                 # offline smoke-test
  python run_struct.py --model qwen25c-0.5b         # real model (needs HF)
  HF_MODEL=/path/to/weights python run_struct.py --model local

Writes:
  results/struct/struct_<model>.json   full result (profiles + raw pair data)
  results/struct/modelinfo_<model>.json model metadata (for Table I)
"""
import argparse
import json
import os

import config
from src.struct.corpus import build_corpus
from src.struct.metrics import StructProfiler, save_result
from src.struct.registry import CODE_MODELS, model_info

OUT_DIR = os.path.join(config.RESULTS_DIR, "struct")


def load_runner(model_key: str, seed: int = 42, hidden: int = 128,
                layers: int = 6, heads: int = 4):
    if model_key == "tiny":
        from src.struct.tiny_model import build_tiny_runner
        return build_tiny_runner(seed=seed, hidden_size=hidden,
                                 num_layers=layers, num_heads=heads)
    from src.model_runner import ModelRunner
    if model_key in CODE_MODELS:
        model_id = CODE_MODELS[model_key]
    elif model_key in config.MODELS:
        model_id = config.MODELS[model_key]
    elif model_key == "local":
        model_id = os.environ["HF_MODEL"]
    else:
        model_id = model_key  # treat as a raw HF id / path
    return ModelRunner(model_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--max-programs", type=int, default=0,
                    help="0 = all corpus programs")
    ap.add_argument("--dist", default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--pooling", default="mean", choices=["mean", "max", "sum", "last"])
    ap.add_argument("--eta", type=int, default=8)
    ap.add_argument("--max-neg", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--seed", type=int, default=42, help="tiny-model seed")
    ap.add_argument("--hidden", type=int, default=128, help="tiny-model hidden size")
    ap.add_argument("--layers", type=int, default=6, help="tiny-model layers")
    ap.add_argument("--heads", type=int, default=4, help="tiny-model heads")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading model '{args.model}' ...", flush=True)
    runner = load_runner(args.model, seed=args.seed, hidden=args.hidden,
                         layers=args.layers, heads=args.heads)

    progs = [p.source for p in build_corpus(config.SEED)]
    if args.max_programs > 0:
        progs = progs[:args.max_programs]
    print(f"Profiling {len(progs)} programs ...", flush=True)

    label = f"{args.model}{('-' + args.tag) if args.tag else ''}"
    prof = StructProfiler(runner, dist=args.dist, pooling=args.pooling,
                          eta=args.eta, max_neg=args.max_neg)
    res = prof.run(progs, label)

    tag = (f"_{args.tag}" if args.tag else "")
    out = os.path.join(OUT_DIR, f"struct_{args.model}{tag}.json")
    save_result(res, out)
    info = model_info(runner)
    info["model_id"] = label
    if args.model == "tiny":  # distinguish panel members by size
        info["params"] = f"{(args.hidden * args.hidden * args.layers) / 1e6:.2f}M"
        info["layers"] = args.layers
        info["hidden"] = args.hidden
    with open(os.path.join(OUT_DIR, f"modelinfo_{args.model}{tag}.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved {out}")
    print(f"  layers(incl emb)={res.num_layers}  meta={res.meta}")
    print(f"  peaks: {res.peak}")
    for name, a in [("SRS", res.SRS), ("CFS", res.CFS), ("DFBS", res.DFBS)]:
        mx = float(a[1:].max()) if len(a) > 1 else float("nan")
        print(f"  max {name} = {mx:.4f}")


if __name__ == "__main__":
    main()
