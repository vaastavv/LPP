#!/usr/bin/env python3
"""
End-to-end real sweep for the CodeLLM structural-profiling study.

Run this where Hugging Face is reachable (Colab/Kaggle/local GPU). It profiles
every model in the panel, runs the downstream task proxies, and builds all
tables (I-VIII) and figures (3-4).

  python scripts/run_colab.py --models qwen25c-0.5b qwen25c-1.5b qwen25c-3b
  python scripts/run_colab.py                 # uses registry.CODE_MODELS

On a GPU the structural pass is minutes per small model; downstream generation
is the slower part (tune --max-new-tokens / --limit).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.struct.corpus import build_corpus
from src.struct.metrics import StructProfiler, save_result
from src.struct.registry import CODE_MODELS, model_info
from src.struct.downstream import run_downstream
from src.model_runner import ModelRunner

OUT = os.path.join(config.RESULTS_DIR, "struct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(CODE_MODELS),
                    help="registry keys or raw HF ids/paths")
    ap.add_argument("--skip-downstream", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    progs = [p.source for p in build_corpus(config.SEED)]

    for key in args.models:
        model_id = CODE_MODELS.get(key, key)
        label = key if key in CODE_MODELS else os.path.basename(model_id)
        print(f"\n=== {label} ({model_id}) ===", flush=True)
        runner = ModelRunner(model_id)

        prof = StructProfiler(runner)
        res = prof.run(progs, label)
        save_result(res, os.path.join(OUT, f"struct_{label}.json"))
        info = model_info(runner); info["model_id"] = label
        with open(os.path.join(OUT, f"modelinfo_{label}.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"  structural peaks: {res.peak}")

        if not args.skip_downstream:
            gen = {"do_sample": False, "max_new_tokens": args.max_new_tokens}
            d = run_downstream(runner, label, gen=gen, limit=args.limit)
            with open(os.path.join(OUT, f"downstream_{label}.json"), "w") as f:
                json.dump(d, f, indent=2)
            print(f"  downstream: {d['scores']}")

        del runner  # free memory before the next model

    # build everything
    os.system(f"{sys.executable} {os.path.join(config.ROOT, 'run_tables.py')}")


if __name__ == "__main__":
    main()
