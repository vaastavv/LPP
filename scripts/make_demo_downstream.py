#!/usr/bin/env python3
"""
Generate ILLUSTRATIVE (synthetic) downstream scores for the smoke-test panel so
Tables VI-VIII can be rendered end to end.

These numbers are NOT measured task performance. They are a deterministic
function of each model's peak latent metrics plus fixed noise, written only so
the table/figure format can be demonstrated on the random tiny panel (whose
real task scores are degenerate). Every file is stamped {"synthetic": true}.

The real Colab sweep (run_downstream.py on pretrained models) overwrites these
with genuine measured scores.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.struct.metrics import load_result

OUT = os.path.join(config.RESULTS_DIR, "struct")


def peak(res, q):
    arr = {"SRS": res.SRS, "CFS": res.CFS, "DFBS": res.DFBS}[q]
    return float(np.max(arr[1:])) if arr is not None and len(arr) > 1 else 0.0


def main():
    results = [load_result(p) for p in sorted(glob.glob(os.path.join(OUT, "struct_*.json")))]
    # normalize each metric across models to [0,1] so the mapping is well-scaled
    def norm(vals):
        v = np.array(vals, float)
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-9)
    srs = norm([peak(r, "SRS") for r in results])
    cfs = norm([peak(r, "CFS") for r in results])
    dfbs = norm([peak(r, "DFBS") for r in results])
    rng = np.random.default_rng(0)
    for i, r in enumerate(results):
        noise = rng.normal(0, 0.05, 5)
        scores = {
            # dependency-sensitive tasks track DFBS (paper's central claim)
            "bug_localization": float(np.clip(0.45 + 0.35 * dfbs[i] + noise[0], 0, 1)),
            "program_repair":   float(np.clip(0.30 + 0.40 * dfbs[i] + 0.1 * cfs[i] + noise[1], 0, 1)),
            # summarization tracks syntax
            "code_summarization": float(np.clip(0.25 + 0.45 * srs[i] + noise[2], 0, 1)),
            # completion/translation mildly track control flow
            "code_completion":  float(np.clip(0.40 + 0.30 * cfs[i] + noise[3], 0, 1)),
            "code_translation": float(np.clip(0.20 + 0.35 * cfs[i] + noise[4], 0, 1)),
        }
        path = os.path.join(OUT, f"downstream_{_fname(r.model)}.json")
        with open(path, "w") as f:
            json.dump({"model": r.model, "scores": scores,
                       "synthetic": True,
                       "note": "ILLUSTRATIVE synthetic scores; replace with real benchmark runs"},
                      f, indent=2)
        print("wrote", path)


def _fname(model):
    # struct files are struct_<key>.json; mirror the key from the model label
    return model.replace("-", "_", 1) if model.startswith("tiny-") else model


if __name__ == "__main__":
    main()
