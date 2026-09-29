#!/usr/bin/env python3
"""
Drift-monitoring experiment (proposed LPP extension).

Tests the paper's untested early-warning claim: as inputs drift out of
distribution, do the LPP metrics (entropy floor, max-ER, max-PR) deviate from
their in-distribution baseline BEFORE task accuracy visibly degrades?

For each drift level (increasing perturbation severity) we take the study's
gradable task prompts, corrupt them, and measure on the SAME corrupted inputs:
  - task accuracy (runner.chat -> extract -> grade), and
  - the LPP metrics (runner.signals).
One row per (base prompt, repetition, level) is written to
results/drift_<model>_<kind>.jsonl. drift_analysis.py then computes the
detection level of each metric and the "lead" of the latent metrics over
accuracy, and plots the overlay.

Usage:
    # real run (needs the model + torch; a few minutes on CPU for qwen-0.5b):
    python run_drift.py --model qwen-0.5b --kind char_noise --reps 3

    # synthetic demonstration of the pipeline (NO model, clearly labelled):
    python run_drift.py --synthetic --kind char_noise

Perturbation kinds: char_noise, word_dropout, word_shuffle, homoglyph,
append_gibberish (see src/drift.py).
"""
import argparse
import json
import os
import random

import config
from src.drift import KINDS, DEFAULT_SEVERITIES, perturb, aligned_by_level, detection_level


def _stable_seed(seed: int, i: int, r: int, lvl: int) -> int:
    return (seed * 1_000_003) + (i * 10_007) + (r * 101) + lvl


def _print_summary(records: list, kind: str) -> None:
    """Console per-level table + detection levels + lead (mirrors drift_analysis)."""
    severities = {}
    for r in records:
        severities[r["level"]] = r["severity"]
    levels = sorted(severities)

    # per-level accuracy over the samples correct at baseline; latent over all.
    correct_rows = [r for r in records if r.get("correct_at_baseline")]
    _, acc = aligned_by_level(correct_rows, "is_correct")
    _, ent = aligned_by_level(records, "min_entropy")
    _, er = aligned_by_level(records, "max_ER")
    _, pr = aligned_by_level(records, "max_PR")

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"\n=== drift summary ({kind}) ===")
    print(f"{'lvl':>3} {'sev':>5} {'acc':>7} {'entropy':>9} {'max_ER':>8} {'max_PR':>8}")
    for lvl in levels:
        a = _mean(acc.get(lvl, [])) if acc else float("nan")
        print(f"{lvl:>3} {severities[lvl]:>5.2f} {a:>7.3f} "
              f"{_mean(ent.get(lvl, [])):>9.3f} {_mean(er.get(lvl, [])):>8.2f} "
              f"{_mean(pr.get(lvl, [])):>8.2f}")

    ent_lvl = detection_level(ent, "up") if ent else None
    er_lvl = detection_level(er, "up") if er else None
    pr_lvl = detection_level(pr, "up") if pr else None
    acc_lvl = detection_level(acc, "down") if acc else None

    def _sev(lvl):
        return severities.get(lvl) if lvl is not None else None

    print("\nDetection level (first level shifted vs baseline, paired bootstrap):")
    print(f"  entropy floor : level {ent_lvl}  (severity {_sev(ent_lvl)})")
    print(f"  max ER        : level {er_lvl}  (severity {_sev(er_lvl)})")
    print(f"  max PR        : level {pr_lvl}  (severity {_sev(pr_lvl)})")
    print(f"  accuracy drop : level {acc_lvl}  (severity {_sev(acc_lvl)})")
    if acc_lvl is not None and ent_lvl is not None:
        lead = acc_lvl - ent_lvl
        verdict = ("entropy LEADS accuracy" if lead > 0 else
                   "entropy lags accuracy" if lead < 0 else
                   "entropy and accuracy trigger together")
        print(f"\n  lead(entropy over accuracy) = {lead} levels -> {verdict}")


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------
def run_real(args) -> list:
    from src.experiment import load_problems, expand_family
    from src.answer_extract import extract, grade
    from src.model_runner import ModelRunner

    families = load_problems(args.data)
    bases = []
    for fam in families:
        for v in expand_family(fam):
            if "answer" in v:
                bases.append({
                    "prompt": v["prompt"], "answer": v["answer"],
                    "answer_type": fam["answer_type"], "family_id": fam["id"],
                })
    if args.n:
        bases = bases[:args.n]

    model_id = config.MODELS[args.model]
    print(f"Loading {args.model} ({model_id}) ...")
    runner = ModelRunner(model_id)
    print(f"Loaded on device: {runner.device}")

    severities = list(DEFAULT_SEVERITIES)
    gen = {"do_sample": False, "max_new_tokens": args.max_new_tokens, "repetition_penalty": 1.0}

    records = []
    total = len(severities) * len(bases) * args.reps
    done = 0
    for lvl, sev in enumerate(severities):
        for i, b in enumerate(bases):
            for r in range(args.reps):
                text = perturb(b["prompt"], args.kind, sev, _stable_seed(args.seed, i, r, lvl))
                raw = runner.chat(text, gen)
                got = extract(raw, b["answer_type"])
                ok = int(bool(grade(got, b["answer"], b["answer_type"])))
                sig = runner.signals(text, use_chat_template=config.LATENT_USE_CHAT_TEMPLATE)
                records.append({
                    "model": args.model, "kind": args.kind,
                    "level": lvl, "severity": sev, "sample_id": [i, r],
                    "family_id": b["family_id"], "answer_type": b["answer_type"],
                    "is_correct": ok,
                    "min_entropy": sig["min_entropy"],
                    "max_ER": sig["max_ER"], "max_PR": sig["max_PR"],
                    "num_tokens": sig["num_tokens"],
                })
                done += 1
        print(f"  level {lvl} (sev {sev}) done — {done}/{total}")

    # tag each row with whether that (base, rep) was correct at level 0
    base_ok = {tuple(r["sample_id"]): r["is_correct"]
               for r in records if r["level"] == 0}
    for r in records:
        r["correct_at_baseline"] = bool(base_ok.get(tuple(r["sample_id"]), False))
    return records


# ---------------------------------------------------------------------------
# Synthetic demonstration (no model). Clearly labelled; fabricated data whose
# only purpose is to exercise the analysis/plot code path end-to-end and show
# what a positive early-warning result looks like.
# ---------------------------------------------------------------------------
def run_synthetic(args) -> list:
    print("!! SYNTHETIC DEMO — fabricated data, NOT a model measurement !!")
    rng = random.Random(args.seed)
    severities = list(DEFAULT_SEVERITIES)
    n_bases, reps = 12, args.reps
    records = []
    # Each (base, rep) gets a FIXED latent "difficulty"; it is correct while the
    # severity-dependent skill threshold stays above it. Correctness is thus
    # deterministic per sample (mirroring greedy decoding), so conditioning on
    # baseline-correct does not create a regression-to-the-mean artifact.
    # Design: entropy rises immediately with severity; accuracy holds until
    # severity passes ~0.2, then falls -> the latent metric should lead.
    difficulty = {(i, r): rng.random() for i in range(n_bases) for r in range(reps)}
    for lvl, sev in enumerate(severities):
        skill = 0.95 if sev <= 0.2 else max(0.05, 0.95 - 2.2 * (sev - 0.2))
        for i in range(n_bases):
            for r in range(reps):
                ent = 0.30 + 3.0 * sev + rng.gauss(0, 0.05)
                er = 8.0 + 6.0 * sev + rng.gauss(0, 0.4)
                pr = 4.0 + 1.5 * sev + rng.gauss(0, 0.3)
                ok = int(difficulty[(i, r)] < skill)
                records.append({
                    "model": "SYNTHDEMO", "kind": args.kind,
                    "level": lvl, "severity": sev, "sample_id": [i, r],
                    "family_id": "synthetic", "answer_type": "numeric",
                    "is_correct": ok,
                    "min_entropy": round(ent, 4), "max_ER": round(er, 3),
                    "max_PR": round(pr, 3), "num_tokens": 40,
                })
    base_ok = {tuple(x["sample_id"]): x["is_correct"] for x in records if x["level"] == 0}
    for x in records:
        x["correct_at_baseline"] = bool(base_ok.get(tuple(x["sample_id"]), False))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODELS))
    ap.add_argument("--kind", default="char_noise", choices=list(KINDS))
    ap.add_argument("--data", default=config.DATA_FILE)
    ap.add_argument("--reps", type=int, default=3,
                    help="random corruptions per base prompt per level (raises n)")
    ap.add_argument("--n", type=int, default=None, help="cap number of base prompts")
    ap.add_argument("--max-new-tokens", type=int, default=config.GEN["max_new_tokens"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--synthetic", action="store_true",
                    help="fabricate data to demo the pipeline without a model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = run_synthetic(args) if args.synthetic else run_real(args)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    label = "SYNTHDEMO" if args.synthetic else args.model
    out = args.out or os.path.join(config.RESULTS_DIR, f"drift_{label}_{args.kind}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} drift records to {out}")

    _print_summary(records, args.kind)
    print("\nNext: python drift_analysis.py")


if __name__ == "__main__":
    main()
