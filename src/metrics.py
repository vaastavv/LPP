"""
Aggregate accuracy-track records into the report's metrics:

  - accuracy per variant kind
  - baseline accuracy (kind == 'baseline')
  - consistency: within a family, how often the model gives the SAME extracted
    answer across variants (mode count / number of variants). 1.0 = perfectly
    invariant to the surface changes.
  - degradation: baseline accuracy minus a variant kind's accuracy, in points.

Pure-Python so it runs with no extra dependencies; use to_csv() if pandas
is available.
"""
from __future__ import annotations
from collections import defaultdict, Counter


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def summarize(records: list) -> dict:
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)

    summary = {}
    for model, recs in by_model.items():
        # group into families
        fams = defaultdict(list)
        for r in recs:
            fams[r["family_id"]].append(r)

        # accuracy per kind
        kind_hits = defaultdict(list)
        for r in recs:
            kind_hits[r["kind"]].append(1.0 if r["is_correct"] else 0.0)
        acc_by_kind = {k: _mean(v) for k, v in kind_hits.items()}

        # consistency per family (over the model's extracted answers)
        consistencies = []
        for fam_id, fr in fams.items():
            answers = [str(r["extracted"]) for r in fr]
            if answers:
                mode_count = Counter(answers).most_common(1)[0][1]
                consistencies.append(mode_count / len(answers))
        consistency = _mean(consistencies)

        # degradation vs baseline
        baseline = acc_by_kind.get("baseline")
        degradation = {}
        if baseline is not None:
            for k, a in acc_by_kind.items():
                if k != "baseline":
                    degradation[k] = round(100 * (baseline - a), 1)

        summary[model] = {
            "n_records": len(recs),
            "n_families": len(fams),
            "baseline_accuracy": round(baseline, 3) if baseline is not None else None,
            "accuracy_by_kind": {k: round(a, 3) for k, a in acc_by_kind.items()},
            "answer_consistency": round(consistency, 3),
            "degradation_points_by_kind": degradation,
        }
    return summary


def print_summary(summary: dict):
    for model, s in summary.items():
        print(f"\n=== {model} ===")
        print(f"  families: {s['n_families']}   records: {s['n_records']}")
        print(f"  baseline accuracy : {s['baseline_accuracy']}")
        print(f"  answer consistency: {s['answer_consistency']}  (1.0 = surface-invariant)")
        print("  accuracy by kind:")
        for k, a in sorted(s["accuracy_by_kind"].items()):
            print(f"      {k:<12} {a}")
        if s["degradation_points_by_kind"]:
            print("  degradation vs baseline (percentage points, higher = more brittle):")
            for k, d in sorted(s["degradation_points_by_kind"].items()):
                print(f"      {k:<12} {d}")
