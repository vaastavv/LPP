"""Reproducibly build data/processed/intent_dataset.csv from the raw seed + curated expansion files.

Steps (every modification is logged to dataset_changelog.json):
  1. load + validate the raw seed (schema, 20 intents)
  2. apply documented label corrections (data/raw/label_corrections.csv)
  3. remove exact / trivial (case+punctuation) duplicates
  4. add curated expansion examples (data/raw/expansion/<intent>.txt)
  5. remove near-duplicates (char n-gram cosine >= threshold); cross-intent near-duplicates are conflicts
  6. enforce 100-150 examples per intent (deterministic down-sampling of *expansion* rows only)
  7. label-consistency check: out-of-fold TF-IDF/LR predictions flag suspicious labels for review
  8. re-validate, save dataset + provenance + changelog, print summary

Usage: python -m src.data.enhance_dataset
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from src.config import INTENTS, load_config, path, set_seed
from src.data.deduplicate import near_duplicate_pairs
from src.data.validate import validate_dataset
from src.text import dedup_key, normalize


def read_expansion(exp_dir, intent: str) -> list[tuple[str, int]]:
    f = exp_dir / f"{intent}.txt"
    if not f.exists():
        return []
    rows = []
    for lineno, line in enumerate(f.read_text().splitlines(), 1):
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append((normalize(line), lineno))
    return rows


def label_consistency(df: pd.DataFrame, seed: int, min_conf: float = 0.5) -> pd.DataFrame:
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                         LogisticRegression(C=10.0, max_iter=3000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, df["text"], df["intent"], cv=cv, method="predict_proba")
    classes = np.array(sorted(df["intent"].unique()))
    pred = classes[proba.argmax(1)]
    conf = proba.max(1)
    out = df.assign(oof_pred=pred, oof_conf=conf.round(3))
    return out[(out["oof_pred"] != out["intent"]) & (out["oof_conf"] >= min_conf)].sort_values("oof_conf", ascending=False)


def main() -> int:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    dcfg = cfg["dataset"]
    log: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "dataset_version": cfg["dataset_version"], "steps": []}

    # 1. raw seed
    raw = pd.read_csv(path(cfg, "raw_dataset"), keep_default_na=False)
    raw["text"] = raw["text"].map(normalize)
    errs = [e for e in validate_dataset(raw) if "duplicate" not in e]
    if errs:
        print("Raw dataset invalid:", errs)
        return 1
    log["original_size"] = len(raw)
    log["original_per_intent"] = dict(Counter(raw["intent"]))
    raw["source"] = "original"
    raw["source_ref"] = [f"original_intent_dataset.csv:{i + 2}" for i in range(len(raw))]

    # 2. label corrections
    corr = pd.read_csv(path(cfg, "label_corrections"), keep_default_na=False)
    relabels = []
    for _, c in corr.iterrows():
        m = (raw["text"] == normalize(c["text"])) & (raw["intent"] == c["old_intent"])
        if m.any():
            raw.loc[m, "intent"] = c["new_intent"]
            relabels.append(dict(text=c["text"], old=c["old_intent"], new=c["new_intent"], reason=c["reason"]))
    log["steps"].append({"step": "label_corrections", "count": len(relabels), "items": relabels})

    # 3. exact / trivial duplicates within the seed
    keys = raw["text"].map(dedup_key)
    dup = keys.duplicated(keep="first")
    log["steps"].append({"step": "seed_exact_or_trivial_duplicates_removed", "count": int(dup.sum()),
                         "items": raw.loc[dup, ["text", "intent"]].to_dict("records")})
    raw = raw[~dup]

    # 4. curated expansion
    exp_rows = []
    for intent in INTENTS:
        for text, lineno in read_expansion(path(cfg, "expansion_dir"), intent):
            exp_rows.append(dict(text=text, intent=intent, source="expansion",
                                 source_ref=f"expansion/{intent}.txt:{lineno}"))
    exp = pd.DataFrame(exp_rows)
    log["expansion_candidates"] = len(exp)
    df = pd.concat([raw, exp], ignore_index=True)
    keys = df["text"].map(dedup_key)
    dup = keys.duplicated(keep="first")  # originals come first, so originals are always preserved
    conflicts = []
    for idx in df.index[dup]:
        first = df.index[(keys == keys[idx])][0]
        if df.at[first, "intent"] != df.at[idx, "intent"]:
            conflicts.append(dict(text=df.at[idx, "text"], kept_intent=df.at[first, "intent"],
                                  dropped_intent=df.at[idx, "intent"]))
    log["steps"].append({"step": "expansion_exact_or_trivial_duplicates_removed", "count": int(dup.sum()),
                         "label_conflicts": conflicts,
                         "items": df.loc[dup, ["text", "intent", "source_ref"]].to_dict("records")})
    df = df[~dup].reset_index(drop=True)

    # 5. near duplicates
    pairs = near_duplicate_pairs(df["text"].tolist(), dcfg["near_duplicate_threshold"])
    drop: set[int] = set()
    near_items = []
    for p in pairs:
        if p.i in drop or p.j in drop:
            continue
        a, b = df.loc[p.i], df.loc[p.j]
        # Prefer keeping the original; otherwise keep the earlier row.
        victim = p.j if not (a["source"] == "expansion" and b["source"] == "original") else p.i
        if df.at[victim, "source"] == "original":
            near_items.append(dict(kept=a["text"], other=b["text"], similarity=round(p.similarity, 3),
                                   action="kept_both_originals", cross_intent=a["intent"] != b["intent"]))
            continue
        drop.add(victim)
        near_items.append(dict(kept=df.at[p.i + p.j - victim, "text"], dropped=df.at[victim, "text"],
                               kept_intent=df.at[p.i + p.j - victim, "intent"],
                               dropped_intent=df.at[victim, "intent"], similarity=round(p.similarity, 3),
                               cross_intent=a["intent"] != b["intent"]))
    log["steps"].append({"step": "near_duplicates_removed", "threshold": dcfg["near_duplicate_threshold"],
                         "count": len(drop), "items": near_items})
    df = df.drop(index=sorted(drop)).reset_index(drop=True)

    # 6. enforce per-intent bounds
    rng = random.Random(seed)
    capped = []
    for intent in INTENTS:
        idx = df.index[df["intent"] == intent].tolist()
        excess = len(idx) - dcfg["max_per_intent"]
        if excess > 0:
            exp_idx = [i for i in idx if df.at[i, "source"] == "expansion"]
            victims = rng.sample(exp_idx, excess)
            capped += victims
    log["steps"].append({"step": "capped_to_max_per_intent", "count": len(capped),
                         "items": df.loc[capped, ["text", "intent"]].to_dict("records")})
    df = df.drop(index=capped).reset_index(drop=True)

    # 7. label consistency (report only - humans decide; the report is reviewed before release)
    flagged = label_consistency(df[["text", "intent"]], seed)
    log["steps"].append({"step": "label_consistency_flags", "count": len(flagged),
                         "note": "out-of-fold TF-IDF/LR disagreement with conf>=0.5; review, not auto-removed",
                         "items": flagged.to_dict("records")})

    # 8. final validation + save
    out = df[["text", "intent"]].copy()
    errors = validate_dataset(out, dcfg["min_per_intent"], dcfg["max_per_intent"])
    if errors:
        print("Enhanced dataset failed validation:")
        for e in errors:
            print("  -", e)
        return 1
    out = out.sort_values(["intent"], kind="stable").reset_index(drop=True)
    p = path(cfg, "processed_dataset")
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False, quoting=1)
    df.sort_values(["intent"], kind="stable")[["text", "intent", "source", "source_ref"]].to_csv(
        path(cfg, "provenance"), index=False, quoting=1)
    counts = Counter(out["intent"])
    src_counts = df.groupby(["intent", "source"]).size().unstack(fill_value=0)
    log["enhanced_size"] = len(out)
    log["enhanced_per_intent"] = {k: counts[k] for k in INTENTS}
    log["per_intent_by_source"] = src_counts.to_dict("index")
    path(cfg, "changelog").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str))

    print(f"Original dataset: {log['original_size']} rows | expansion candidates: {len(exp)}")
    for s in log["steps"]:
        print(f"  {s['step']}: {s['count']}")
    print(f"\n{'Intent':<30}{'Original':>10}{'Added':>8}{'Total':>8}")
    print("-" * 56)
    for k in INTENTS:
        o = int(src_counts.loc[k].get("original", 0))
        e = int(src_counts.loc[k].get("expansion", 0))
        print(f"{k:<30}{o:>10}{e:>8}{counts[k]:>8}")
    print("-" * 56)
    print(f"{'TOTAL':<30}{int(src_counts.get('original', pd.Series()).sum()):>10}"
          f"{int(src_counts.get('expansion', pd.Series()).sum()):>8}{len(out):>8}")
    print(f"\nSaved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
