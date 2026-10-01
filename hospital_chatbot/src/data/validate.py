"""Validation for the intent dataset and the response-template matrix.

CLI:
    python -m src.data.validate                # validate processed dataset + templates
    python -m src.data.validate --dataset PATH --templates PATH
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import GOEMOTIONS_LABELS, INTENTS, load_config, path
from src.text import dedup_key


def validate_dataset(df: pd.DataFrame, min_per_intent: int | None = None,
                     max_per_intent: int | None = None, required_intents=INTENTS) -> list[str]:
    errors: list[str] = []
    if list(df.columns) != ["text", "intent"]:
        errors.append(f"schema must be [text, intent], got {list(df.columns)}")
        return errors
    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        errors.append("empty text values present")
    if df["intent"].isna().any():
        errors.append("empty intent values present")
    unknown = sorted(set(df["intent"].dropna()) - set(required_intents))
    if unknown:
        errors.append(f"unknown intents: {unknown}")
    missing = sorted(set(required_intents) - set(df["intent"]))
    if missing:
        errors.append(f"missing intents: {missing}")
    dup_exact = df["text"][df["text"].duplicated()].tolist()
    if dup_exact:
        errors.append(f"{len(dup_exact)} exact duplicate texts, e.g. {dup_exact[:3]}")
    keys = df["text"].astype(str).map(dedup_key)
    dup_trivial = df["text"][keys.duplicated()].tolist()
    if dup_trivial:
        errors.append(f"{len(dup_trivial)} case/punctuation-only duplicates, e.g. {dup_trivial[:3]}")
    counts = Counter(df["intent"])
    for intent in required_intents:
        n = counts.get(intent, 0)
        if min_per_intent is not None and n < min_per_intent:
            errors.append(f"{intent}: {n} examples < min {min_per_intent}")
        if max_per_intent is not None and n > max_per_intent:
            errors.append(f"{intent}: {n} examples > max {max_per_intent}")
    return errors


def validate_templates(templates: dict, intents=INTENTS, emotions=GOEMOTIONS_LABELS) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Duplicate strings are warnings ('where avoidable')."""
    errors: list[str] = []
    warnings: list[str] = []
    missing_intents = sorted(set(intents) - set(templates))
    extra_intents = sorted(set(templates) - set(intents) - {"_meta"})
    if missing_intents:
        errors.append(f"missing intents: {missing_intents}")
    if extra_intents:
        errors.append(f"unexpected top-level keys: {extra_intents}")
    seen: Counter = Counter()
    for intent in intents:
        block = templates.get(intent, {})
        if not isinstance(block, dict):
            errors.append(f"{intent}: expected an object of emotion -> response")
            continue
        missing = [e for e in emotions if e not in block]
        if missing:
            errors.append(f"{intent}: missing emotions {missing}")
        extra = sorted(set(block) - set(emotions))
        if extra:
            errors.append(f"{intent}: unknown emotions {extra}")
        for emo, resp in block.items():
            if not isinstance(resp, str) or not resp.strip():
                errors.append(f"{intent}/{emo}: empty response")
            else:
                seen[resp.strip()] += 1
    dups = [r for r, c in seen.items() if c > 1]
    if dups:
        warnings.append(f"{len(dups)} response strings reused across combinations, e.g. {dups[:2]}")
    return errors, warnings


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(path(cfg, "processed_dataset")))
    ap.add_argument("--templates", default=str(path(cfg, "response_templates")))
    args = ap.parse_args(argv)
    rc = 0
    if Path(args.dataset).exists():
        df = pd.read_csv(args.dataset, keep_default_na=False)
        errs = validate_dataset(df, cfg["dataset"]["min_per_intent"], cfg["dataset"]["max_per_intent"])
        print(f"[dataset] {args.dataset}: {len(df)} rows, {df['intent'].nunique()} intents")
        for e in errs:
            print("  ERROR:", e)
        print("  OK" if not errs else f"  {len(errs)} error(s)")
        rc |= bool(errs)
    else:
        print(f"[dataset] not found: {args.dataset}")
        rc = 1
    if Path(args.templates).exists():
        templates = json.load(open(args.templates))
        errs, warns = validate_templates(templates)
        n = sum(len(v) for k, v in templates.items() if k != "_meta" and isinstance(v, dict))
        print(f"[templates] {args.templates}: {n} responses "
              f"({len([k for k in templates if k != '_meta'])} intents x {len(GOEMOTIONS_LABELS)} emotions expected)")
        for e in errs:
            print("  ERROR:", e)
        for w in warns:
            print("  WARN:", w)
        print("  OK" if not errs else f"  {len(errs)} error(s)")
        rc |= bool(errs)
    else:
        print(f"[templates] not found: {args.templates}")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
