"""Validate ``data/intents.csv`` and create stratified train/val/test splits.

Steps
-----
1. Check required columns and missing values (rows with missing text/intent are dropped).
2. Normalise whitespace and remove exact + case/punctuation-insensitive duplicates.
3. Detect label conflicts (same text with different intents) and drop them.
4. Report class distribution and imbalance ratio (max/min class count).
5. Stratified split into train / val / test and save CSVs.
6. Write a JSON validation report to ``reports/dataset_report.json``.

Usage::

    python -m scripts.validate_dataset --val-size 0.15 --test-size 0.15 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    INTENTS,
    INTENTS_CSV,
    REPORTS_DIR,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    dedup_key,
    get_logger,
    normalize_text,
    save_json,
)

logger = get_logger(__name__)

REQUIRED_COLUMNS = ("text", "intent")


def check_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    df = df.copy()
    df["text"] = df["text"].astype("string").map(lambda t: normalize_text(t) if pd.notna(t) else t)
    df["intent"] = df["intent"].astype("string").str.strip()
    df = df.replace({"": pd.NA})

    missing = {col: int(df[col].isna().sum()) for col in REQUIRED_COLUMNS}
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    logger.info("Missing values: %s -> dropped %d rows", missing, before - len(df))
    return df, {"missing_values": missing, "rows_dropped_missing": before - len(df)}


def check_unknown_labels(df: pd.DataFrame, allowed: list[str]) -> tuple[pd.DataFrame, dict]:
    unknown = sorted(set(df["intent"]) - set(allowed))
    if unknown:
        logger.warning("Dropping rows with unknown intents: %s", unknown)
    df = df[df["intent"].isin(allowed)]
    absent = sorted(set(allowed) - set(df["intent"]))
    if absent:
        logger.warning("Intents with zero examples: %s", absent)
    return df, {"unknown_labels": unknown, "absent_labels": absent}


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["_key"] = df["text"].map(dedup_key)

    # conflicting labels: same normalised text, several intents -> ambiguous, drop all
    label_counts = df.groupby("_key")["intent"].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)
    conflicts = int(df["_key"].isin(conflicting_keys).sum())
    df = df[~df["_key"].isin(conflicting_keys)]

    before = len(df)
    exact_dups = int(df.duplicated(subset=["text", "intent"]).sum())
    df = df.drop_duplicates(subset=["_key", "intent"], keep="first")
    near_dups = before - len(df) - exact_dups

    logger.info(
        "Duplicates removed: exact=%d, case/punct-insensitive=%d, label-conflict rows=%d",
        exact_dups, near_dups, conflicts,
    )
    return df.drop(columns="_key"), {
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
        "label_conflict_rows": conflicts,
    }


def check_class_imbalance(df: pd.DataFrame, warn_ratio: float = 1.5) -> dict:
    counts = df["intent"].value_counts().sort_index()
    ratio = float(counts.max() / counts.min()) if len(counts) else float("nan")
    logger.info("Class distribution:\n%s", counts.to_string())
    if ratio > warn_ratio:
        logger.warning("Class imbalance ratio %.2f exceeds %.2f - consider class weights / resampling", ratio, warn_ratio)
    else:
        logger.info("Class imbalance ratio (max/min): %.2f - balanced", ratio)
    return {
        "class_counts": {k: int(v) for k, v in counts.items()},
        "imbalance_ratio": ratio,
        "is_balanced": ratio <= warn_ratio,
    }


def text_statistics(df: pd.DataFrame) -> dict:
    lengths = df["text"].str.split().str.len()
    return {
        "n_rows": int(len(df)),
        "n_intents": int(df["intent"].nunique()),
        "words_min": int(lengths.min()),
        "words_mean": round(float(lengths.mean()), 2),
        "words_max": int(lengths.max()),
    }


def stratified_split(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be in (0, 1)")
    train_df, holdout_df = train_test_split(
        df, test_size=val_size + test_size, stratify=df["intent"], random_state=seed
    )
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        holdout_df, test_size=rel_test, stratify=holdout_df["intent"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def check_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    keys = {name: set(d["text"].map(dedup_key)) for name, d in (("train", train_df), ("val", val_df), ("test", test_df))}
    leakage = {
        "train_val": len(keys["train"] & keys["val"]),
        "train_test": len(keys["train"] & keys["test"]),
        "val_test": len(keys["val"] & keys["test"]),
    }
    if any(leakage.values()):
        raise AssertionError(f"Data leakage between splits: {leakage}")
    return leakage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default=str(INTENTS_CSV))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-clean", action="store_true", help="Write the cleaned dataset back to --input")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    report: dict = {"input": str(Path(args.input).name), "raw_rows": int(len(raw))}

    df, info = check_missing_values(raw)
    report.update(info)
    df, info = check_unknown_labels(df, INTENTS)
    report.update(info)
    df, info = remove_duplicates(df)
    report.update(info)
    report["imbalance"] = check_class_imbalance(df)
    report["text_stats"] = text_statistics(df)

    if args.overwrite_clean:
        df.to_csv(args.input, index=False)
        logger.info("Cleaned dataset written back to %s", args.input)

    train_df, val_df, test_df = stratified_split(df, args.val_size, args.test_size, args.seed)
    report["leakage"] = check_leakage(train_df, val_df, test_df)
    report["splits"] = {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
        "seed": args.seed,
    }

    for split_df, path in ((train_df, TRAIN_CSV), (val_df, VAL_CSV), (test_df, TEST_CSV)):
        split_df.to_csv(path, index=False)
        logger.info("Saved %-5s split: %4d rows -> %s", path.stem, len(split_df), path)

    save_json(report, REPORTS_DIR / "dataset_report.json")
    logger.info("Validation report saved to %s", REPORTS_DIR / "dataset_report.json")


if __name__ == "__main__":
    main()
