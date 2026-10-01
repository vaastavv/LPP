"""Leakage-aware, stratified 70/15/15 split.

Exact and near-duplicates were already removed by enhance_dataset. To stop *moderately* similar
paraphrases (e.g. "Bye for now" / "Bye for today") from straddling train and test, items whose char
n-gram cosine >= leakage_group_threshold are linked into groups (union-find), and whole groups are
assigned to a split. Assignment is done per intent, so the split stays stratified by intent.

Also splits the out-of-domain query set into ood_validation / ood_test.

Usage: python -m src.data.split_dataset
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import INTENTS, load_config, path, set_seed
from src.data.deduplicate import connected_groups, near_duplicate_pairs, similarity_matrix
from src.text import dedup_key, normalize


def assign_groups(df: pd.DataFrame, fractions: dict, threshold: float, seed: int) -> pd.Series:
    rng = random.Random(seed)
    split = pd.Series("", index=df.index)
    # One global vectorizer so similarities match those used by leakage_report().
    S_all = similarity_matrix(df["text"].tolist())
    for intent in INTENTS:
        sub = df[df["intent"] == intent]
        texts = sub["text"].tolist()
        pos = df.index.get_indexer(sub.index)
        gids = connected_groups(len(texts), near_duplicate_pairs(texts, threshold, S_all[np.ix_(pos, pos)]))
        groups: dict[int, list] = defaultdict(list)
        for local, g in enumerate(gids):
            groups[g].append(sub.index[local])
        order = list(groups.values())
        rng.shuffle(order)
        # Large groups first so they don't overflow the small splits.
        order.sort(key=len, reverse=True)
        n = len(sub)
        target = {"test": round(n * fractions["test"]), "validation": round(n * fractions["validation"])}
        filled = Counter()
        for members in order:
            for name in ("test", "validation"):
                if filled[name] + len(members) <= target[name]:
                    filled[name] += len(members)
                    split.loc[members] = name
                    break
            else:
                split.loc[members] = "train"
    return split


def leakage_report(train: pd.DataFrame, other: pd.DataFrame, all_texts: list[str]) -> dict:
    texts = train["text"].tolist() + other["text"].tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(
        [dedup_key(t) for t in all_texts])
    X = vec.transform([dedup_key(t) for t in texts])
    S = (X[len(train):] @ X[:len(train)].T).toarray()
    same = other["intent"].to_numpy()[:, None] == train["intent"].to_numpy()[None, :]
    best_any = S.max(1)
    best_same = np.where(same, S, 0).max(1)
    return {"max_similarity_to_train_same_intent": float(best_same.max()),
            "mean_max_similarity_to_train_same_intent": float(best_same.mean()),
            # Cross-intent similarity is expected: hard negatives are deliberately close to other intents.
            "max_similarity_to_train_any_intent": float(best_any.max()),
            "n_exact_overlap": int(len(set(train["text"]) & set(other["text"])))}


def main() -> int:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    dcfg = cfg["dataset"]
    df = pd.read_csv(path(cfg, "processed_dataset"), keep_default_na=False)
    df["split"] = assign_groups(df, dcfg["split"], dcfg["leakage_group_threshold"], seed)
    parts = {name: df[df["split"] == name][["text", "intent"]].sample(frac=1.0, random_state=seed)
             for name in ("train", "validation", "test")}
    for name, part in parts.items():
        part.to_csv(path(cfg, name), index=False, quoting=1)

    # OOD split
    lines = [normalize(l) for l in path(cfg, "ood_source").read_text().splitlines()]
    ood = [l for l in lines if l and not l.startswith("#")]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ood))
    k = int(round(len(ood) * dcfg["ood_validation_fraction"]))
    ood_val = pd.DataFrame({"text": [ood[i] for i in perm[:k]]})
    ood_test = pd.DataFrame({"text": [ood[i] for i in perm[k:]]})
    path(cfg, "ood_validation").parent.mkdir(parents=True, exist_ok=True)
    ood_val.to_csv(path(cfg, "ood_validation"), index=False, quoting=1)
    ood_test.to_csv(path(cfg, "ood_test"), index=False, quoting=1)

    summary = {
        "seed": seed,
        "sizes": {k: len(v) for k, v in parts.items()},
        "per_intent": {k: dict(Counter(v["intent"])) for k, v in parts.items()},
        "leakage": {"validation": leakage_report(parts["train"], parts["validation"], df["text"].tolist()),
                    "test": leakage_report(parts["train"], parts["test"], df["text"].tolist())},
        "ood_sizes": {"validation": len(ood_val), "test": len(ood_test)},
        "leakage_group_threshold": dcfg["leakage_group_threshold"],
    }
    (path(cfg, "train").parent / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("sizes", "leakage", "ood_sizes")}, indent=2))
    assert summary["leakage"]["test"]["n_exact_overlap"] == 0
    assert summary["leakage"]["validation"]["n_exact_overlap"] == 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
