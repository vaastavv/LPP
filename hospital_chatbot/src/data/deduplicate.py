"""Exact and near-duplicate detection.

* exact duplicates: identical `dedup_key` (case/punctuation/whitespace-insensitive), e.g.
  "I need an appointment." vs "I need an appointment!"
* near duplicates: character 3-5-gram TF-IDF cosine similarity >= threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.text import dedup_key


@dataclass
class Pair:
    i: int
    j: int
    similarity: float


def exact_duplicate_groups(texts: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, t in enumerate(texts):
        groups.setdefault(dedup_key(t), []).append(idx)
    return {k: v for k, v in groups.items() if len(v) > 1}


def similarity_matrix(texts: list[str]) -> np.ndarray:
    keys = [dedup_key(t) for t in texts]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    X = vec.fit_transform(keys)
    return (X @ X.T).toarray()


def near_duplicate_pairs(texts: list[str], threshold: float, S: np.ndarray | None = None) -> list[Pair]:
    """Pairs with similarity >= threshold. Pass a precomputed `S` to keep IDF weights consistent
    with another measurement (e.g. the split leakage report)."""
    if len(texts) < 2:
        return []
    if S is None:
        S = similarity_matrix(texts)
    iu = np.triu_indices(len(texts), k=1)
    mask = S[iu] >= threshold
    return sorted(
        (Pair(int(i), int(j), float(s)) for i, j, s in zip(iu[0][mask], iu[1][mask], S[iu][mask])),
        key=lambda p: -p.similarity,
    )


def connected_groups(n: int, pairs: list[Pair]) -> list[int]:
    """Union-find over pairs; returns a group id per item."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        ri, rj = find(p.i), find(p.j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(n)]
