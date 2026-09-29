"""
Distribution-drift monitoring experiment for LPP (proposed extension).

The paper claims (Discussion, "Implications for model selection and monitoring",
page 11) that intrinsic LPP metrics can act as an EARLY-WARNING signal:

    "if a deployed model that normally maintains a certain entropy floor
     suddenly exhibits a rising entropy trend on incoming data, it may indicate
     domain drift ... Such latent shifts could presage performance degradation
     ... long before these issues surface in output accuracy."

The paper never tests this. This module provides the machinery to test it
directly and falsifiably: apply progressively stronger out-of-distribution
perturbations to task inputs, measure BOTH task accuracy and the LPP metrics
(entropy floor, max-ER, max-PR) on the SAME perturbed inputs at each drift
level, and ask a precise question:

    Does a latent metric deviate from its in-distribution baseline at a LOWER
    drift level than accuracy does?  (i.e. does it "lead"?)

Everything here is pure-Python (no torch), so the perturbations and the
detection statistics are unit-testable offline. `run_drift.py` drives a model;
`drift_analysis.py` plots the result.
"""
from __future__ import annotations

import random
import statistics

# ---------------------------------------------------------------------------
# Perturbations. Each takes (text, severity in [0,1], rng) and returns a
# corrupted string. severity=0 must return the text unchanged.
# ---------------------------------------------------------------------------
_TYPO_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

# A small latin -> confusable-unicode map (homoglyphs). Visually similar,
# tokenizes very differently -> strong distribution shift with little visual cue.
_HOMOGLYPHS = {
    "a": "а", "c": "с", "e": "е", "i": "і", "o": "о", "p": "р",
    "s": "ѕ", "x": "х", "y": "у", "A": "А", "B": "В", "C": "С",
    "E": "Е", "H": "Н", "O": "О", "P": "Р", "T": "Т", "X": "Х",
}

_GIBBERISH = [
    "qux", "zad", "vom", "bleen", "flarn", "wodge", "trisk", "plemn",
    "grock", "shiv", "dworp", "klat", "myrn", "pilv", "zesh", "obru",
]


def char_noise(text: str, severity: float, rng: random.Random) -> str:
    """Replace each character with prob `severity` by a random lowercase letter."""
    if severity <= 0:
        return text
    out = []
    for ch in text:
        if ch != " " and rng.random() < severity:
            out.append(rng.choice(_TYPO_ALPHABET))
        else:
            out.append(ch)
    return "".join(out)


def word_dropout(text: str, severity: float, rng: random.Random) -> str:
    """Drop each whitespace-separated token with probability `severity`."""
    if severity <= 0:
        return text
    words = text.split()
    kept = [w for w in words if rng.random() >= severity]
    if not kept and words:                     # never return an empty prompt
        kept = [rng.choice(words)]
    return " ".join(kept)


def word_shuffle(text: str, severity: float, rng: random.Random) -> str:
    """Apply ~severity*len local adjacent swaps, scrambling word order gradually."""
    if severity <= 0:
        return text
    words = text.split()
    n_swaps = int(round(severity * len(words)))
    for _ in range(n_swaps):
        if len(words) < 2:
            break
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def homoglyph(text: str, severity: float, rng: random.Random) -> str:
    """Replace latin chars with confusable unicode homoglyphs with prob `severity`."""
    if severity <= 0:
        return text
    out = []
    for ch in text:
        sub = _HOMOGLYPHS.get(ch)
        if sub and rng.random() < severity:
            out.append(sub)
        else:
            out.append(ch)
    return "".join(out)


def append_gibberish(text: str, severity: float, rng: random.Random) -> str:
    """Append ~severity*20 nonsense words (domain / topical drift)."""
    if severity <= 0:
        return text
    n = int(round(severity * 20))
    extra = " ".join(rng.choice(_GIBBERISH) for _ in range(n))
    return f"{text} {extra}" if extra else text


KINDS = {
    "char_noise": char_noise,
    "word_dropout": word_dropout,
    "word_shuffle": word_shuffle,
    "homoglyph": homoglyph,
    "append_gibberish": append_gibberish,
}

# Increasing severity schedule; index 0 is in-distribution (severity 0).
DEFAULT_SEVERITIES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60)


def perturb(text: str, kind: str, severity: float, seed: int) -> str:
    """Deterministically corrupt `text` at the given severity for the named kind."""
    if kind not in KINDS:
        raise ValueError(f"unknown drift kind {kind!r}; expected one of {sorted(KINDS)}")
    return KINDS[kind](text, severity, random.Random(seed))


# ---------------------------------------------------------------------------
# Detection statistics. The core question is: at which drift level does a
# metric deviate significantly from its in-distribution (level-0) baseline?
#
# We use a PAIRED bootstrap over sample ids (each id is one base-prompt/rep,
# shared across all levels), so accuracy (binary) and the continuous latent
# metrics go through one code path. A metric is "detected" at a level when the
# bootstrap CI of the mean difference (level - baseline) excludes 0 in the
# hypothesized direction ('up' for entropy/ER/PR, 'down' for accuracy).
# ---------------------------------------------------------------------------
def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def paired_bootstrap_detect(baseline: list[float], level: list[float],
                            direction: str, alpha: float = 0.05,
                            n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Paired bootstrap of mean(level) - mean(baseline). `baseline` and `level` are
    aligned element-for-element (same sample ids). Returns:
      {'detected': bool, 'effect': mean diff, 'ci': (lo, hi)}
    'detected' is True when the (1-alpha) CI excludes 0 in `direction`
    ('up' => lo > 0, 'down' => hi < 0).
    """
    n = min(len(baseline), len(level))
    if n == 0:
        return {"detected": False, "effect": 0.0, "ci": (0.0, 0.0)}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        d = sum(level[j] - baseline[j] for j in idxs) / n
        diffs.append(d)
    diffs.sort()
    lo = _percentile(diffs, alpha / 2)
    hi = _percentile(diffs, 1 - alpha / 2)
    effect = statistics.fmean(level[:n]) - statistics.fmean(baseline[:n])
    if direction == "up":
        detected = lo > 0
    elif direction == "down":
        detected = hi < 0
    else:
        raise ValueError("direction must be 'up' or 'down'")
    return {"detected": detected, "effect": effect, "ci": (lo, hi)}


def detection_level(per_level: dict[int, list[float]], direction: str,
                    baseline_level: int = 0, alpha: float = 0.05,
                    n_boot: int = 2000, seed: int = 0) -> "int | None":
    """
    First level index > baseline_level at which the metric is detected as shifted
    (per paired_bootstrap_detect). Returns None if never detected. `per_level`
    lists must be aligned by sample id across levels.
    """
    if baseline_level not in per_level:
        return None
    base = per_level[baseline_level]
    for lvl in sorted(k for k in per_level if k > baseline_level):
        res = paired_bootstrap_detect(base, per_level[lvl], direction,
                                      alpha=alpha, n_boot=n_boot, seed=seed)
        if res["detected"]:
            return lvl
    return None


def aligned_by_level(rows: list[dict], value_key: str,
                     level_key: str = "level", id_key: str = "sample_id"
                     ) -> tuple[list[int], dict[int, list[float]]]:
    """
    Group `rows` into {level: [values]} aligned by the sample ids common to every
    level (so bootstrap pairs are valid). Returns (sorted_levels, per_level).
    """
    by_level: dict[int, dict] = {}
    for r in rows:
        sid = tuple(r[id_key]) if isinstance(r[id_key], list) else r[id_key]
        by_level.setdefault(r[level_key], {})[sid] = r[value_key]
    if not by_level:
        return [], {}
    common = set.intersection(*(set(d) for d in by_level.values()))
    common_sorted = sorted(common)
    levels = sorted(by_level)
    per_level = {lvl: [by_level[lvl][sid] for sid in common_sorted] for lvl in levels}
    return levels, per_level
