"""
Program corpus for structural profiling.

The CodeLLM paper partitions its corpus by repository so that structurally
related programs never straddle the dev/cal/test split. Real code datasets
(MBPP/HumanEval) are not reachable in every environment, so this module
provides a deterministic *synthetic* corpus of small, self-contained, and
*executable* Python functions grouped into repositories ("families"). Each
program exposes:

  * source          -- the function source (what the model sees)
  * entry           -- the callable name
  * repo            -- the partition id (family)
  * tests           -- (args, expected) pairs used by the downstream tasks

Programs are written to exercise the structural features the metrics need:
`range` loops (so for->while fires), augmented assignment and list
comprehensions (so the syntax transform fires), and clear definition->use
chains.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Program:
    pid: str
    repo: str
    entry: str
    source: str
    tests: list = field(default_factory=list)


def _mk(pid, repo, entry, source, tests):
    src = source.strip() + "\n"
    # sanity: must parse and the entry must be defined
    import ast
    ast.parse(src)
    return Program(pid=pid, repo=repo, entry=entry, source=src, tests=tests)


# ---------------------------------------------------------------------------
# Repository generators (each returns several related programs)
# ---------------------------------------------------------------------------
def _repo_aggregate(rng) -> list[Program]:
    progs = []
    for k in (2, 3, 4, 5):
        src = f"""
def sum_multiples(n):
    total = 0
    for i in range(n):
        if i % {k} == 0:
            total += i
    return total
"""
        tests = [((10,), sum(i for i in range(10) if i % k == 0)),
                 ((20,), sum(i for i in range(20) if i % k == 0))]
        progs.append(_mk(f"agg_mult{k}", "aggregate", "sum_multiples", src, tests))
    for t in (0, 5, 10):
        src = f"""
def count_above(values):
    count = 0
    for v in values:
        if v > {t}:
            count += 1
    return count
"""
        tests = [(([1, 6, 11, 3],), sum(1 for v in [1, 6, 11, 3] if v > t)),
                 (([t, t + 1],), sum(1 for v in [t, t + 1] if v > t))]
        progs.append(_mk(f"agg_above{t}", "aggregate", "count_above", src, tests))
    return progs


def _repo_search(rng) -> list[Program]:
    progs = []
    src = """
def find_index(values, target):
    idx = -1
    for i in range(len(values)):
        if values[i] == target:
            idx = i
            return idx
    return idx
"""
    progs.append(_mk("srch_find", "search", "find_index", src,
                     [(([3, 7, 9], 7), 1), (([3, 7, 9], 4), -1)]))
    src = """
def max_value(values):
    best = values[0]
    for v in values:
        if v > best:
            best = v
    return best
"""
    progs.append(_mk("srch_max", "search", "max_value", src,
                     [(([3, 9, 2],), 9), (([-1, -5],), -1)]))
    src = """
def count_until(values, limit):
    acc = 0
    n = 0
    for v in values:
        acc += v
        if acc >= limit:
            return n
        n += 1
    return n
"""
    progs.append(_mk("srch_until", "search", "count_until", src,
                     [(([1, 2, 3, 4], 5), 2), (([10], 100), 1)]))
    return progs


def _repo_transform(rng) -> list[Program]:
    progs = []
    for s in (2, 3, 10):
        src = f"""
def scale_list(values):
    scaled = [v * {s} for v in values]
    return scaled
"""
        progs.append(_mk(f"xf_scale{s}", "transform", "scale_list", src,
                         [(([1, 2, 3],), [i * s for i in [1, 2, 3]])]))
    src = """
def keep_positive(values):
    result = [v for v in values]
    out = []
    for v in result:
        if v > 0:
            out.append(v)
    return out
"""
    progs.append(_mk("xf_pos", "transform", "keep_positive", src,
                     [(([-1, 2, -3, 4],), [2, 4])]))
    src = """
def running_total(values):
    out = []
    total = 0
    for v in values:
        total += v
        out.append(total)
    return out
"""
    progs.append(_mk("xf_runtot", "transform", "running_total", src,
                     [(([1, 2, 3],), [1, 3, 6])]))
    return progs


def _repo_strings(rng) -> list[Program]:
    progs = []
    src = """
def count_vowels(text):
    vowels = "aeiou"
    count = 0
    for ch in text:
        if ch in vowels:
            count += 1
    return count
"""
    progs.append(_mk("str_vowel", "strings", "count_vowels", src,
                     [(("hello",), 2), (("xyz",), 0)]))
    src = """
def reverse_text(text):
    out = ""
    for ch in text:
        out = ch + out
    return out
"""
    progs.append(_mk("str_rev", "strings", "reverse_text", src,
                     [(("abc",), "cba"), (("",), "")]))
    for target in ("a", "e", "z"):
        src = f"""
def count_char(text):
    total = 0
    for i in range(len(text)):
        if text[i] == "{target}":
            total += 1
    return total
"""
        progs.append(_mk(f"str_cnt_{target}", "strings", "count_char", src,
                         [(("banana",), "banana".count(target))]))
    return progs


def _repo_math(rng) -> list[Program]:
    progs = []
    src = """
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""
    progs.append(_mk("math_fact", "math", "factorial", src,
                     [((5,), 120), ((0,), 1)]))
    src = """
def gcd(a, b):
    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a
"""
    progs.append(_mk("math_gcd", "math", "gcd", src,
                     [((12, 8), 4), ((7, 3), 1)]))
    src = """
def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True
"""
    progs.append(_mk("math_prime", "math", "is_prime", src,
                     [((7,), True), ((8,), False)]))
    src = """
def power(base, exp):
    result = 1
    for i in range(exp):
        result = result * base
    return result
"""
    progs.append(_mk("math_pow", "math", "power", src,
                     [((2, 5), 32), ((3, 0), 1)]))
    return progs


def _repo_nested(rng) -> list[Program]:
    progs = []
    src = """
def grid_sum(rows):
    total = 0
    for row in rows:
        for v in row:
            total += v
    return total
"""
    progs.append(_mk("nest_sum", "nested", "grid_sum", src,
                     [(([[1, 2], [3, 4]],), 10)]))
    src = """
def count_pairs(values, target):
    count = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] + values[j] == target:
                count += 1
    return count
"""
    progs.append(_mk("nest_pairs", "nested", "count_pairs", src,
                     [(([1, 2, 3, 4], 5), 2)]))
    src = """
def diagonal(matrix):
    out = []
    for i in range(len(matrix)):
        out.append(matrix[i][i])
    return out
"""
    progs.append(_mk("nest_diag", "nested", "diagonal", src,
                     [(([[1, 2], [3, 4]],), [1, 4])]))
    return progs


def _repo_stats(rng) -> list[Program]:
    progs = []
    src = """
def mean_value(values):
    total = 0
    for v in values:
        total += v
    avg = total / len(values)
    return avg
"""
    progs.append(_mk("st_mean", "stats", "mean_value", src,
                     [(([2, 4, 6],), 4.0)]))
    src = """
def variance_num(values):
    total = 0
    for v in values:
        total += v
    avg = total / len(values)
    acc = 0
    for v in values:
        acc += (v - avg) * (v - avg)
    return acc / len(values)
"""
    progs.append(_mk("st_var", "stats", "variance_num", src,
                     [(([1, 2, 3],), 2.0 / 3.0)]))
    src = """
def clamp_all(values, lo, hi):
    out = []
    for v in values:
        if v < lo:
            out.append(lo)
        elif v > hi:
            out.append(hi)
        else:
            out.append(v)
    return out
"""
    progs.append(_mk("st_clamp", "stats", "clamp_all", src,
                     [(([-1, 5, 12], 0, 10), [0, 5, 10])]))
    return progs


_REPOS = [_repo_aggregate, _repo_search, _repo_transform, _repo_strings,
          _repo_math, _repo_nested, _repo_stats]


def build_corpus(seed: int = 42) -> list[Program]:
    rng = random.Random(seed)
    progs: list[Program] = []
    for gen in _REPOS:
        progs.extend(gen(rng))
    return progs


def partition_corpus(progs: list[Program], seed: int = 42):
    """Repository-level split into dev/cal/test (no repo shared across sets)."""
    rng = random.Random(seed)
    repos = sorted({p.repo for p in progs})
    rng.shuffle(repos)
    n = len(repos)
    n_dev = max(1, int(round(0.4 * n)))
    n_cal = max(1, int(round(0.3 * n)))
    dev = set(repos[:n_dev])
    cal = set(repos[n_dev:n_dev + n_cal])
    test = set(repos[n_dev + n_cal:]) or {repos[-1]}
    split = {"dev": [], "cal": [], "test": []}
    for p in progs:
        if p.repo in dev:
            split["dev"].append(p)
        elif p.repo in cal:
            split["cal"].append(p)
        else:
            split["test"].append(p)
    return split


if __name__ == "__main__":
    c = build_corpus()
    print(f"{len(c)} programs across {len({p.repo for p in c})} repos")
    sp = partition_corpus(c)
    for k, v in sp.items():
        print(f"  {k}: {len(v)} programs, repos={sorted({p.repo for p in v})}")
