"""
Offline tests for the CodeLLM structural-profiling pipeline. No model download
needed: structural extraction, transforms, stats math, and a full end-to-end
run on the tiny random model.

Run: python tests/test_struct_offline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.struct.extract import extract_struct, delta_cf
from src.struct.transforms import make_contrast_pairs, rename_identifiers
from src.struct.corpus import build_corpus, partition_corpus
from src.struct import stats as S

_FAILS = []


def check(cond, msg):
    if cond:
        print(f"  ok: {msg}")
    else:
        print(f"  FAIL: {msg}")
        _FAILS.append(msg)


SAMPLE = """
def f(a, b):
    total = a + b
    for i in range(total):
        total += i
    if total > 10:
        return total
    return 0
"""


def test_extraction():
    print("test_extraction")
    ps = extract_struct(SAMPLE)
    check(ps.cfg.n_nodes > 3, "CFG has nodes")
    check(len(ps.def_uses) >= 4, "def-use pairs found")
    check(len(ps.cf_units) >= 2, "control-flow units found")
    names = {du.name for du in ps.def_uses}
    check("total" in names and "i" in names, "def-use names include total,i")


def test_delta_cf():
    print("test_delta_cf")
    a = extract_struct("def g():\n    for i in range(3):\n        x = i\n    return x\n")
    b = extract_struct("def g():\n    x = 0\n    return x\n")
    r = rename_identifiers("def g():\n    for i in range(3):\n        x = i\n    return x\n")
    ra = extract_struct(r)
    check(delta_cf(a.cfg, b.cfg) > 0.05, "CFG-different programs have delta>0")
    check(delta_cf(a.cfg, ra.cfg) < 1e-6, "rename preserves CFG (delta≈0)")


def test_transforms():
    print("test_transforms")
    pr = make_contrast_pairs(SAMPLE)
    check(pr["syn_pos"] is not None, "syntax positive generated")
    check(pr["syn_ctrl"] is not None, "syntax control generated")
    check(pr["cf_pos"] is not None, "control-flow positive generated")
    # shapes: syn_pos changes AST, syn_ctrl preserves it
    base = extract_struct(SAMPLE).structural_hash
    sp = extract_struct(pr["syn_pos"][1]).structural_hash
    sc = extract_struct(pr["syn_ctrl"][1]).structural_hash
    check(sp != base, "syn_pos changes AST shape")
    check(sc == base, "syn_ctrl preserves AST shape")


def test_corpus():
    print("test_corpus")
    c = build_corpus()
    check(len(c) >= 20, "corpus has >=20 programs")
    # all programs parse and pass their own tests
    import ast
    allok = True
    for p in c:
        ast.parse(p.source)
        ns = {}
        exec(p.source, ns)
        fn = ns[p.entry]
        for args, exp in p.tests:
            if fn(*args) != exp:
                allok = False
    check(allok, "all corpus programs pass their reference tests")
    sp = partition_corpus(c)
    repos = [set(x.repo for x in sp[k]) for k in ("dev", "cal", "test")]
    check(repos[0].isdisjoint(repos[1]) and repos[0].isdisjoint(repos[2])
          and repos[1].isdisjoint(repos[2]), "partitions are repo-disjoint")


def test_stats():
    print("test_stats")
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 200); b = rng.normal(0.0, 1.0, 200)
    d = S.cohens_d(a, b)
    check(0.7 < d < 1.3, f"cohens_d recovers ~1.0 (got {d:.2f})")
    adj = S.benjamini_hochberg([0.001, 0.5, 0.9])
    check(adj[0] <= 0.003 and adj[2] <= 1.0, "BH adjusts p-values")
    lo, hi = S.bootstrap_ci(a)
    check(lo < a.mean() < hi, "bootstrap CI brackets the mean")
    rho, _ = S.spearman([1, 2, 3, 4, 5], [2, 4, 5, 4, 6])
    check(rho > 0.5, "spearman positive for increasing data")


def test_end_to_end_tiny():
    print("test_end_to_end_tiny (loads tiny random model)")
    try:
        from src.struct.tiny_model import build_tiny_runner
        from src.struct.metrics import StructProfiler
    except Exception as e:
        check(False, f"imports for tiny run failed: {e!r}")
        return
    runner = build_tiny_runner(seed=1, hidden_size=64, num_layers=4)
    progs = [p.source for p in build_corpus()][:8]
    res = StructProfiler(runner).run(progs, "tiny-test", progress=False)
    check(res.num_layers == 5, "tiny model exposes 5 hidden-state layers")
    check(res.SRS is not None and len(res.SRS) == 5, "SRS profile length matches")
    check(res.DFBS is not None and np.isfinite(res.DFBS).all(), "DFBS finite")
    check(all(k in res.peak for k in ("syn", "cf", "df")), "peaks computed")


def main():
    for t in [test_extraction, test_delta_cf, test_transforms, test_corpus,
              test_stats, test_end_to_end_tiny]:
        t()
    print()
    if _FAILS:
        print(f"{len(_FAILS)} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL STRUCT OFFLINE TESTS PASSED")


if __name__ == "__main__":
    main()
