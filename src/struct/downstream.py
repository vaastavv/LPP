"""
Downstream software-engineering task proxies (RQ3).

These are deliberately lightweight, deterministic, CPU-friendly proxies for the
five SE tasks the paper studies. They are NOT the full benchmarks (Defects4J,
CodeXGLUE, HumanEval, ...), which need large datasets and execution
infrastructure out of scope here; each proxy is self-contained on the synthetic
corpus and produces a single per-model score in [0, 1]:

  * bug_localization  -- top-1 localization of a mutated line via token
                         surprisal (uses logits only, no generation)
  * program_repair    -- fix a one-operator bug; graded by test execution
  * code_completion   -- complete a masked final body line; test execution
  * code_summarization-- generate a short description; ROUGE-1 F1 vs reference
  * code_translation  -- style transpilation (for-loop <-> while); test execution

The absolute scores are only meaningful for pretrained models; on the random
tiny model they are near chance, which is expected.
"""
from __future__ import annotations

import ast
import re
import signal

import numpy as np
import torch

import config
from .corpus import build_corpus, partition_corpus, Program


# ---------------------------------------------------------------------------
# Safe execution
# ---------------------------------------------------------------------------
class _Timeout(Exception):
    pass


def _run_tests(source: str, entry: str, tests, timeout: int = 2) -> float:
    """Fraction of tests passed by `source`'s `entry` function."""
    def handler(signum, frame):
        raise _Timeout()
    ns: dict = {}
    try:
        old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
    except (ValueError, AttributeError):
        old = None
    try:
        exec(source, ns)
        fn = ns.get(entry)
        if fn is None:
            return 0.0
        ok = 0
        for args, exp in tests:
            try:
                if fn(*args) == exp:
                    ok += 1
            except Exception:
                pass
        return ok / len(tests) if tests else 0.0
    except Exception:
        return 0.0
    finally:
        if old is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Token surprisal (for bug localization)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _line_surprisals(runner, source: str) -> dict[int, float]:
    """Mean next-token NLL per 1-indexed source line."""
    enc = runner.tokenizer(source, return_offsets_mapping=True,
                           add_special_tokens=False, return_tensors="pt",
                           truncation=True, max_length=512)
    ids = enc["input_ids"].to(runner.device)
    offsets = enc["offset_mapping"][0].tolist()
    if ids.shape[1] < 2:
        return {}
    logits = runner.model(ids).logits[0].float()
    logp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = ids[0, 1:]
    nll = -logp[torch.arange(tgt.shape[0]), tgt]           # [T-1]
    # map char offset -> line number
    starts = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            starts.append(i + 1)
    def char_to_line(c):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= c:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1
    per_line: dict[int, list] = {}
    for t in range(tgt.shape[0]):
        a, b = offsets[t + 1]
        if b <= a:
            continue
        ln = char_to_line(a)
        per_line.setdefault(ln, []).append(float(nll[t]))
    return {ln: float(np.mean(v)) for ln, v in per_line.items()}


# ---------------------------------------------------------------------------
# Bug injection
# ---------------------------------------------------------------------------
_OP_SWAPS = [("+", "-"), ("-", "+"), ("*", "+"), (">", "<"), ("<", ">"),
             (">=", "<="), ("<=", ">="), ("==", "!="), (" and ", " or ")]


def _inject_bug(source: str):
    """Return (buggy_source, buggy_lineno) or (None, None)."""
    lines = source.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(("def ", "#", "return 0", "import")):
            continue
        for a, b in _OP_SWAPS:
            if a in line and ("=" in line or "if" in line or "while" in line or "return" in line):
                buggy = line.replace(a, b, 1)
                if buggy == line:
                    continue
                cand = lines[:i] + [buggy] + lines[i + 1:]
                src = "\n".join(cand)
                try:
                    ast.parse(src)
                    return src, i + 1
                except SyntaxError:
                    continue
    return None, None


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def _extract_function(text: str, entry: str) -> str | None:
    m = re.search(rf"(def\s+{re.escape(entry)}\s*\(.*)", text, re.S)
    if not m:
        return None
    body = m.group(1)
    # keep until a line that is not indented and not the def line (end of func)
    out = []
    for i, line in enumerate(body.split("\n")):
        if i > 0 and line and not line[0].isspace() and not line.startswith("def"):
            break
        out.append(line)
    src = "\n".join(out)
    try:
        ast.parse(src)
        return src
    except SyntaxError:
        return None


def _rouge1_f1(pred: str, ref: str) -> float:
    pt = re.findall(r"[a-z]+", pred.lower())
    rt = re.findall(r"[a-z]+", ref.lower())
    if not pt or not rt:
        return 0.0
    from collections import Counter
    cp, cr = Counter(pt), Counter(rt)
    overlap = sum((cp & cr).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(cp.values())
    rec = overlap / sum(cr.values())
    return 2 * prec * rec / (prec + rec)


def _ref_summary(p: Program) -> str:
    words = re.findall(r"[a-z]+", p.entry.lower())
    return " ".join(words)


# ---------------------------------------------------------------------------
# The five tasks
# ---------------------------------------------------------------------------
def task_bug_localization(runner, programs) -> float:
    correct = total = 0
    for p in programs:
        buggy, ln = _inject_bug(p.source)
        if buggy is None:
            continue
        surpr = _line_surprisals(runner, buggy)
        if not surpr:
            continue
        pred = max(surpr, key=surpr.get)
        correct += int(pred == ln)
        total += 1
    return correct / total if total else float("nan")


def task_program_repair(runner, programs, gen) -> float:
    ok = total = 0
    for p in programs:
        buggy, ln = _inject_bug(p.source)
        if buggy is None:
            continue
        prompt = (f"The following Python function has a bug. Return the "
                  f"corrected function only.\n\n{buggy}\n")
        resp = runner.chat(prompt, gen)
        fixed = _extract_function(resp, p.entry) or resp
        ok += int(_run_tests(fixed, p.entry, p.tests) == 1.0)
        total += 1
    return ok / total if total else float("nan")


def task_code_completion(runner, programs, gen) -> float:
    ok = total = 0
    for p in programs:
        lines = p.source.rstrip().split("\n")
        if len(lines) < 3:
            continue
        prefix = "\n".join(lines[:-1]) + "\n"
        prompt = (f"Complete the final line of this Python function. Return the "
                  f"full function.\n\n{prefix}")
        resp = runner.chat(prompt, gen)
        completed = _extract_function(prefix + resp, p.entry) or _extract_function(resp, p.entry)
        if completed is None:
            total += 1
            continue
        ok += int(_run_tests(completed, p.entry, p.tests) == 1.0)
        total += 1
    return ok / total if total else float("nan")


def task_code_summarization(runner, programs, gen) -> float:
    scores = []
    for p in programs:
        prompt = (f"Describe in a few words what this Python function does.\n\n"
                  f"{p.source}\n")
        resp = runner.chat(prompt, gen)
        scores.append(_rouge1_f1(resp, _ref_summary(p)))
    return float(np.mean(scores)) if scores else float("nan")


def task_code_translation(runner, programs, gen) -> float:
    ok = total = 0
    for p in programs:
        if "for " not in p.source:
            continue
        prompt = (f"Rewrite this Python function using while loops instead of "
                  f"for loops, preserving behavior. Return the function only.\n\n"
                  f"{p.source}\n")
        resp = runner.chat(prompt, gen)
        trans = _extract_function(resp, p.entry) or resp
        ok += int(_run_tests(trans, p.entry, p.tests) == 1.0)
        total += 1
    return ok / total if total else float("nan")


TASKS = ["bug_localization", "program_repair", "code_completion",
         "code_summarization", "code_translation"]

TASK_METRIC = {
    "bug_localization": "Top-1 Loc. Acc.",
    "program_repair": "Repair Rate",
    "code_completion": "pass@1",
    "code_summarization": "ROUGE-1 F1",
    "code_translation": "Transpile pass@1",
}


def run_downstream(runner, model_name: str, gen: dict | None = None,
                   limit: int = 0) -> dict:
    gen = gen or {"do_sample": False, "max_new_tokens": 96}
    progs = partition_corpus(build_corpus(config.SEED))["test"]
    if not progs:  # fall back to whole corpus if the test split is empty
        progs = build_corpus(config.SEED)
    if limit > 0:
        progs = progs[:limit]
    scores = {
        "bug_localization": task_bug_localization(runner, progs),
        "program_repair": task_program_repair(runner, progs, gen),
        "code_completion": task_code_completion(runner, progs, gen),
        "code_summarization": task_code_summarization(runner, progs, gen),
        "code_translation": task_code_translation(runner, progs, gen),
    }
    return {"model": model_name, "scores": scores, "n_programs": len(progs)}
