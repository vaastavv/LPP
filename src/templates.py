"""
GSM-Symbolic-style template expansion.

You write ONE template with placeholders and an answer formula; this module
samples many instantiations and builds a variant family:

  baseline : one instantiation (the reference)
  numeric  : same structure, resampled numbers, answer RECOMPUTED from the formula
  rename   : same numbers, different names (answer unchanged)
  noop     : baseline plus an irrelevant-but-plausible clause (answer unchanged)

The answer/constraint formulas are evaluated with a restricted AST evaluator
(only arithmetic, comparisons, and min/max/abs/round) so no arbitrary code runs.

Template schema (one JSON object per line in data/templates.jsonl):

  {
    "id": "apples",
    "capability": "arithmetic",
    "answer_type": "numeric",
    "template": "{name} has {a} apples.{noop} {name} buys {b} more, then gives away {c}. How many apples does {name} have now? End with 'Answer: <number>'.",
    "vars": {
      "name": {"type": "name"},
      "a": {"type": "int", "low": 5, "high": 30},
      "b": {"type": "int", "low": 1, "high": 15},
      "c": {"type": "int", "low": 1, "high": 10}
    },
    "constraints": ["a >= c", "a + b - c > 0"],
    "answer": "a + b - c",
    "noop_templates": ["The apples are a mix of red and green."]
  }

Notes:
  - Put a {noop} placeholder where a distractor sentence should go (it is empty
    for every non-noop variant). Leading space is added automatically.
  - Every var used in the template must appear in "vars".
"""
from __future__ import annotations
import ast

NAME_BANK = ["Ravi", "Meera", "Arjun", "Priya", "Sam", "Lena",
             "Omar", "Nadia", "Tom", "Aisha", "Ken", "Rosa"]

_ALLOWED_FUNCS = {"min": min, "max": max, "abs": abs, "round": round}
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_CMP = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


def safe_eval(expr: str, env: dict):
    """Evaluate an arithmetic/boolean expression using only names in env."""
    return _ev(ast.parse(expr, mode="eval").body, env)


def _ev(node, env):
    if isinstance(node, ast.BinOp) and isinstance(node.op, _BINOPS):
        return _binop(node.op, _ev(node.left, env), _ev(node.right, env))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _ev(node.operand, env)
        return +v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants allowed")
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(f"unknown name in formula: {node.id}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _ALLOWED_FUNCS:
        return _ALLOWED_FUNCS[node.func.id](*[_ev(a, env) for a in node.args])
    if isinstance(node, ast.Compare):
        left = _ev(node.left, env)
        ok = True
        for op, comp in zip(node.ops, node.comparators):
            if not isinstance(op, _CMP):
                raise ValueError("disallowed comparison")
            right = _ev(comp, env)
            ok = ok and _cmp(op, left, right)
            left = right
        return ok
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        vals = [_ev(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    raise ValueError(f"disallowed expression: {ast.dump(node)}")


def _binop(op, l, r):
    if isinstance(op, ast.Add): return l + r
    if isinstance(op, ast.Sub): return l - r
    if isinstance(op, ast.Mult): return l * r
    if isinstance(op, ast.Div): return l / r
    if isinstance(op, ast.FloorDiv): return l // r
    if isinstance(op, ast.Mod): return l % r
    if isinstance(op, ast.Pow): return l ** r
    raise ValueError("bad binop")


def _cmp(op, l, r):
    if isinstance(op, ast.Lt): return l < r
    if isinstance(op, ast.LtE): return l <= r
    if isinstance(op, ast.Gt): return l > r
    if isinstance(op, ast.GtE): return l >= r
    if isinstance(op, ast.Eq): return l == r
    if isinstance(op, ast.NotEq): return l != r
    raise ValueError("bad comparator")


def _sample_var(spec: dict, rng):
    t = spec.get("type", "int")
    if t == "int":
        step = spec.get("step", 1)
        return rng.choice(list(range(spec["low"], spec["high"] + 1, step)))
    if t == "choice":
        return rng.choice(spec["choices"])
    if t == "name":
        return rng.choice(spec.get("choices", NAME_BANK))
    raise ValueError(f"unknown var type: {t}")


def _sample_assignment(vars, constraints, rng, max_tries=500):
    for _ in range(max_tries):
        env = {k: _sample_var(v, rng) for k, v in vars.items()}
        if all(safe_eval(c, env) for c in constraints):
            return env
    raise RuntimeError("could not satisfy constraints; loosen ranges/constraints")


def _fmt(x):
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x)


def generate_family(t: dict, rng, n_numeric: int = 5, n_rename: int = 1,
                    do_noop: bool = True) -> dict:
    vars = t["vars"]
    constraints = t.get("constraints", [])
    ans_expr = t["answer"]
    name_keys = [k for k, v in vars.items() if v.get("type") == "name"]
    num_keys = [k for k in vars if k not in name_keys]

    def render(env, kind):
        e = dict(env)
        e.setdefault("noop", "")
        prompt = t["template"].format(**e)
        return {"kind": kind, "prompt": prompt,
                "answer": _fmt(safe_eval(ans_expr, e))}

    base = _sample_assignment(vars, constraints, rng)
    variants = [render(base, "baseline")]

    # numeric variants: same names, resampled numbers, recomputed answer
    seen = {tuple(base[k] for k in num_keys)}
    tries = 0
    while sum(v["kind"] == "numeric" for v in variants) < n_numeric \
            and tries < n_numeric * 100:
        tries += 1
        env = dict(base)
        for k in num_keys:
            env[k] = _sample_var(vars[k], rng)
        if not all(safe_eval(c, env) for c in constraints):
            continue
        sig = tuple(env[k] for k in num_keys)
        if sig in seen:
            continue
        seen.add(sig)
        variants.append(render(env, "numeric"))

    # rename variants: same numbers, different name(s)
    for _ in range(n_rename):
        if not name_keys:
            break
        env = dict(base)
        for k in name_keys:
            for _try in range(20):
                cand = _sample_var(vars[k], rng)
                if cand != base[k]:
                    env[k] = cand
                    break
        variants.append(render(env, "rename"))

    # noop variants: baseline numbers + an irrelevant clause
    if do_noop:
        for noop in t.get("noop_templates", []):
            env = dict(base)
            env["noop"] = " " + noop.format(**base)
            variants.append(render(env, "noop"))

    return {
        "id": t["id"],
        "capability": t.get("capability", ""),
        "answer_type": t.get("answer_type", "numeric"),
        "variants": variants,
    }
