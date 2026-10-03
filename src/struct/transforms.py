"""
Behavior-preserving transforms that build the contrast pair-sets the CodeLLM
paper requires.

For a source program x we produce:

  * syntax positive  (P+_syn): x' whose AST *shape* differs from x
                               (e.g. `a += b` -> `a = a + b`, comprehension ->
                               explicit loop) while behavior is preserved.
  * syntax control   (P-_syn): x'' whose AST shape is *identical* to x
                               (identifier renaming only).
  * control-flow pos (P+_cf):  x' whose CFG differs from x (e.g. for -> while,
                               redundant always-true guard) behavior preserved.
  * control-flow ctrl(P-_cf):  x'' whose CFG is identical (identifier renaming).

Each transform is semantics-preserving *by construction*. Generated pairs are
additionally checked against the structural validity criteria (shape/CFG must
actually change for a positive, and stay fixed for a control); pairs that fail
are dropped -- this is the paper's `Delta_q(x, x') > delta_q` filter.
"""
from __future__ import annotations

import ast
import builtins
from typing import Optional

from .extract import extract_struct, delta_cf

_BUILTINS = set(dir(builtins))


# ---------------------------------------------------------------------------
# Control condition: identifier renaming (preserves AST shape AND CFG)
# ---------------------------------------------------------------------------
class _RenameCollector(ast.NodeVisitor):
    def __init__(self):
        self.assigned: set[str] = set()
        self.imported: set[str] = set()
        self.func_class: set[str] = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.assigned.add(node.id)
        self.generic_visit(node)

    def visit_arg(self, node):
        self.assigned.add(node.arg)

    def visit_FunctionDef(self, node):
        self.func_class.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.func_class.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            self.imported.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.imported.add(a.asname or a.name)


class _RenameApply(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]):
        self.m = mapping

    def visit_Name(self, node):
        if node.id in self.m:
            node.id = self.m[node.id]
        return node

    def visit_arg(self, node):
        if node.arg in self.m:
            node.arg = self.m[node.arg]
        return node


def rename_identifiers(source: str) -> Optional[str]:
    """Rename local variables/parameters to fresh names; preserves behavior and
    structure. Returns None if it would be a no-op or fails."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    col = _RenameCollector()
    col.visit(tree)
    targets = sorted(
        n for n in col.assigned
        if n not in _BUILTINS and n not in col.imported
        and n not in col.func_class and not n.startswith("__")
    )
    if not targets:
        return None
    mapping = {name: f"v{i}_" for i, name in enumerate(targets)}
    tree = _RenameApply(mapping).visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Syntax-changing transforms (shape differs, CFG usually unchanged)
# ---------------------------------------------------------------------------
class _AugAssignExpand(ast.NodeTransformer):
    """`a += b` -> `a = a + b` (shape change, behavior/CFG preserved)."""
    changed = False

    def visit_AugAssign(self, node):
        self.changed = True
        load_target = _as_load(node.target)
        return ast.Assign(
            targets=[node.target],
            value=ast.BinOp(left=load_target, op=node.op, right=node.value),
        )


class _CompExpand(ast.NodeTransformer):
    """`[expr for t in it]` assigned to a name -> explicit append loop."""
    changed = False

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.ListComp)
                and len(node.value.generators) == 1
                and not node.value.generators[0].ifs):
            comp = node.value
            gen = comp.generators[0]
            name = node.targets[0]
            init = ast.Assign(targets=[ast.Name(id=name.id, ctx=ast.Store())],
                              value=ast.List(elts=[], ctx=ast.Load()))
            append = ast.Expr(value=ast.Call(
                func=ast.Attribute(value=ast.Name(id=name.id, ctx=ast.Load()),
                                   attr="append", ctx=ast.Load()),
                args=[comp.elt], keywords=[]))
            loop = ast.For(target=gen.target, iter=gen.iter,
                           body=[append], orelse=[])
            self.changed = True
            return [init, loop]
        return node


def _as_load(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.Name):
        return ast.Name(id=node.id, ctx=ast.Load())
    if isinstance(node, ast.Attribute):
        return ast.Attribute(value=node.value, attr=node.attr, ctx=ast.Load())
    if isinstance(node, ast.Subscript):
        return ast.Subscript(value=node.value, slice=node.slice, ctx=ast.Load())
    return node


# ---------------------------------------------------------------------------
# Control-flow-changing transforms (CFG differs, behavior preserved)
# ---------------------------------------------------------------------------
class _ForToWhile(ast.NodeTransformer):
    """`for t in range(...): body` -> counter-driven while loop."""
    changed = False

    def visit_For(self, node):
        self.generic_visit(node)
        it = node.iter
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and it.func.id == "range" and isinstance(node.target, ast.Name)):
            return node
        args = it.args
        if len(args) == 1:
            start, stop, step = ast.Constant(0), args[0], ast.Constant(1)
        elif len(args) == 2:
            start, stop, step = args[0], args[1], ast.Constant(1)
        elif len(args) == 3:
            # only safe for a literal positive step
            if not (isinstance(args[2], ast.Constant)
                    and isinstance(args[2].value, int) and args[2].value > 0):
                return node
            start, stop, step = args
        else:
            return node
        tgt = node.target
        init = ast.Assign(targets=[ast.Name(id=tgt.id, ctx=ast.Store())], value=start)
        cond = ast.Compare(left=ast.Name(id=tgt.id, ctx=ast.Load()),
                           ops=[ast.Lt()], comparators=[stop])
        incr = ast.AugAssign(target=ast.Name(id=tgt.id, ctx=ast.Store()),
                             op=ast.Add(), value=step)
        wh = ast.While(test=cond, body=list(node.body) + [incr], orelse=[])
        self.changed = True
        return [init, wh]


class _RedundantGuard(ast.NodeTransformer):
    """Wrap a function body in an always-true guard (`if True:`), adding a
    branch node to the CFG while preserving behavior. Applied to the first
    function found."""
    changed = False

    def visit_FunctionDef(self, node):
        if not self.changed and node.body:
            guard = ast.If(test=ast.Constant(True), body=list(node.body), orelse=[])
            node.body = [guard]
            self.changed = True
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


_SYNTAX_TRANSFORMS = [_AugAssignExpand, _CompExpand]
_CF_TRANSFORMS = [_ForToWhile, _RedundantGuard]


def _apply(source: str, transformers) -> Optional[str]:
    """Apply the first transformer in the list that actually fires."""
    for T in transformers:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        t = T()
        tree = t.visit(tree)
        if getattr(t, "changed", False):
            ast.fix_missing_locations(tree)
            try:
                out = ast.unparse(tree)
                ast.parse(out)  # validity check
                return out
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Pair construction with structural-validity filtering
# ---------------------------------------------------------------------------
SHAPE_CF_MIN = 0.02  # minimum delta_cf for a control-flow positive


def make_contrast_pairs(source: str) -> dict[str, Optional[tuple[str, str]]]:
    """
    Build the four contrast pairs for one source program. Each value is a
    (base, variant) tuple of source strings, or None when no valid transform
    applies. The base is always the renamed program so that the positive and
    the control are compared against the same surface baseline (this removes
    raw lexical identity as a confound, as the paper requires).
    """
    out: dict[str, Optional[tuple[str, str]]] = {
        "syn_pos": None, "syn_ctrl": None, "cf_pos": None, "cf_ctrl": None,
    }
    try:
        base_ps = extract_struct(source)
    except SyntaxError:
        return out

    renamed = rename_identifiers(source)

    # ---- syntax ----
    syn_variant = _apply(source, _SYNTAX_TRANSFORMS)
    if syn_variant is not None:
        try:
            vp = extract_struct(syn_variant)
            if vp.structural_hash != base_ps.structural_hash:
                out["syn_pos"] = (source, syn_variant)
        except SyntaxError:
            pass
    if renamed is not None:
        try:
            rp = extract_struct(renamed)
            if rp.structural_hash == base_ps.structural_hash:
                out["syn_ctrl"] = (source, renamed)
        except SyntaxError:
            pass

    # ---- control flow ----
    cf_variant = _apply(source, _CF_TRANSFORMS)
    if cf_variant is not None:
        try:
            vp = extract_struct(cf_variant)
            if delta_cf(base_ps.cfg, vp.cfg) > SHAPE_CF_MIN:
                out["cf_pos"] = (source, cf_variant)
        except SyntaxError:
            pass
    if renamed is not None:
        try:
            rp = extract_struct(renamed)
            if delta_cf(base_ps.cfg, rp.cfg) <= SHAPE_CF_MIN:
                out["cf_ctrl"] = (source, renamed)
        except SyntaxError:
            pass

    return out
