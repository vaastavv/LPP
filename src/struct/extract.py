"""
Structural extraction for Python source, stdlib-only (`ast`).

For a program x we extract the three structural views the CodeLLM paper needs:

  * syntax units   -- token spans over which Phi_syn aggregates hidden states
  * control-flow   -- a CFG (for the normalized graph distance delta_cf) and
                      the token spans of control-flow headers (Phi_cf)
  * data flow      -- definition->use pairs (positives) and the full list of
                      use occurrences (from which token-distance-matched
                      negatives are drawn for DFBS)

Everything is expressed as *character offsets* into the original source string
so it can later be aligned to model tokens via the tokenizer offset mapping,
independent of any particular tokenizer.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Spans and char-offset conversion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """A half-open character range [start, end) into the source string."""
    start: int
    end: int

    def __post_init__(self):
        if self.end < self.start:
            object.__setattr__(self, "end", self.start)

    @property
    def length(self) -> int:
        return self.end - self.start

    def text(self, source: str) -> str:
        return source[self.start:self.end]


def line_start_offsets(source: str) -> list[int]:
    """`offsets[L-1]` is the char offset at the start of 1-indexed line L."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _col_to_char(lineno: int, col: int, line_starts: list[int]) -> int:
    idx = lineno - 1
    if idx < 0:
        idx = 0
    if idx >= len(line_starts):
        idx = len(line_starts) - 1
    return line_starts[idx] + col


def node_span(node: ast.AST, line_starts: list[int], source: str) -> Optional[Span]:
    """Full character span of an AST node, or None if location is unavailable."""
    if not hasattr(node, "lineno"):
        return None
    start = _col_to_char(node.lineno, node.col_offset, line_starts)
    end_lineno = getattr(node, "end_lineno", None)
    end_col = getattr(node, "end_col_offset", None)
    if end_lineno is None or end_col is None:
        end = min(start + 1, len(source))
    else:
        end = _col_to_char(end_lineno, end_col, line_starts)
    return Span(start, min(end, len(source)))


def header_span(node: ast.AST, line_starts: list[int], source: str) -> Optional[Span]:
    """
    Span of a compound statement's *header* only (the `if ...:`, `for ...:`,
    `while ...:` line), rather than the whole block. Used for Phi_cf so the
    control-flow aggregation focuses on branch/loop tokens.
    """
    full = node_span(node, line_starts, source)
    if full is None:
        return None
    # The header ends at the colon that opens the suite. Find the first ':'
    # at paren-depth 0 after the keyword, within the node span.
    seg = source[full.start:full.end]
    depth = 0
    for i, ch in enumerate(seg):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            return Span(full.start, full.start + i + 1)
    # Fallback: first physical line of the node.
    nl = seg.find("\n")
    if nl == -1:
        return full
    return Span(full.start, full.start + nl)


# ---------------------------------------------------------------------------
# Control-flow graph (lightweight, stdlib-only)
# ---------------------------------------------------------------------------
@dataclass
class CFG:
    """A minimal control-flow graph: integer node ids + directed edges."""
    n_nodes: int = 0
    edges: list[tuple[int, int]] = field(default_factory=list)

    def add_node(self) -> int:
        nid = self.n_nodes
        self.n_nodes += 1
        return nid

    def add_edge(self, a: int, b: int) -> None:
        if a is not None and b is not None:
            self.edges.append((a, b))

    def feature_vector(self) -> list[float]:
        """
        Structural signature used by `delta_cf`. Captures the quantities that
        distinguish control-flow shapes: size, branching, merging, looping and
        cyclomatic complexity.
        """
        n = self.n_nodes
        e = len(self.edges)
        out_deg: dict[int, int] = {}
        in_deg: dict[int, int] = {}
        for a, b in self.edges:
            out_deg[a] = out_deg.get(a, 0) + 1
            in_deg[b] = in_deg.get(b, 0) + 1
        branches = sum(1 for d in out_deg.values() if d >= 2)
        merges = sum(1 for d in in_deg.values() if d >= 2)
        back_edges = sum(1 for a, b in self.edges if b <= a)  # loop/back edges
        cyclomatic = e - n + 2  # single connected component assumption
        return [float(n), float(e), float(branches), float(merges),
                float(back_edges), float(cyclomatic)]


def delta_cf(g1: CFG, g2: CFG) -> float:
    """
    Normalized control-flow graph distance in [0, 1]. A feature-based proxy for
    graph edit distance (exact GED is intractable); the features are the
    size/branch/merge/loop/cyclomatic signature of each CFG. Normalized L1.
    """
    f1 = g1.feature_vector()
    f2 = g2.feature_vector()
    num = sum(abs(a - b) for a, b in zip(f1, f2))
    den = sum(abs(a) + abs(b) for a, b in zip(f1, f2)) + 1e-9
    return num / den


class _CFGBuilder:
    """
    Builds an approximate statement-level CFG from an AST. Captures sequential
    flow, if/else branching, loop back-edges, and early exits well enough for
    the structural feature signature used by delta_cf.
    """

    def __init__(self):
        self.cfg = CFG()

    def build(self, body: list[ast.stmt]) -> CFG:
        entry = self.cfg.add_node()
        exits = self._block(body, [entry])
        term = self.cfg.add_node()
        for e in exits:
            self.cfg.add_edge(e, term)
        return self.cfg

    def _block(self, body: list[ast.stmt], preds: list[int]) -> list[int]:
        """Wire a statement list; return the list of exit node ids."""
        cur = preds
        for stmt in body:
            cur = self._stmt(stmt, cur)
        return cur

    def _stmt(self, stmt: ast.stmt, preds: list[int]) -> list[int]:
        node = self.cfg.add_node()
        for p in preds:
            self.cfg.add_edge(p, node)

        if isinstance(stmt, (ast.If,)):
            then_exits = self._block(stmt.body, [node])
            if stmt.orelse:
                else_exits = self._block(stmt.orelse, [node])
            else:
                else_exits = [node]
            return then_exits + else_exits

        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            body_exits = self._block(stmt.body, [node])
            for e in body_exits:
                self.cfg.add_edge(e, node)  # back edge to loop header
            exits = [node]  # loop may not execute / falls through
            if stmt.orelse:
                exits = self._block(stmt.orelse, exits)
            return exits

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Nested definition: profile its body as a sub-region but keep it
            # attached so its complexity is reflected in the graph.
            self._block(stmt.body, [node])
            return [node]

        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            return self._block(stmt.body, [node])

        if isinstance(stmt, ast.Try):
            body_exits = self._block(stmt.body, [node])
            all_exits = list(body_exits)
            for handler in stmt.handlers:
                all_exits += self._block(handler.body, [node])
            if stmt.orelse:
                all_exits = self._block(stmt.orelse, all_exits)
            if stmt.finalbody:
                all_exits = self._block(stmt.finalbody, all_exits)
            return all_exits

        if isinstance(stmt, (ast.Return, ast.Raise)):
            return []  # terminates this path

        # Break/Continue: simplified -- terminate the local path.
        if isinstance(stmt, (ast.Break, ast.Continue)):
            return []

        return [node]


def build_cfg(tree: ast.Module) -> CFG:
    return _CFGBuilder().build(tree.body)


# ---------------------------------------------------------------------------
# Data flow: definition -> use pairs
# ---------------------------------------------------------------------------
@dataclass
class DefUse:
    name: str
    def_span: Span
    use_span: Span


@dataclass
class UseOcc:
    name: str
    span: Span
    def_span: Optional[Span]  # the definition it reads, if resolvable


@dataclass
class NameOcc:
    name: str
    span: Span
    is_def: bool
    scope: int
    order: int  # source-order rank


def _collect_names(tree: ast.Module, line_starts: list[int], source: str) -> list[NameOcc]:
    """
    Collect definition and use occurrences of identifiers in source order,
    tagged with an enclosing-scope id (module=0, each function gets its own).
    """
    occ: list[NameOcc] = []
    scope_counter = [0]

    def visit(node: ast.AST, scope: int):
        # Assignment: value (uses) is read before targets (defs).
        if isinstance(node, ast.Assign):
            visit(node.value, scope)
            for t in node.targets:
                visit(t, scope)
            return
        if isinstance(node, ast.AugAssign):
            # target is both a use and a def
            visit(node.value, scope)
            if isinstance(node.target, ast.Name):
                sp = node_span(node.target, line_starts, source)
                if sp:
                    occ.append(NameOcc(node.target.id, sp, False, scope, 0))
                    occ.append(NameOcc(node.target.id, sp, True, scope, 0))
            else:
                visit(node.target, scope)
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                visit(node.value, scope)
            visit(node.target, scope)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # the function name is a def in the enclosing scope
            sp = node_span(node, line_starts, source)
            if sp:
                occ.append(NameOcc(node.name, Span(sp.start, min(sp.start + len(node.name), sp.end)),
                                   True, scope, 0))
            scope_counter[0] += 1
            inner = scope_counter[0]
            # decorators/defaults evaluated in enclosing scope
            for d in node.decorator_list:
                visit(d, scope)
            for default in node.args.defaults + [d for d in node.args.kw_defaults if d]:
                visit(default, scope)
            # parameters are defs in the inner scope
            for a in _iter_args(node.args):
                sp = node_span(a, line_starts, source)
                if sp:
                    occ.append(NameOcc(a.arg, sp, True, inner, 0))
            for s in node.body:
                visit(s, inner)
            return
        if isinstance(node, ast.ClassDef):
            sp = node_span(node, line_starts, source)
            if sp:
                occ.append(NameOcc(node.name, Span(sp.start, min(sp.start + len(node.name), sp.end)),
                                   True, scope, 0))
            for s in node.body:
                visit(s, scope)
            return
        if isinstance(node, ast.Name):
            sp = node_span(node, line_starts, source)
            if sp:
                occ.append(NameOcc(node.id, sp, isinstance(node.ctx, ast.Store), scope, 0))
            return

        # Generic descent in child order.
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    for s in tree.body:
        visit(s, 0)

    # Assign source-order ranks by span start.
    occ.sort(key=lambda o: (o.span.start, 0 if o.is_def else 1))
    for i, o in enumerate(occ):
        o.order = i
    return occ


def _iter_args(args: ast.arguments):
    seen = []
    for a in getattr(args, "posonlyargs", []) or []:
        seen.append(a)
    for a in args.args:
        seen.append(a)
    if args.vararg:
        seen.append(args.vararg)
    for a in args.kwonlyargs:
        seen.append(a)
    if args.kwarg:
        seen.append(args.kwarg)
    return seen


def resolve_def_use(occ: list[NameOcc]) -> tuple[list[DefUse], list[UseOcc]]:
    """
    Match each use to the latest preceding definition of the same name, within
    the same scope first, then the module scope (lexical fallback).
    """
    def_uses: list[DefUse] = []
    uses: list[UseOcc] = []
    # latest def span per (scope, name)
    latest: dict[tuple[int, str], Span] = {}
    for o in occ:
        if o.is_def:
            latest[(o.scope, o.name)] = o.span
        else:
            d = latest.get((o.scope, o.name))
            if d is None:
                d = latest.get((0, o.name))  # module-scope fallback
            uses.append(UseOcc(o.name, o.span, d))
            if d is not None:
                def_uses.append(DefUse(o.name, d, o.span))
    return def_uses, uses


# ---------------------------------------------------------------------------
# Unit span collectors (for Phi_syn / Phi_cf aggregation)
# ---------------------------------------------------------------------------
_CF_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
             ast.With, ast.AsyncWith, ast.Return, ast.Break,
             ast.Continue, ast.Raise)


def syntax_units(tree: ast.Module, line_starts: list[int], source: str) -> list[Span]:
    """Phi_syn aggregates over the whole program's code span (global syntax)."""
    # Use the span from the first to the last statement (skips leading comments).
    spans = [node_span(s, line_starts, source) for s in tree.body]
    spans = [s for s in spans if s]
    if not spans:
        return [Span(0, len(source))]
    return [Span(min(s.start for s in spans), max(s.end for s in spans))]


def controlflow_units(tree: ast.Module, line_starts: list[int], source: str) -> list[Span]:
    """Phi_cf aggregates over control-flow statement headers."""
    spans: list[Span] = []
    for node in ast.walk(tree):
        if isinstance(node, _CF_NODES):
            sp = header_span(node, line_starts, source)
            if sp:
                spans.append(sp)
    return spans


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------
@dataclass
class ProgramStruct:
    source: str
    tree: ast.Module
    cfg: CFG
    def_uses: list[DefUse]
    uses: list[UseOcc]
    syn_units: list[Span]
    cf_units: list[Span]

    @property
    def structural_hash(self) -> str:
        """A shape signature (identifier-insensitive) used to test whether two
        programs differ syntactically. Dumps the AST with field structure but
        without identifier names or constant values."""
        return _ast_shape(self.tree)


def _ast_shape(tree: ast.AST) -> str:
    parts: list[str] = []

    def walk(node: ast.AST):
        parts.append(type(node).__name__)
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(tree)
    return ",".join(parts)


def extract_struct(source: str) -> ProgramStruct:
    """Parse `source` and extract all structural views. Raises SyntaxError on
    unparsable input (callers should filter the corpus accordingly)."""
    tree = ast.parse(source)
    line_starts = line_start_offsets(source)
    cfg = build_cfg(tree)
    occ = _collect_names(tree, line_starts, source)
    def_uses, uses = resolve_def_use(occ)
    syn = syntax_units(tree, line_starts, source)
    cf = controlflow_units(tree, line_starts, source)
    return ProgramStruct(
        source=source, tree=tree, cfg=cfg,
        def_uses=def_uses, uses=uses, syn_units=syn, cf_units=cf,
    )
