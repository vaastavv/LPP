"""
Per-program hidden-state profiling for the structural metrics.

A single forward pass over a source program yields, for every transformer
layer l:

  * z_syn^(l)(x)  -- hidden states pooled over the syntax units
  * z_cf^(l)(x)   -- hidden states pooled over the control-flow headers
  * for every definition->use pair, the per-layer similarity of the
    definition and use token representations, plus matched non-dependent
    negatives (used by DFBS and its distance strata)

The source is tokenized WITHOUT special tokens or a chat template so the
tokenizer offset mapping aligns directly with character offsets in the source,
making the span->token mapping tokenizer-independent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .extract import ProgramStruct, Span


# ---------------------------------------------------------------------------
# Aggregation operators (Phi_q). Configurable for the robustness analysis.
# ---------------------------------------------------------------------------
def _pool(vectors: np.ndarray, how: str) -> np.ndarray:
    if vectors.shape[0] == 0:
        return np.zeros(vectors.shape[1], dtype=np.float32)
    if how == "mean":
        return vectors.mean(axis=0)
    if how == "max":
        return vectors.max(axis=0)
    if how == "sum":
        return vectors.sum(axis=0)
    if how == "last":
        return vectors[-1]
    raise ValueError(f"unknown pooling {how!r}")


def _span_token_idx(offsets: list[tuple[int, int]], spans: list[Span]) -> list[int]:
    """Token indices whose character span overlaps any of `spans`."""
    idxs: list[int] = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:  # special / empty token
            continue
        for sp in spans:
            if a < sp.end and b > sp.start:  # overlap
                idxs.append(i)
                break
    return idxs


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
@dataclass
class HiddenPass:
    hidden: list[np.ndarray]          # L+1 arrays, each [T, d] float32
    offsets: list[tuple[int, int]]    # per-token (char_start, char_end)
    num_layers: int                   # L+1 including embedding layer

    def pooled(self, spans: list[Span], how: str = "mean") -> np.ndarray:
        """Return [L+1, d] pooling of hidden states over the given spans."""
        idx = _span_token_idx(self.offsets, spans)
        if not idx:
            idx = list(range(len(self.offsets)))  # fallback: whole sequence
        out = np.stack([_pool(layer[idx], how) for layer in self.hidden], axis=0)
        return out

    def token_vectors(self, span: Span) -> Optional[np.ndarray]:
        """[L+1, d] representation of a single span (mean over its tokens)."""
        idx = _span_token_idx(self.offsets, [span])
        if not idx:
            return None
        return np.stack([layer[idx].mean(axis=0) for layer in self.hidden], axis=0)


@torch.no_grad()
def forward_pass(runner, source: str, max_tokens: int = 512) -> Optional[HiddenPass]:
    """Run one forward pass; return hidden states + char-offset mapping."""
    tok = runner.tokenizer
    try:
        enc = tok(source, return_offsets_mapping=True,
                  add_special_tokens=False, return_tensors="pt",
                  truncation=True, max_length=max_tokens)
    except (TypeError, NotImplementedError):
        return None  # slow tokenizer without offset mapping
    offsets = [tuple(int(x) for x in o) for o in enc["offset_mapping"][0].tolist()]
    input_ids = enc["input_ids"].to(runner.device)
    if input_ids.shape[1] < 2:
        return None
    out = runner.model(input_ids, output_hidden_states=True)
    hidden = [h.squeeze(0).float().cpu().numpy() for h in out.hidden_states]
    return HiddenPass(hidden=hidden, offsets=offsets, num_layers=len(hidden))


# ---------------------------------------------------------------------------
# Similarity / distance
# ---------------------------------------------------------------------------
def cosine_sim_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of two [L, d] matrices -> [L]."""
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.where((na * nb) > 1e-12, na * nb, 1e-12)
    return (a * b).sum(axis=1) / denom


def cosine_dist_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """d(a,b) = 1 - cos(a,b), row-wise over layers. [L]."""
    return 1.0 - cosine_sim_rows(a, b)


def euclidean_dist_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


# ---------------------------------------------------------------------------
# Data-flow binding for one program
# ---------------------------------------------------------------------------
@dataclass
class DefUseRecord:
    tok_dist: int                 # |token(def) - token(use)|
    B_by_layer: np.ndarray        # [L+1] positive binding similarity
    N_by_layer: np.ndarray        # [L+1] mean negative similarity
    n_negatives: int


def dataflow_records(hp: HiddenPass, struct: ProgramStruct,
                     eta: int = 8, max_neg: int = 8) -> list[DefUseRecord]:
    """
    For each definition->use positive, compute the per-layer binding similarity
    and the matched negative similarity. Negatives are other use occurrences
    NOT reading from the same definition, matched on token distance within eta.
    """
    # Representative token index per span = first overlapping token.
    def rep_tok(span: Span) -> Optional[int]:
        idx = _span_token_idx(hp.offsets, [span])
        return idx[0] if idx else None

    # Precompute token vectors and rep-token index for all use occurrences.
    use_tok = []
    for u in struct.uses:
        rt = rep_tok(u.span)
        if rt is not None:
            use_tok.append((u, rt))

    records: list[DefUseRecord] = []
    for du in struct.def_uses:
        d_vec = hp.token_vectors(du.def_span)
        u_vec = hp.token_vectors(du.use_span)
        d_tok = rep_tok(du.def_span)
        u_tok = rep_tok(du.use_span)
        if d_vec is None or u_vec is None or d_tok is None or u_tok is None:
            continue
        pos_dist = abs(u_tok - d_tok)
        B = cosine_sim_rows(d_vec, u_vec)

        # candidate negatives: uses not at the same position as this use and
        # not reading the same definition, matched on |dist - pos_dist| <= eta
        negs = []
        for u, rt in use_tok:
            if rt == u_tok:
                continue
            if u.def_span is not None and u.def_span == du.def_span:
                continue  # dependent on same def -> not a valid negative
            nd = abs(rt - d_tok)
            if abs(nd - pos_dist) <= eta:
                negs.append(rt)
        if not negs:
            continue
        negs = negs[:max_neg]
        neg_sims = []
        for rt in negs:
            nv = np.stack([layer[rt] for layer in hp.hidden], axis=0)  # [L+1,d]
            neg_sims.append(cosine_sim_rows(d_vec, nv))
        N = np.mean(neg_sims, axis=0)
        records.append(DefUseRecord(tok_dist=pos_dist, B_by_layer=B,
                                    N_by_layer=N, n_negatives=len(negs)))
    return records
