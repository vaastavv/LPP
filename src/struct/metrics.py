"""
Structural metrics over a corpus for one model: SRS, CFS (+normalized), DFBS.

Reads a list of source programs, builds contrast pairs, runs the model, and
aggregates layer-wise profiles plus the raw per-pair data the statistical
tables need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .extract import extract_struct, delta_cf
from .transforms import make_contrast_pairs
from .profile import (forward_pass, cosine_dist_rows, euclidean_dist_rows,
                      dataflow_records, DefUseRecord)

EPS = 1e-8


@dataclass
class StructResult:
    model: str
    num_layers: int                          # L+1 (includes embedding layer)
    SRS: np.ndarray = None                    # [L+1]
    CFS: np.ndarray = None
    CFS_norm: np.ndarray = None
    DFBS: np.ndarray = None
    # raw per-pair layerwise distances (for stats / effect sizes)
    srs_pos: np.ndarray = None                # [n+, L+1]
    srs_ctrl: np.ndarray = None
    cfs_pos: np.ndarray = None
    cfs_ctrl: np.ndarray = None
    cf_deltas: np.ndarray = None              # [n_cf_pos] graph distances
    dfbs_records: list = field(default_factory=list)   # list[DefUseRecord]
    peak: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def L(self) -> int:
        return self.num_layers - 1

    def layer_axis(self) -> np.ndarray:
        return np.arange(self.num_layers) / max(self.L, 1)


class StructProfiler:
    def __init__(self, runner, dist: str = "cosine", pooling: str = "mean",
                 eta: int = 8, max_neg: int = 8, max_tokens: int = 512):
        self.runner = runner
        self.dist = dist
        self.pooling = pooling
        self.eta = eta
        self.max_neg = max_neg
        self.max_tokens = max_tokens
        self._zcache: dict[str, Optional[tuple]] = {}

    def _dist_rows(self, a, b):
        if self.dist == "cosine":
            return cosine_dist_rows(a, b)
        if self.dist == "euclidean":
            return euclidean_dist_rows(a, b)
        raise ValueError(self.dist)

    def _z(self, source: str):
        """(z_syn, z_cf) pooled layer matrices for a source, cached."""
        if source in self._zcache:
            return self._zcache[source]
        try:
            st = extract_struct(source)
        except SyntaxError:
            self._zcache[source] = None
            return None
        hp = forward_pass(self.runner, source, self.max_tokens)
        if hp is None:
            self._zcache[source] = None
            return None
        z_syn = hp.pooled(st.syn_units, self.pooling)
        z_cf = hp.pooled(st.cf_units, self.pooling)
        self._zcache[source] = (z_syn, z_cf)
        return self._zcache[source]

    def _pair_dist(self, pair, which: str) -> Optional[np.ndarray]:
        base, var = pair
        zb = self._z(base)
        zv = self._z(var)
        if zb is None or zv is None:
            return None
        i = 0 if which == "syn" else 1
        return self._dist_rows(zb[i], zv[i])

    def run(self, programs: list[str], model_name: str,
            progress: bool = True) -> StructResult:
        srs_pos, srs_ctrl = [], []
        cfs_pos, cfs_ctrl, cf_deltas = [], [], []
        dfbs_records: list[DefUseRecord] = []
        num_layers = None

        it = enumerate(programs)
        for i, src in it:
            if progress and i % 10 == 0:
                print(f"  [{model_name}] program {i+1}/{len(programs)}", flush=True)
            try:
                st = extract_struct(src)
            except SyntaxError:
                continue

            # --- data flow (needs full hidden states of the base program) ---
            hp = forward_pass(self.runner, src, self.max_tokens)
            if hp is None:
                continue
            if num_layers is None:
                num_layers = hp.num_layers
            recs = dataflow_records(hp, st, self.eta, self.max_neg)
            dfbs_records.extend(recs)
            # seed the z-cache for the base from this pass (avoid re-running it)
            if src not in self._zcache:
                self._zcache[src] = (hp.pooled(st.syn_units, self.pooling),
                                     hp.pooled(st.cf_units, self.pooling))
            del hp

            # --- syntax / control-flow contrast pairs ---
            pairs = make_contrast_pairs(src)
            if pairs["syn_pos"]:
                d = self._pair_dist(pairs["syn_pos"], "syn")
                if d is not None:
                    srs_pos.append(d)
            if pairs["syn_ctrl"]:
                d = self._pair_dist(pairs["syn_ctrl"], "syn")
                if d is not None:
                    srs_ctrl.append(d)
            if pairs["cf_pos"]:
                d = self._pair_dist(pairs["cf_pos"], "cf")
                if d is not None:
                    cfs_pos.append(d)
                    try:
                        g1 = extract_struct(pairs["cf_pos"][0]).cfg
                        g2 = extract_struct(pairs["cf_pos"][1]).cfg
                        cf_deltas.append(delta_cf(g1, g2))
                    except SyntaxError:
                        cf_deltas.append(np.nan)
            if pairs["cf_ctrl"]:
                d = self._pair_dist(pairs["cf_ctrl"], "cf")
                if d is not None:
                    cfs_ctrl.append(d)

        if num_layers is None:
            raise RuntimeError("no program produced hidden states")

        res = StructResult(model=model_name, num_layers=num_layers)
        res.srs_pos = _stack(srs_pos, num_layers)
        res.srs_ctrl = _stack(srs_ctrl, num_layers)
        res.cfs_pos = _stack(cfs_pos, num_layers)
        res.cfs_ctrl = _stack(cfs_ctrl, num_layers)
        res.cf_deltas = np.array(cf_deltas, dtype=np.float32)
        res.dfbs_records = dfbs_records

        res.SRS = _ratio(res.srs_pos, res.srs_ctrl)
        res.CFS = _ratio(res.cfs_pos, res.cfs_ctrl)
        res.CFS_norm = _cfs_norm(res.cfs_pos, res.cf_deltas)
        res.DFBS = _dfbs_profile(dfbs_records, num_layers)

        res.peak = _peaks(res)
        res.meta = {
            "n_syn_pos": len(srs_pos), "n_syn_ctrl": len(srs_ctrl),
            "n_cf_pos": len(cfs_pos), "n_cf_ctrl": len(cfs_ctrl),
            "n_defuse": len(dfbs_records),
            "dist": self.dist, "pooling": self.pooling,
            "eta": self.eta, "max_neg": self.max_neg,
        }
        return res


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _stack(rows: list[np.ndarray], num_layers: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, num_layers), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def _ratio(pos: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """SRS/CFS^(l) = E[D over P+] / (E[D over P-] + eps)."""
    if pos.shape[0] == 0:
        return np.zeros(pos.shape[1] if pos.ndim == 2 else 0, dtype=np.float32)
    ep = pos.mean(axis=0)
    ec = ctrl.mean(axis=0) if ctrl.shape[0] else np.zeros_like(ep)
    return ep / (ec + EPS)


def _cfs_norm(cf_pos: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """CFS_norm^(l) = E[ D_cf^(l) / (delta_cf + eps) ] over P+ pairs."""
    if cf_pos.shape[0] == 0:
        return np.zeros(cf_pos.shape[1] if cf_pos.ndim == 2 else 0, dtype=np.float32)
    d = deltas.copy()
    d = np.where(np.isnan(d), np.nanmean(d) if np.isfinite(np.nanmean(d)) else 1.0, d)
    scale = (d + EPS).reshape(-1, 1)
    return (cf_pos / scale).mean(axis=0)


def _dfbs_profile(records: list[DefUseRecord], num_layers: int) -> np.ndarray:
    if not records:
        return np.zeros(num_layers, dtype=np.float32)
    deltas = np.stack([r.B_by_layer - r.N_by_layer for r in records], axis=0)
    return deltas.mean(axis=0).astype(np.float32)


def save_result(res: StructResult, path: str) -> None:
    import json
    def arr(a):
        return None if a is None else np.asarray(a).tolist()
    payload = {
        "model": res.model, "num_layers": res.num_layers,
        "SRS": arr(res.SRS), "CFS": arr(res.CFS),
        "CFS_norm": arr(res.CFS_norm), "DFBS": arr(res.DFBS),
        "srs_pos": arr(res.srs_pos), "srs_ctrl": arr(res.srs_ctrl),
        "cfs_pos": arr(res.cfs_pos), "cfs_ctrl": arr(res.cfs_ctrl),
        "cf_deltas": arr(res.cf_deltas),
        "dfbs_records": [
            {"tok_dist": int(r.tok_dist),
             "B": np.asarray(r.B_by_layer).tolist(),
             "N": np.asarray(r.N_by_layer).tolist(),
             "n_neg": int(r.n_negatives)}
            for r in res.dfbs_records
        ],
        "peak": res.peak, "meta": res.meta,
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def load_result(path: str) -> StructResult:
    import json
    with open(path) as f:
        d = json.load(f)
    def arr(x):
        return None if x is None else np.array(x, dtype=np.float32)
    res = StructResult(model=d["model"], num_layers=d["num_layers"])
    res.SRS = arr(d["SRS"]); res.CFS = arr(d["CFS"])
    res.CFS_norm = arr(d["CFS_norm"]); res.DFBS = arr(d["DFBS"])
    res.srs_pos = arr(d["srs_pos"]); res.srs_ctrl = arr(d["srs_ctrl"])
    res.cfs_pos = arr(d["cfs_pos"]); res.cfs_ctrl = arr(d["cfs_ctrl"])
    res.cf_deltas = arr(d["cf_deltas"])
    res.dfbs_records = [
        DefUseRecord(tok_dist=r["tok_dist"],
                     B_by_layer=np.array(r["B"], dtype=np.float32),
                     N_by_layer=np.array(r["N"], dtype=np.float32),
                     n_negatives=r["n_neg"])
        for r in d["dfbs_records"]
    ]
    res.peak = {k: v for k, v in d["peak"].items()}
    res.meta = d["meta"]
    return res


def _peaks(res: StructResult) -> dict:
    """l* = argmax over transformer layers (excluding embedding layer 0)."""
    def peak(arr):
        if arr is None or len(arr) <= 1:
            return 0, 0.0
        sub = arr[1:]                      # skip embedding layer
        l = int(np.argmax(sub)) + 1
        return l, l / max(res.L, 1)
    out = {}
    for q, arr in [("syn", res.SRS), ("cf", res.CFS), ("df", res.DFBS)]:
        l, lam = peak(arr)
        out[q] = {"l_star": l, "lambda_star": lam}
    return out
