"""Calibration and confidence-threshold selection (model-agnostic: works on any logits).

Procedure (validation data only - the test set is never touched here):
  1. Temperature scaling: fit a single T > 0 minimising NLL on in-domain validation logits.
  2. Threshold on the calibrated max-softmax probability. A prediction should be *accepted* iff it is an
     in-domain query that the model classifies correctly; in-domain errors and out-of-domain queries
     should be *rejected*. We pick the threshold maximising balanced accuracy (Youden's J) of that
     accept/reject decision, subject to in-domain coverage >= min_in_domain_coverage.
  3. Emergency threshold: a separate, lower threshold on P(emergency_assistance), chosen as the lowest
     value whose false-emergency rate on non-emergency validation (+OOD) stays <= the configured maximum.
     Missing an emergency is costlier than a false alarm, so this maximises emergency recall.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(logits, T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    res = minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            bins.append(dict(lo=float(lo), hi=float(hi), count=int(m.sum()),
                             accuracy=float(correct[m].mean()), confidence=float(conf[m].mean())))
        else:
            bins.append(dict(lo=float(lo), hi=float(hi), count=0, accuracy=None, confidence=None))
    return float(ece), bins


def threshold_curve(conf_in: np.ndarray, correct_in: np.ndarray, conf_ood: np.ndarray,
                    step: float = 0.01) -> list[dict]:
    rows = []
    for t in np.round(np.arange(0.0, 1.0 + 1e-9, step), 4):
        acc_in = conf_in >= t
        acc_ood = conf_ood >= t if len(conf_ood) else np.array([], bool)
        n_acc = acc_in.sum()
        positives = correct_in  # should be accepted
        tp = (acc_in & positives).sum()
        tn = (~acc_in & ~positives).sum() + (~acc_ood).sum()
        n_pos = positives.sum()
        n_neg = (~positives).sum() + len(conf_ood)
        tpr = tp / n_pos if n_pos else 0.0
        tnr = tn / n_neg if n_neg else 0.0
        rows.append(dict(
            threshold=float(t),
            in_domain_coverage=float(acc_in.mean()),
            selective_accuracy=float((acc_in & correct_in).sum() / n_acc) if n_acc else 1.0,
            ood_false_accept_rate=float(acc_ood.mean()) if len(conf_ood) else 0.0,
            balanced_accuracy=float((tpr + tnr) / 2),
        ))
    return rows


def select_threshold(curve: list[dict], min_coverage: float) -> dict:
    feasible = [r for r in curve if r["in_domain_coverage"] >= min_coverage] or curve
    best = max(feasible, key=lambda r: (round(r["balanced_accuracy"], 6), r["in_domain_coverage"]))
    return best


def select_emergency_threshold(p_emerg_neg: np.ndarray, p_emerg_pos: np.ndarray, max_fpr: float,
                               default: float, step: float = 0.01) -> dict:
    """Lowest threshold with false-emergency rate <= max_fpr; never above the general threshold."""
    for t in np.round(np.arange(step, 1.0 + 1e-9, step), 4):
        if t > default:
            break
        fpr = float((p_emerg_neg >= t).mean()) if len(p_emerg_neg) else 0.0
        if fpr <= max_fpr:
            rec = float((p_emerg_pos >= t).mean()) if len(p_emerg_pos) else 0.0
            return dict(threshold=float(t), false_emergency_rate=fpr, emergency_recall=rec)
    rec = float((p_emerg_pos >= default).mean()) if len(p_emerg_pos) else 0.0
    fpr = float((p_emerg_neg >= default).mean()) if len(p_emerg_neg) else 0.0
    return dict(threshold=float(default), false_emergency_rate=fpr, emergency_recall=rec)


@dataclass
class Calibration:
    temperature: float
    threshold: float
    emergency_threshold: float
    selection: dict
    emergency_selection: dict
    ece_before: float
    ece_after: float
    method: str = ("temperature scaling (val NLL) + max-softmax threshold maximising balanced accept/reject "
                   "accuracy on validation + OOD-validation, subject to min in-domain coverage")

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate(val_logits: np.ndarray, val_y: np.ndarray, ood_logits: np.ndarray, labels: list[str],
              cfg: dict) -> tuple[Calibration, dict]:
    ccfg = cfg["confidence"]
    T = fit_temperature(val_logits, val_y)
    p_raw = softmax(val_logits)
    p_val = softmax(val_logits, T)
    p_ood = softmax(ood_logits, T) if len(ood_logits) else np.zeros((0, len(labels)))
    ece_b, bins_b = expected_calibration_error(p_raw, val_y, ccfg["ece_bins"])
    ece_a, bins_a = expected_calibration_error(p_val, val_y, ccfg["ece_bins"])
    curve = threshold_curve(p_val.max(1), p_val.argmax(1) == val_y, p_ood.max(1), ccfg["threshold_grid_step"])
    sel = select_threshold(curve, ccfg["min_in_domain_coverage"])
    e = labels.index("emergency_assistance")
    is_e = val_y == e
    neg = np.concatenate([p_val[~is_e, e], p_ood[:, e]])
    esel = select_emergency_threshold(neg, p_val[is_e, e], ccfg["max_emergency_false_positive_rate"],
                                      sel["threshold"], ccfg["threshold_grid_step"])
    cal = Calibration(temperature=T, threshold=sel["threshold"], emergency_threshold=esel["threshold"],
                      selection=sel, emergency_selection=esel, ece_before=ece_b, ece_after=ece_a)
    return cal, {"curve": curve, "reliability_before": bins_b, "reliability_after": bins_a}
