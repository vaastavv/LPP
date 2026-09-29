"""
Latent (intrinsic) metrics, following the Latent Performance Profiling paper.

Three quantities are computed from a single forward pass:

  1. Next-token entropy  H(x<=t) = -sum_v p(v|x<=t) log p(v|x<=t)
     We report the MINIMUM entropy across token positions (the "uncertainty
     floor": the least uncertain the model becomes as context accumulates).

  2. Effective Rank (ER) of the hidden-state covariance C:
        s_i  = eigenvalues of C (nonneg, since C is PSD symmetric)
        s~_i = s_i / sum_j s_j
        ER   = exp( -sum_i s~_i log s~_i )
     ER estimates how many dimensions are actively used. Reported as the
     MAXIMUM across layers.

  3. Participation Ratio (PR) of C:
        PR = (sum_i s_i)^2 / (sum_i s_i^2)
     PR measures how evenly variance is spread. Reported as the MAXIMUM
     across layers.

All heavy math is done in float32 for numerical stability, whatever dtype the
model ran in.
"""
from __future__ import annotations
import torch

EPS = 1e-12


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """logits: [T, V] -> per-position next-token entropy [T] (nats)."""
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def _covariance(H: torch.Tensor) -> torch.Tensor:
    """H: [T, d] hidden states -> [d, d] covariance across the T tokens."""
    H = H.float()
    H = H - H.mean(dim=0, keepdim=True)
    T = H.shape[0]
    return (H.t() @ H) / max(T - 1, 1)


def _eigenvalues(H: torch.Tensor) -> torch.Tensor:
    C = _covariance(H)
    ev = torch.linalg.eigvalsh(C)          # ascending, real (C symmetric)
    return ev.clamp(min=0.0)               # remove tiny negatives from fp error


def effective_rank(H: torch.Tensor) -> float:
    s = _eigenvalues(H)
    total = s.sum()
    if total <= EPS:
        return 0.0
    p = s / total
    p = p[p > EPS]
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy))


def participation_ratio(H: torch.Tensor) -> float:
    s = _eigenvalues(H)
    num = s.sum() ** 2
    den = (s ** 2).sum()
    if den <= EPS:
        return 0.0
    return float(num / den)


def profile_hidden_states(hidden_states) -> dict:
    """
    hidden_states: tuple of tensors each [1, T, d] (transformers,
    output_hidden_states=True). Returns per-layer ER/PR and their maxima.
    """
    er_by_layer, pr_by_layer = [], []
    for layer in hidden_states:
        H = layer.squeeze(0)               # [T, d]
        if H.shape[0] < 2:                 # need >=2 tokens for a covariance
            er_by_layer.append(0.0)
            pr_by_layer.append(0.0)
            continue
        er_by_layer.append(effective_rank(H))
        pr_by_layer.append(participation_ratio(H))
    return {
        "ER_by_layer": er_by_layer,
        "PR_by_layer": pr_by_layer,
        "max_ER": max(er_by_layer) if er_by_layer else 0.0,
        "max_PR": max(pr_by_layer) if pr_by_layer else 0.0,
        "last_layer_ER": er_by_layer[-1] if er_by_layer else 0.0,
        "last_layer_PR": pr_by_layer[-1] if pr_by_layer else 0.0,
    }
