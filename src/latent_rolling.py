"""
Rolling-context entropy schedule (LPP paper item #5).

Paper: Chakraborty et al., *Latent Performance Profiling of Large Language
Models*, arXiv:2605.30018v2.
  - Table 1, "Extremal statistic and interpretation": "minimum entropy over the
    rolling-context schedule".
  - Materials & Methods, "Uncertainty floor" (page 12):
        H_t = -sum_v P_theta(v | x_<=t) log P_theta(v | x_<=t)
  - Materials & Methods, "Datasets and prompts for intrinsic metrics" (page 12):
    default prefix schedule {10, 20, 30, 40, 50, 60, 70, 80, 90, 100} tokens.

Difference from ModelRunner.signals(): signals() runs ONE forward pass over the
full prompt and reports min entropy across all token positions. The paper
instead evaluates entropy at position L across a SERIES of forward passes, one
per prefix length L in the schedule (each on text[:L]), then reports the min
across L. This module implements that rolling schedule and, for free (the
forward pass is already run), also records the ER/PR profile of each partial
context so item #7 (prefix-length sensitivity) can read them straight from the
results file.
"""
from __future__ import annotations

import torch

from .latent_metrics import token_entropy, profile_hidden_states

DEFAULT_PREFIX_LENGTHS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


@torch.no_grad()
def signals_rolling(
    runner: "ModelRunner",
    text: str,
    prefix_lengths: tuple[int, ...] = DEFAULT_PREFIX_LENGTHS,
    use_chat_template: bool = False,
) -> dict:
    """
    For each L in prefix_lengths, run one forward pass on the first L tokens of
    `text` and record the next-token entropy at the last position (index L-1),
    plus the max ER/PR of that partial context.

    Paper: Materials & Methods "Uncertainty floor" / "Datasets and prompts for
    intrinsic metrics" (page 12).

    Returns:
      entropy_by_prefix     {L: float}  entropy at position L for each L
      max_ER_by_prefix      {L: float}  max effective rank over layers at prefix L
      max_PR_by_prefix      {L: float}  max participation ratio over layers at L
      min_entropy_rolling   float       min entropy over the schedule
      mean_entropy_rolling  float       mean entropy over the schedule
      prefix_lengths_used   list[int]   L values actually reached (an L larger
                                        than the tokenized length is skipped)
    """
    # Tokenize once, then slice per prefix length. Chat-templating is optional;
    # the paper profiles raw task-agnostic text, so the default is off.
    if use_chat_template:
        try:
            enc = runner.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
            input_ids = enc["input_ids"].to(runner.device)
        except Exception:
            input_ids = runner.tokenizer(text, return_tensors="pt")["input_ids"].to(runner.device)
    else:
        input_ids = runner.tokenizer(text, return_tensors="pt")["input_ids"].to(runner.device)

    total = input_ids.shape[1]

    entropy_by_prefix: dict[int, float] = {}
    max_er_by_prefix: dict[int, float] = {}
    max_pr_by_prefix: dict[int, float] = {}
    used: list[int] = []

    for L in prefix_lengths:
        if L < 1 or L > total:
            continue  # not enough tokens to reach this prefix length
        prefix_ids = input_ids[:, :L]
        out = runner.model(prefix_ids, output_hidden_states=True)
        # Entropy of the next-token distribution at the LAST prefix position.
        last_logits = out.logits.squeeze(0)[L - 1]        # [V]
        entropy_by_prefix[L] = float(token_entropy(last_logits))
        prof = profile_hidden_states(out.hidden_states)
        max_er_by_prefix[L] = prof["max_ER"]
        max_pr_by_prefix[L] = prof["max_PR"]
        used.append(L)

    ent_vals = list(entropy_by_prefix.values())
    return {
        "entropy_by_prefix": entropy_by_prefix,
        "max_ER_by_prefix": max_er_by_prefix,
        "max_PR_by_prefix": max_pr_by_prefix,
        "min_entropy_rolling": min(ent_vals) if ent_vals else 0.0,
        "mean_entropy_rolling": (sum(ent_vals) / len(ent_vals)) if ent_vals else 0.0,
        "prefix_lengths_used": used,
    }
