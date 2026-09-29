"""
Run the accuracy / robustness track: for every problem family, present each
variant to the model, record the raw answer, extract and grade it.

A "family" is one base problem plus its meaning-preserving and stress variants.
The baseline variant (kind == "baseline") is the untransformed reference.
"""
from __future__ import annotations
import json
from .answer_extract import extract, grade
from .transforms import auto_variants


def load_problems(path: str) -> list:
    families = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                families.append(json.loads(line))
    return families


def expand_family(family: dict) -> list:
    """Return the full list of variants, adding any auto-generated ones."""
    variants = list(family.get("variants", []))
    base = next((v for v in variants if v.get("kind") == "baseline"), None)
    if base and family.get("auto"):
        for v in auto_variants(base["prompt"], family["auto"]):
            v = {**v, "answer": base["answer"]}     # answer-preserving by construction
            variants.append(v)
    return variants


def run_accuracy(runner, families: list, gen: dict, model_label: str) -> list:
    records = []
    for fam in families:
        for v in expand_family(fam):
            raw = runner.chat(v["prompt"], gen)
            got = extract(raw, fam["answer_type"])
            ok = grade(got, v["answer"], fam["answer_type"])
            records.append({
                "model": model_label,
                "family_id": fam["id"],
                "capability": fam.get("capability", ""),
                "kind": v.get("kind", "variant"),
                "answer_type": fam["answer_type"],
                "prompt": v["prompt"],
                "gold": v["answer"],
                "raw_output": raw,
                "extracted": got,
                "is_correct": ok,
            })
    return records


def run_latent(runner, families: list, use_chat_template: bool, model_label: str) -> list:
    """Profile intrinsic signals on each variant prompt (Track 2)."""
    records = []
    for fam in families:
        for v in expand_family(fam):
            sig = runner.signals(v["prompt"], use_chat_template=use_chat_template)
            records.append({
                "model": model_label,
                "family_id": fam["id"],
                "kind": v.get("kind", "variant"),
                "min_entropy": sig["min_entropy"],
                "mean_entropy": sig["mean_entropy"],
                "max_ER": sig["max_ER"],
                "max_PR": sig["max_PR"],
                "last_layer_ER": sig["last_layer_ER"],
                "last_layer_PR": sig["last_layer_PR"],
                # Per-layer profiles — kept so #2 (hourglass plot) and every
                # later layerwise analysis can read them straight from the
                # results JSONL, no re-running the model. LPP paper
                # §"Formal definition"; Supplement §2.2, Figure 4.
                "ER_by_layer": sig["ER_by_layer"],
                "PR_by_layer": sig["PR_by_layer"],
                "num_tokens": sig["num_tokens"],
                "num_layers": sig["num_layers"],
            })
    return records


def save_jsonl(records: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
