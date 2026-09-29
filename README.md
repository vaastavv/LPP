# LLM Understanding vs Statistical Pattern Prediction — Experiment Codebase

Starter codebase for the progress-report project "Investigating Understanding
and Reasoning in Large Language Models". It runs the two tracks from the
methodology on one model object:

- Track 1 (accuracy / robustness): give the model a base problem and its
  meaning-preserving variants, then measure whether the correct answer survives
  the surface change. This tests understanding vs surface pattern-matching.
- Track 2 (latent profiling): from a single forward pass, compute the
  uncertainty floor (min next-token entropy), effective rank (ER) and
  participation ratio (PR) of the hidden states, following the Latent
  Performance Profiling paper.

You can start today with the Llama 1B you already have; add the other models to
`config.py` as they finish downloading.

## Layout

    config.py            models, decoding settings, paths
    data/problems.jsonl  base problems + their variants (5 seed families)
    src/
      model_runner.py    load a model; chat() for text, signals() for latents
      transforms.py      answer-preserving surface variants (rename, NoOp, reorder)
      answer_extract.py  pull the final answer from a reply and grade it
      latent_metrics.py  entropy, effective rank, participation ratio
      experiment.py      orchestrates both tracks over the dataset
      metrics.py         accuracy / consistency / degradation aggregation
    run_experiments.py   CLI: Track 1
    run_latent.py        CLI: Track 2
    tests/test_offline.py tests that need no model download

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Llama 3.2 is a gated model on Hugging Face: request access on its model page,
then `huggingface-cli login`. Alternatively point `HF_MODEL` at a local folder
of the weights you already downloaded:

    export HF_MODEL=/path/to/Llama-3.2-1B-Instruct

## Run

Track 1 (accuracy / robustness):

    python run_experiments.py --model llama-1b

Writes `results/accuracy_llama-1b.jsonl` (one row per variant, raw output kept)
and prints a summary: baseline accuracy, accuracy by variant kind, answer
consistency (1.0 = surface-invariant), and degradation in points vs baseline.

Track 2 (latent profiling):

    python run_latent.py --model llama-1b

Writes `results/latent_llama-1b.jsonl` and prints the LPP-style extremal
profile (uncertainty floor, max ER, max PR).

Offline tests (no download needed):

    python tests/test_offline.py

## Adding models

Uncomment lines in `config.MODELS` as each downloads, keeping the Llama / Qwen /
Mistral families so results stay comparable to the LPP paper. Small models
(<=3B) profile fine on a laptop CPU; push 7B+ latent runs to a free Colab or
Kaggle GPU. On 8 GB RAM stick to 1.5B-3B; on 16 GB an 8B model at load is fine
for Track 1.

## Metric definitions (as implemented)

- Accuracy: fraction of correct final answers per variant kind.
- Consistency: within a family, (most common extracted answer count) / (number
  of variants). High = the model's answer does not change with the surface.
- Degradation: baseline accuracy minus a variant kind's accuracy, in points.
- Uncertainty floor: minimum next-token entropy across positions.
- Effective rank / participation ratio: from the eigenvalues of the hidden-state
  covariance; see `src/latent_metrics.py` for the exact formulas.

## The seed dataset

Five small families, one per planned experiment: arithmetic with paraphrase /
number-swap / NoOp variants, an ambiguity-resolution multiple-choice item, a
context-following (counter-factual) item, and a novel-combination ("blip of a
blip") item. Replace and grow these — they exist to prove the pipeline runs end
to end.

## Notes and honest limitations

- Answer extraction and grading are heuristic; the raw output is always saved so
  you can spot-check and improve the parser.
- Track 2 needs open-weight models (hidden states); it will not work through a
  chat-only API.
- Greedy decoding is used so any variation comes from the prompt, not sampling.
