# Remaining work to close the gap with the LPP paper

Companion to [`PLAN.md`](PLAN.md) (the gap analysis). This file is the
step-by-step build guide for the eleven items still open. Each section is
self-contained: paper citation, exact files to touch, function signatures,
integration points, runtime cost, and how to verify it works.

Paper: Chakraborty et al., *Latent Performance Profiling of Large Language
Models*, arXiv:2605.30018v2, 29 May 2026 — full PDF at
[`docs/LPP_paper.pdf`](docs/LPP_paper.pdf).

## Progress so far

| # | Item | Status |
|---|------|--------|
| 1 | Per-layer ER/PR persisted in latent records | **Done** ([`16704d0`](https://github.com/vaastavv/LPP/commit/16704d0)) |
| 5 | Rolling-context entropy schedule | **Code done** — `src/latent_rolling.py`, `run_latent_rolling.py` |
| 6 | Real calibration set: 100 Alpaca prompts | **Code done** — `src/calibration.py`, `run_calibration.py` |
| 11 | SPC synthetic task | **Code done** — `src/spc_task.py`, `run_spc.py` |
| 7 | Sensitivity: prefix length | **Code done** — `plot_prefix_sensitivity` in `latent_analysis.py` |
| 2 | Layerwise ER/PR plot + hourglass detector | **Code done** — `latent_analysis.py` |
| 8 | Sensitivity: context length | **Code done** — `plot_context_sensitivity` in `latent_analysis.py` |
| 9 | Sensitivity: sample size | **Code done** — `plot_sample_size_sensitivity` in `latent_analysis.py` |
| 10 | Sensitivity: dataset (Alpaca/Dolly/WikiText) | **Code done** — `load_dataset_sample` + `plot_dataset_sensitivity` |
| 3 | Cross-model Spearman correlations | **Code done** — `correlation_table` in `analyze_results.py`; needs ≥3 models |
| 4 | Aggregation-invariance check | **Code done** — `aggregation_invariance` in `latent_analysis.py`; needs ≥3 models |
| 12 | AR synthetic task | **Code done** — `src/ar_task.py`, `run_ar.py` |

> **Code done** means the module and CLI are written, byte-compile clean, and
> the pure-Python logic is covered by `tests/test_tasks.py` (14 tests, all
> passing offline). The model-driven *runs* that produce `results/*.jsonl` and
> the plots still need to be launched by hand — install the deps
> (`pip install -r requirements.txt`, which now includes `scipy` and
> `datasets`) and run the commands in each item's section below. #3 and #4 only
> become meaningful once ≥3 models have results.

## Recommended order

`#5 → #6 → #2 → #11 → #7 → #8 → #9 → #10 → #3 → #4 → #12`

`#5` and `#6` together are the paper's actual definition of the latent
profile — do them first. `#2` is a cheap payoff on #1's data. `#11` is the
easier synthetic task; `#12` is the heavy one and can wait. `#3`/`#4` are
one-file additions to `analyze_results.py` and only need enough models to be
meaningful.

## Ground rules for every item

- **Do not rewrite** working code in `src/latent_metrics.py`,
  `src/model_runner.py`, or the runners. If a change is needed there, show
  the diff first and get approval.
- Reuse existing functions: `src.metrics.summarize` for aggregation,
  `src.latent_metrics.token_entropy` / `.profile_hidden_states` for the
  three metrics, `src.model_runner.ModelRunner.signals` for forward passes.
- Follow the paper's exact formulas and defaults. Cite the section, equation
  or figure in a comment next to the code that implements it.
- Only add libraries already sensible for this project: `numpy`, `pandas`,
  `matplotlib`, `scipy`, `datasets` (for #6). Add them to
  `requirements.txt` when used.
- Keep everything CPU-runnable on the two small open models
  (`qwen-0.5b`, `qwen-1.5b`). Note in the item where a step realistically
  needs GPU (Colab / Kaggle).
- Do not run long experiments unattended. Write the code, print the command,
  and let the user launch the run.

## Environment prerequisites

- Two Qwen models are cached and wired into `config.MODELS`:
  `qwen-0.5b` and `qwen-1.5b`. Both run without HF auth.
- Llama-3.2-1B is in `config.MODELS` but gated on HuggingFace. Run
  `huggingface-cli login` once, then either `run_experiments.py --model
  llama-1b` and `run_latent.py --model llama-1b` work.
- The `.venv` inside the project is a fresh Python 3.12 environment with
  `torch`, `transformers`, `accelerate`, `numpy`, `tqdm`, `pandas`,
  `matplotlib`. Add new deps to `requirements.txt` and `pip install` them
  into `.venv`.
- Windows: HF cache lives at `~/.cache/huggingface/hub/`. Symlink warnings
  are harmless (already visible on runs).

---

## #5 · Rolling-context entropy schedule

**Paper.** Table 1, "Extremal statistic and interpretation": *"minimum
entropy over the rolling-context schedule"*. Materials & Methods
§"Uncertainty floor" (page 12), equation
$H_t = -\sum_v P_\theta(v \mid x_{\le t}) \log P_\theta(v \mid x_{\le t})$.
Materials & Methods §"Datasets and prompts for intrinsic metrics" (page 12):
default prefix schedule `{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}` tokens.

**Why this is not the same as today's `min_entropy`.** `signals()` runs one
forward pass on the full prompt, computes entropy at every token position,
and returns `min` across positions. The paper computes entropy at position L
across a **series of forward passes on `text[:L]`** for L in the schedule,
then reports `min` across L.

**Files to create (no edits to existing files).**

- `src/latent_rolling.py` — new module. One function:

  ```python
  def signals_rolling(
      runner: "ModelRunner",
      text: str,
      prefix_lengths: tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
      use_chat_template: bool = False,
  ) -> dict:
      """
      For each L in prefix_lengths, run one forward pass on the first L tokens
      of `text` and record the next-token entropy at the last position.
      Returns:
        entropy_by_prefix     {L: float}  entropy at position L for each L
        min_entropy_rolling   float       min over the schedule
        mean_entropy_rolling  float       mean over the schedule
        prefix_lengths_used   list[int]   L values actually reached (some may
                                          be skipped if the input tokenizes to
                                          fewer tokens)
      """
  ```

  Implementation notes:
  - Reuse `runner.tokenizer` to tokenize once, then slice `input_ids[:, :L]`
    for each L.
  - Reuse `src.latent_metrics.token_entropy` on the logits row at position
    `L - 1` (last position of the prefix).
  - Cite the paper section in a comment above the function.

- `run_latent_rolling.py` — new project-root CLI, mirrors `run_latent.py`:

  ```
  python run_latent_rolling.py --model qwen-0.5b [--data data/problems.jsonl]
  ```

  Iterates families / variants, calls `signals_rolling`, writes
  `results/latent_rolling_<model>.jsonl` with one row per variant:

  ```json
  {"model": "qwen-0.5b", "family_id": "arith_apples", "kind": "baseline",
   "entropy_by_prefix": {"10": 2.13, "20": 1.87, ...},
   "min_entropy_rolling": 0.31, "mean_entropy_rolling": 1.42,
   "prefix_lengths_used": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
  ```

**Cost.** 10 forward passes per prompt × 18 prompts × qwen-0.5b on CPU:
~3–5 minutes. qwen-1.5b: ~10 min. 7B+: GPU.

**Verify.**
```
python run_latent_rolling.py --model qwen-0.5b
python -c "import json; r=json.loads(open('results/latent_rolling_qwen-0.5b.jsonl').readline()); print(sorted(r['entropy_by_prefix'].keys()), r['min_entropy_rolling'])"
```
Should print `['10', '20', '30', ..., '100']` (as strings) and a positive
float.

---

## #6 · Real calibration set — 100 Alpaca prompts

**Paper.** Materials & Methods §"Datasets and prompts for intrinsic metrics"
(page 12): *"Intrinsic metrics are computed using 100 randomly-sampled
task-agnostic texts from the Alpaca dataset. […] All experiments set the
random seed to 42."* §"Implementation details" (page 12): *"maximum context
length of 200 tokens, with a designated prefix length of 100."*

**Why.** Current Track 2 runs on our five problem families. That inflates
`min_entropy` because arithmetic prompts push the model to a confident
numeric prediction. The paper's latent profile is measured on generic text.

**Files to create.**

- `src/calibration.py` — new module. Loads and caches 100 Alpaca prompts
  offline:

  ```python
  ALPACA_HF_ID = "tatsu-lab/alpaca"
  CACHE_DIR = os.path.join(config.ROOT, "data", "calibration")

  def load_alpaca(n: int = 100, seed: int = 42) -> list[str]:
      """Return `n` `instruction + input` strings from Alpaca, cached locally
      as data/calibration/alpaca_<n>_seed<seed>.jsonl."""
  ```

  Implementation:
  - `pip install datasets`; add to `requirements.txt`.
  - `from datasets import load_dataset; ds = load_dataset("tatsu-lab/alpaca")`.
  - `random.Random(seed).sample(range(len(ds["train"])), n)`.
  - `text = row["instruction"] + ("\n" + row["input"] if row["input"] else "")`.
  - Write the sampled list to the cache file; on later calls, read the cache
    directly (offline reproducibility).

- `run_calibration.py` — new project-root CLI:

  ```
  python run_calibration.py --model qwen-0.5b [--n 100] [--context-length 200]
  ```

  Runs `runner.signals(text)` (and, once #5 lands, also `signals_rolling`)
  on each of the 100 prompts, truncated to `--context-length` tokens.
  Writes `results/calibration_<dataset>_<model>.jsonl` with the same shape
  as latent records plus `prompt_id` and `dataset`.

**Cost.** 100 forward passes on qwen-0.5b at 200 tokens on CPU: ~2–5 min.

**Verify.**
```
python run_calibration.py --model qwen-0.5b --n 10   # smoke test
python -c "import json; recs=[json.loads(l) for l in open('results/calibration_alpaca_qwen-0.5b.jsonl')]; print(len(recs), recs[0]['dataset'])"
```

---

## #2 · Layerwise ER / PR plot + hourglass detector

**Paper.** Results §"Stable latent patterns and their origins" (page 10);
Supplement §2.2, Figure 4 (page 20): *"PR and ER as functions of normalized
layer depth"* with a pronounced *"hourglass"* dip.

**Files to create (no edits to existing).**

- `latent_analysis.py` — new project-root script. Reads
  `results/latent_*.jsonl` (and, when #6 exists, `results/calibration_*.jsonl`),
  averages `ER_by_layer` and `PR_by_layer` across records per model, and:
  - Plots ER vs. normalized depth (`layer_index / (num_layers - 1)`), one
    line per model, in `results/plots/layerwise_ER.png`.
  - Same for PR in `results/plots/layerwise_PR.png`.
  - Computes a per-model **hourglass score**:

    ```python
    def hourglass_score(vals: list[float]) -> float:
        """Positive when endpoints are higher than middle third — i.e. an
        hourglass. LPP paper Supp. §2.2."""
        n = len(vals)
        endpoints_mean = (vals[0] + vals[-1]) / 2
        mid = vals[n // 3 : 2 * n // 3]
        middle_min = min(mid) if mid else vals[n // 2]
        return endpoints_mean - middle_min
    ```

  - Emits `results/hourglass.csv` with `model, hourglass_ER, hourglass_PR`.

**Cost.** Pure post-hoc on existing JSONLs. Seconds.

**Verify.**
```
python latent_analysis.py
open results/plots/layerwise_ER.png results/plots/layerwise_PR.png
cat results/hourglass.csv
```
For qwen-0.5b (25 hidden states) the ER curve should show high values at
the ends, lower in the middle. `hourglass_ER` should be positive.

---

## #11 · Symbolic Pattern Completion (SPC) task

**Paper.** Results §"LPP-informed tasks for model evaluation" (page 6);
Supplement §1.1.2 (page 18) and Algorithm 2; Table 2 example (page 9);
Evaluation: Materials & Methods §"Evaluating LLMs" (page 13): *"For SPC we
compute character-level F1 and average across samples. […] 10 in-context
examples. `max_new_tokens=8`."*

**Files to create.**

- `src/spc_task.py` — new module.

  ```python
  ALPHABET = ["A", "B", "C", "D", "E", "F"]
  SEQ_LEN = 12         # paper: 12
  MASK_LEN = 3         # paper: last 3 tokens are masked

  def generate_spc(n: int = 100, seed: int = 42) -> list[dict]:
      """Return n items: {'prompt': str, 'gold': str, 'rule': str}."""
      # rules per paper Alg 2 / §1.1.2:
      #   'alternation'      ABABABAB…            length L over 2 symbols
      #   'mirror'           ABCCBA               palindrome
      #   'progression'      ABCD ABCD            fixed cycle over k symbols
      #   'modular_increment' e.g. A B D G K … step increases by 1 each time
      #   'nested'           compose two of the above
      ...

  def prompt_template(sequence_prefix: str, in_context_examples: list[tuple[str, str]]) -> str:
      """Table 2 format:
           You are given a symbolic sequence. Continue it by writing exactly
           the next 3 symbols, without spaces or explanations.
           Sequence: <prefix>
           Answer:
      Prepend `in_context_examples` (10 per paper, each a full
      Sequence/Answer pair)."""

  def char_f1(pred: str, gold: str) -> float:
      """Character-level F1 over the multiset of characters (paper §Evaluating LLMs)."""

  def grade_spc(records: list[dict]) -> dict:
      return {"n": len(records), "mean_char_f1": mean(r["char_f1"] for r in records)}
  ```

- `run_spc.py` — new CLI:

  ```
  python run_spc.py --model qwen-0.5b [--n 100] [--seed 42]
  ```

  For each generated item, build a 10-shot prompt (10 other items sampled
  from the same generator, seed-reproducible), call `runner.chat(prompt,
  {"do_sample": False, "max_new_tokens": 8, "repetition_penalty": 1.0})`,
  compute char-F1, write `results/spc_<model>.jsonl` and print the summary.

**Cost.** 100 short generations on qwen-0.5b on CPU: ~5–10 min.

**Verify.**
```
python run_spc.py --model qwen-0.5b --n 20      # smoke test
python -c "import json; r=json.loads(open('results/spc_qwen-0.5b.jsonl').readline()); print(r['rule'], r['prompt'][-40:], r['pred'], r['gold'], r['char_f1'])"
```

---

## #7 · Sensitivity: prefix length

**Paper.** Materials & Methods §"Datasets and prompts" (page 12): schedule
`{10, 20, …, 100}`. Figure 4A (page 8): entropy / PR / ER as functions of
prefix length, one line per model.

**Depends on #5** (rolling schedule) and **#6** (Alpaca set) — the plot
reads `entropy_by_prefix` from `results/latent_rolling_*.jsonl` or
`results/calibration_*.jsonl` (whichever ran with the rolling schedule).

**Extend** `latent_analysis.py` (created in #2) with:

```python
def plot_prefix_sensitivity(rolling_by_model: dict[str, list[dict]],
                            out_dir: str) -> None:
    """Figure 4A analogue. One line per model per metric. LPP §Datasets."""
```

Reads `entropy_by_prefix` per prompt, averages across prompts per model,
and plots mean entropy vs. prefix length. Same shape for PR and ER once
`signals_rolling` also collects those per prefix (extension: have
`signals_rolling` return `max_ER_by_prefix` and `max_PR_by_prefix` by
calling `profile_hidden_states` on each partial forward pass — cheap since
the forward pass is already being done).

Output: `results/plots/sensitivity_prefix.png`.

---

## #8 · Sensitivity: context length

**Paper.** Materials & Methods §"Datasets and prompts": context lengths
`{50, 100, 200, 500}`. Figure 4B (page 8).

**Depends on #6.** Add `--context-length` sweep to `run_calibration.py`:

```
for L in 50 100 200 500; do
  python run_calibration.py --model qwen-0.5b --context-length $L
done
```

Each writes `results/calibration_alpaca_qwen-0.5b_ctx<L>.jsonl`. Then
extend `latent_analysis.py`:

```python
def plot_context_sensitivity(models: list[str],
                             context_lengths=(50, 100, 200, 500)) -> None:
    """Figure 4B analogue."""
```

Output: `results/plots/sensitivity_context.png`.

---

## #9 · Sensitivity: sample size

**Paper.** Supplement §2.1, Figure 2 (page 21): sample sizes
`{10, 100, 500, 1000}`, Alpaca. *"Values stabilize quickly with increasing
sample size."*

**Depends on #6.** Two ways:

1. **Cheap (recommended if the calibration set is exactly 100):**
   subsample `results/calibration_alpaca_qwen-0.5b.jsonl` post-hoc at
   sizes `{10, 100}` (can't go higher than what was run).
2. **Faithful:** re-run `run_calibration.py --n 1000` once, then
   subsample to `{10, 100, 500, 1000}` from that pool.

Extend `latent_analysis.py`:

```python
def plot_sample_size_sensitivity(records: list[dict],
                                 sizes=(10, 100, 500, 1000)) -> None:
```

Output: `results/plots/sensitivity_sample_size.png`.

**Note.** 1000 Alpaca forward passes on qwen-1.5b on CPU is ~30 min; on
qwen-0.5b ~10 min. GPU-optional.

---

## #10 · Sensitivity: dataset

**Paper.** Supplement §2.1, Figure 3 (page 21): Alpaca / Dolly / WikiText,
same 100 prompts each. *"Relative model rankings remain stable across
datasets."*

**Depends on #6.** Extend `src/calibration.py`:

```python
def load_dataset_sample(name: str, n: int = 100, seed: int = 42) -> list[str]:
    # name in {"alpaca", "dolly", "wikitext"}
    # dolly:     databricks/databricks-dolly-15k     (instruction + context)
    # wikitext:  wikitext / wikitext-103-raw-v1     (train text, filter empties)
```

Run:
```
python run_calibration.py --model qwen-0.5b --dataset alpaca
python run_calibration.py --model qwen-0.5b --dataset dolly
python run_calibration.py --model qwen-0.5b --dataset wikitext
```

Extend `latent_analysis.py` with `plot_dataset_sensitivity` — grouped bar
of entropy / PR / ER per dataset per model, or a 1×3 grid of one-line-per-
model plots.

Output: `results/plots/sensitivity_dataset.png`.

---

## #3 · Cross-model Spearman correlations

**Paper.** Results §"Intrinsic metrics versus extrinsic performance"
(page 4); Figure 2D (page 5): *"Spearman correlations between LPP metrics
(entropy floor, max-ER, max-PR) and scores on MMLU-PRO, BBH, and IFEval."*

**Depends on nothing structurally; needs ≥3 models in `results/` to be
meaningful.**

**Extend** `analyze_results.py`:

```python
from scipy.stats import spearmanr   # add scipy>=1.10 to requirements.txt

def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (latent metric × extrinsic metric) with rho and p."""
    latent = ["min_entropy", "max_ER", "max_PR"]
    extrinsic = ["baseline_accuracy", "answer_consistency"]
    # optionally: read external MMLU-PRO / BBH / IFEval from a CSV the user
    # populates by hand (paper values from open-llm-leaderboard).
    ...
```

Emit `results/correlations.csv` and print to console. Skip with a message
if fewer than 3 models are present.

---

## #4 · Aggregation-invariance check

**Paper.** Discussion §"LPP complements traditional benchmarks" (page 9);
Supplement §2.3, Figure 5: *"model rankings … invariant to the choice of
aggregation method (means, medians, or extremal values)"*.

**Depends on nothing structurally; needs ≥3 models to be meaningful.**

**Extend** `latent_analysis.py`:

```python
from scipy.stats import kendalltau

def aggregation_invariance(latent_by_model: dict[str, list[dict]]) -> pd.DataFrame:
    """For each metric ∈ {entropy, ER, PR} × aggregation ∈ {min, mean, median, max},
    rank the models and compare rankings across aggregators with Kendall-τ.
    Paper claim: τ ≈ 1.0 within a metric."""
```

Emit `results/aggregation_invariance.csv`.

---

## #12 · Ambiguous Reasoning (AR) task

**Paper.** Results §"LPP-informed tasks" (page 6); Supplement §1.1.1 (page 18)
and Algorithm 1; Table 2 (page 9). Grading: mean of (i) ambiguity-flag
accuracy and (ii) answer-choice accuracy. 100 items, 10 in-context examples,
`max_new_tokens=16`.

**Depends on nothing strictly; benefits from #6 for the Alpaca loader.**

Note: the paper's Algorithm 1 requires:
1. A corpus of natural prefixes (Alpaca or similar).
2. A per-prefix entropy scan (`H(p) > threshold`) to keep only ambiguous-
   looking prefixes.
3. Two curated completions per prefix — the paper hand-writes these; a
   procedural version needs an ambiguity bank.

**Files to create.**

- `src/ar_task.py`:

  ```python
  # Small hand-written ambiguity bank — paper does not release theirs.
  # Format: {prefix_stub, sense_a, sense_b, hint_a, hint_b}.
  AMBIGUITY_BANK = [
      {"prefix": "She deposited money at the",
       "a": "bank branch",  "b": "river bank",
       "hint_a": "The teller printed a receipt.",
       "hint_b": "The muddy shore was slippery after the rain."},
      {"prefix": "The bark was loud",
       "a": "of the dog",   "b": "of the tree cracking",
       "hint_a": "The neighbours complained about the puppy.",
       "hint_b": "A storm had felled it overnight."},
      # target: ~20 curated items so we can sample 100 with variation
      ...
  ]

  def generate_ar(n: int = 100, seed: int = 42) -> list[dict]:
      """Sample from AMBIGUITY_BANK with randomized A/B ordering and
      alternating correct sense."""

  def prompt_template(item: dict, in_context: list[dict]) -> str:
      """Table 2 format:
        Consider the ambiguous prefix and two possible senses. First judge
        the prefix alone as AMBIGUOUS or NOT AMBIGUOUS. Then, after reading
        the hint, choose the correct option A or B. Respond strictly as:
          ambiguous status=AMBIGUOUS or NOT AMBIGUOUS
          answer=A or B
        Prefix: <p>, Options: A. <sa> or B. <sb>. Hint: <h>. Your response:
      Prepend 10 in-context examples with expected responses."""

  def parse_ar_response(text: str) -> dict:
      """Extract status ∈ {AMBIGUOUS, NOT_AMBIGUOUS} and answer ∈ {A, B}
      by regex. Returns {'status': ..., 'answer': ...} or Nones on failure."""

  def grade_ar(records: list[dict]) -> dict:
      return {
          "n": len(records),
          "status_accuracy": mean(r["status_correct"] for r in records),
          "answer_accuracy": mean(r["answer_correct"] for r in records),
          "mean_accuracy":   mean((r["status_correct"] + r["answer_correct"]) / 2
                                  for r in records),
      }
  ```

- `run_ar.py` — mirrors `run_spc.py` in shape. `max_new_tokens=16`.

**Cost.** 100 short generations on qwen-0.5b on CPU: ~5–10 min. If you also
add the paper's per-prefix entropy scan on Alpaca (Algorithm 1 step 6), that
adds ~30–60 min of `signals()` calls on CPU — GPU-recommended.

**Scope note.** Writing a good ambiguity bank is the real work here. Aim for
~20 curated ambiguities and sample with rotations; the paper does not open-
source theirs. Cite the paper's algorithm but call out in the module
docstring that the bank is our own.

---

## After all items land

- Re-run `analyze_results.py` — the summary CSV will now have entries for
  every model that has both accuracy and latent data, plus a synthetic-task
  column each for AR and SPC once those runners write per-model summaries.
- Consider a top-level `run_all.py` that chains
  `expand_templates.py → run_experiments.py → run_latent.py →
  run_latent_rolling.py → run_calibration.py → run_spc.py → run_ar.py →
  analyze_results.py → latent_analysis.py` for a single model. Optional
  polish.
