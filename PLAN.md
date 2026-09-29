# Gap between this codebase and the full LPP paper

Reference: Chakraborty et al., *Latent Performance Profiling of Large Language
Models* (arXiv:2605.30018v2, 29 May 2026), `docs/LPP_paper.pdf`.

Ordered easiest → hardest. Each item is a stand-alone deliverable: pick one and
we build only that. Every item cites the paper section it comes from and shows
where it plugs into the current code. Items that require an earlier item to be
useful are marked with **Depends on: #N**.

Current baseline (what we already have):
- `src/latent_metrics.py`: `token_entropy`, `effective_rank`,
  `participation_ratio`, `profile_hidden_states` (already returns
  `ER_by_layer` / `PR_by_layer`).
- `src/model_runner.signals`: one forward pass per prompt, returns
  `min_entropy`, `mean_entropy`, `max_ER`, `max_PR`, `last_layer_ER`,
  `last_layer_PR`, `num_layers`, `num_tokens`.
- `src/experiment.run_latent`, `run_latent.py`: profile the problem-set
  variants and report per-model extremal (min-entropy / max-ER / max-PR).
- `analyze_results.py`: per-model summary table + two plots
  (`degradation_by_transform.png`, `accuracy_vs_latent.png`).

---

## 1. Save per-layer ER / PR in latent runs
**Paper**: Materials & Methods §"Formal definition"; Supplement §2.2, Figure 4
("PR and ER as functions of normalized layer depth").

**What**: The per-layer arrays are already computed in
`profile_hidden_states` but dropped in `src/model_runner.signals` (only
`max_ER`, `last_layer_ER`, etc. are kept). Persist `ER_by_layer` and
`PR_by_layer` in each latent record so every later layerwise analysis can read
them from `results/latent_*.jsonl` without re-running the model.

**Maps to**: two extra keys added in `src/model_runner.signals` and passed
through `src/experiment.run_latent`. No new files. Minimal change, so I would
propose a diff before touching either file.

**Cost**: ~zero — the values are already in memory.

---

## 2. Layer-wise ER / PR plot + hourglass detector
**Paper**: Results §"Stable latent patterns and their origins";
Supplement §2.2 and Figure 4.

**What**: For each model, plot ER and PR against **normalized depth**
(layer index / `num_layers - 1`, so profiles align across model sizes), and
flag the "hourglass" shape (high at 0.0 and 1.0, dip in the middle). A simple
detector: report `hourglass_score = (ER[0] + ER[-1]) / 2 - min(ER[middle_third])`
per model, positive means an hourglass.

**Maps to**: new file `latent_analysis.py` that reads `results/latent_*.jsonl`
and writes `results/plots/layerwise_ER_PR.png` + a row per model in
`results/summary.csv` (or a sibling CSV) with the hourglass score.

**Depends on**: #1.

---

## 3. Cross-model Spearman correlations
**Paper**: Results §"Intrinsic metrics versus extrinsic performance",
Figure 2D (Spearman ρ between LPP metrics and MMLU-PRO / IFEval / BBH).

**What**: With ≥3 models in `results/`, compute Spearman ρ between each
latent metric (min-entropy, max-ER, max-PR) and baseline accuracy /
answer-consistency across models, and print a small ρ / p-value table. With
one model this is a no-op that says "need ≥3 models".

**Maps to**: extension inside `analyze_results.py`; uses `scipy.stats.spearmanr`
(new dep, add to `requirements.txt`).

**Cost**: trivial code; the honest requirement is that we have run 3+ models
first.

---

## 4. Aggregation-invariance check
**Paper**: Discussion §"LPP complements traditional benchmarks"; Supplement
§2.3, Figure 5 ("model rankings … invariant to the choice of aggregation
method").

**What**: For the same latent records, aggregate each metric with min / mean /
median / max and check that per-model rankings are the same across aggregators
(rank correlation of the resulting orderings). Confirms our choice of the
extremal summary is not the story.

**Maps to**: a short function in `latent_analysis.py`; outputs a per-metric
Kendall-τ table (`results/aggregation_invariance.csv`).

**Depends on**: nothing strictly, but is only interesting once ≥3 models exist.

---

## 5. Rolling-context entropy schedule
**Paper**: Table 1 ("minimum entropy over the rolling-context schedule");
Materials & Methods §"Uncertainty floor"; Figure 4B.

**What**: The paper's "uncertainty floor" is the min next-token entropy taken
over a **schedule of growing prefixes**, not over the token positions of a
single forward pass. Current `signals()` runs one pass on the whole prompt and
takes `min(H_t)` across positions — related but not the same thing. Add a
`signals_rolling(text, prefix_lengths)` that, for each L in the schedule
(default `{10, 20, …, 100}` per the paper), runs a forward pass on `text[:L]`,
records the entropy at position L, and returns `min` / mean over the schedule.
Existing `signals()` stays as it is.

**Maps to**: new function in a new file `src/latent_rolling.py` so
`src/model_runner.py` isn't rewritten; a new runner `run_latent_rolling.py`;
new output `results/latent_rolling_<model>.jsonl` (one row per prompt with
`entropy_by_prefix`, `min_entropy_rolling`).

**Cost**: 10× the forward passes of Track 2 per prompt. On CPU with a 1B model
and 100 Alpaca prompts, roughly 15–30 minutes; 7B+ realistically needs a
Colab / Kaggle GPU.

---

## 6. Real calibration set (100 Alpaca prompts)
**Paper**: Materials & Methods §"Datasets and prompts for intrinsic metrics"
("100 randomly-sampled task-agnostic texts from the Alpaca dataset"; seed 42).

**What**: The paper's latent profile is computed on **task-agnostic** text, not
on our problem-set variants. Current Track 2 runs latent metrics on the same
problems as Track 1 — that inflates min-entropy because arithmetic prompts
push the model to a confident numeric prediction. Add a calibration loader
that downloads / caches 100 Alpaca prompts (seed 42, HF `tatsu-lab/alpaca`),
runs `signals()` (or rolling-#5 once that exists), and writes
`results/calibration_alpaca_<model>.jsonl`.

**Maps to**: new file `src/calibration.py` (dataset loader + iterator, offline
cache under `data/calibration/`), new `run_calibration.py` runner. Reuses
`src.model_runner.signals` unchanged.

**Cost**: 100 forward passes on a 1B model, ~2–5 minutes CPU. Larger models
push to GPU.

**Depends on**: nothing strictly, but pairs naturally with #5.

---

## 7. Sensitivity: prefix length {10, 20, …, 100}
**Paper**: Materials & Methods §"Datasets and prompts for intrinsic metrics"
("prefix length {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}"); Figure 4A.

**What**: For each model, plot the three metrics as a function of the prefix
length used to compute them, one line per model. This is what Figure 4A shows
and it's the direct output of #5's `entropy_by_prefix` array.

**Maps to**: analysis function in `latent_analysis.py`; produces
`results/plots/sensitivity_prefix.png`.

**Depends on**: #5.

---

## 8. Sensitivity: context length {50, 100, 200, 500}
**Paper**: Materials & Methods §"Datasets and prompts for intrinsic metrics";
Figure 4B.

**What**: Same three-metric plot as #7 but varying the **maximum context
length** on the full calibration set (truncate each Alpaca prompt to 50 / 100
/ 200 / 500 tokens, recompute metrics, plot).

**Maps to**: extra loop in `run_calibration.py` behind a `--context-length`
argument (or a wrapper `run_sensitivity_context.py`); one PNG per plot.

**Depends on**: #6.

---

## 9. Sensitivity: sample size {10, 100, 500, 1000}
**Paper**: Supplement §2.1, Figure 2 ("Stability of LPP metrics across varying
sample sizes").

**What**: Prove metric stability at small `n` by subsampling the calibration
set at sizes 10 / 100 / 500 / 1000 and plotting each metric per model. Paper's
claim is that ~100 is enough.

**Maps to**: small analysis added to `latent_analysis.py` (no new model runs
needed if the calibration set already has ≥1000 prompts) or, if
`n_calibration=100`, a `--n` sweep in `run_calibration.py`.

**Depends on**: #6.

---

## 10. Sensitivity: datasets (Alpaca, Dolly, WikiText)
**Paper**: Supplement §2.1, Figure 3 ("Dataset Sensitivity").

**What**: Repeat #6 for Dolly (`databricks/databricks-dolly-15k`) and
WikiText-103 (`wikitext`), 100 prompts each, and plot side-by-side to show
model rankings hold across datasets (paper's Supp. Figure 3).

**Maps to**: extend `src/calibration.py` with `dataset in {"alpaca", "dolly",
"wikitext"}` selector; three calibration JSONLs per model; one comparison PNG.

**Depends on**: #6.

---

## 11. Symbolic Pattern Completion (SPC) task
**Paper**: Results §"LPP-informed tasks"; Supplement §1.1.2 and Algorithm 2;
Table 2.

**What**: Generator produces sequences of length 12 over a small alphabet
under rules (alternation `ABAB…`, mirroring `ABBA`, progression `ABCD`, nested
combinations), masks the last 3 tokens, asks the model to continue. Grading is
**character-level F1** averaged over samples (Supp. §"Evaluating LLMs"). 100
items, 10 in-context examples per prompt, `max_new_tokens=8`.

**Maps to**: new file `src/spc_task.py` (generator + evaluator), a new runner
`run_spc.py`, output `results/spc_<model>.jsonl`. Reuses
`src.model_runner.chat` unchanged.

**Cost**: 100 short generations on a 1B model — a few minutes on CPU.

**Depends on**: nothing.

---

## 12. Ambiguous Reasoning (AR) task
**Paper**: Results §"LPP-informed tasks"; Supplement §1.1.1 and Algorithm 1;
Table 2.

**What**: For each of ~100 Alpaca-derived prefixes, filter by high predictive
entropy under the target model, pair with two plausible completions and a
disambiguating hint, ask the model to output
`ambiguous status={AMBIGUOUS|NOT_AMBIGUOUS}; answer={A|B}`. Score is the mean
of (i) ambiguity-flag accuracy and (ii) answer-choice accuracy.

**Maps to**: new file `src/ar_task.py` (prefix scanner + hint bank + item
generator + evaluator), new runner `run_ar.py`, output
`results/ar_<model>.jsonl`. Uses `src.model_runner.signals` for the entropy
filter (reuse; no rewrite) and `src.model_runner.chat` for the answer.

**Cost**: the entropy scan is the bottleneck — need many candidate prefixes
per selected AR item. Roughly an hour on a 1B model on CPU; push 7B+ to GPU.
Also, the paper hand-curates the two completions and the hint; a purely
procedural generator needs a small pattern bank of ambiguities (bank/river-
bank, bat/animal-vs-bat/baseball, bark/tree-vs-dog, …). I'd write this bank
first and note it as a scope decision before implementing.

**Depends on**: nothing strictly; benefits from #6 for the Alpaca loader.

---

## Notes on the ordering
- **#1** is the single-highest-leverage change: two lines of code, unlocks #2
  and every layerwise question later.
- **#5 + #6** are the paper's actual definition of the latent profile — until
  we have them we're computing "our own thing", not what the paper does. They
  are the ones I'd do next after #1.
- **#11 (SPC)** is a much smaller task than **#12 (AR)** because it has no
  entropy pre-scan and needs no natural-language ambiguity bank; SPC first,
  then AR.
- **#3, #4, #7–#10** are cheap **once the underlying data exists**; each is
  really a "produce the plot from the paper" task.
- CPU realism: 1B model on Windows CPU handles calibration on 100 Alpaca
  prompts and full SPC. Rolling schedule and AR entropy scans get slow; 7B+
  needs Colab / Kaggle GPU.

## Extras I noticed but didn't add above
- Paper uses **greedy `temperature=0`, `top_p=1.0`, batch size 8, left padding,
  seed 42** (Materials & Methods §"Implementation details"). We already use
  greedy (`config.GEN["do_sample"] = False`) and seed 42 (`config.SEED`), but
  we do **not** batch, do **not** force `top_p`, and `signals()` runs one
  prompt at a time. Batching is a throughput improvement, not a methodology
  gap, so I left it off the checklist — flag it if you want it in.
- Table 1 also mentions reporting extrema **across layers *and* contexts**;
  currently we only take the extremum across variants in a single forward
  pass. #1 + #5 together close this.
