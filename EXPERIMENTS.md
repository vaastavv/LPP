# Extension experiment: does LPP give an early warning of drift?

The LPP paper claims (Discussion, *"Implications for model selection and
monitoring"*, page 11) that intrinsic metrics can act as an **early-warning
signal** — a rising entropy floor or effective rank on incoming data "could
presage performance degradation … long before these issues surface in output
accuracy." The paper never tests this. This experiment does, and makes the claim
falsifiable.

## Hypothesis

As inputs drift out of distribution, a latent metric (entropy floor, max-ER, or
max-PR) deviates from its in-distribution baseline at a **lower drift severity**
than task accuracy does. If true, the metric "leads" accuracy and is usable as a
monitor. If false (it moves at the same severity or later), the monitoring claim
does not hold — a genuine, reportable negative result.

## Design

For a schedule of increasing perturbation severities
`{0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60}` we take the study's gradable
task prompts and, at each level, corrupt every prompt and measure on the **same
corrupted inputs**:

- **task accuracy** — `runner.chat` → `extract` → `grade`;
- **the LPP metrics** — `runner.signals` (min entropy, max-ER, max-PR).

Because decoding is greedy, level-0 correctness is deterministic; we measure the
accuracy decline on the subset of samples the model got right at level 0, so the
decline is well defined (starts at 1.0 by construction). `--reps` applies several
independent random corruptions per prompt per level to raise the sample count.

Perturbation kinds (`src/drift.py`): `char_noise` (typos), `word_dropout`,
`word_shuffle`, `homoglyph` (confusable-unicode substitution — strong tokenizer
shift with little visual cue), `append_gibberish` (topical drift).

## Detection statistic

For each metric we find the **detection level**: the first severity whose value
separates from the level-0 baseline under a **paired bootstrap** of the mean
difference (CI excludes 0 — direction *up* for the latent metrics, *down* for
accuracy). The pairing is over sample ids shared across levels, so accuracy
(binary) and the continuous metrics use one code path. The **lead** is
`accuracy_detection_level − metric_detection_level` (positive ⇒ the metric leads).

## Running it

```bash
# real run (needs the model + torch; a few minutes on CPU for qwen-0.5b)
python run_drift.py --model qwen-0.5b --kind char_noise --reps 3
python drift_analysis.py

# sweep perturbation kinds
for k in char_noise word_dropout homoglyph append_gibberish; do
  python run_drift.py --model qwen-0.5b --kind $k --reps 3
done
python drift_analysis.py
```

Outputs: `results/drift_<model>_<kind>.jsonl`, `results/drift_summary.csv`, and
overlay plots `results/plots/drift_<model>_<kind>.png`.

## Synthetic demonstration (no model)

`python run_drift.py --synthetic --kind char_noise` fabricates clearly-labelled
data (`model = "SYNTHDEMO"`, plot titled *SYNTHETIC DEMO*) whose only purpose is
to exercise the analysis/plot end-to-end and illustrate what a positive result
looks like. It is **not** a measurement. In the demo, accuracy holds flat until
severity 0.2 then falls (detected at 0.30) while the entropy floor is already
detectable at severity 0.05 — a 4-level lead. Replace it with a real run to test
the hypothesis on an actual model.

## Adding more models

`config.MODELS` now includes `smollm-1.7b` (`HuggingFaceTB/SmolLM2-1.7B-Instruct`,
fully open) and `gemma2-2b` (`google/gemma-2-2b-it`, **gated** — accept the
license on HuggingFace and `huggingface-cli login` once). Every runner takes
`--model <label>`, and all analyses key off the model label, so a new model flows
into the sensitivity plots (`latent_analysis.py`), the correlations
(`analyze_results.py`), and the drift plots automatically once its result files
exist. With ≥3 models present, the cross-model correlations (#3/#4) become
meaningful.

```bash
for m in qwen-0.5b qwen-1.5b smollm-1.7b gemma2-2b; do
  python run_latent.py       --model $m
  python run_calibration.py  --model $m
  python run_drift.py        --model $m --kind char_noise --reps 3
done
python latent_analysis.py
python analyze_results.py
python drift_analysis.py     # per-model drift plots + drift_lead_comparison_<kind>.png
```

`drift_analysis.py` also emits `results/plots/drift_lead_comparison_<kind>.png`
— a bar of the entropy early-warning lead per model — whenever ≥2 models have
drift results for a kind, so gemma2-2b / smollm-1.7b / qwen appear side by side.

> Note: downloading weights needs network access to `huggingface.co`. If a run
> fails to reach it, the environment's network policy is blocking that host — add
> it under the cloud environment's **Network access** settings.

## Caveats / honest limitations

- A single small model (qwen-0.5b) can only show the *method*; the paper's claim
  is about models generally, so a real result needs several models.
- `char_noise`/`homoglyph` are input-level corruptions; "drift" in deployment is
  usually semantic/topical — `append_gibberish` and swapping the calibration
  corpus (Alpaca→Dolly→WikiText via `run_calibration.py`) probe that better.
- The ER/PR covariance is estimated per-prompt and is rank-limited by token
  count (see the geometry caveat in the review notes); the entropy floor is the
  cleaner monitor here.
