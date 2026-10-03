# Structural Property Profiling for Code LLMs (`src/struct/`)

This subproject implements **Meher & Mall, _Latent Structural Property Profiling
in Code Large Language Model_** (the "CodeLLM paper", `docs/CodeLLM_paper.pdf`),
which is a *different* paper from the LPP performance-profiling work the rest of
this repo implements. It computes the paper's three latent structural metrics,
runs the downstream analyses, and fills Tables I–VIII and Figures 3–4.

> This is kept entirely under `src/struct/` + `run_struct.py`,
> `run_downstream.py`, `run_tables.py`, `scripts/` so it does not disturb the
> existing LPP code.

## The three metrics

Hidden states `H^(l)(x) ∈ R^{T×d}` are read from every transformer layer `l`.
A structure-conditioned representation `z_q^(l)(x)` is pooled from the tokens
that participate in structural dimension `q`. Cosine distance
`d(a,b) = 1 − a·b/(‖a‖‖b‖)`.

| Metric | Definition (per layer `l`) | File |
|---|---|---|
| **SRS** (syntax) | `E[D over syntactically-contrasting pairs] / (E[D over matched controls] + ε)` | `metrics.py` |
| **CFS** (control flow) | same ratio over CFG-contrasting vs CFG-preserving pairs; `CFS_norm` divides by graph distance `δ_cf` | `metrics.py` |
| **DFBS** (data flow) | `mean_{(d,u)}( sim(h(d),h(u)) − mean_{neg} sim(h(d),h(u')) )`, negatives matched on token distance ≤ η | `profile.py`, `metrics.py` |

Peak layer `l*_q = argmax_l R_q^(l)` and normalized depth `λ* = l*/L`.

## Pipeline (how the paper's Fig 2 maps to code)

1. **Structural analysis** — `extract.py`: `ast`→ CFG, def-use pairs, syntax/CF
   unit spans, char-offset spans.
2. **Contrast / dependency construction** — `transforms.py`: behavior-preserving
   transforms build `P⁺`/`P⁻` for syntax (`+=`↔`=…+…`, comprehension→loop) and
   control flow (for→while, redundant guard); renaming is the matched control;
   def-use pairs + token-distance-matched negatives for data flow.
3. **Forward pass + hidden-state extraction** — `profile.py`: one pass per
   program, tokenizer offset mapping aligns spans→tokens (tokenizer-independent).
4. **Metric computation** — `metrics.py`: SRS/CFS/DFBS layer-wise profiles + raw
   per-pair data.
5. **Statistics** — `stats.py`: bootstrap CIs, Cohen's d, Mann-Whitney/Wilcoxon,
   Benjamini-Hochberg, Spearman + partial correlation, OLS.
6. **Tables + figures** — `tables.py` (I–V), `downstream_tables.py` (VI–VIII),
   `plots.py` (Fig 3–4), `report.py` (assembles everything).

## Tables produced

| Table | Content | Builder |
|---|---|---|
| I | Model & corpus characteristics | `tables.table1` + `corpus_stats` |
| II | Overall structural separability (effect sizes, BH-p) | `tables.table2` |
| III | Peak layers `l*`, `λ*` per model | `tables.table3` |
| IV | Cross-dimension layer-wise emergence | `tables.table4` |
| V | DFBS across token-distance strata | `tables.table5` |
| VI | Metric↔SE-task Spearman association | `downstream_tables.table6` |
| VII | Regression (task ~ metric) | `downstream_tables.table7` |
| VIII | Partial correlation controlling model scale | `downstream_tables.table8` |

Tables IX–XI (causal intervention / restoration / robustness, RQ4) are **out of
scope** for this build (you chose Tables I–VIII). The hooks to add them live in
the same modules.

Each table is emitted as CSV, LaTeX (booktabs, `tables/paper_tables.tex`), and
Markdown (`tables/paper_tables.md`).

## Running

### Offline smoke-test (no downloads — random tiny models)

```bash
python run_struct.py --model tiny --tag A --seed 1 --hidden 96  --layers 4
python run_struct.py --model tiny --tag B --seed 2 --hidden 128 --layers 6
# … a panel of tiny models …
python scripts/make_demo_downstream.py     # ILLUSTRATIVE synthetic SE scores
python run_tables.py                        # build all tables + figures + report
```

The smoke-test proves the pipeline runs and emits every table/figure. **Its
numbers are meaningless** (random weights; synthetic downstream scores stamped
`synthetic:true`).

### Real sweep (needs Hugging Face access → Colab / Kaggle / GPU box)

The sandbox this was built in firewalls `huggingface.co`, so real weights can't
be fetched here. On a machine with HF access:

```bash
pip install -r requirements.txt
python scripts/run_colab.py --models qwen25c-0.5b qwen25c-1.5b qwen25c-3b
# or point at any HF ids / local folders; edit registry.CODE_MODELS for the panel
```

That writes real `results/struct/struct_*.json` and `downstream_*.json`,
overwrites the synthetic demo scores, and rebuilds every table and figure with
genuine numbers. Copy the filled `tables/paper_tables.tex` blocks into the paper.

To profile weights you already have locally (no network at all):

```bash
HF_MODEL=/path/to/Qwen2.5-Coder-1.5B python run_struct.py --model local
```

## Scope / honesty notes

- **Scale**: the paper uses 6 models up to 34B and ~43k programs; this runs on a
  small CPU-friendly panel and a ~30-program synthetic, repo-partitioned corpus
  (`corpus.py`). The methodology is faithful; the N is smaller, so absolute
  magnitudes will differ from the paper's (which are placeholder dashes anyway).
- **Language**: Python only (`ast`). The extractor is structured so a
  tree-sitter backend could add more languages later.
- **δ_cf** is a feature-based normalized graph distance (size/branch/merge/loop/
  cyclomatic), a deterministic proxy for intractable exact graph-edit distance.
- **Downstream tasks** are lightweight self-contained proxies, not the full
  benchmarks; see `downstream.py` docstring.
