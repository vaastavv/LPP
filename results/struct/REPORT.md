# Latent Structural Property Profiling in Code LLMs — Results report

Implementation of Meher & Mall, *Latent Structural Property Profiling in Code Large Language Model*. This report is regenerated from `results/struct/` by `src.struct.report`.

> **PROVENANCE / CAVEATS**
>

> - Structural metrics below were computed on a **randomly initialised tiny model panel** (pipeline smoke-test). The numbers are real outputs of the pipeline but have **no scientific meaning**; run pretrained CodeLLMs for real values.
>

> - Downstream scores (Tables VI-VIII, Fig 4) are **ILLUSTRATIVE synthetic placeholders** (stamped `synthetic:true`), present only to demonstrate the table format. Replace with real benchmark runs.
>


**Panel:** tiny, tiny-A, tiny-B, tiny-C, tiny-D, tiny-E, tiny-F


## Research questions and where they are answered

- **RQ1 (do CodeLLMs encode structure?)** → Table II (contrast vs control separability, effect sizes, BH-adjusted p).
- **RQ2 (at which layers?)** → Table III (peak layers $l^*$, normalized depth $\lambda^*$), Table IV (cross-dimension emergence), Fig 3 (layer-wise profiles).
- **RQ3 (association with SE tasks?)** → Tables VI-VIII, Fig 4.
- **RQ4 (causal components)** → not in this build (scope: Tables I-VIII).


## Figures

- `results/struct/plots/figure3_layerwise.png` — SRS/CFS/DFBS vs normalized depth.

- `results/struct/plots/figure4_assoc.png` — latent metric vs downstream performance.


## Tables

All tables are in `tables/` as CSV, LaTeX (`paper_tables.tex`), and Markdown (`paper_tables.md`). Summary:


### Table II — Overall structural separability

| Structural Dimension   |   Contrast Condition |   Control Condition |   Mean Diff. |   Effect Size | Adjusted p-value   |
|:-----------------------|---------------------:|--------------------:|-------------:|--------------:|:-------------------|
| Syntax                 |                0.017 |               0.353 |       -0.336 |         -3.72 | <0.001             |
| Control flow           |                0.187 |               0.47  |       -0.282 |         -2.91 | <0.001             |
| Data flow              |                0.727 |               0.285 |        0.442 |          1.41 | <0.001             |


### Table III — Peak layers

| Model   |   l*_syn |   lambda*_syn |   l*_cf |   lambda*_cf |   l*_df |   lambda*_df |
|:--------|---------:|--------------:|--------:|-------------:|--------:|-------------:|
| tiny    |        1 |          0.17 |       1 |         0.17 |       1 |         0.17 |
| tiny-A  |        1 |          0.25 |       1 |         0.25 |       1 |         0.25 |
| tiny-B  |        1 |          0.17 |       1 |         0.17 |       1 |         0.17 |
| tiny-C  |        1 |          0.17 |       1 |         0.17 |       1 |         0.17 |
| tiny-D  |        1 |          0.12 |       1 |         0.12 |       1 |         0.12 |
| tiny-E  |        1 |          0.1  |       1 |         0.1  |       1 |         0.1  |
| tiny-F  |        1 |          0.08 |       1 |         0.08 |       1 |         0.08 |


### Table VI — Downstream association

| Task               | Performance Metric   |   SRS |   CFS |   DFBS | Best Latent Predictor   |
|:-------------------|:---------------------|------:|------:|-------:|:------------------------|
| Bug Localization   | Top-1 Loc. Acc.      |  0.61 | -0.07 |   1    | DFBS                    |
| Program Repair     | Repair Rate          |  0.61 | -0.07 |   1    | DFBS                    |
| Code Completion    | pass@1               | -0.25 |  0.96 |  -0.18 | CFS                     |
| Code Summarization | ROUGE-1 F1           |  0.93 | -0.18 |   0.5  | SRS                     |
| Code Translation   | Transpile pass@1     | -0.25 |  0.96 |  -0.18 | CFS                     |


## Reproducing with real models

See `docs/CODELLM_README.md` and `scripts/run_colab.py`. The sweep is: `run_struct.py` + `run_downstream.py` per model, then `run_tables.py` to rebuild every table and figure.
