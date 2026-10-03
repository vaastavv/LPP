# CodeLLM paper — computed tables

> **PROVENANCE / CAVEATS**
>

> - Structural metrics below were computed on a **randomly initialised tiny model panel** (pipeline smoke-test). The numbers are real outputs of the pipeline but have **no scientific meaning**; run pretrained CodeLLMs for real values.
>

> - Downstream scores (Tables VI-VIII, Fig 4) are **ILLUSTRATIVE synthetic placeholders** (stamped `synthetic:true`), present only to demonstrate the table format. Replace with real benchmark runs.
>


## Table I — Characteristics of the evaluated CodeLLMs

| Model   | Family            | Parameters   |   Layers |   Hidden Size |   Context Length | Language Coverage   |
|:--------|:------------------|:-------------|---------:|--------------:|-----------------:|:--------------------|
| tiny    | TinyLlama(random) | <0.01B       |        6 |           128 |             1024 | n/a                 |
| tiny-A  | TinyLlama(random) | 0.04M        |        4 |            96 |             1024 | n/a                 |
| tiny-B  | TinyLlama(random) | 0.10M        |        6 |           128 |             1024 | n/a                 |
| tiny-C  | TinyLlama(random) | 0.15M        |        6 |           160 |             1024 | n/a                 |
| tiny-D  | TinyLlama(random) | 0.29M        |        8 |           192 |             1024 | n/a                 |
| tiny-E  | TinyLlama(random) | 0.50M        |       10 |           224 |             1024 | n/a                 |
| tiny-F  | TinyLlama(random) | 0.79M        |       12 |           256 |             1024 | n/a                 |


**Corpus**

| Corpus Component            |   Number |
|:----------------------------|---------:|
| Source programs             |       30 |
| Syntactic contrast pairs    |       23 |
| Control-flow contrast pairs |       30 |
| Definition-use pairs        |      168 |
| Repository-level partitions |        7 |
| Programming languages       |        1 |


## Table II — Overall structural separability

| Structural Dimension   |   Contrast Condition |   Control Condition |   Mean Diff. |   Effect Size | Adjusted p-value   |
|:-----------------------|---------------------:|--------------------:|-------------:|--------------:|:-------------------|
| Syntax                 |                0.017 |               0.353 |       -0.336 |         -3.72 | <0.001             |
| Control flow           |                0.187 |               0.47  |       -0.282 |         -2.91 | <0.001             |
| Data flow              |                0.727 |               0.285 |        0.442 |          1.41 | <0.001             |


## Table III — Maximum structural representation locations

| Model   |   l*_syn |   lambda*_syn |   l*_cf |   lambda*_cf |   l*_df |   lambda*_df |
|:--------|---------:|--------------:|--------:|-------------:|--------:|-------------:|
| tiny    |        1 |          0.17 |       1 |         0.17 |       1 |         0.17 |
| tiny-A  |        1 |          0.25 |       1 |         0.25 |       1 |         0.25 |
| tiny-B  |        1 |          0.17 |       1 |         0.17 |       1 |         0.17 |
| tiny-C  |        1 |          0.17 |       1 |         0.17 |       1 |         0.17 |
| tiny-D  |        1 |          0.12 |       1 |         0.12 |       1 |         0.12 |
| tiny-E  |        1 |          0.1  |       1 |         0.1  |       1 |         0.1  |
| tiny-F  |        1 |          0.08 |       1 |         0.08 |       1 |         0.08 |


## Table IV — Layer-wise structural emergence

| Comparison                |   Mean Diff. | 95% CI         | Test Statistic   | Adjusted p-value   |   Effect Size |
|:--------------------------|-------------:|:---------------|:-----------------|:-------------------|--------------:|
| Syntax vs control flow    |            0 | [0.000, 0.000] | --               | --                 |             0 |
| Syntax vs data flow       |            0 | [0.000, 0.000] | --               | --                 |             0 |
| Control flow vs data flow |            0 | [0.000, 0.000] | --               | --                 |             0 |


## Table V — DFBS across token-distance strata

| Distance Stratum   | Positive Binding   | Negative Binding   | DFBS   | 95% CI         | Adjusted p-value   |
|:-------------------|:-------------------|:-------------------|:-------|:---------------|:-------------------|
| 1-10 tokens        | 0.846              | 0.325              | 0.521  | [0.494, 0.548] | <0.001             |
| 11-25 tokens       | 0.581              | 0.253              | 0.328  | [0.302, 0.357] | <0.001             |
| 26-50 tokens       | 0.573              | 0.222              | 0.351  | [0.288, 0.408] | <0.001             |
| 51-100 tokens      | 0.638              | 0.132              | 0.506  | [0.361, 0.659] | 0.016              |
| >100 tokens        | --                 | --                 | --     | --             | --                 |


## Table VI — Association with downstream SE performance

| Task               | Performance Metric   |   SRS |   CFS |   DFBS | Best Latent Predictor   |
|:-------------------|:---------------------|------:|------:|-------:|:------------------------|
| Bug Localization   | Top-1 Loc. Acc.      |  0.61 | -0.07 |   1    | DFBS                    |
| Program Repair     | Repair Rate          |  0.61 | -0.07 |   1    | DFBS                    |
| Code Completion    | pass@1               | -0.25 |  0.96 |  -0.18 | CFS                     |
| Code Summarization | ROUGE-1 F1           |  0.93 | -0.18 |   0.5  | SRS                     |
| Code Translation   | Transpile pass@1     | -0.25 |  0.96 |  -0.18 | CFS                     |


## Table VII — Regression analysis

| Task             | Predictor   |   beta |   Standard Error | 95% CI        | p-value   |   Partial R2 |
|:-----------------|:------------|-------:|-----------------:|:--------------|:----------|-------------:|
| Bug Localization | SRS         |   0.53 |             0.38 | [-0.45, 1.50] | 0.222     |         0.28 |
| Bug Localization | CFS         |  -0.03 |             0.45 | [-1.18, 1.12] | 0.945     |         0    |
| Bug Localization | DFBS        |   0.99 |             0.07 | [0.81, 1.17]  | <0.001    |         0.98 |
| Program Repair   | SRS         |   0.6  |             0.36 | [-0.31, 1.52] | 0.151     |         0.36 |
| Program Repair   | CFS         |   0.05 |             0.45 | [-1.09, 1.20] | 0.910     |         0    |
| Program Repair   | DFBS        |   0.97 |             0.1  | [0.71, 1.24]  | <0.001    |         0.95 |


## Table VIII — Partial association controlling for model scale

| Task               | Latent Metric   |   Raw Correlation | Partial Correlation   | Control Variables   | p-value   |
|:-------------------|:----------------|------------------:|:----------------------|:--------------------|:----------|
| Bug Localization   | DFBS            |              1    | --                    | Parameters          | --        |
| Bug Localization   | CFS             |             -0.43 | --                    | Parameters          | --        |
| Program Repair     | DFBS            |              1    | --                    | Parameters          | --        |
| Code Summarization | SRS             |              0.89 | 0.89                  | Parameters          | 0.017     |