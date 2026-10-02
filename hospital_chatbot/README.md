# Hospital Intent + Emotion Chatbot

A production-oriented NLP pipeline for a hospital / medical-care assistant. It classifies **what the user
wants from the hospital** (20 intents) and, separately, **how the user feels** (28 GoEmotions labels). A
policy layer decides how to behave (normal, clarification, emergency), and a controlled template layer
produces the final response.

> **BERT determines what the user wants. RoBERTa/GoEmotions determines how the user feels. The policy layer
> determines how the system should behave. The response template layer determines what controlled response
> should be delivered.**

> **Current status:** every component is implemented and tested. The **TF-IDF + Logistic Regression**
> baseline is trained and evaluated. **DistilBERT fine-tuning and the RoBERTa GoEmotions model have not
> been run yet**, because the build environment's network policy blocks `huggingface.co`. All measured
> numbers are in [`evaluation/REPORT.md`](evaluation/REPORT.md), generated from saved artifacts. To finish
> the transformer steps, see [Reproduce everything](#reproduce-everything).

---

## 1. Architecture

```text
                         ┌──────────────────────┐
                         │      USER QUERY      │
                         │ "Can I see a heart   │
                         │  specialist tomorrow?"│
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  Input Validation    │  src/text.py
                         │  + Basic Safety      │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Minimal Text         │  src/text.py (NFKC, whitespace, quotes)
                         │ Normalization        │
                         └──────────┬───────────┘
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
          ┌──────────────────────┐         ┌──────────────────────┐
          │ INTENT CLASSIFIER    │         │ EMOTION CLASSIFIER   │
          │ Fine-tuned DistilBERT│         │ RoBERTa-base         │
          │ (or TF-IDF baseline) │         │ GoEmotions           │
          │ src/intent/          │         │ src/emotion/         │
          └──────────┬───────────┘         └──────────┬───────────┘
                     ▼                                ▼
          ┌──────────────────────┐         ┌──────────────────────┐
          │ Intent + Confidence  │         │ Emotion + Confidence │
          └──────────┬───────────┘         └──────────┬───────────┘
                     └──────────────┬─────────────────┘
                                    ▼
                         ┌────────────────────────┐
                         │ Confidence / OOD       │  src/policy/confidence.py
                         │ / Policy Layer         │  src/policy/router.py
                         └────────────┬───────────┘  src/policy/emergency.py
                       ┌──────────────┼──────────────┐
                       ▼              ▼              ▼
                  Confident       Uncertain       Emergency
                       ▼              ▼              ▼
                 Normal Intent    Clarification   Emergency
                       │              │           Handler
                       └──────────────┼──────────────┘
                                      ▼
                         ┌────────────────────────┐
                         │ Response Selector      │  src/response/selector.py
                         │ intent + emotion       │
                         └────────────┬───────────┘
                                      ▼
                         ┌────────────────────────┐
                         │ response_templates.json│  data/response_templates.json
                         └────────────┬───────────┘
                                      ▼
                         ┌────────────────────────┐
                         │    FINAL RESPONSE      │  src/pipeline.py
                         └────────────────────────┘
```

Intent and emotion stay **separate classification tasks**. There are no combined classes such as
`appointment_booking_sadness`: the response layer combines `intent = appointment_booking` with
`emotion = sadness`.

## 2. Project layout

```text
hospital_chatbot/
├── configs/training.yaml            all paths + hyper-parameters (single source of truth)
├── data/
│   ├── raw/
│   │   ├── original_intent_dataset.csv      seed dataset (20 / intent; see "Provenance")
│   │   ├── original_response_templates.json seed templates (6 emotions)
│   │   ├── expansion/<intent>.txt           curated, hand-written expansion examples
│   │   └── label_corrections.csv            documented relabels applied to the seed
│   ├── processed/
│   │   ├── intent_dataset.csv               enhanced dataset  (text,intent)
│   │   ├── train.csv / validation.csv / test.csv
│   │   ├── dataset_provenance.csv           where every row came from
│   │   ├── dataset_changelog.json           every modification, logged
│   │   └── split_summary.json               split sizes + leakage checks
│   ├── ood/                                  out-of-domain queries (calibration only, not a class)
│   └── response_templates.json              20 intents x 28 emotions = 560 templates
├── models/
│   ├── intent/{model,tokenizer,label_mapping.json,calibration.json,training_config.json}
│   ├── baseline/{tfidf_vectorizer.pkl,logistic_regression.pkl,calibration.json,training_config.json}
│   ├── emotion/                              local copy of the GoEmotions checkpoint (after download)
│   └── selected_model.json                   validation-based model selection
├── src/
│   ├── data/{validate,deduplicate,enhance_dataset,split_dataset}.py
│   ├── intent/{dataset,train,baseline,evaluate,predictor}.py
│   ├── emotion/predictor.py
│   ├── policy/{confidence,emergency,router}.py
│   ├── response/{selector,build_templates}.py
│   ├── pipeline.py   text.py   config.py   report.py
├── evaluation/   classification_report.json, confusion_matrix.png, baseline_results.json,
│                 error_analysis.csv, reliability_diagram.png, threshold_curve.png,
│                 experiments.{json,md}, REPORT.md, <model>/...
└── tests/        test_dataset, test_response, test_intent, test_emotion, test_pipeline
```

## 3. Dataset

### Provenance (important)

The task assumed an existing `intent_dataset.csv` (about 20 examples per intent) and a
`response_templates.json` (6 emotions). **Neither file existed anywhere in this repository**, so with the
project owner's approval they were **reconstructed** as `data/raw/original_*` (400 examples, 20 per intent,
plus 6-emotion templates). The rest of the pipeline treats them as the original data: they are preserved,
corrected only through logged relabels, and never silently dropped (enforced by
`tests/test_dataset.py::test_original_examples_preserved_or_logged`). If the real originals turn up, drop
them into `data/raw/` and rerun the pipeline.

### Schema

```csv
text,intent
"I need to schedule a cardiology appointment","appointment_booking"
```

Provenance is kept in a **separate** file (`dataset_provenance.csv`), so the training schema stays
`text,intent`.

### Intent taxonomy (20, names unchanged)

`appointment_booking`, `appointment_cancellation`, `appointment_rescheduling`, `doctor_search`,
`hospital_timings`, `emergency_assistance`, `billing_query`, `insurance_query`, `lab_reports`,
`pharmacy_query`, `department_information`, `contact_information`, `facility_information`,
`admission_query`, `discharge_query`, `medical_records`, `prescription_query`, `greeting`, `thank_you`,
`goodbye`.

The chatbot classifies **service intent**, not disease. There are no diagnosis intents.

### Boundary rules (used when writing and correcting examples)

| pair | rule |
|---|---|
| booking / rescheduling / cancellation | no existing appointment → **booking**; has one and wants it **moved** → **rescheduling**; has one and wants it **removed** → **cancellation**. Cancellation/reschedule *policy* questions (fees, refunds) stay with the action. |
| doctor_search / department_information | a **person / specialist type** ("I need a cardiologist", "Is Dr. X in today?") → doctor_search; a **unit** and its services/location → department_information |
| billing / insurance | money owed, paid, refunded or estimated → billing; coverage, claims, cashless, TPA, policy → insurance |
| lab_reports / medical_records | result of a **specific test** → lab_reports; the **whole file**, history, old records → medical_records |
| admission / discharge | getting **in** vs getting **out**. Deposits are billing. |
| pharmacy / prescription | obtaining/buying the **product** → pharmacy; the doctor-issued **document** (renewal, copy, correction) → prescription |
| hospital_timings / facility_information | **when** something is open (any service, including the pharmacy) → timings; **whether** an amenity exists / what it offers → facility |
| greeting / thank_you / goodbye vs. tasks | "Hi, I want to book…" is **booking**, not greeting; "Thanks for moving my appointment" is **thank_you** |

Each intent file in `data/raw/expansion/` ends with a `hard negatives` block that targets these boundaries.

### Emotion taxonomy (28)

`admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment,
disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism,
pride, realization, relief, remorse, sadness, surprise, neutral`

This is the canonical GoEmotions inventory used by `SamLowe/roberta-base-go_emotions`. Templates are keyed
by label **name**, never by class id. `EmotionClassifier` reads `config.id2label` from the downloaded
checkpoint and **raises** if its label set differs from the templates. `python -m src.emotion.predictor
download` prints the actual mapping.

## 4. Dataset enhancement (`python -m src.data.enhance_dataset`)

1. load and validate the seed (schema, 20 intents)
2. apply documented relabels (`label_corrections.csv`). Three seed rows broke the boundary rules, e.g.
   `"Is the pharmacy open 24 hours?"` was `pharmacy_query` while its twin `"When is the pharmacy open?"` was
   `hospital_timings`.
3. remove exact and trivial duplicates (case, punctuation and whitespace-insensitive key, e.g. `Hello` /
   `Hello?`)
4. add curated, hand-written expansion examples. These are not template-generated paraphrases: they mix
   formal and casual wording, short and long queries, indirect requests, abbreviations (`appt`, `pls`,
   `OPD`, `TPA`), mild typos (`apointment`, `reshedule`) and code-mixed phrases (`discharge kab hoga`).
5. remove near-duplicates (char 3–5-gram TF-IDF cosine ≥ 0.90). The original is always kept. Cross-intent
   near-duplicates are logged as conflicts.
6. enforce 100–150 per intent (down-sampling would only ever touch expansion rows)
7. **label-consistency audit**: out-of-fold TF-IDF/LR predictions flag rows whose label disagrees with
   high confidence. Flags are reviewed by a person, never auto-removed. The 84 flags in v1.0 were reviewed:
   nearly all are deliberate hard negatives ("I need to cancel, not reschedule"), and all labels were
   kept.
8. re-validate, save, and print a per-intent summary

Every step and every affected row is written to `data/processed/dataset_changelog.json`.

## 5. Train / validation / test methodology (`python -m src.data.split_dataset`)

* stratified **70 / 15 / 15** by intent, fixed seed (42)
* **leakage control:** after deduplication, items within an intent whose similarity is ≥ 0.75 are linked
  into groups (union-find), and **whole groups** go to one split. The split summary records the maximum
  same-intent similarity between any test item and any train item, which stays below the grouping
  threshold, and asserts zero exact overlap.
* a separate **out-of-domain (OOD) set** (jokes, weather, coding, general medical-knowledge questions…)
  is split 50/50 into OOD-validation, used for threshold selection, and OOD-test, used for reporting. OOD
  is not an intent class.
* the test set is read only by `src/intent/evaluate.py`, and only **after** the model is selected on
  validation.

## 6. Intent models

### Fine-tuned compact BERT (`src/intent/train.py`)

```text
Raw text → BERT tokenizer → input ids / attention mask → pretrained encoder (DistilBERT)
        → [CLS] representation → dropout → linear → 20 logits → softmax → intent probabilities
```

* default checkpoint `distilbert-base-uncased`, configurable as `intent_model.model_name`. DistilBERT's
  stock head adds one pre-classifier dense layer before dropout and the final linear layer.
* fine-tunes a pretrained checkpoint; never trains from scratch
* AdamW (lr 2e-5, weight decay 0.01 except bias/LayerNorm), linear warm-up (10%), batch 16, up to 5
  epochs, gradient clipping 1.0, early stopping on **validation macro-F1** (patience 2), best checkpoint
  restored
* **class weighting `auto`**: enabled only if the max/min class ratio in train exceeds 1.5. The v1.0 train
  split ratio is 95/84 ≈ 1.13, so standard cross-entropy is used, and the decision is recorded in
  `training_config.json`.
* saved in Hugging Face format (safetensors) under `models/intent/`, alongside the tokenizer, label
  mapping, calibration and training config. No pickle is used for the Transformer.

### TF-IDF + Logistic Regression baseline (`src/intent/baseline.py`)

Word 1–2-gram TF-IDF (sublinear tf) + multinomial LR. `C` is chosen on **validation** macro-F1 over
`baseline.C_grid`. Artifacts: `models/baseline/tfidf_vectorizer.pkl`, `logistic_regression.pkl`.

### Model selection (`src/intent/evaluate.py`)

1. compute validation metrics for every trained candidate
2. select by highest validation macro-F1; if candidates are within 0.005, pick the faster one. Write
   `models/selected_model.json`.
3. only then evaluate every candidate on the held-out test set: accuracy, macro P/R/F1, weighted F1,
   per-intent report, confusion matrix, error analysis, most-confused pairs, calibration (ECE +
   reliability diagram), selective prediction and OOD false-accepts, emergency recall, latency, model size

BERT is **not** assumed to be better: if the baseline wins on validation, it is selected.

## 7. Confidence handling and unknown intents (`src/policy/confidence.py`)

The threshold is **not** an arbitrary 0.5. On validation data only:

1. **Temperature scaling**: fit a single temperature T by minimizing validation NLL. ECE before and after
   is reported, on validation during training and on test in the report.
2. **Threshold**: a prediction should be *accepted* iff it is an in-domain query the model gets right.
   In-domain errors and OOD queries should be *rejected*. The threshold maximizes the balanced accuracy of
   that accept/reject decision over validation + OOD-validation, subject to in-domain coverage ≥ 0.90
   (`confidence.min_in_domain_coverage`). The full threshold → coverage / selective accuracy / OOD
   false-accept curve is saved and plotted (`threshold_curve.png`).
3. **Emergency threshold**: a separate, lower threshold on P(emergency_assistance): the lowest value
   that keeps false emergency routes ≤ 2% on non-emergency validation + OOD-validation. Missing an
   emergency costs more than a false alarm.

Predictor output:

```json
{"intent": "appointment_booking", "confidence": 0.96, "status": "confident", "predicted_intent": "appointment_booking", "emergency_probability": 0.001, "top_k": [...]}
{"intent": null, "confidence": 0.38, "status": "uncertain", "predicted_intent": "hospital_timings", ...}
```

An uncertain result goes to the **clarification** handler. It asks "Is your question about X or Y?" using
the top-2 candidates, or shows the general menu.

## 8. Emergency routing (`src/policy/emergency.py`, `src/policy/router.py`)

Priority, first match wins:

1. invalid input → ask the user to type a question
2. **self-harm language** → crisis message: contact the local emergency number, a local crisis helpline
   or someone trusted
3. **emergency** → if P(emergency) ≥ the emergency threshold **or** a high-severity red-flag phrase
   (chest pain, not breathing, unconscious, heavy bleeding, stroke, seizure, overdose, choking…),
   ordinary template selection is bypassed and the emergency handler responds. Negations such as "this is
   not an emergency" are respected.
4. confident intent → `response_templates.json[intent][emotion]`
5. otherwise → clarification / unknown

The emergency response tells the user to call **their local emergency number** or go to the nearest
emergency department. It never diagnoses, never provides treatment, and never invents phone numbers or
locations. Emotion only changes a short calming prefix.

## 9. Response templates (`data/response_templates.json`)

`python -m src.response.build_templates` generates 20 × 28 = **560** templates as *opener(emotion) + body
(intent)*:

* **intent determines content:** one body per intent that asks for the details the hospital workflow
  needs and names the team that confirms facts
* **emotion determines tone:** one opener per emotion (task, social and emergency variants)
* rules, partly enforced by tests: no diagnoses, no prescribing, no invented availability, results,
  coverage, prices, phone numbers, URLs or locations. Every response for an intent ends with the same
  body, so meaning stays the same across emotions.

Example (`appointment_booking`):

* `fear`: "I understand this can feel stressful, and I'm here to help. I can help you request a new
  appointment. Please share the department or doctor you'd like to see, …"
* `joy`: "Absolutely! I'd be happy to help. I can help you request a new appointment. …"

Validate with `python -m src.data.validate`: 20 intents, every label present, no empty strings, no reused
strings.

## 10. Emotion model (`src/emotion/predictor.py`)

Pretrained `SamLowe/roberta-base-go_emotions`, used as-is with no training. It is multi-label (sigmoid). The
top label is used; if its score is below 0.5 (the model's default decision threshold), the result falls
back to `neutral`. Output: `{"emotion": "fear", "confidence": 0.88, "status": "confident", "top_k": [...]}`.

## Reproduce everything

```bash
cd hospital_chatbot
pip install -r requirements.txt

python -m src.data.enhance_dataset          # build data/processed/intent_dataset.csv (+ changelog)
python -m src.data.split_dataset            # train/validation/test + OOD splits
python -m src.response.build_templates      # 560 templates
python -m src.data.validate                 # dataset + template validation

python -m src.intent.baseline               # TF-IDF + LR  (C chosen on validation)
python -m src.intent.train                  # fine-tune DistilBERT (needs huggingface.co)
python -m src.emotion.predictor download    # save GoEmotions RoBERTa to models/emotion (needs huggingface.co)

python -m src.intent.evaluate               # select on validation, then report on test
python -m src.report                        # regenerate evaluation/REPORT.md from artifacts
python -m pytest -q                         # full test suite
```

Inference:

```python
from src.intent.predictor import load_intent_classifier
from src.pipeline import HospitalChatbot

clf = load_intent_classifier()                 # model from models/selected_model.json
clf.predict("Can I book a cardiology appointment?")

bot = HospitalChatbot.load()
bot.respond("My father is having severe chest pain and I'm terrified.")
# -> route "emergency", final_intent "emergency_assistance", emotion from RoBERTa, emergency template
```

CLI: `python -m src.pipeline "I want to cancel my appointment"`. With no arguments it opens an
interactive prompt.

Tests run offline. Transformer interface tests use tiny, randomly initialized local checkpoints
(`tests/tiny_models.py`). Checks on the trained BERT and the real emotion model run automatically once
those artifacts exist, and are skipped otherwise.

## Results

See **[`evaluation/REPORT.md`](evaluation/REPORT.md)**. It is regenerated from saved artifacts, so no
number in it is typed by hand. Current state: baseline trained and evaluated; BERT row marked *not run*.

## Limitations

* **Synthetic data.** Every example was written for this project. The seed itself is a reconstruction.
  Real user traffic will have more noise, more multi-intent messages and more code-mixing. Test scores
  therefore measure performance on held-out *authored* queries and are likely optimistic.
* **Small held-out sets:** about 19 test examples per intent, so per-intent recall moves by about 0.05 per
  error.
* **OOD detection** relies on calibrated max-softmax plus a small authored OOD set. The baseline still
  accepts a large share of OOD-test queries (see the report). Max-softmax OOD detection is known to be
  weak.
* **Emotion threshold not validated on hospital data:** there are no emotion labels for this domain, so
  the 0.5 fallback uses the model's default. GoEmotions is Reddit-trained and may misread terse
  service-style messages.
* **Red-flag patterns** are a hand-written, English-only safety net. They were written alongside the data,
  so their measured recall is not a blind estimate.
* Single-turn only: no dialogue state, slot filling or real booking/report integration. Templates ask for
  details but nothing downstream consumes them yet.
* Multi-intent messages ("cancel Monday and book Friday") get a single label.

## Future improvements

* collect and label real, de-identified queries; re-audit boundaries with the confusion matrix
* add slot extraction (department, doctor, date) and backend integration for availability and reports
* stronger OOD detection: energy score, Mahalanobis distance on encoder features, or an explicit
  out-of-scope class trained on real negatives
* per-label thresholds for emotions calibrated on annotated hospital messages
* multilingual / code-mixed support (e.g. a multilingual DistilBERT) and human-handoff routing
* compare other compact encoders (MiniLM, BERT-small) and distillation or quantization for latency
