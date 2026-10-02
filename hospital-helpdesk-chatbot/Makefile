PY ?= python

.PHONY: install data train evaluate test app docker all

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m scripts.generate_dataset --per-intent 150 --seed 42
	$(PY) -m scripts.validate_dataset --val-size 0.15 --test-size 0.15 --seed 42

train:
	$(PY) -m src.train_intent

evaluate:
	$(PY) -m src.evaluate_intent --prefix eval

test:
	$(PY) -m pytest

app:
	streamlit run app.py

docker:
	docker build -t hospital-helpdesk-chatbot .

all: data train evaluate test
