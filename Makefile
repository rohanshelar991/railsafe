.PHONY: test scenario evaluation

PYTHON ?= .venv/bin/python

test:
	$(PYTHON) -m pytest -q

scenario:
	$(PYTHON) evaluation/run_scenarios.py

evaluation:
	$(PYTHON) evaluation/ablation.py
	$(PYTHON) evaluation/evaluate.py
