# Shortcuts for common tasks. Run `make help` to list them.
# Use another Python with e.g.  make test PYTHON=python3.11
PYTHON ?= python

.PHONY: help install test lint profile-sample api

help:            ## show this list
	@grep -E "^[a-z-]+:.*##" Makefile | sed "s/:.*##/ -/"

install:         ## install the project requirements (inside your venv)
	$(PYTHON) -m pip install -r requirements.txt

test:            ## run all tests
	$(PYTHON) -m pytest -q

lint:            ## check the code with ruff
	$(PYTHON) -m ruff check .

profile-sample:  ## profile the committed 1,000-row bank sample -> results/sample/
	$(PYTHON) run_profiling.py --input data/raw/bank_marketing_sample.csv --output results/sample

api:             ## run the API on http://127.0.0.1:8000 (docs at /docs)
	$(PYTHON) -m uvicorn src.api.app:app --reload
