.PHONY: help setup smoke tables full dynamic portfolio static manuscript checksums verify-checksums clean-generated

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

help:
	@echo "Available targets:"
	@echo "  setup            Create/update Python dependencies from requirements.txt"
	@echo "  smoke            Run lightweight import/artifact smoke test"
	@echo "  tables           Rebuild tables and manuscript from existing artifacts"
	@echo "  full             Run full reproduction pipeline"
	@echo "  static           Run static ZDT sanity benchmarks"
	@echo "  dynamic          Run dynamic DF benchmark suite"
	@echo "  portfolio        Run walk-forward portfolio stress-test suite"
	@echo "  manuscript       Rebuild graphical abstract and manuscript PDF"
	@echo "  checksums        Regenerate SHA256SUMS.txt"
	@echo "  verify-checksums Verify SHA256SUMS.txt"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

smoke:
	$(PYTHON) experiments/smoke_test.py

tables:
	bash reproduce_all.sh --tables-only

full:
	bash reproduce_all.sh

static:
	(cd experiments && $(PYTHON) run_benchmarks.py)

dynamic:
	(cd experiments && $(PYTHON) debug_predictive_baselines.py && $(PYTHON) run_dynamic_asoc_suite.py && $(PYTHON) validate_dynamic_activation_audit.py && $(PYTHON) run_dynamic_pitstop_budget_probe.py && $(PYTHON) rebuild_dynamic_asoc_reports.py && $(PYTHON) run_dynamic_sensitivity_asoc.py)

portfolio:
	(cd experiments && $(PYTHON) run_portfolio_asoc_suite.py && $(PYTHON) validate_portfolio_results.py && $(PYTHON) rebuild_portfolio_reports.py)

manuscript:
	(cd manuscript && $(PYTHON) build_graphical_abstract.py && pdflatex -interaction=nonstopmode -halt-on-error asoc_fito.tex && pdflatex -interaction=nonstopmode -halt-on-error asoc_fito.tex)

checksums:
	$(PYTHON) scripts/generate_checksums.py

verify-checksums:
	$(PYTHON) scripts/verify_checksums.py

clean-generated:
	@echo "Use with care: generated results are part of the submission snapshot."
	@echo "No destructive cleanup is performed by default."
