PYTHON ?= python3

.PHONY: help install-dev install-runtime test lint validate-configs check-docs verify-results check

help:
	@echo "install-dev      Install the lightweight code-reading and test environment"
	@echo "install-runtime  Install the validated vLLM experiment stack and local package"
	@echo "test             Run CPU unit and contract tests"
	@echo "lint             Run Ruff over maintained Python code"
	@echo "validate-configs Validate maintained experiment JSON files"
	@echo "check-docs       Check local Markdown links"
	@echo "verify-results   Verify tracked result bundle checksums"
	@echo "check            Run all local, non-GPU checks"

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-runtime:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q -p no:cacheprovider

lint:
	$(PYTHON) -m ruff check waveslice experiments profiler scripts tests tools

validate-configs:
	$(PYTHON) tools/validate_configs.py

check-docs:
	$(PYTHON) tools/check_markdown_links.py

verify-results:
	$(PYTHON) tools/verify_result_bundles.py

check: lint test validate-configs check-docs verify-results
