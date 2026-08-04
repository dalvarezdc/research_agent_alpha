.PHONY: help setup router run check-llms api api-dev test test-verbose lint clean

# Default target
help:
	@echo "🏥 Medical Analysis Agents — Commands"
	@echo "=================================================="
	@echo "  make setup        Run automated setup script"
	@echo "  make run          Start interactive router (default)"
	@echo "  make check-llms   Verify configured LLM provider API keys"
	@echo "  make api          Start REST API server (port 8000)"
	@echo "  make api-dev      Start REST API server with auto-reload"
	@echo "  make test         Run test suite"
	@echo "  make test-verbose Run test suite in verbose mode"
	@echo "  make lint         Run ruff linter checks"
	@echo "  make clean        Remove build artifacts and bytecode caches"

setup:
	@chmod +x setup.sh
	@./setup.sh

run: router

router:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python router.py; \
	else \
		python3 router.py; \
	fi

check-llms:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python router.py --check-llms; \
	else \
		python3 router.py --check-llms; \
	fi

api:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python api.py --port 8000; \
	else \
		python3 api.py --port 8000; \
	fi

api-dev:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python api.py --reload --port 8000; \
	else \
		python3 api.py --reload --port 8000; \
	fi

test:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m pytest tests/; \
	else \
		python3 -m pytest tests/; \
	fi

test-verbose:
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m pytest tests/ -v; \
	else \
		python3 -m pytest tests/ -v; \
	fi

lint:
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff check .; \
	else \
		ruff check .; \
	fi

clean:
	@echo "🧹 Cleaning temporary files and caches..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache 2>/dev/null || true
	@echo "✅ Clean completed"
