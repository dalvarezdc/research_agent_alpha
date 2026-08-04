#!/usr/bin/env bash
# Usage: ./setup.sh
# Automated setup script for Medical Analysis Agents (Research Agent Alpha)
# Checks Python 3.12+, syncs dependencies via uv (or venv/pip fallback),
# initializes .env template, and prepares workspace directories.

set -e

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: ./setup.sh"
  echo ""
  echo "Automated setup script that:"
  echo "  1. Checks for Python 3.12+"
  echo "  2. Installs dependencies using 'uv' (or creates .venv with pip)"
  echo "  3. Generates a template .env file if missing"
  echo "  4. Ensures outputs/ and cache/ directories exist"
  exit 0
fi

echo "🏥 Medical Analysis Agents — Setup"
echo "=================================="

# --- Check Python ---
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found. Install Python 3.12+ from https://python.org"
  exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]; }; then
  echo "❌ Python 3.12+ required (found $PYTHON_VERSION)"
  exit 1
fi
echo "✅ Python $PYTHON_VERSION"

# --- Python Dependencies & Environment ---
if command -v uv &>/dev/null; then
  echo "✅ uv package manager found"
  echo "📦 Syncing dependencies with uv..."
  uv sync --extra parsing-extras --extra dev
  echo "✅ Python dependencies installed via uv"
else
  echo "⚠️  uv not found (recommended: https://docs.astral.sh/uv/)"
  if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
  fi
  echo "📦 Installing Python dependencies with pip..."
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -e ".[dev,parsing-extras]"
  echo "✅ Python dependencies installed via pip"
fi

# --- Environment File (.env) ---
if [ ! -f ".env" ]; then
  echo "🔑 Creating .env template file..."
  cat << 'EOF' > .env
# Primary LLM API Keys (Set at least one)
XAI_API_KEY=""           # Grok-4.5 (default), Grok-4.3
ANTHROPIC_API_KEY=""      # Claude 3.5 Sonnet / Claude 3 Opus
OPENAI_API_KEY=""         # GPT-4o

# Web Search API Keys (Optional - Tavily recommended; DuckDuckGo works with no key)
TAVILY_API_KEY=""
SERPAPI_API_KEY=""

# Vertex AI / GCP (Optional)
# VERTEX_PROJECT=""
# VERTEX_LOCATION="us-east5"
# GLOBAL_SERVICE=true
EOF
  echo "✅ .env created (please edit to add your API key)"
else
  echo "✅ .env already exists — skipping"
fi

# --- Output & Cache Directories ---
echo "📁 Ensuring output and cache directories exist..."
mkdir -p outputs cache
echo "✅ Workspace directories ready"

echo ""
echo "✨ Setup complete!"
echo ""
echo "To check configured LLM providers:"
echo "  uv run python router.py --check-llms"
echo ""
echo "To start the interactive router:"
echo "  uv run python router.py"
echo ""
echo "To start the REST API server:"
echo "  uv run python api.py --port 8000"
echo ""
echo "To run tests:"
echo "  uv run python -m pytest tests/"
