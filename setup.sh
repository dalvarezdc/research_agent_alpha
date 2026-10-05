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

# --- Ensure pyenv shims are in PATH if pyenv exists ---
if [ -d "$HOME/.pyenv/shims" ] && [[ ":$PATH:" != *":$HOME/.pyenv/shims:"* ]]; then
  export PATH="$HOME/.pyenv/shims:$PATH"
fi
if command -v pyenv &>/dev/null; then
  eval "$(pyenv init - 2>/dev/null)" || true
fi

# --- Check & Locate Python 3.12+ ---
PYTHON_BIN=""

for candidate in python3.12 python3.13 python3 python "$HOME/.pyenv/shims/python3" "$HOME/.pyenv/shims/python"; do
  if command -v "$candidate" &>/dev/null; then
    BIN_PATH=$(command -v "$candidate")
    if "$BIN_PATH" -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
      PYTHON_BIN="$BIN_PATH"
      break
    fi
  elif [ -x "$candidate" ]; then
    if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  CURRENT_PY=$(command -v python3 &>/dev/null && python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" || echo "unknown")
  echo "❌ Python 3.12+ required (found $CURRENT_PY at $(command -v python3 || echo 'none'))"
  echo "💡 Tip: If using pyenv, ensure 'pyenv global 3.12.3' is set and shims are in PATH (~/.pyenv/shims)."
  exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "✅ Found Python $PYTHON_VERSION ($PYTHON_BIN)"

# --- Python Dependencies & Environment ---
if command -v uv &>/dev/null; then
  echo "✅ uv package manager found"
  echo "📦 Syncing dependencies with uv..."
  uv sync --python "$PYTHON_BIN" --extra parsing-extras --extra dev
  echo "✅ Python dependencies installed via uv"
else
  echo "⚠️  uv not found (recommended: https://docs.astral.sh/uv/)"
  if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    "$PYTHON_BIN" -m venv .venv
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

# --- Agent-ignore pre-commit hook ---
# Installs .githooks/pre-commit into the active hooks directory. A hook that is
# already there (for example a beads shim) is kept; the checker runs first.
ROOT="$(cd "$(dirname "$0")" && pwd)"
if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  (
    cd "$ROOT"
    hooks_dir="$(git rev-parse --git-path hooks)"
    mkdir -p "$hooks_dir"
    dest="$hooks_dir/pre-commit"
    prepend_checker() {
      tmp="$(mktemp)"
      cat > "$tmp" << 'EOF'
#!/bin/sh
# .agentignore gate. Runs before any hook that was already installed.
root=$(git rev-parse --show-toplevel)
checker="$root/scripts/check_agentignore.py"
if [ -f "$checker" ]; then
  python3 "$checker" --staged || exit $?
fi

EOF
      cat "$dest" >> "$tmp"
      mv "$tmp" "$dest"
      chmod 755 "$dest"
    }
    if [ -L "$dest" ]; then
      rm -f "$dest"
    fi
    if [ -f "$dest" ]; then
      if grep -q "check_agentignore.py" "$dest"; then
        echo "✅ .agentignore pre-commit hook already installed"
      else
        prepend_checker
        echo "✅ Chained .agentignore check in front of the existing pre-commit hook"
      fi
    elif command -v bd >/dev/null 2>&1 && printf '%s' "$hooks_dir" | grep -q '\.beads/hooks'; then
      cat > "$dest" << 'EOF'
#!/bin/sh
# .agentignore gate, then the beads pre-commit shim.
root=$(git rev-parse --show-toplevel)
checker="$root/scripts/check_agentignore.py"
if [ -f "$checker" ]; then
  python3 "$checker" --staged || exit $?
fi
if command -v bd >/dev/null 2>&1; then
  export BD_GIT_HOOK=1
  bd hooks run pre-commit "$@"
fi
EOF
      chmod 755 "$dest"
      echo "✅ Installed .agentignore check chained with beads"
    else
      ln -sfn "$ROOT/.githooks/pre-commit" "$dest"
      echo "✅ Installed .agentignore pre-commit hook"
    fi
  )
fi

# --- Optional System Dependencies (WeasyPrint PDF export on macOS) ---
if [[ "$OSTYPE" == "darwin"* ]]; then
  if ! command -v brew &>/dev/null || ! brew list pango &>/dev/null 2>&1; then
    echo "💡 Note: For optional PDF report generation via WeasyPrint on macOS, install C-libraries with Homebrew:"
    echo "   brew install pango gdk-pixbuf cairo libffi"
  fi
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "Quick start with Makefile:"
echo "  make run          # Start interactive router"
echo "  make api          # Start REST API server (http://localhost:8080)"
echo "  make check-llms   # Check configured LLM providers"
echo "  make test         # Run test suite"
echo ""
echo "Or run directly:"
echo "  uv run python router.py"
echo "  uv run python api.py --port 8080"
