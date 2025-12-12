#!/usr/bin/env python3
"""
Check which LLM providers are available
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 70)
print("🤖 AVAILABLE LLM PROVIDERS")
print("=" * 70)
print()

# Check each provider
providers = [
    {
        "name": "Claude Sonnet 4.5",
        "cli": "claude-sonnet",
        "env_var": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-5-20250929",
        "cost": "$3 input / $15 output per 1M tokens"
    },
    {
        "name": "Claude Opus 4.5",
        "cli": "claude-opus",
        "env_var": "ANTHROPIC_API_KEY",
        "model": "claude-opus-4-5-20251101",
        "cost": "$15 input / $75 output per 1M tokens"
    },
    {
        "name": "OpenAI GPT-4",
        "cli": "openai",
        "env_var": "OPENAI_API_KEY",
        "model": "gpt-4-turbo-preview",
        "cost": "$3 input / $15 output per 1M tokens"
    },
    {
        "name": "Grok 4.1 Fast",
        "cli": "grok-4-1-fast",
        "env_var": "GROK_API_KEY",
        "model": "grok-4-1-fast-non-reasoning-latest",
        "cost": "$0.20 input / $0.50 output per 1M tokens"
    },
    {
        "name": "Grok 4.1 Reasoning",
        "cli": "grok-4-1-reasoning",
        "env_var": "GROK_API_KEY",
        "model": "grok-4-1-fast-reasoning-latest",
        "cost": "$0.20 input / $0.50 output per 1M tokens"
    },
    {
        "name": "Grok 4.1 Code",
        "cli": "grok-4-1-code",
        "env_var": "GROK_API_KEY",
        "model": "grok-code-fast",
        "cost": "$0.20 input / $1.50 output per 1M tokens"
    },
    {
        "name": "Ollama (local)",
        "cli": "ollama",
        "env_var": None,
        "model": "llama2:13b (or other local models)",
        "cost": "Free (runs locally)"
    }
]

available = []
unavailable = []

for provider in providers:
    if provider["env_var"]:
        api_key = os.getenv(provider["env_var"])
        if api_key:
            status = "✅ Available"
            key_preview = f" ({api_key[:8]}...)" if len(api_key) > 8 else ""
            available.append(provider)
        else:
            status = "❌ Not configured"
            key_preview = f" (Set {provider['env_var']})"
            unavailable.append(provider)
    else:
        status = "⚙️  Local"
        key_preview = " (No API key needed)"
        available.append(provider)

    print(f"{status} {provider['name']}")
    print(f"    CLI: --llm {provider['cli']}")
    print(f"    Model: {provider['model']}")
    print(f"    Cost: {provider['cost']}")
    print(f"    {key_preview}")
    print()

print("=" * 70)
print(f"📊 SUMMARY: {len(available)} available, {len(unavailable)} not configured")
print("=" * 70)
print()

if available:
    print("✅ Ready to use:")
    for p in available:
        print(f"   uv run python run_analysis.py factcheck --subject \"topic\" --llm {p['cli']}")
    print()

if unavailable:
    print("⚙️  To enable additional providers:")
    for p in unavailable:
        print(f"   export {p['env_var']}=\"your-api-key\"")
    print()
