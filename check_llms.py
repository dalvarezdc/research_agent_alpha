#!/usr/bin/env python3
"""
Check which LLM providers are available
"""

import os
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is None:
        return
    load_dotenv()

def get_llm_provider_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "Claude Sonnet 4.5",
            "cli": "claude-sonnet",
            "env_var": "ANTHROPIC_API_KEY",
            "model": "claude-sonnet-4-5-20250929",
            "cost": "$3 input / $15 output per 1M tokens",
        },
        {
            "name": "Claude Opus 4.5",
            "cli": "claude-opus",
            "env_var": "ANTHROPIC_API_KEY",
            "model": "claude-opus-4-5-20251101",
            "cost": "$15 input / $75 output per 1M tokens",
        },
        {
            "name": "OpenAI GPT-4",
            "cli": "openai",
            "env_var": "OPENAI_API_KEY",
            "model": "gpt-4-turbo-preview",
            "cost": "$3 input / $15 output per 1M tokens",
        },
        {
            "name": "Grok 4.1 Fast",
            "cli": "grok-4-1-fast",
            "env_var": "GROK_API_KEY",
            "model": "grok-4-1-fast-non-reasoning-latest",
            "cost": "$0.20 input / $0.50 output per 1M tokens",
        },
        {
            "name": "Grok 4.1 Reasoning",
            "cli": "grok-4-1-reasoning",
            "env_var": "GROK_API_KEY",
            "model": "grok-4-1-fast-reasoning-latest",
            "cost": "$0.20 input / $0.50 output per 1M tokens",
        },
        {
            "name": "Grok 4.1 Code",
            "cli": "grok-4-1-code",
            "env_var": "GROK_API_KEY",
            "model": "grok-code-fast",
            "cost": "$0.20 input / $1.50 output per 1M tokens",
        },
        {
            "name": "Ollama (local)",
            "cli": "ollama",
            "env_var": None,
            "model": "llama2:13b (or other local models)",
            "cost": "Free (runs locally)",
        },
    ]

def get_llm_provider_status(load_env: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if load_env:
        _load_env()

    available: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []

    for provider in get_llm_provider_definitions():
        env_var = provider.get("env_var")
        if env_var:
            api_key = os.getenv(env_var)
            if api_key:
                available.append(provider)
            else:
                unavailable.append(provider)
        else:
            available.append(provider)

    return available, unavailable

def print_llm_status(load_env: bool = True) -> int:
    available, unavailable = get_llm_provider_status(load_env=load_env)

    print("=" * 70)
    print("🤖 AVAILABLE LLM PROVIDERS")
    print("=" * 70)
    print()

    for provider in available + unavailable:
        env_var = provider.get("env_var")
        if env_var:
            api_key = os.getenv(env_var)
            if api_key:
                status = "✅ Available"
                key_preview = f" ({api_key[:8]}...)" if len(api_key) > 8 else ""
            else:
                status = "❌ Not configured"
                key_preview = f" (Set {env_var})"
        else:
            status = "⚙️  Local"
            key_preview = " (No API key needed)"

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
            print(
                f"   uv run python run_analysis.py factcheck --subject \"topic\" --llm {p['cli']}"
            )
        print()

    if unavailable:
        print("⚙️  To enable additional providers:")
        for p in unavailable:
            env_var = p.get("env_var")
            if env_var:
                print(f"   export {env_var}=\"your-api-key\"")
        print()

    return 0


def main() -> int:
    return print_llm_status(load_env=True)


if __name__ == "__main__":
    raise SystemExit(main())
