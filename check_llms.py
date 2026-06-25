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


# Provider-level presentation metadata. The actual model IDs and pricing are
# pulled from the canonical registries (llm_integrations.get_available_models +
# cost_tracker.PRICING) so this file can never drift from runtime behavior.
#
# Each entry maps a CLI provider key -> (display name, env var, representative
# model ID). The representative model is the one create_llm_manager() actually
# instantiates for that provider.
_PROVIDER_PRESENTATION: list[dict[str, Any]] = [
    {
        "name": "Claude Sonnet",
        "cli": "claude-sonnet",
        "env_var": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
    },
    {
        "name": "Claude Opus (current flagship)",
        "cli": "claude-opus",
        "env_var": "ANTHROPIC_API_KEY",
        "model": "claude-opus-4-8",
    },
    {
        "name": "OpenAI GPT-4o",
        "cli": "openai",
        "env_var": "OPENAI_API_KEY",
        "model": "gpt-4o",
    },
    {
        "name": "Grok 4.3 (current xAI flagship)",
        "cli": "grok-4.3",
        "env_var": "GROK_API_KEY",
        "model": "grok-4.3",
    },
    {
        "name": "Ollama (local)",
        "cli": "ollama",
        "env_var": None,
        "model": "llama2:13b",
    },
    {
        "name": "Gemini 3.5 Flash (Vertex AI, reasoning levels)",
        "cli": "gemini-vertex",
        "env_var": "VERTEX_PROJECT",
        "model": "gemini-3.5-flash",
    },
]


def _format_cost(model: str) -> str:
    """Derive a human-readable cost string from the canonical PRICING table."""
    try:
        from cost_tracker import PRICING
    except Exception:  # pragma: no cover - cost_tracker should always import
        return "Pricing unavailable"

    pricing = PRICING.get(model)
    if pricing is None:
        return "Free (runs locally)" if model.startswith("llama") else "Pricing unavailable"

    return (
        f"${pricing['input']:g} input / ${pricing['output']:g} output per 1M tokens"
    )


def get_llm_provider_definitions() -> list[dict[str, Any]]:
    """Build provider definitions from the canonical model + pricing registries.

    The representative model IDs are validated against
    ``llm_integrations.get_available_models()`` so any drift surfaces as a
    ``(unknown model)`` marker rather than silently disagreeing with runtime.
    """
    try:
        from llm_integrations import get_available_models

        known_models = get_available_models()
    except Exception:  # pragma: no cover
        known_models = {}

    definitions: list[dict[str, Any]] = []
    for entry in _PROVIDER_PRESENTATION:
        model = entry["model"]
        # Ollama is local and intentionally absent from get_available_models cost-wise
        model_label = model
        if known_models and model not in known_models and not model.startswith("llama"):
            model_label = f"{model} (unknown model)"

        definitions.append(
            {
                "name": entry["name"],
                "cli": entry["cli"],
                "env_var": entry["env_var"],
                "model": model_label,
                "cost": _format_cost(model),
            }
        )

    return definitions

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
