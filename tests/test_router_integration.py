#!/usr/bin/env python3
"""
Test script for router with real LLM integration.
This tests the call_model function and router agent selection.

NOTE: Requires API keys to be set:
- GROK_API_KEY for Grok models
- OPENAI_API_KEY for OpenAI models
- ANTHROPIC_API_KEY for Claude models
"""

import os
from router import route_agent, AgentSpec

def test_router_with_llm():
    """Test router with real LLM call"""

    # Define test agents
    test_agents = [
        AgentSpec(
            id="medication_agent",
            name="Medication Specialist",
            description="Handles queries about medications, drugs, dosages, side effects, and prescriptions",
            routing_notes="Use for pharmaceutical and medication-related questions"
        ),
        AgentSpec(
            id="procedure_agent",
            name="Medical Procedure Specialist",
            description="Handles queries about medical procedures, surgeries, and treatments",
            routing_notes="Use for procedural and interventional medical questions"
        ),
        AgentSpec(
            id="diagnostic_agent",
            name="Diagnostic Specialist",
            description="Handles queries about symptoms, diagnoses, and medical conditions",
            routing_notes="Use for diagnostic and condition-related questions"
        ),
        AgentSpec(
            id="general_agent",
            name="General Medical Assistant",
            description="Handles general medical queries that don't fit other specialized categories"
        )
    ]

    # Test queries
    test_queries = [
        ("What is paracetamol used for?", "medication_agent"),
        ("How is appendectomy performed?", "procedure_agent"),
        ("What causes fever?", "diagnostic_agent"),
    ]

    print("Testing Router with Real LLM Integration")
    print("=" * 60)

    # Check for API keys
    has_grok = os.getenv("GROK_API_KEY") is not None
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    has_anthropic = os.getenv("ANTHROPIC_API_KEY") is not None

    print(f"\nAPI Keys Status:")
    print(f"  GROK_API_KEY: {'✓ Set' if has_grok else '✗ Not set'}")
    print(f"  OPENAI_API_KEY: {'✓ Set' if has_openai else '✗ Not set'}")
    print(f"  ANTHROPIC_API_KEY: {'✓ Set' if has_anthropic else '✗ Not set'}")

    if not (has_grok or has_openai or has_anthropic):
        print("\n⚠ WARNING: No API keys set. Tests will fail.")
        print("Set at least one API key to test the router.")
        return

    # Select model based on available API keys
    if has_grok:
        model = "grok-4-1-fast-non-reasoning-latest"
    elif has_openai:
        model = "gpt-4o-mini"
    elif has_anthropic:
        model = "claude-sonnet-4-5-20250929"

    print(f"\nUsing model: {model}")
    print("\nRunning tests...\n")

    for query, expected_agent in test_queries:
        try:
            print(f"Query: {query}")
            result = route_agent(
                query,
                test_agents,
                default_agent_id="general_agent",
                model=model
            )
            agent = next(a for a in test_agents if a.id == result)

            status = "✓" if result == expected_agent else "⚠"
            print(f"{status} Routed to: {result} ({agent.name})")
            if result != expected_agent:
                print(f"  Expected: {expected_agent}")
            print()

        except Exception as e:
            print(f"✗ Error: {e}\n")


if __name__ == "__main__":
    test_router_with_llm()
