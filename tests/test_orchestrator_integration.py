#!/usr/bin/env python3
"""
Test suite for Router + Orchestrator Integration
Tests end-to-end flow: routing → execution → result formatting
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from router import route_agent, AgentSpec
from orchestrator import AgentOrchestrator, AgentExecutionResult


def get_test_agents():
    """Get standard test agent configuration"""
    return [
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


def check_api_keys():
    """Check which API keys are available"""
    has_grok = os.getenv("GROK_API_KEY") is not None
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    has_anthropic = os.getenv("ANTHROPIC_API_KEY") is not None

    print("API Keys Status:")
    print(f"  GROK_API_KEY: {'✓ Set' if has_grok else '✗ Not set'}")
    print(f"  OPENAI_API_KEY: {'✓ Set' if has_openai else '✗ Not set'}")
    print(f"  ANTHROPIC_API_KEY: {'✓ Set' if has_anthropic else '✗ Not set'}")

    if not (has_grok or has_openai or has_anthropic):
        print("\n⚠ WARNING: No API keys set. Tests will be skipped.")
        print("Set at least one API key to run integration tests:")
        print("  export GROK_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export ANTHROPIC_API_KEY='your-key'")
        return None

    # Select model based on available API keys
    if has_grok:
        return "grok-4-1-fast-non-reasoning-latest"
    elif has_openai:
        return "gpt-4o-mini"
    elif has_anthropic:
        return "claude-sonnet-4-5-20250929"


def test_routing_only():
    """Test routing without execution"""
    print("\n" + "=" * 60)
    print("TEST 1: Routing Only")
    print("=" * 60)

    model = check_api_keys()
    if not model:
        print("SKIPPED: No API keys available")
        return False

    agents = get_test_agents()

    test_cases = [
        ("paracetamol", "medication_agent"),
        ("appendectomy", "procedure_agent"),
        ("fever symptoms", "diagnostic_agent"),
    ]

    print(f"\nUsing model: {model}")

    all_passed = True
    for query, expected_agent in test_cases:
        try:
            print(f"\nQuery: '{query}'")
            result = route_agent(query, agents, model=model)
            status = "✓" if result == expected_agent else "⚠"
            print(f"{status} Routed to: {result} (expected: {expected_agent})")

            if result != expected_agent:
                all_passed = False

        except Exception as e:
            print(f"✗ Error: {e}")
            all_passed = False

    return all_passed


def test_full_execution():
    """Test full routing + execution flow"""
    print("\n" + "=" * 60)
    print("TEST 2: Full Execution (Routing + Agent)")
    print("=" * 60)

    model = check_api_keys()
    if not model:
        print("SKIPPED: No API keys available")
        return False

    agents = get_test_agents()
    orchestrator = AgentOrchestrator(llm_model=model)

    test_query = "paracetamol"

    print(f"\nUsing model: {model}")
    print(f"Test Query: '{test_query}'")

    try:
        # Step 1: Route
        print("\n[1/2] Routing query...")
        agent_id = route_agent(test_query, agents, model=model)
        agent = next(a for a in agents if a.id == agent_id)
        print(f"✓ Routed to: {agent_id} ({agent.name})")

        # Step 2: Execute
        print(f"\n[2/2] Executing {agent.name}...")
        result = orchestrator.execute_agent(agent_id, test_query)

        if result.success:
            print("✓ Execution successful!")
            print("\n" + "-" * 60)
            print("SUMMARY:")
            print("-" * 60)
            print(result.summary)
            print("-" * 60)
            return True
        else:
            print(f"✗ Execution failed: {result.error_message}")
            return False

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator_initialization():
    """Test orchestrator can be initialized with different models"""
    print("\n" + "=" * 60)
    print("TEST 3: Orchestrator Initialization")
    print("=" * 60)

    test_models = [
        "grok-4-1-fast-non-reasoning-latest",
        "gpt-4o-mini",
        "claude-sonnet-4-5-20250929"
    ]

    all_passed = True
    for model in test_models:
        try:
            orchestrator = AgentOrchestrator(llm_model=model)
            print(f"✓ Initialized with {model}")
            print(f"  Agents registered: {len(orchestrator.agent_metadata)}")
        except Exception as e:
            print(f"✗ Failed with {model}: {e}")
            all_passed = False

    return all_passed


def run_all_tests():
    """Run all integration tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  Router + Orchestrator Integration Test Suite" + " " * 11 + "║")
    print("╚" + "=" * 58 + "╝")

    results = []

    # Test 1: Routing only
    results.append(("Routing Only", test_routing_only()))

    # Test 2: Full execution
    results.append(("Full Execution", test_full_execution()))

    # Test 3: Orchestrator initialization
    results.append(("Orchestrator Init", test_orchestrator_initialization()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
