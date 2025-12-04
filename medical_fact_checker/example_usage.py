#!/usr/bin/env python3
"""
Example usage of the Medical Fact Checker Agent
Demonstrates both interactive and programmatic usage patterns.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_fact_checker_agent import MedicalFactChecker, OutputType


def example_interactive():
    """Example: Interactive mode with user prompts"""
    print("=== Example 1: Interactive Mode ===\n")

    agent = MedicalFactChecker(
        primary_llm_provider="claude",
        interactive=True  # Will prompt user at each phase
    )

    session = agent.start_analysis(
        subject="Omega-6 fatty acids",
        clarifying_info="inflammation and cardiovascular health"
    )

    print(f"\nFinal output length: {len(session.final_output)} characters")
    print(f"Phases completed: {len(session.phase_results)}")


def example_non_interactive():
    """Example: Non-interactive mode with defaults"""
    print("=== Example 2: Non-Interactive Mode ===\n")

    agent = MedicalFactChecker(
        primary_llm_provider="claude",
        interactive=False  # Uses default choices
    )

    session = agent.start_analysis(
        subject="Red light therapy",
        clarifying_info="skin health and mitochondrial function"
    )

    # Display output
    print("ANALYSIS RESULT:")
    print("="*80)
    print(session.final_output)
    print("="*80)

    # Export session
    os.makedirs("outputs", exist_ok=True)
    agent.export_session("outputs/red_light_therapy_example.json")
    print("\nSession exported to outputs/red_light_therapy_example.json")


def example_programmatic():
    """Example: Programmatic usage with custom processing"""
    print("=== Example 3: Programmatic Usage ===\n")

    subjects = [
        ("Coffee consumption", "cognitive performance"),
        ("Intermittent fasting", "metabolic health"),
        ("Cold exposure", "brown fat activation")
    ]

    agent = MedicalFactChecker(
        primary_llm_provider="claude",
        interactive=False,
        enable_logging=False  # Reduce console output
    )

    results = []

    for subject, context in subjects:
        print(f"Analyzing: {subject} ({context})")

        try:
            session = agent.start_analysis(subject, context)
            results.append({
                'subject': subject,
                'context': context,
                'output_length': len(session.final_output),
                'phases': len(session.phase_results)
            })
            print(f"  ✓ Complete ({len(session.final_output)} chars)\n")
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            results.append({
                'subject': subject,
                'context': context,
                'error': str(e)
            })

    # Summary
    print("\nSummary:")
    print("-" * 60)
    for result in results:
        if 'error' not in result:
            print(f"{result['subject']}: {result['output_length']} chars, {result['phases']} phases")
        else:
            print(f"{result['subject']}: ERROR - {result['error']}")


def example_phase_inspection():
    """Example: Inspect individual phase results"""
    print("=== Example 4: Phase Inspection ===\n")

    agent = MedicalFactChecker(
        primary_llm_provider="claude",
        interactive=False
    )

    session = agent.start_analysis(
        subject="Saturated fat",
        clarifying_info="cardiovascular disease risk"
    )

    # Inspect each phase
    print("Phase-by-Phase Analysis:")
    print("="*80)

    for phase_result in session.phase_results:
        print(f"\nPhase: {phase_result.phase.value}")
        print(f"Timestamp: {phase_result.timestamp}")
        print(f"User choice: {phase_result.user_choice}")

        if phase_result.token_usage:
            print(f"Token usage: {phase_result.token_usage.total_tokens} total")

        print(f"Content keys: {list(phase_result.content.keys())}")

        # Display a sample of the content
        for key, value in phase_result.content.items():
            if value:
                preview = value[:200] + "..." if len(value) > 200 else value
                print(f"  {key}: {preview}")

        print("-" * 80)


def example_error_handling():
    """Example: Proper error handling"""
    print("=== Example 5: Error Handling ===\n")

    try:
        agent = MedicalFactChecker(
            primary_llm_provider="claude",
            interactive=False
        )

        # Try with empty subject (should fail validation)
        session = agent.start_analysis(
            subject="",
            clarifying_info=""
        )

    except ValueError as e:
        print(f"Validation error caught: {e}")
    except Exception as e:
        print(f"Error caught: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("Medical Fact Checker - Example Usage Patterns")
    print("="*80)
    print()

    # Run examples
    examples = [
        ("1", "Interactive Mode", example_interactive),
        ("2", "Non-Interactive Mode", example_non_interactive),
        ("3", "Programmatic Batch Processing", example_programmatic),
        ("4", "Phase Inspection", example_phase_inspection),
        ("5", "Error Handling", example_error_handling)
    ]

    print("Available examples:")
    for num, name, _ in examples:
        print(f"  [{num}] {name}")
    print()

    choice = input("Select example to run (1-5, or 'all'): ").strip()

    print("\n" + "="*80 + "\n")

    if choice.lower() == "all":
        for _, _, func in examples:
            try:
                func()
                print("\n" + "="*80 + "\n")
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"\nExample failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        for num, _, func in examples:
            if choice == num:
                try:
                    func()
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                except Exception as e:
                    print(f"\nExample failed: {e}")
                    import traceback
                    traceback.print_exc()
                break
        else:
            print(f"Invalid choice: {choice}")
