#!/usr/bin/env python3
"""
Simple runner script for the Medical Fact Checker Agent
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_fact_checker_agent import MedicalFactChecker


def main():
    print("="*80)
    print("MEDICAL FACT CHECKER - Independent Bio-Investigator")
    print("="*80)
    print()
    print("This agent investigates health subjects with a skeptical eye on")
    print("corporate interests and institutional bias.")
    print()

    # Get subject from user
    subject = input("Enter the health subject to investigate: ").strip()

    if not subject:
        print("Error: Subject cannot be empty")
        return

    # Optional context
    print("\nOptional: Provide clarifying context or scope")
    print("(e.g., 'for vitamin D synthesis', 'skin cancer risk', etc.)")
    context = input("Context (press Enter to skip): ").strip()

    print("\nInitializing agent...")

    try:
        # Initialize agent in interactive mode
        agent = MedicalFactChecker(
            primary_llm_provider="claude",
            fallback_providers=["openai"],
            interactive=True
        )

        # Run analysis
        print(f"\nStarting analysis for: {subject}")
        print("="*80)

        session = agent.start_analysis(subject, context)

        # Display final output
        print("\n" + "="*80)
        print("FINAL OUTPUT")
        print("="*80)
        print()
        print(session.final_output)
        print()
        print("="*80)

        # Ask if user wants to export
        export_choice = input("\nWould you like to export this session? (y/n): ").strip().lower()
        if export_choice == 'y':
            filename = f"fact_check_{subject.replace(' ', '_')}_{session.started_at.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join("outputs", filename)

            # Create outputs directory if it doesn't exist
            os.makedirs("outputs", exist_ok=True)

            agent.export_session(filepath)
            print(f"\nSession exported to: {filepath}")

        print("\nAnalysis complete!")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
