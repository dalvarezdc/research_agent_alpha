#!/usr/bin/env python3
"""
Medical Fact Checker Agent
An independent bio-investigator that uncovers unfiltered biological reality.
Uses DSPy for structured LLM interactions and supports interactive phase-based analysis.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from medical_procedure_analyzer.llm_integrations import LLMManager, create_llm_manager
from medical_procedure_analyzer.medical_reasoning_agent import TokenUsage

# Add parent directory to path for cost_tracker import
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_tracker import track_cost, print_cost_summary, reset_tracking


class AnalysisPhase(Enum):
    """Phases of the medical fact checking process"""
    CONFLICT_SCAN = "conflict_scan"
    EVIDENCE_STRESS_TEST = "evidence_stress_test"
    SYNTHESIS_MENU = "synthesis_menu"
    COMPLEX_OUTPUT = "complex_output"
    SIMPLIFIED_OUTPUT = "simplified_output"


class OutputType(Enum):
    """Types of final output formats"""
    EVOLUTIONARY = "A"  # Nature-first, evolutionary guide
    BIOHACKER = "B"  # Optimization-focused, cutting-edge
    PARADIGM_SHIFT = "C"  # Corrected text showing how consensus is wrong
    VILLAGE_WISDOM = "D"  # Simplified, traditional knowledge
    PROCEED = "P"  # Direct simplified output


@dataclass
class PhaseResult:
    """Result from a phase of analysis"""
    phase: AnalysisPhase
    timestamp: datetime
    content: Dict[str, Any]
    user_choice: Optional[str] = None
    token_usage: Optional[TokenUsage] = None


@dataclass
class FactCheckSession:
    """Complete fact-checking session data"""
    subject: str
    started_at: datetime = field(default_factory=datetime.now)
    phase_results: List[PhaseResult] = field(default_factory=list)
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    final_output: Optional[str] = None
    validation_report: Optional[Any] = None  # Reference validation report


# DSPy Signatures for different phases
class ConflictScanSignature(dspy.Signature):
    """Identify the official vs counter-narrative for a health subject"""
    subject = dspy.InputField(desc="Health subject to investigate")
    context = dspy.InputField(desc="Additional context or clarifying information")

    official_narrative = dspy.OutputField(desc="What mainstream medicine and authorities say")
    counter_narrative = dspy.OutputField(desc="What independent researchers and biohackers suspect")
    key_conflicts = dspy.OutputField(desc="Main points of disagreement between narratives")


class EvidenceAnalysisSignature(dspy.Signature):
    """Analyze evidence with focus on funding sources and methodology"""
    subject = dspy.InputField(desc="Health subject being investigated")
    angle = dspy.InputField(desc="Which perspective to prioritize: Official/Independent/Both")

    industry_funded_studies = dspy.OutputField(desc="Key studies funded by manufacturers")
    independent_research = dspy.OutputField(desc="Small lab or independent studies")
    methodology_quality = dspy.OutputField(desc="Methodological strengths and weaknesses")
    anecdotal_signals = dspy.OutputField(desc="Patterns from clinical observation and user reports")
    time_weighted_evidence = dspy.OutputField(desc="Most recent vs older research comparison")


class SynthesisSignature(dspy.Signature):
    """Synthesize findings into actionable insights"""
    subject = dspy.InputField(desc="Health subject")
    evidence_summary = dspy.InputField(desc="Summary of evidence analysis")

    biological_truth = dspy.OutputField(desc="Most plausible reality based on evidence and evolutionary logic")
    industry_bias = dspy.OutputField(desc="Where profit motives may distort safety/efficacy data")
    grey_zone = dspy.OutputField(desc="Promising hypotheses that lack gold standard proof but are safe")


class MedicalFactChecker:
    """
    Interactive medical fact checker following the "Independent Bio-Investigator" protocol.
    """

    def __init__(self,
                 primary_llm_provider: str = "claude",
                 fallback_providers: List[str] = None,
                 enable_logging: bool = True,
                 interactive: bool = True,
                 enable_reference_validation: bool = False):
        """
        Initialize the medical fact checker.

        Args:
            primary_llm_provider: Primary LLM to use (claude, openai, etc.)
            fallback_providers: List of fallback LLM providers
            enable_logging: Whether to enable detailed logging
            interactive: Whether to pause for user input between phases
            enable_reference_validation: Validate evidence sources
        """
        self.interactive = interactive
        self.enable_reference_validation = enable_reference_validation
        self.reference_validator = None

        # Setup logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True

        # Initialize reference validator if enabled
        if enable_reference_validation:
            try:
                from reference_validation import ReferenceValidator, ValidationConfig
                self.reference_validator = ReferenceValidator(ValidationConfig(
                    cache_backend="sqlite",
                    min_credibility_score=75  # Higher for fact-checking
                ))
            except ImportError:
                self.logger.warning("Reference validation not available") if enable_logging else None

        # Initialize LLM manager
        try:
            self.llm_manager = create_llm_manager(
                primary_provider=primary_llm_provider,
                fallback_providers=fallback_providers or ["openai"]
            )
            self.logger.info(f"LLM manager initialized with {primary_llm_provider}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM manager: {e}")
            raise

        # Session tracking
        self.current_session: Optional[FactCheckSession] = None

    def start_analysis(self, subject: str, clarifying_info: str = "") -> FactCheckSession:
        """
        Start a new fact-checking analysis session.

        Args:
            subject: The health subject to investigate
            clarifying_info: Optional clarifying information about scope

        Returns:
            FactCheckSession: The analysis session object
        """
        self.logger.info(f"Starting analysis for subject: {subject}")
        self.current_session = FactCheckSession(subject=subject)

        # Reset cost tracking for new analysis
        reset_tracking()

        # Initialize token usage tracker for this analysis
        self.total_token_usage = TokenUsage()

        # Phase 1: Conflict & Hypothesis Scan
        phase1_result = self._phase1_conflict_scan(subject, clarifying_info)
        self.current_session.phase_results.append(phase1_result)

        if self.interactive:
            user_choice = self._prompt_user_phase1()
            phase1_result.user_choice = user_choice
        else:
            phase1_result.user_choice = "Both"

        # Phase 2: Evidence Stress-Test
        phase2_result = self._phase2_evidence_stress_test(
            subject,
            phase1_result.content,
            phase1_result.user_choice
        )
        self.current_session.phase_results.append(phase2_result)

        if self.interactive:
            user_choice = self._prompt_user_phase2(phase2_result.content)
            phase2_result.user_choice = user_choice
        else:
            phase2_result.user_choice = "Proceed"

        # Phase 3: Synthesis & Menu
        phase3_result = self._phase3_synthesis_menu(
            subject,
            phase1_result.content,
            phase2_result.content
        )
        self.current_session.phase_results.append(phase3_result)

        if self.interactive:
            output_choice = self._prompt_user_phase3()
            phase3_result.user_choice = output_choice
        else:
            phase3_result.user_choice = "P"  # Default to simplified output

        # Phase 4: Generate output based on choice
        output_type = OutputType(phase3_result.user_choice)
        final_output = self._phase4_generate_output(
            subject,
            phase3_result.content,
            output_type
        )

        # Phase 5: Simplified output (if needed)
        if output_type != OutputType.PROCEED:
            final_output = self._phase5_simplify_output(final_output)

        self.current_session.final_output = final_output

        # Validate references if enabled
        if self.enable_reference_validation and self.reference_validator:
            self.current_session.validation_report = self.reference_validator.validate_analysis(self.current_session)

        # Print cost summary
        print_cost_summary()

        return self.current_session

    @track_cost("Phase 1: Conflict Scan")
    def _phase1_conflict_scan(self, subject: str, context: str) -> PhaseResult:
        """Phase 1: Identify official vs counter-narrative"""
        self.logger.info("=== PHASE 1: Conflict & Hypothesis Scan ===")

        prompt = f"""
        Analyze the health subject: {subject}

        Context: {context if context else "General investigation"}

        Provide:
        1. Official Narrative: What mainstream medicine/government agencies say
        2. Counter-Narrative: What independent researchers/biohackers suspect
        3. Key Conflicts: Main points of disagreement

        Apply these biases:
        - Prioritize recent research (last 5-10 years) over old dogma
        - Penalize studies with financial conflicts of interest
        - Consider evolutionary biology as a tie-breaker
        - Do not dismiss anecdotal evidence - label it as "Emerging Signal"
        - Favor natural mechanisms over synthetic when efficacy is comparable
        """

        system_prompt = """You are an independent medical researcher and "Red Teamer"
        uncovering unfiltered biological reality. You are skeptical of consensus driven
        by inertia or corporate interest. Weight methodological quality over institutional authority."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            # Parse the response
            content = self._parse_conflict_scan_response(response)

            return PhaseResult(
                phase=AnalysisPhase.CONFLICT_SCAN,
                timestamp=datetime.now(),
                content=content,
                token_usage=token_usage
            )
        except Exception as e:
            self.logger.error(f"Phase 1 failed: {e}")
            raise

    @track_cost("Phase 2: Evidence Stress Test")
    def _phase2_evidence_stress_test(self, subject: str, phase1_content: Dict, angle: str) -> PhaseResult:
        """Phase 2: Deep evidence analysis with funding and methodology focus"""
        self.logger.info("=== PHASE 2: Evidence Stress-Test ===")

        prompt = f"""
        Deep evidence analysis for: {subject}
        Priority angle: {angle}

        Official narrative: {phase1_content.get('official_narrative', '')}
        Counter-narrative: {phase1_content.get('counter_narrative', '')}

        Analyze:
        1. Funding Filter: Flag studies funded by manufacturers
        2. Methodology Audit: Evaluate independent studies from small labs
        3. Time Weighting: Prioritize 2020-2025 research over older beliefs
        4. Anecdotal Forensics: Search for clinical pearls and user experience patterns

        If small lab study has rigorous design but contradicts industry-funded study,
        highlight the small lab finding as "Priority Signal".
        """

        system_prompt = """You are a medical research auditor focused on uncovering
        methodological quality and funding biases. Small, well-designed independent studies
        trump large industry-funded studies. Recent evidence overturns old dogma."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            content = self._parse_evidence_response(response)

            return PhaseResult(
                phase=AnalysisPhase.EVIDENCE_STRESS_TEST,
                timestamp=datetime.now(),
                content=content,
                token_usage=token_usage
            )
        except Exception as e:
            self.logger.error(f"Phase 2 failed: {e}")
            raise

    @track_cost("Phase 3: Synthesis Menu")
    def _phase3_synthesis_menu(self, subject: str, phase1_content: Dict, phase2_content: Dict) -> PhaseResult:
        """Phase 3: Synthesize findings and prepare menu options"""
        self.logger.info("=== PHASE 3: Synthesis & Menu ===")

        prompt = f"""
        Synthesize findings for: {subject}

        Official vs Counter-Narrative:
        {phase1_content.get('official_narrative', '')}
        vs
        {phase1_content.get('counter_narrative', '')}

        Evidence Analysis:
        - Industry-funded: {phase2_content.get('industry_funded_studies', '')}
        - Independent: {phase2_content.get('independent_research', '')}
        - Anecdotal: {phase2_content.get('anecdotal_signals', '')}

        Provide:
        1. Biological Truth: Most plausible reality based on physics, evolution, and independent data
        2. Industry Bias: Where profit motives distort safety/efficacy data
        3. Grey Zone: Promising hypotheses lacking gold standard proof but safe to try
        """

        system_prompt = """You synthesize medical evidence through the lens of evolutionary
        biology and independent science. You identify where corporate interests distort truth
        and highlight promising approaches that lack corporate funding for large trials."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            content = self._parse_synthesis_response(response)

            return PhaseResult(
                phase=AnalysisPhase.SYNTHESIS_MENU,
                timestamp=datetime.now(),
                content=content,
                token_usage=token_usage
            )
        except Exception as e:
            self.logger.error(f"Phase 3 failed: {e}")
            raise

    @track_cost("Phase 4: Complex Output Generation")
    def _phase4_generate_output(self, subject: str, synthesis: Dict, output_type: OutputType) -> str:
        """Phase 4: Generate complex output based on user selection"""
        self.logger.info(f"=== PHASE 4: Complex Output Generation ({output_type.value}) ===")

        template_prompts = {
            OutputType.EVOLUTIONARY: f"""
            Write "The Evolutionary Protocol" for {subject} in first person.

            Structure:
            # The Evolutionary Protocol: {subject}

            ## The Ancestral Logic
            Why this matters evolutionarily (e.g., "Humans have always...")

            ## The Toxic Load
            Modern ingredients/factors to eliminate (focus on disruptors)

            ## The Bio-Identical Swap
            Natural alternatives to the chemical standard

            ## The Protocol
            A routine aligned with circadian/biological rhythms

            ## References
            APA 7 format with URLs

            Synthesis data:
            {json.dumps(synthesis, indent=2)}

            Write densely, first person, collaborative investigative tone.
            """,

            OutputType.BIOHACKER: f"""
            Write "The Bio-Hacker's Guide" for {subject} in first person.

            Structure:
            # The Bio-Hacker's Optimization Guide: {subject}

            ## The Optimization Target
            What mechanism are we exploiting?

            ## The "Underground" Data
            Findings from small labs/new research the mainstream ignores

            ## The Stack
            Specific compounds/routines (including promising but insufficient data)

            ## Risk Management
            How to mitigate side effects while pushing boundaries

            ## References
            APA 7 format with URLs

            Synthesis data:
            {json.dumps(synthesis, indent=2)}

            Write densely, first person, optimization-focused tone.
            """,

            OutputType.PARADIGM_SHIFT: f"""
            Write "The Paradigm Shift" article for {subject} in first person.

            Structure:
            # The New Science of {subject}

            ## The Old Dogma
            "We were told that..."

            ## The Conflict
            "However, when we remove industry-funded data, we see..."

            ## The New Reality
            "Recent independent research (2020-2025) suggests..."

            ## The Takeaway
            Actionable advice based on the new reality

            ## References
            APA 7 format with URLs

            Synthesis data:
            {json.dumps(synthesis, indent=2)}

            Write densely, first person, paradigm-shifting tone.
            """,

            OutputType.VILLAGE_WISDOM: f"""
            Write "The Village Wisdom Guide" for {subject} in first person.

            Structure:
            # Village Wisdom: {subject}

            ## The Story
            A simple analogy (e.g., "Your gut is like a garden, not a battlefield")

            ## The Misunderstanding
            Why the "sterile/chemical" approach fails

            ## The Return to Roots
            3 simple, natural behavioral changes

            ## References
            APA 7 format with URLs

            Synthesis data:
            {json.dumps(synthesis, indent=2)}

            Write simply, first person, friendly teaching tone. Use analogies.
            """,

            OutputType.PROCEED: f"""
            Write a simplified guide for {subject} in friendly teaching tone.

            Use simple language (high school level), explain complex ideas simply,
            use analogies for abstract concepts.

            Include:
            - Key findings in plain language
            - Practical recommendations
            - What to avoid and why

            ## References
            APA 7 format with URLs

            Synthesis data:
            {json.dumps(synthesis, indent=2)}
            """
        }

        prompt = template_prompts.get(output_type, template_prompts[OutputType.PROCEED])

        system_prompt = """You are a medical writer creating evidence-based guides.
        Write in first person, as the user's private researcher. Be collaborative,
        investigative, slightly conspiratorial but grounded in data. Include proper APA 7
        references with URLs to actual papers (PubMed/Nature preferred)."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            return response
        except Exception as e:
            self.logger.error(f"Phase 4 failed: {e}")
            raise

    @track_cost("Phase 5: Simplified Output")
    def _phase5_simplify_output(self, complex_output: str) -> str:
        """Phase 5: Simplify output for general audience"""
        self.logger.info("=== PHASE 5: Simplified Output Generation ===")

        prompt = f"""
        Simplify this medical guide for a general audience:

        {complex_output}

        Requirements:
        - High school reading level
        - Simple, friendly language
        - Analogies for complex ideas
        - Teaching tone
        - Preserve all references
        """

        system_prompt = """You are a medical educator translating complex medical
        information for the general public. Make it accessible without losing accuracy."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            return response
        except Exception as e:
            self.logger.warning(f"Phase 5 simplification failed: {e}, returning original")
            return complex_output

    # Response parsing helpers
    def _parse_conflict_scan_response(self, response: str) -> Dict[str, Any]:
        """Parse Phase 1 response into structured data"""
        # Simple parsing - look for sections
        lines = response.split('\n')
        content = {
            'official_narrative': '',
            'counter_narrative': '',
            'key_conflicts': ''
        }

        current_section = None
        for line in lines:
            line_lower = line.lower().strip()
            if 'official' in line_lower or 'mainstream' in line_lower:
                current_section = 'official_narrative'
            elif 'counter' in line_lower or 'independent' in line_lower or 'alternative' in line_lower:
                current_section = 'counter_narrative'
            elif 'conflict' in line_lower or 'disagree' in line_lower:
                current_section = 'key_conflicts'
            elif current_section and line.strip():
                content[current_section] += line + '\n'

        # Fallback: if sections not found, put everything in official_narrative
        if not any(content.values()):
            content['official_narrative'] = response

        return content

    def _parse_evidence_response(self, response: str) -> Dict[str, Any]:
        """Parse Phase 2 response into structured data"""
        content = {
            'industry_funded_studies': '',
            'independent_research': '',
            'methodology_quality': '',
            'anecdotal_signals': '',
            'time_weighted_evidence': ''
        }

        current_section = None
        lines = response.split('\n')

        for line in lines:
            line_lower = line.lower().strip()
            if 'industry' in line_lower or 'funded' in line_lower or 'manufacturer' in line_lower:
                current_section = 'industry_funded_studies'
            elif 'independent' in line_lower or 'small lab' in line_lower:
                current_section = 'independent_research'
            elif 'method' in line_lower or 'quality' in line_lower:
                current_section = 'methodology_quality'
            elif 'anecdot' in line_lower or 'clinical' in line_lower or 'user' in line_lower:
                current_section = 'anecdotal_signals'
            elif 'recent' in line_lower or 'time' in line_lower or '202' in line:
                current_section = 'time_weighted_evidence'
            elif current_section and line.strip():
                content[current_section] += line + '\n'

        if not any(content.values()):
            content['independent_research'] = response

        return content

    def _parse_synthesis_response(self, response: str) -> Dict[str, Any]:
        """Parse Phase 3 response into structured data"""
        content = {
            'biological_truth': '',
            'industry_bias': '',
            'grey_zone': ''
        }

        current_section = None
        lines = response.split('\n')

        for line in lines:
            line_lower = line.lower().strip()
            if 'biological' in line_lower or 'truth' in line_lower or 'reality' in line_lower:
                current_section = 'biological_truth'
            elif 'industry' in line_lower or 'bias' in line_lower or 'profit' in line_lower:
                current_section = 'industry_bias'
            elif 'grey' in line_lower or 'promising' in line_lower or 'hypothesis' in line_lower:
                current_section = 'grey_zone'
            elif current_section and line.strip():
                content[current_section] += line + '\n'

        if not any(content.values()):
            content['biological_truth'] = response

        return content

    # Interactive prompts
    def _prompt_user_phase1(self) -> str:
        """Prompt user for Phase 1 decision"""
        print("\n" + "="*80)
        print("PHASE 1 COMPLETE: Conflict & Hypothesis Scan")
        print("="*80)
        print("\nI have identified the official narrative vs. the emerging independent view.")
        print("\nWhich angle do you want me to prioritize in the stress test?")
        print("  [Official] - Focus on mainstream medicine perspective")
        print("  [Independent] - Focus on independent/biohacker perspective")
        print("  [Both] - Analyze both perspectives equally")
        print()

        while True:
            choice = input("Your choice (Official/Independent/Both): ").strip()
            if choice.lower() in ['official', 'independent', 'both']:
                return choice.capitalize()
            print("Invalid choice. Please enter: Official, Independent, or Both")

    def _prompt_user_phase2(self, phase2_content: Dict) -> str:
        """Prompt user for Phase 2 decision"""
        print("\n" + "="*80)
        print("PHASE 2 COMPLETE: Evidence Stress-Test")
        print("="*80)
        print("\nKey findings:")
        print(f"- Industry-funded studies: {phase2_content.get('industry_funded_studies', 'N/A')[:100]}...")
        print(f"- Independent research: {phase2_content.get('independent_research', 'N/A')[:100]}...")
        print(f"- Anecdotal signals: {phase2_content.get('anecdotal_signals', 'N/A')[:100]}...")
        print("\nDo you want me to dig deeper into the mechanism, or proceed to the conclusion?")
        print("  [Dig] - Investigate mechanisms in more detail")
        print("  [Proceed] - Move to synthesis and conclusions")
        print()

        while True:
            choice = input("Your choice (Dig/Proceed): ").strip()
            if choice.lower() in ['dig', 'proceed']:
                return choice.capitalize()
            print("Invalid choice. Please enter: Dig or Proceed")

    def _prompt_user_phase3(self) -> str:
        """Prompt user for Phase 3 output format selection"""
        print("\n" + "="*80)
        print("PHASE 3 COMPLETE: Synthesis & Menu")
        print("="*80)
        print("\nHow would you like to proceed with the Final Guide?")
        print()
        print("  [A] The 'Evolutionary' Protocol")
        print("      A strict, nature-first guide prioritizing biological compatibility")
        print()
        print("  [B] The 'Bio-Hacker's' Guide")
        print("      Optimization-focused using cutting-edge science and anecdotal signals")
        print()
        print("  [C] The 'Paradigm Shift' Artifact")
        print("      Shows how the official consensus is likely wrong")
        print()
        print("  [D] The 'Village Wisdom' Bridge")
        print("      Simplified narrative using common sense and traditional knowledge")
        print()
        print("  [P] Simple Proceed")
        print("      Direct simplified output without complex structure")
        print()

        while True:
            choice = input("Your choice (A/B/C/D/P): ").strip().upper()
            if choice in ['A', 'B', 'C', 'D', 'P']:
                return choice
            print("Invalid choice. Please enter: A, B, C, D, or P")

    def export_session(self, filepath: str):
        """Export the complete session to JSON"""
        if not self.current_session:
            self.logger.warning("No active session to export")
            return

        session_data = {
            'subject': self.current_session.subject,
            'started_at': self.current_session.started_at.isoformat(),
            'phases': [],
            'final_output': self.current_session.final_output
        }

        for phase_result in self.current_session.phase_results:
            phase_data = {
                'phase': phase_result.phase.value,
                'timestamp': phase_result.timestamp.isoformat(),
                'content': phase_result.content,
                'user_choice': phase_result.user_choice
            }
            if phase_result.token_usage:
                phase_data['token_usage'] = {
                    'input': phase_result.token_usage.input_tokens,
                    'output': phase_result.token_usage.output_tokens,
                    'total': phase_result.token_usage.total_tokens
                }
            session_data['phases'].append(phase_data)

        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

        self.logger.info(f"Session exported to {filepath}")


def main():
    """Main entry point for the medical fact checker"""
    import argparse

    parser = argparse.ArgumentParser(description="Medical Fact Checker - Independent Bio-Investigator")
    parser.add_argument('subject', type=str, help='Health subject to investigate')
    parser.add_argument('--context', type=str, default='', help='Clarifying context or scope')
    parser.add_argument('--non-interactive', action='store_true', help='Run without user prompts')
    parser.add_argument('--llm', type=str, default='claude', help='Primary LLM provider')
    parser.add_argument('--export', type=str, help='Export session to JSON file')

    args = parser.parse_args()

    # Initialize agent
    agent = MedicalFactChecker(
        primary_llm_provider=args.llm,
        interactive=not args.non_interactive
    )

    # Run analysis
    try:
        session = agent.start_analysis(args.subject, args.context)

        # Display final output
        print("\n" + "="*80)
        print("FINAL OUTPUT")
        print("="*80)
        print(session.final_output)
        print("\n" + "="*80)

        # Export if requested
        if args.export:
            agent.export_session(args.export)
            print(f"\nSession exported to: {args.export}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
