# Medical Fact Checker Agent

An independent bio-investigator AI agent that uncovers the **unfiltered biological reality** behind health subjects. This agent is designed to be skeptical of consensus driven by inertia or corporate interest, weighing **methodological quality** over **institutional authority**.

## Philosophy & Core Biases

1. **New vs. Old**: Prioritizes recent research (last 5-10 years) over older dogma
2. **Source Weighting**: Penalizes studies with financial conflicts of interest; boosts independent lab findings
3. **Anecdotal Signals**: Does not dismiss anecdotal evidence; labels patterns as "Emerging Signals"
4. **Evolutionary Logic**: Uses evolutionary biology as a tie-breaker when data conflicts
5. **Natural Preference**: Favors natural/bio-identical mechanisms over synthetic when efficacy is comparable

## Architecture

The agent uses **DSPy** for structured LLM interactions and follows a **5-phase interactive protocol**:

### Phase 1: Conflict & Hypothesis Scan
- Identifies the "Official Narrative" (mainstream medicine)
- Contrasts with "Counter-Narrative" (independent researchers, biohackers)
- **User Decision**: Choose to prioritize Official/Independent/Both perspectives

### Phase 2: Evidence Stress-Test
- **Funding Filter**: Flags manufacturer-funded studies
- **Methodology Audit**: Evaluates independent studies for rigor
- **Time Weighting**: Prioritizes 2020-2025 research
- **Anecdotal Forensics**: Identifies clinical patterns
- **User Decision**: Dig deeper into mechanisms or proceed

### Phase 3: Synthesis & Menu
- **Biological Truth**: Most plausible reality based on evidence
- **Industry Bias**: Where profit motives distort data
- **Grey Zone**: Promising hypotheses lacking gold-standard proof
- **User Decision**: Select output format (A/B/C/D/P)

### Phase 4: Complex Output Generation
Generates output based on user selection:
- **[A] Evolutionary Protocol**: Nature-first, ancestral logic guide
- **[B] Bio-Hacker's Guide**: Optimization-focused, cutting-edge approach
- **[C] Paradigm Shift**: Shows how official consensus is wrong
- **[D] Village Wisdom**: Simplified, traditional knowledge narrative
- **[P] Simple Proceed**: Direct simplified output

### Phase 5: Simplified Output (if needed)
Translates complex output to high school reading level with analogies

## Features

- **Interactive Stops**: User makes decisions at each phase
- **Token Usage Tracking**: Monitors LLM usage across the session
- **Session Export**: Save complete analysis to JSON
- **LLM Fallback**: Supports multiple LLM providers (Claude, OpenAI, Ollama)
- **Evidence-Based**: References in APA 7 format with URLs

## Installation

This project uses **UV** for dependency management. Install dependencies from the root directory:

```bash
# Install main dependencies
uv sync

# Install with dev dependencies (includes pytest)
uv sync --extra dev
```

Set up your API keys:

```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

Alternatively, create a `.env` file in the root directory:
```bash
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
```

## Usage

### Interactive Mode (Recommended)

```bash
python run_fact_checker.py
```

Follow the prompts to:
1. Enter your health subject
2. Provide optional context
3. Make decisions at each phase
4. Choose output format
5. Export session if desired

### Command Line Mode

```bash
# Basic usage
python medical_fact_checker_agent.py "Sunlight exposure"

# With context
python medical_fact_checker_agent.py "Sunlight exposure" --context "for vitamin D synthesis"

# Non-interactive mode (uses defaults)
python medical_fact_checker_agent.py "Coffee consumption" --non-interactive

# Export session
python medical_fact_checker_agent.py "Intermittent fasting" --export session.json

# Use different LLM
python medical_fact_checker_agent.py "Red meat" --llm openai
```

### Programmatic Usage

```python
from medical_fact_checker_agent import MedicalFactChecker, OutputType

# Initialize agent
agent = MedicalFactChecker(
    primary_llm_provider="claude",
    interactive=True  # Set to False for automated analysis
)

# Run analysis
session = agent.start_analysis(
    subject="Seed oils",
    clarifying_info="cardiovascular health impact"
)

# Access results
print(session.final_output)

# Export session
agent.export_session("seed_oils_analysis.json")
```

## Example Session

```
=== PHASE 1: Conflict & Hypothesis Scan ===
Analyzing: Vitamin D supplementation

Official Narrative:
- RDA of 600-800 IU sufficient for most people
- Get from food and limited sun exposure
- High doses potentially harmful

Counter-Narrative:
- RDA levels maintain minimal function, not optimal health
- Modern indoor lifestyle creates widespread deficiency
- Doses of 4000-5000 IU often needed for optimal blood levels
- Toxicity concerns overblown below 10,000 IU daily

Which angle do you want me to prioritize? [Official/Independent/Both]
> Both

=== PHASE 2: Evidence Stress-Test ===
[Analysis continues...]
```

## Simplifications from Original Prompt

This implementation streamlines the original design while maintaining core functionality:

1. **Combined Phase 4 & 5**: Original had separate complex and simplified outputs; this version can optionally simplify any output
2. **Simpler Parsing**: Uses text-based parsing instead of complex structured extraction
3. **Flexible Output**: All output types follow similar workflow
4. **Token Awareness**: Tracks but doesn't over-optimize token usage
5. **Clean State Management**: Uses dataclasses for session state

## Output Formats Explained

### [A] Evolutionary Protocol
Best for: Understanding biological compatibility, ancestral health perspective
- Focuses on what humans evolved eating/doing
- Identifies modern toxins and disruptions
- Provides natural alternatives
- Aligns recommendations with circadian biology

### [B] Bio-Hacker's Guide
Best for: Optimization, cutting-edge approaches, experimental protocols
- Targets specific biological mechanisms
- Includes promising but not-yet-proven interventions
- Details specific compounds and dosing
- Covers risk management

### [C] Paradigm Shift
Best for: Understanding how consensus may be wrong, critical analysis
- Contrasts old dogma with new evidence
- Highlights industry-funded bias
- Shows recent independent findings
- Provides corrected understanding

### [D] Village Wisdom
Best for: Simple, accessible guidance using traditional knowledge
- Uses analogies and stories
- No medical jargon
- Common sense approach
- 3 simple behavioral changes

### [P] Simple Proceed
Best for: Quick, practical guidance without complex frameworks
- High school reading level
- Friendly teaching tone
- Key findings + practical recommendations
- What to avoid and why

## Design Notes

### What This Agent Does Well
- **Critical Analysis**: Questions mainstream consensus when evidence warrants
- **Funding Bias Detection**: Identifies corporate influence in research
- **Evolutionary Context**: Applies biological first principles
- **Anecdotal Integration**: Values clinical observation and user reports
- **Recent Evidence**: Prioritizes current research over outdated beliefs

### What This Agent Doesn't Do
- **Medical Advice**: Not a replacement for professional medical consultation
- **Diagnostic**: Does not diagnose conditions
- **Treatment Plans**: Does not prescribe specific treatments
- **Personalization**: Does not account for individual medical history

### Potential Over-Engineering in Original Prompt
The original prompt had some complexity that may not be necessary:
1. **5 separate phases**: Could be consolidated to 3 (scan → analyze → output)
2. **Multiple decision points**: Some could be automated based on context
3. **Complex output structures**: Simpler formats might be more useful
4. **Strict phase separation**: Could allow more fluid analysis flow

## Future Enhancements

Potential improvements (not yet implemented):
- [ ] Web research integration for recent papers
- [ ] PubMed API integration for reference validation
- [ ] Structured output with citations database
- [ ] Comparison mode (analyze multiple subjects side-by-side)
- [ ] Research quality scoring algorithm
- [ ] Automated funding source detection
- [ ] Historical timeline of belief changes
- [ ] Interactive reference exploration

## Troubleshooting

**Issue**: Agent hangs at Phase 1
- **Solution**: Check API key is valid and LLM provider is accessible

**Issue**: Poor quality responses
- **Solution**: Try switching LLM provider with `--llm openai`

**Issue**: Import errors
- **Solution**: Ensure parent directory is in Python path or install as package

**Issue**: No references in output
- **Solution**: This is a known limitation; LLMs may not always include proper citations

## Contributing

To improve this agent:
1. Enhance response parsing for better structured extraction
2. Add web research capabilities
3. Integrate with medical literature databases
4. Improve reference validation and formatting
5. Add citation tracking across phases

## License

Part of the medical reasoning agent research project.

## Testing

This project includes comprehensive pytest tests. Run tests using UV:

### Run All Tests

```bash
# From project root
uv run pytest medical_fact_checker/test_medical_fact_checker.py -v

# Or with coverage
uv run pytest medical_fact_checker/test_medical_fact_checker.py --cov=medical_fact_checker --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
uv run pytest medical_fact_checker/ -v -m unit

# Integration tests only
uv run pytest medical_fact_checker/ -v -m integration

# Exclude slow tests
uv run pytest medical_fact_checker/ -v -m "not slow"

# Run specific test class
uv run pytest medical_fact_checker/test_medical_fact_checker.py::TestPhase1ConflictScan -v

# Run specific test
uv run pytest medical_fact_checker/test_medical_fact_checker.py::TestPhase1ConflictScan::test_phase1_execution -v
```

### Test Coverage

```bash
# Generate HTML coverage report
uv run pytest medical_fact_checker/ --cov=medical_fact_checker --cov-report=html

# View coverage report
# Open htmlcov/index.html in your browser
```

### Test Structure

The test suite includes:
- **Initialization Tests**: Agent setup and configuration
- **Phase Tests**: Individual testing of all 5 phases
- **Response Parsing Tests**: Validation of LLM response parsing
- **Workflow Tests**: End-to-end analysis workflows
- **Session Management Tests**: Export and session tracking
- **Error Handling Tests**: Edge cases and failure scenarios
- **Interactive Mode Tests**: Mocked user input scenarios

All tests use mocked LLM responses, so no API keys are required for testing.

## Disclaimer

This tool is for research and educational purposes. It provides analysis of medical literature and health topics but does not provide medical advice. Always consult qualified healthcare professionals for medical decisions.
