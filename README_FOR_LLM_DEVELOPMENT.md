# LLM-Assisted Development Guide

This document provides rules, patterns, and best practices for maintaining and extending this medical AI agent repository. It's designed for both LLM assistants (like Claude, GPT-4, etc.) and human developers using AI coding tools.

---

## 📋 Table of Contents

1. [Core Development Principles](#core-development-principles)
2. [Code Consistency Rules](#code-consistency-rules)
3. [Architecture Patterns](#architecture-patterns)
4. [Testing Requirements](#testing-requirements)
5. [Documentation Standards](#documentation-standards)
6. [Performance Considerations](#performance-considerations)
7. [Security & Safety](#security--safety)
8. [Features Roadmap](#features-roadmap)
9. [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
10. [LLM-Specific Best Practices](#llm-specific-best-practices)

---

## Core Development Principles

### 1. **Medical Safety First**
- All agents must include disclaimers that output is for research/educational purposes
- Never claim agents provide medical advice or replace healthcare professionals
- Validate medical terminology and avoid making definitive diagnostic claims
- Include evidence quality ratings (Strong/Moderate/Limited/Poor)
- **MANDATORY: Every report must include a References section at the end**
- **MANDATORY: Enable reference validation when generating reports** (see reference_validation/)

### 2. **Transparency Over Black Boxes**
- Export reasoning traces for all multi-step analyses
- **MANDATORY: Track and report cost analysis for EVERY phase**
- Log token usage for cost tracking
- Provide confidence scores with explanations
- Make agent decisions auditable
- Display per-phase cost breakdown in reports

### 3. **Modularity & Reusability**
- Each agent should be independently usable
- Share common components (LLM integrations, input validation, etc.)
- Avoid tight coupling between agents
- Use dependency injection for flexibility

### 4. **Progressive Enhancement**
- Start with working MVP, then add sophistication
- Don't over-engineer on first iteration
- Prefer simple solutions that work over complex ones that might work

---

## Code Consistency Rules

### File Organization

```
medical_reasoning_agent/
├── medical_procedure_analyzer/     # Procedure analysis agent
│   ├── __init__.py
│   ├── medical_reasoning_agent.py  # Main agent class
│   ├── llm_integrations.py         # Shared LLM code
│   ├── input_validation.py         # Input validation
│   ├── web_research.py             # Web search integration
│   └── report_generator.py         # Output formatting
├── medical_fact_checker/           # Fact checking agent
│   ├── __init__.py
│   ├── medical_fact_checker_agent.py
│   ├── test_medical_fact_checker.py
│   └── README.md
├── run_analysis.py                 # Unified orchestrator
├── tests/                          # Integration tests
└── outputs/                        # Generated reports
```

### Naming Conventions

**Classes:**
- Agent classes: `{Purpose}Agent` (e.g., `MedicalReasoningAgent`, `MedicalFactChecker`)
- Data classes: `{Domain}{Type}` (e.g., `MedicalInput`, `PhaseResult`)
- Enums: `{Domain}{Category}` (e.g., `AnalysisPhase`, `OutputType`)

**Functions:**
- Private methods: `_verb_noun` (e.g., `_parse_response`, `_save_analysis`)
- Public methods: `verb_noun` (e.g., `analyze_procedure`, `start_analysis`)
- Phase methods: `_phase{N}_{description}` (e.g., `_phase1_conflict_scan`)

**Files:**
- Agent modules: `{purpose}_agent.py`
- Tests: `test_{module_name}.py`
- Utilities: `{function}.py` (e.g., `input_validation.py`)

**Variables:**
- Use descriptive names: `medical_input` not `mi`
- Enums: `SCREAMING_SNAKE_CASE`
- Constants: `SCREAMING_SNAKE_CASE`
- Regular variables: `snake_case`

### Import Order

```python
# 1. Standard library
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional

# 2. Third-party packages
import dspy
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic

# 3. Local imports
from medical_procedure_analyzer import MedicalReasoningAgent
from .llm_integrations import create_llm_manager
```

### Type Hints

**Always use type hints for:**
- Function parameters
- Return values
- Class attributes
- Complex data structures

```python
# Good
def analyze_procedure(self, medical_input: MedicalInput) -> MedicalOutput:
    pass

# Bad
def analyze_procedure(self, medical_input):
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def phase1_analysis(self, subject: str, context: str) -> PhaseResult:
    """
    Execute phase 1 analysis with conflict scanning.

    Args:
        subject: Health subject to investigate
        context: Additional context or scope information

    Returns:
        PhaseResult containing analysis outputs and metadata

    Raises:
        ValueError: If subject is empty or invalid
        APITimeoutError: If LLM request times out
    """
    pass
```

---

## Architecture Patterns

### 1. **Agent Structure Pattern**

All agents should follow this structure:

```python
class NewAgent:
    """Agent description and purpose"""

    def __init__(self, primary_llm_provider: str = "claude", **kwargs):
        """Initialize with LLM manager and configuration"""
        self.llm_manager = create_llm_manager(primary_llm_provider)
        self.logger = logging.getLogger(__name__)

    def analyze(self, input_data: InputType) -> OutputType:
        """Main entry point - coordinates analysis phases"""
        # Phase 1: Initial processing
        # Phase 2: Core analysis
        # Phase 3: Output generation
        pass

    def _phase1_name(self, data) -> PhaseResult:
        """Private method for each phase"""
        pass

    def export_results(self, filepath: str):
        """Export analysis results"""
        pass
```

### 2. **LLM Integration Pattern**

**DO:**
- Use `llm_integrations.py` for all LLM calls
- Support multiple providers with fallback
- Track token usage
- Handle timeouts gracefully
- Use consistent error handling

```python
# Good - centralized LLM management
response, token_usage = self.llm_manager.get_available_provider().generate_response(
    prompt, system_prompt
)

# Accumulate token usage for cost tracking
if token_usage:
    self.total_token_usage.add(token_usage)

# Bad - direct API calls
response = anthropic.messages.create(...)
```

### 3. **DSPy Structured Output Pattern** ⭐ **CRITICAL**

**ALWAYS use DSPy for structured outputs.** Manual JSON parsing is fragile and leads to errors.

#### Why DSPy?
- **Type Safety**: Pydantic models enforce structure
- **Validation**: Automatic validation of LLM outputs
- **Reliability**: No more JSON parsing errors
- **Documentation**: Schema serves as documentation
- **Maintainability**: Changes to schema propagate automatically

#### Step 1: Define Pydantic Schemas

Create structured models in a dedicated schema file (e.g., `dspy_schemas.py`):

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class PharmacologyData(BaseModel):
    """Structured pharmacology information"""
    drug_class: str = Field(description="Pharmacologic and therapeutic class")
    mechanism_of_action: str = Field(description="Detailed mechanism at molecular level")
    absorption: str = Field(description="Bioavailability, onset, peak concentration")
    metabolism: str = Field(description="CYP enzymes, active metabolites")
    elimination: str = Field(description="Primary route, elimination half-life")
    half_life: str = Field(description="Elimination half-life value")
    approved_indications: List[str] = Field(description="FDA-approved indications")
    off_label_uses: List[str] = Field(default_factory=list)
    standard_dosing: str = Field(description="Standard adult dosing")
    dose_adjustments: dict = Field(default_factory=dict)
```

#### Step 2: Setup DSPy in Agent Initialization

```python
class MedicationAnalyzer(MedicalReasoningAgent):
    def __init__(self, primary_llm_provider: str = "claude", **kwargs):
        super().__init__(primary_llm_provider, **kwargs)

        # Setup DSPy for structured output
        try:
            self.llm_manager.setup_dspy_integration()
            self.use_dspy = True
            self.logger.info("DSPy structured output enabled")
        except Exception as e:
            self.logger.warning(f"DSPy setup failed: {e}. Falling back to manual parsing.")
            self.use_dspy = False
```

#### Step 3: Use Pydantic Schemas in Prompts

Include the JSON schema in your prompts:

```python
import json

def _analyze_pharmacology(self, medication: str) -> Dict[str, Any]:
    # Get schema from Pydantic model
    schema = PharmacologyData.model_json_schema()

    prompt = f"""
    Provide comprehensive pharmacology information for {medication}.

    CRITICAL: Respond with ONLY valid JSON matching this exact schema:
    {json.dumps(schema, indent=2)}

    Include detailed information for all fields.
    Start with {{ and end with }}. No other text.
    """

    system_prompt = """You are a clinical pharmacologist.
    Respond ONLY with valid JSON matching the schema. No explanatory text."""

    response, token_usage = self.llm_manager.get_available_provider().generate_response(
        prompt, system_prompt
    )

    # Accumulate token usage
    if token_usage:
        self.total_token_usage.add(token_usage)

    # Parse with Pydantic validation
    pharmacology = self._parse_with_pydantic(response, PharmacologyData)
    return pharmacology.model_dump() if pharmacology else {}
```

#### Step 4: Create Robust Parsing Helper

```python
def _parse_with_pydantic(self, response: str, model_class: type, fallback_value: Any = None) -> Any:
    """
    Parse LLM response using Pydantic model validation.

    Args:
        response: Raw LLM response
        model_class: Pydantic model class to validate against
        fallback_value: Value to return if parsing fails

    Returns:
        Validated Pydantic model instance or fallback value
    """
    import re

    # Extract JSON from response
    json_str = None

    # Try code block first
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', response, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        # Try to find JSON object/array
        json_match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)

    if not json_str:
        self.logger.warning(f"No JSON found in response for {model_class.__name__}")
        return fallback_value

    # Clean JSON
    json_str = re.sub(r'//.*?\n', '\n', json_str)  # Remove comments
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    json_str = re.sub(r',\s*}', '}', json_str)  # Trailing commas
    json_str = re.sub(r',\s*]', ']', json_str)

    # Parse and validate with Pydantic
    try:
        data = json.loads(json_str)
        validated = model_class.model_validate(data)
        self.logger.info(f"Successfully parsed {model_class.__name__}")
        return validated
    except json.JSONDecodeError as e:
        self.logger.warning(f"JSON decode error for {model_class.__name__}: {e}")
        return fallback_value
    except Exception as e:
        self.logger.warning(f"Pydantic validation error for {model_class.__name__}: {e}")
        return fallback_value
```

#### ❌ **NEVER Do This** (Manual JSON Parsing)

```python
# BAD - Fragile manual parsing
def _parse_response(self, response: str) -> dict:
    json_match = re.search(r'\{.*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            return {}  # Silent failure!
    return {}
```

#### ✅ **ALWAYS Do This** (DSPy + Pydantic)

```python
# GOOD - Robust structured parsing
def _analyze_medication(self, medication: str) -> Dict[str, Any]:
    schema = MedicationData.model_json_schema()

    prompt = f"""
    Analyze {medication}.

    Respond with ONLY valid JSON matching this schema:
    {json.dumps(schema, indent=2)}
    """

    response, token_usage = self.llm_manager.generate_response(prompt, system_prompt)

    # Accumulate token usage
    if token_usage:
        self.total_token_usage.add(token_usage)

    # Parse with validation
    result = self._parse_with_pydantic(response, MedicationData, fallback_value=None)
    return result.model_dump() if result else self._get_fallback_data()
```

#### DSPy Structured Output Checklist

Before implementing LLM calls, ensure:

- [ ] Created Pydantic model for expected output structure
- [ ] DSPy is initialized in agent `__init__`
- [ ] JSON schema is included in prompt
- [ ] Using `_parse_with_pydantic()` helper
- [ ] Have fallback values for parsing failures
- [ ] Logging parsing success/failures
- [ ] Token usage is accumulated after each call
- [ ] Tests cover both successful and failed parsing

### 4. **Phase-Based Analysis Pattern**

Complex agents should use phases with **automatic cost tracking** via the existing `cost_tracker.py`:

```python
# Import from existing cost_tracker.py
from cost_tracker import track_cost, print_cost_summary, reset_tracking

class MultiPhaseAgent:
    def __init__(self):
        self.total_token_usage = TokenUsage()
        self.phase_results = []

    def analyze(self, input_data):
        # Reset tracking at start of new analysis
        reset_tracking()

        # Phase 1 with automatic cost tracking
        phase1 = self._phase1_analysis(input_data)
        self.phase_results.append(phase1)

        # Phase 2 with automatic cost tracking
        phase2 = self._phase2_analysis(input_data, phase1.content)
        self.phase_results.append(phase2)

        # Print cost summary at end
        print_cost_summary()

        return self._synthesize_results()

    @track_cost("Phase 1: Initial Analysis")
    def _phase1_analysis(self, input_data):
        """
        Phase 1 with automatic cost tracking.
        The decorator captures token usage before/after automatically!
        """
        response, token_usage = self.llm_manager.generate_response(prompt)
        # IMPORTANT: Still accumulate tokens - decorator reads from this
        if token_usage:
            self.total_token_usage.add(token_usage)
        return result

    @track_cost("Phase 2: Deep Analysis")
    def _phase2_analysis(self, input_data, phase1_data):
        """Phase 2 - decorator handles cost tracking automatically"""
        response, token_usage = self.llm_manager.generate_response(prompt)
        if token_usage:
            self.total_token_usage.add(token_usage)
        return result
```

**How `cost_tracker.py` Works:**
- ✅ Decorator captures `self.total_token_usage` state BEFORE phase runs
- ✅ Decorator captures state AFTER phase runs
- ✅ Automatically calculates: cost = (tokens_after - tokens_before) × pricing
- ✅ Automatically prints: `💰 Phase 1: $0.05 (15.2s)`
- ✅ Stores all phase costs in global `_phase_costs` list

**MANDATORY Phase Requirements:**
- ✅ Call `reset_tracking()` at start of each new analysis
- ✅ Use `@track_cost("Phase Name")` decorator on ALL phase methods
- ✅ Accumulate tokens as usual: `self.total_token_usage.add(token_usage)`
- ✅ Call `print_cost_summary()` at end of analysis
- ✅ The decorator handles everything else automatically!

### 5. **Data Flow Pattern**

```
Input Validation → LLM Analysis → Response Parsing → Output Formatting → Export
      ↓                 ↓               ↓                   ↓              ↓
 Reject invalid    Track tokens    Handle errors      Add metadata    Save JSON/MD
```

### 6. **Orchestrator Pattern**

`run_analysis.py` demonstrates the orchestrator pattern:

```python
class AgentOrchestrator:
    """Central hub for running multiple agents"""

    AGENTS = {
        "agent_id": {
            "name": "Display Name",
            "description": "What it does",
            "class": AgentClass,
        }
    }

    def run_agent_name(self, **kwargs):
        """Run specific agent with common interface"""
        agent = AgentClass(**kwargs)
        result = agent.analyze()
        files = self._save_results(result)
        return result, files
```

---

## Testing Requirements

### Unit Tests

**Every module must have unit tests covering:**
- Initialization
- Each major function
- Error handling
- Edge cases

```python
class TestMedicalAgent:
    def test_initialization_success(self):
        """Test agent initializes correctly"""
        pass

    def test_initialization_failure(self):
        """Test agent handles init errors"""
        pass

    def test_analysis_with_valid_input(self):
        """Test analysis with valid data"""
        pass

    def test_analysis_with_invalid_input(self):
        """Test analysis rejects bad input"""
        pass
```

### Integration Tests

Test full workflows:

```python
@pytest.mark.integration
def test_full_analysis_workflow(self):
    """Test complete analysis from input to output"""
    agent = MedicalAgent()
    result = agent.analyze(valid_input)

    assert result is not None
    assert len(result.outputs) > 0
    assert result.confidence_score > 0
```

### Mocking LLM Calls

**Always mock LLM calls in tests:**

```python
@pytest.fixture
def mock_llm_manager():
    """Mock LLM with predictable responses"""
    manager = Mock()
    provider = Mock()

    def mock_response(prompt, system_prompt=None):
        return "Mocked response", TokenUsage(100, 200, 300)

    provider.generate_response = Mock(side_effect=mock_response)
    manager.get_available_provider = Mock(return_value=provider)
    return manager
```

### Test Coverage Goals

- **Minimum:** 70% coverage
- **Target:** 85% coverage
- **Critical paths:** 100% coverage (input validation, LLM calls, output generation)

### Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=. --cov-report=html

# Specific test file
uv run pytest medical_fact_checker/test_medical_fact_checker.py -v

# Exclude slow tests
uv run pytest -m "not slow"
```

---

## Documentation Standards

### README Files

Each agent directory must have a README.md with:

1. **Purpose**: What the agent does in 1-2 sentences
2. **Architecture**: How it works (phases, workflow)
3. **Installation**: How to install dependencies
4. **Usage**: Examples of running the agent
5. **Output**: What files are generated
6. **Testing**: How to run tests
7. **Configuration**: Available options
8. **Limitations**: Known issues or constraints

### Report Generation Standards ⭐ **MANDATORY**

**Every report generated by any module MUST include:**

1. **Emoji Formatting for Titles and Subtitles** ⭐ **NEW**
   - All report titles and section headers MUST use relevant emojis
   - This improves readability and visual organization
   - Consistent emoji usage across all analysis types

   **Standard Emoji Mapping:**
   ```markdown
   # 🔬 Main Report Title (Procedure)
   # 💊 Main Report Title (Medication)
   # 🔎 Main Report Title (Fact Check)

   ## 📋 Overview/Summary/Subject
   ## 🫀 Organ Analysis
   ## 🧬 Mechanism of Action
   ## ⚗️ Pharmacokinetics/Pharmacology
   ## 💉 Clinical Use
   ## 🔗 Interactions
   ## ⚠️ Safety Information
   ## 🚨 Black Box Warnings
   ## 💡 Recommendations/General Recommendations
   ## ✅ Evidence-Based Recommendations
   ## 🔬 Research Gaps/Investigational Approaches
   ## ❌ Debunked Claims
   ## 📚 References
   ## 💰 Cost Analysis
   ## 📊 Monitoring Requirements
   ## 📄 Final Output
   ## 🔍 Detailed Analysis
   ## 🎯 Quality Assessment

   ### Subsection emojis:
   - 🔴 High risk
   - 🟡 Moderate risk
   - 🟢 Low risk
   - 💊 Drug interactions
   - 🍎 Food interactions
   ```

   **Implementation:**
   ```python
   def generate_report(self, analysis_result) -> str:
       """Generate report with mandatory emoji formatting"""
       report = f"""
   # 🔬 Analysis Report

   ## 📋 Overview
   {analysis_result.summary}

   ## 💡 Recommendations
   {self._format_recommendations(analysis_result.recommendations)}

   ## 📚 References
   {self._format_references(analysis_result.sources)}

   ## 💰 Cost Analysis
   {self._format_cost_analysis()}
   """
       return report
   ```

2. **References Section** (at the end of report)
   ```markdown
   ## 📚 References
   [1] Smith J, et al. (2020). Study Title. Journal Name. DOI: 10.1234/example
   [2] Jones A, et al. (2021). Another Study. PMID: 12345678
   [3] CDC Guidelines. https://www.cdc.gov/resource
   ```

3. **Cost Analysis Section** (in report metadata or footer)
   ```markdown
   ## 💰 Analysis Cost Summary
   - Total Cost: $0.15
   - Duration: 45.2s
   - Token Usage: 5,234 input, 2,100 output

   ### Phase Breakdown:
   - Phase 1 (Evidence Gathering): $0.05 (15.2s)
   - Phase 2 (Risk Assessment): $0.04 (12.1s)
   - Phase 3 (Recommendations): $0.06 (18.0s)
   ```

4. **Reference Validation Status** (if validation enabled)
   ```markdown
   ## Reference Validation
   - Valid References: 8/10 (80%)
   - Overall Credibility Score: 82/100
   - ⚠️ 2 references could not be verified
   ```

**Implementation:**
```python
def generate_report(self, analysis_result) -> str:
    """Generate report with mandatory sections"""
    report = f"""
# Analysis Report

{analysis_result.content}

## References
{self._format_references(analysis_result.sources)}

## Cost Analysis
- Total Cost: ${self._calculate_cost():.4f}
- Duration: {analysis_result.duration}s

### Phase Breakdown:
{self._format_phase_costs()}

## Reference Validation
{self._format_validation_report(analysis_result.validation_report)}
"""
    return report
```

### Inline Comments

**Use comments for:**
- Complex logic that isn't obvious
- Why a particular approach was chosen
- Workarounds for known issues
- TODO items for future improvements

**Don't comment:**
- Obvious code (`i += 1  # increment i`)
- What the code does (use descriptive names instead)

```python
# Good - explains WHY
# Use fallback parsing because LLM may not return valid JSON
content = self._parse_text_response(response)

# Bad - explains WHAT (obvious from code)
# Create a new list
my_list = []
```

### API Documentation

For functions that will be used by other modules:

```python
def medical_analysis_with_fallback(
    self,
    medical_input: Dict[str, Any],
    stage: str
) -> Dict[str, Any]:
    """
    Perform medical analysis with automatic provider fallback.

    This method attempts to run analysis with each configured LLM provider
    until one succeeds. It automatically tracks token usage across all attempts.

    Args:
        medical_input: Dictionary containing:
            - procedure: str, name of medical procedure
            - details: str, additional context
            - organ: str, specific organ if applicable
        stage: Current analysis stage (e.g., "evidence_gathering")

    Returns:
        Dictionary containing:
            - analysis: str, LLM-generated analysis
            - confidence: float, confidence score 0.0-1.0
            - sources_needed: list, additional sources required
            - provider_used: str, which LLM provider succeeded
            - token_usage: TokenUsage object

    Raises:
        RuntimeError: If all LLM providers fail
        ValueError: If medical_input is invalid

    Example:
        >>> result = llm_manager.medical_analysis_with_fallback(
        ...     {"procedure": "MRI", "details": "with contrast"},
        ...     "risk_assessment"
        ... )
        >>> print(result["analysis"])
    """
    pass
```

---

## Performance Considerations

### 1. **LLM Call Optimization**

**Problem:** LLM calls are expensive (time and money)

**Solutions:**
- Cache LLM responses where appropriate
- Use `@lru_cache` for deterministic functions
- Batch similar queries when possible
- Set reasonable token limits (`max_tokens: 4096`)
- Use appropriate timeouts (default: 300s)

```python
# Good - caching immutable operations
@lru_cache(maxsize=64)
def _gather_evidence(self, medical_input: MedicalInput, organs: tuple) -> Dict:
    pass

# Bad - caching mutable operations
@lru_cache(maxsize=64)
def _gather_evidence(self, medical_input: MedicalInput, organs: list) -> Dict:
    pass  # Won't work - list isn't hashable
```

### 2. **Token Usage Tracking** ⭐ **MANDATORY**

**Every agent MUST use the existing `cost_tracker.py` system:**

```python
# Required imports (use existing cost_tracker.py)
from medical_procedure_analyzer.medical_reasoning_agent import TokenUsage
from cost_tracker import track_cost, print_cost_summary, reset_tracking, get_cost_summary

class Agent:
    def __init__(self):
        # MANDATORY: Initialize token tracker
        self.total_token_usage = TokenUsage()

    # MANDATORY: Use @track_cost decorator on ALL phase methods
    @track_cost("Phase 1: Analysis")
    def _phase1_analysis(self, input_data):
        """
        Phase with automatic cost tracking via decorator.
        The @track_cost decorator automatically:
        - Captures token state before function runs
        - Captures token state after function runs
        - Calculates: cost = (tokens_after - tokens_before) × model_pricing
        - Prints: 💰 Phase 1: Analysis: $0.0234 (12.5s)
        - Stores phase cost in global _phase_costs list
        """
        response, token_usage = self.llm_manager.generate_response(prompt)

        # MANDATORY: Accumulate tokens (decorator reads from self.total_token_usage)
        if token_usage:
            self.total_token_usage.add(token_usage)

        return response

    @track_cost("Phase 2: Synthesis")
    def _phase2_synthesis(self, input_data):
        """Another phase - decorator handles cost tracking automatically"""
        response, token_usage = self.llm_manager.generate_response(prompt)
        if token_usage:
            self.total_token_usage.add(token_usage)
        return response

    def analyze(self, input_data):
        """Main analysis with automatic cost reporting"""
        # MANDATORY: Reset tracking at start
        reset_tracking()

        # Run phases (decorator tracks each automatically)
        phase1 = self._phase1_analysis(input_data)
        phase2 = self._phase2_synthesis(input_data)

        # MANDATORY: Print cost summary at end
        print_cost_summary()

        # MANDATORY: Get cost data for report
        cost_data = get_cost_summary()

        result.total_cost = cost_data['total_cost']
        result.total_duration = cost_data['total_duration']
        result.phase_costs = cost_data['phases']

        return result
```

**What `cost_tracker.py` Provides:**
- ✅ `@track_cost(phase_name)` - Decorator that auto-tracks costs
- ✅ `reset_tracking()` - Clear costs (call at start of new analysis)
- ✅ `print_cost_summary()` - Print formatted cost summary to console
- ✅ `get_cost_summary()` - Get cost data dict for reports

**Cost Tracking Checklist** (MANDATORY for all agents):
- [ ] Initialize `TokenUsage()` in `__init__`
- [ ] Call `reset_tracking()` at start of each analysis
- [ ] Use `@track_cost("Phase Name")` on ALL phase methods
- [ ] Accumulate tokens after EVERY LLM call: `self.total_token_usage.add(token_usage)`
- [ ] Call `print_cost_summary()` at end of analysis
- [ ] Get cost data with `get_cost_summary()` for reports
- [ ] Include cost breakdown in final report

### 3. **Timeout Configuration**

- **Simple queries**: 120s
- **Standard queries**: 300s (default)
- **Complex multi-phase**: 600s
- **Research-heavy**: 900s

```python
# Allow users to configure
agent = MedicalAgent(timeout=600)  # 10 minutes

# Or via CLI
python run_analysis.py factcheck --subject "Complex" --timeout 600
```

### 4. **Parallel Processing**

When phases are independent, run them in parallel:

```python
# Sequential (slow)
organ1_analysis = analyze_organ("kidney")
organ2_analysis = analyze_organ("liver")

# Parallel (fast)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(analyze_organ, organ) for organ in organs]
    results = [f.result() for f in futures]
```

---

## Security & Safety

### 1. **Input Validation**

**Always validate user inputs:**

```python
from .input_validation import InputValidator, ValidationError

def analyze(self, subject: str):
    # Validate before processing
    result = InputValidator.validate_medical_procedure(subject)
    if not result.is_valid:
        raise ValueError(f"Invalid input: {', '.join(result.errors)}")

    # Proceed with validated input
    pass
```

### 2. **API Key Handling**

**DO:**
- Use environment variables
- Support `.env` files
- Never hardcode keys
- Never log API keys

```python
# Good
api_key = os.getenv("ANTHROPIC_API_KEY")

# Bad
api_key = "sk-ant-api03-..."
```

### 3. **Prompt Injection Prevention**

Be cautious with user input in prompts:

```python
# Good - structured, limited user input
prompt = f"""Analyze this medical subject: {subject}

Guidelines:
1. Focus on evidence-based medicine
2. Cite sources
3. Note limitations
"""

# Risky - raw user input in system prompt
system_prompt = user_provided_instructions  # Could contain injection
```

### 4. **Output Sanitization**

Sanitize outputs before saving:

```python
def _sanitize_filename(self, name: str) -> str:
    """Remove unsafe characters from filename"""
    # Remove path separators and special chars
    safe_name = name.replace("/", "_").replace("\\", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_ -")
    return safe_name[:100]  # Limit length
```

---

## Features Roadmap

### High Priority (Next 1-2 Months)

#### 1. **Web Research Integration**
**Status:** Partially implemented, needs enhancement

**Tasks:**
- [ ] Integrate Tavily API for recent research papers
- [ ] Add PubMed API integration for medical literature
- [ ] Implement citation extraction and validation
- [ ] Add source quality scoring
- [ ] Cache research results to reduce API calls

**Implementation Guide:**
```python
class WebResearchEnhancer:
    """Enhance agents with web research capabilities"""

    def search_pubmed(self, query: str, max_results: int = 5) -> List[Paper]:
        """Search PubMed for recent papers"""
        pass

    def validate_citation(self, citation: str) -> bool:
        """Verify citation exists and is correctly formatted"""
        pass

    def score_source_quality(self, source: dict) -> float:
        """Rate source quality based on journal, citations, etc."""
        pass
```

#### 2. **MCP Server Implementation**
**Status:** Architecture ready, not implemented

**Tasks:**
- [ ] Define MCP tool schemas for each agent
- [ ] Implement MCP protocol handlers
- [ ] Add streaming support for long-running analyses
- [ ] Create MCP server configuration
- [ ] Test with Claude Desktop and other MCP clients

**Structure:**
```python
# mcp_server/
# ├── __init__.py
# ├── server.py              # Main MCP server
# ├── tools.py               # Tool definitions
# └── schemas.py             # JSON schemas
```

#### 3. **Interactive Web UI**
**Status:** Not started

**Tasks:**
- [ ] Create FastAPI backend
- [ ] Build React/Vue frontend
- [ ] Add real-time progress updates (websockets)
- [ ] Implement agent comparison view
- [ ] Add result visualization (charts, graphs)
- [ ] Enable editing and re-running analyses

#### 4. **Result Comparison Tool**
**Status:** Not started

**Tasks:**
- [ ] Compare outputs from different LLMs
- [ ] Compare different agent runs on same subject
- [ ] Highlight differences in recommendations
- [ ] Generate comparative reports
- [ ] Add A/B testing framework

### Medium Priority (3-6 Months)

#### 5. **Advanced Prompt Optimization**
**Tasks:**
- [ ] Implement DSPy optimizers
- [ ] A/B test different prompt strategies
- [ ] Auto-tune prompts based on output quality
- [ ] Create prompt library with version control

#### 6. **Multi-Modal Analysis**
**Tasks:**
- [ ] Support PDF upload and analysis
- [ ] Image analysis (medical scans, charts)
- [ ] Video transcript analysis
- [ ] Audio medical notes transcription

#### 7. **Collaborative Features**
**Tasks:**
- [ ] Multi-user workspaces
- [ ] Shared analysis sessions
- [ ] Comment and annotation system
- [ ] Export to collaborative formats (Notion, Google Docs)

#### 8. **Quality Assurance Pipeline**
**Tasks:**
- [ ] Automated fact-checking against databases
- [ ] Contradiction detection within outputs
- [ ] Reference validation
- [ ] Hallucination detection
- [ ] Expert review workflow

### Low Priority / Future Ideas

#### 9. **Advanced Analytics**
- Usage analytics dashboard
- Cost tracking per agent/user
- Performance metrics (accuracy, speed)
- Agent effectiveness scoring

#### 10. **Domain Expansion**
- Veterinary medicine agent
- Nutrition analysis agent
- Exercise/fitness recommendation agent
- Mental health information agent

#### 11. **Integration Ecosystem**
- Obsidian plugin
- Notion integration
- Slack bot
- Discord bot
- VSCode extension

#### 12. **Research Pipeline**
- Automated literature reviews
- Meta-analysis generation
- Study quality assessment
- Evidence synthesis tools

---

## Common Pitfalls to Avoid

### 1. **Over-Engineering Early**

**Bad:**
```python
# Don't create complex abstractions before they're needed
class AbstractMedicalAnalysisStrategyFactoryBuilder:
    def create_strategy_factory(self, strategy_type):
        # 200 lines of abstraction for 2 use cases
        pass
```

**Good:**
```python
# Start simple, refactor when patterns emerge
def analyze_procedure(procedure: str) -> dict:
    """Direct, simple implementation"""
    pass
```

### 2. **Ignoring Error Cases**

**Bad:**
```python
response = self.llm_manager.generate(prompt)
result = json.loads(response)  # What if it's not JSON?
```

**Good:**
```python
try:
    response = self.llm_manager.generate(prompt)
    result = json.loads(response)
except json.JSONDecodeError:
    self.logger.warning("LLM returned invalid JSON, using fallback parser")
    result = self._parse_text_response(response)
```

### 3. **Not Testing with Real Data**

**Bad:**
```python
# Only test with perfect mock data
mock_response = "Perfect JSON response"
```

**Good:**
```python
# Test with realistic LLM responses
mock_responses = [
    "Valid JSON response",
    "Response with extra text before JSON",
    "Malformed JSON {incomplete",
    "",  # Empty response
    "No JSON at all, just text"
]
```

### 4. **Hardcoding Configuration**

**Bad:**
```python
def __init__(self):
    self.timeout = 60  # Hardcoded
    self.model = "claude-sonnet-4-5-20250929"  # Hardcoded
```

**Good:**
```python
def __init__(self, timeout: int = 300, model: str = "claude-sonnet-4-5-20250929"):
    self.timeout = timeout
    self.model = model
```

### 5. **Not Tracking Costs**

**Bad:**
```python
# Make LLM calls without tracking
response = llm.generate(prompt)
```

**Good:**
```python
# Initialize token tracker at start of analysis
self.total_token_usage = TokenUsage()

# Always accumulate token usage after EVERY LLM call
response, token_usage = llm.generate(prompt)
if token_usage:
    self.total_token_usage.add(token_usage)

# Use @track_cost decorator for phase-based cost tracking
@track_cost("Phase 1: Analysis")
def _phase1_analysis(self, input_data):
    # Token accumulation happens here
    response, token_usage = self.llm_manager.generate_response(prompt)
    if token_usage:
        self.total_token_usage.add(token_usage)
    return result
```

**Requirements for Cost Tracking:**
1. Initialize `self.total_token_usage = TokenUsage()` at the start of each analysis
2. Add `if token_usage: self.total_token_usage.add(token_usage)` after EVERY `generate_response()` call
3. Use `@track_cost("Phase Name")` decorator on phase methods
4. The decorator will automatically calculate costs based on token differences

### 6. **Forgetting Disclaimers**

**Bad:**
```python
summary = "You should take X medication for Y condition."
```

**Good:**
```python
summary = """Analysis Summary:
[Content]

⚠️ DISCLAIMER: This analysis is for educational and research purposes only.
It does not constitute medical advice. Always consult qualified healthcare
professionals for medical decisions."""
```

### 7. **Manual JSON Parsing Instead of DSPy**

**Bad:**
```python
# Fragile manual parsing
json_match = re.search(r'\{.*\}', response)
if json_match:
    try:
        return json.loads(json_match.group(0))
    except:
        return {}
```

**Good:**
```python
# Use DSPy + Pydantic for structured output
schema = MedicationData.model_json_schema()
prompt = f"Analyze {med}. Respond with JSON matching: {json.dumps(schema)}"
response, token_usage = self.llm_manager.generate_response(prompt)
if token_usage:
    self.total_token_usage.add(token_usage)
result = self._parse_with_pydantic(response, MedicationData)
return result.model_dump() if result else {}
```

### 8. **Synchronous Processing of Independent Tasks**

**Bad:**
```python
# Sequential when could be parallel
result1 = analyze_organ("kidney")
result2 = analyze_organ("liver")
result3 = analyze_organ("brain")
```

**Good:**
```python
# Parallel processing
with ThreadPoolExecutor() as executor:
    results = list(executor.map(analyze_organ, ["kidney", "liver", "brain"]))
```

---

## LLM-Specific Best Practices

### For AI Assistants (Claude, GPT-4, etc.) Working on This Codebase

#### 1. **Always Read Before Editing**

```python
# Before editing a file, read it first
Read("/path/to/file.py")

# Then make informed edits
Edit("/path/to/file.py", old_string="...", new_string="...")
```

#### 2. **Use Existing Patterns**

When adding new features, follow existing patterns:

```python
# If adding a new agent, look at MedicalFactChecker structure
# If adding a new phase, look at how existing phases are implemented
# If adding tests, look at test_medical_fact_checker.py
```

#### 3. **Preserve User Intent**

If a user has specific requirements (like "keep the biases"), respect them:

```python
# User said: "keep the biases"
# Don't simplify or remove the philosophical biases from the fact checker
```

#### 4. **Progressive Implementation**

Build features incrementally:

1. **First:** Get basic functionality working
2. **Second:** Add error handling
3. **Third:** Add tests
4. **Fourth:** Add documentation
5. **Fifth:** Optimize

#### 5. **Ask When Uncertain**

If design decisions have trade-offs:

```python
# Instead of guessing, ask:
"Should the timeout be configurable via CLI, environment variable, or both?"
"Do you want this to run in parallel or sequential?"
"Should I create a new module or extend existing one?"
```

#### 6. **Respect Project Tools**

This project uses **UV** for dependency management:

```bash
# Good
uv sync
uv run pytest

# Bad
pip install pytest
python -m pytest
```

#### 7. **Update Related Documentation**

When adding features, update:
- README.md (if user-facing)
- This file (if changing patterns)
- USAGE.md (if changing CLI)
- Docstrings (always)

#### 8. **Test Before Claiming Success**

```python
# Don't just write code, verify it works:
uv run pytest medical_fact_checker/test_medical_fact_checker.py -v
```

#### 9. **Provide Usage Examples**

When adding new features, show how to use them:

```bash
# Example of running new feature
uv run python run_analysis.py factcheck --subject "New Topic" --new-flag value
```

#### 10. **Consider Token Usage**

You're using the user's tokens, so:
- Read only files you need
- Don't generate unnecessarily verbose code
- Consolidate tool calls when possible
- Use efficient search patterns (Grep > Read entire file)

---

## Code Quality Checklist

Before considering a feature complete, verify:

- [ ] **Functionality**: Does it work as intended?
- [ ] **Tests**: Are there unit tests with >80% coverage?
- [ ] **Documentation**: README, docstrings, and inline comments?
- [ ] **Type Hints**: All functions have proper type annotations?
- [ ] **Error Handling**: Graceful handling of edge cases?
- [ ] **Logging**: Appropriate info/warning/error logs?
- [ ] **Performance**: Reasonable execution time?
- [ ] **Security**: No hardcoded secrets or injection vulnerabilities?
- [ ] **Consistency**: Follows existing code patterns?
- [ ] **Integration**: Works with existing agents/tools?

---

## Getting Help

### Common Issues

**"Import errors"**
```bash
uv sync --reinstall
```

**"Tests failing"**
```bash
# Check if you're mocking LLM calls
# Check if test fixtures are set up correctly
```

**"Timeout errors"**
```bash
# Increase timeout: --timeout 600
# Check network connectivity
# Verify API keys are valid
```

### Resources

- **Project Structure**: See existing agents as examples
- **LLM Integration**: Check `llm_integrations.py`
- **Testing**: See `test_medical_fact_checker.py`
- **CLI**: See `run_analysis.py`
- **DSPy**: https://github.com/stanfordnlp/dspy
- **LangChain**: https://python.langchain.com/

---

## Contributing Guidelines

### Before Starting Work

1. Read this entire document
2. Review existing code structure
3. Check if similar functionality exists
4. Plan your approach (write it down)
5. Discuss major changes with team/user

### While Working

1. Commit frequently with clear messages
2. Write tests as you go
3. Document as you code
4. Run tests before pushing
5. Update relevant documentation

### Before Submitting

1. Run full test suite: `uv run pytest`
2. Check code formatting: `uv run black .`
3. Verify type hints: `uv run mypy .`
4. Update CHANGELOG.md if significant
5. Ensure all documentation is current

### Commit Message Format

```
type(scope): brief description

Detailed explanation if needed

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Examples:
```
feat(fact-checker): add web research integration
fix(llm): increase timeout to prevent API errors
docs(readme): add usage examples for new CLI flags
test(procedure): add integration tests for full workflow
```

---

## Final Notes

### Philosophy

This repository aims to:
1. **Assist, not replace** - Medical professionals remain essential
2. **Transparency** - All reasoning should be auditable
3. **Safety** - Never claim to provide medical advice
4. **Quality** - Evidence-based, well-tested code
5. **Usability** - Easy to use, extend, and understand

### Continuous Improvement

This document should evolve with the project:
- Add new patterns as they emerge
- Remove outdated guidance
- Clarify ambiguous sections
- Incorporate lessons learned

### Questions?

If something isn't clear:
1. Check existing code for examples
2. Read the documentation
3. Ask the maintainer/user
4. Document the answer here for others

---

**Last Updated:** 2025-12-10
**Version:** 1.3.0
**Maintainer:** Research Agent Alpha Team

**Recent Changes:**
- ⭐ **MANDATORY: Added emoji formatting standard for all report titles and subtitles**
- ⭐ **MANDATORY: Added requirement for References section in all reports**
- ⭐ **MANDATORY: Added requirement for Cost Analysis in every phase**
- Added comprehensive DSPy Structured Output Pattern (Section 3)
- Added cost tracking requirements and best practices
- Added pitfall: Manual JSON parsing instead of DSPy
- Updated all examples to include token accumulation
- Added Report Generation Standards section with mandatory requirements
- Emphasized @track_cost decorator usage for all phases
- Updated run_analysis.py and report_generator.py with consistent emoji usage
