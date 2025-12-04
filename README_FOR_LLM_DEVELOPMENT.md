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

### 2. **Transparency Over Black Boxes**
- Export reasoning traces for all multi-step analyses
- Log token usage for cost tracking
- Provide confidence scores with explanations
- Make agent decisions auditable

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

# Bad - direct API calls
response = anthropic.messages.create(...)
```

### 3. **Phase-Based Analysis Pattern**

Complex agents should use phases:

```python
class MultiPhaseAgent:
    def analyze(self, input_data):
        self.phase_results = []

        # Phase 1
        phase1 = self._phase1_analysis(input_data)
        self.phase_results.append(phase1)

        # Phase 2 (depends on phase 1)
        phase2 = self._phase2_analysis(input_data, phase1.content)
        self.phase_results.append(phase2)

        # Continue phases...
        return self._synthesize_results()
```

### 4. **Data Flow Pattern**

```
Input Validation → LLM Analysis → Response Parsing → Output Formatting → Export
      ↓                 ↓               ↓                   ↓              ↓
 Reject invalid    Track tokens    Handle errors      Add metadata    Save JSON/MD
```

### 5. **Orchestrator Pattern**

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

### 2. **Token Usage Tracking**

Always track and report token usage:

```python
class Agent:
    def __init__(self):
        self.total_token_usage = TokenUsage()

    def _make_llm_call(self, prompt):
        response, token_usage = self.llm_manager.generate(prompt)
        self.total_token_usage.add(token_usage)
        return response

    def get_cost_estimate(self):
        """Estimate cost based on token usage"""
        # Claude Sonnet 4: ~$3/$15 per 1M tokens (input/output)
        input_cost = (self.total_token_usage.input_tokens / 1_000_000) * 3
        output_cost = (self.total_token_usage.output_tokens / 1_000_000) * 15
        return input_cost + output_cost
```

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
# Always track token usage
response, token_usage = llm.generate(prompt)
self.total_usage.add(token_usage)
self.logger.info(f"Tokens used: {token_usage.total_tokens}")
```

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

### 7. **Synchronous Processing of Independent Tasks**

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

**Last Updated:** 2025-12-05
**Version:** 1.0.0
**Maintainer:** Research Agent Alpha Team
