# Medical Analysis Agents - Usage Guide

This repository contains multiple medical AI agents that can be run through a unified interface.

## Quick Start

### List Available Agents

```bash
uv run python run_analysis.py --list
```

Output:
```
================================================================================
Available Medical Analysis Agents
================================================================================

📌 procedure
   Name: Medical Procedure Analyzer
   Description: Analyzes medical procedures with organ-focused reasoning

📌 factcheck
   Name: Medical Fact Checker
   Description: Independent bio-investigator for health subjects
```

## Running Agents

### 1. Medical Procedure Analyzer

Analyzes medical procedures with organ-focused reasoning, identifying risks and evidence-based recommendations.

```bash
# Basic usage
uv run python run_analysis.py procedure \
  --subject "MRI Scanner" \
  --details "With gadolinium contrast"

# With different LLM
uv run python run_analysis.py procedure \
  --subject "CT Scan" \
  --details "Contrast-enhanced" \
  --llm openai

# Custom output directory
uv run python run_analysis.py procedure \
  --subject "Cardiac Catheterization" \
  --details "With contrast" \
  --output-dir my_analysis_results
```

**Output Files:**
- `{procedure}_reasoning_trace_{timestamp}.json` - Detailed step-by-step reasoning
- `{procedure}_analysis_result_{timestamp}.json` - Structured analysis results
- `{procedure}_summary_report_{timestamp}.md` - Human-readable summary

### 2. Medical Fact Checker

Independent bio-investigator that critically analyzes health subjects, skeptical of corporate-funded research.

```bash
# Basic usage
uv run python run_analysis.py factcheck \
  --subject "Vitamin D supplementation"

# With context
uv run python run_analysis.py factcheck \
  --subject "Omega-6 fatty acids" \
  --context "inflammation and cardiovascular health"

# With different LLM
uv run python run_analysis.py factcheck \
  --subject "Intermittent fasting" \
  --context "metabolic health" \
  --llm openai
```

**Output Files:**
- `{subject}_session_{timestamp}.json` - Complete session data with all phases
- `{subject}_output_{timestamp}.md` - Final analysis output
- `{subject}_summary_{timestamp}.md` - Summary of phases and findings

## Examples

### Example 1: Analyze MRI Procedure
```bash
uv run python run_analysis.py procedure \
  --subject "MRI Scanner" \
  --details "With gadolinium contrast"
```

### Example 2: Fact Check Coffee Health Claims
```bash
uv run python run_analysis.py factcheck \
  --subject "Coffee consumption" \
  --context "cognitive performance and longevity"
```

### Example 3: Analyze Multiple Procedures (Batch)
```bash
# Create a simple script
for procedure in "MRI Scanner" "CT Scan" "X-Ray"; do
  uv run python run_analysis.py procedure \
    --subject "$procedure" \
    --details "Standard protocol"
done
```

### Example 4: Compare Different LLMs
```bash
# Run with Claude
uv run python run_analysis.py factcheck \
  --subject "Red meat" \
  --context "health impact" \
  --llm claude \
  --output-dir results_claude

# Run with OpenAI
uv run python run_analysis.py factcheck \
  --subject "Red meat" \
  --context "health impact" \
  --llm openai \
  --output-dir results_openai
```

## Command-Line Arguments

### Common Arguments (All Agents)

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `agent` | Which agent to run (`procedure` or `factcheck`) | - | Yes |
| `--subject` | Subject to analyze | - | Yes |
| `--llm` | LLM provider (`claude`, `openai`, `ollama`) | `claude` | No |
| `--output-dir` | Directory for output files | `outputs/` | No |
| `--list` | List all available agents | - | No |

### Procedure Analyzer Specific

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--details` | Additional details about the procedure | `""` | No |

### Fact Checker Specific

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--context` | Context or scope for the investigation | `""` | No |

## Environment Setup

### Prerequisites

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <repo-url>
cd research_agent_alpha

# Install dependencies
uv sync

# Install with dev dependencies (for testing)
uv sync --extra dev
```

### API Keys

Set up your API keys as environment variables:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

Or create a `.env` file in the root directory:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

## Output Structure

All agents save outputs to the `outputs/` directory (or custom directory specified with `--output-dir`).

### Procedure Analyzer Output Structure

```
outputs/
├── mri_scanner_reasoning_trace_20250512_143022.json
├── mri_scanner_analysis_result_20250512_143022.json
└── mri_scanner_summary_report_20250512_143022.md
```

**reasoning_trace.json** - Step-by-step reasoning through 6 stages:
1. Input Analysis
2. Organ Identification
3. Evidence Gathering
4. Risk Assessment
5. Recommendation Synthesis
6. Critical Evaluation

**analysis_result.json** - Structured data including:
- Organs analyzed with risk levels
- Evidence-based recommendations
- Investigational approaches
- Debunked claims
- Research gaps

**summary_report.md** - Human-readable markdown report

### Fact Checker Output Structure

```
outputs/
├── vitamin_d_session_20250512_143022.json
├── vitamin_d_output_20250512_143022.md
└── vitamin_d_summary_20250512_143022.md
```

**session.json** - Complete session data with all 5 phases:
1. Conflict & Hypothesis Scan
2. Evidence Stress-Test
3. Synthesis & Menu
4. Complex Output Generation
5. Simplified Output

**output.md** - Final analysis output (full report)

**summary.md** - Summary of phases and key findings

## Advanced Usage

### Using in Python Scripts

```python
from run_analysis import AgentOrchestrator

# Initialize orchestrator
orchestrator = AgentOrchestrator(output_dir="my_outputs")

# Run procedure analyzer
result, files = orchestrator.run_procedure_analyzer(
    procedure="MRI Scanner",
    details="With gadolinium contrast",
    llm_provider="claude"
)

# Run fact checker
session, files = orchestrator.run_fact_checker(
    subject="Vitamin D",
    context="optimal dosing",
    llm_provider="claude"
)

# Access results
print(f"Procedure confidence: {result.confidence_score}")
print(f"Fact check phases: {len(session.phase_results)}")
```

### Adding New Agents

To add a new agent to the orchestrator:

1. Create your agent class in a new module
2. Add it to `AgentOrchestrator.AGENTS` in `run_analysis.py`:

```python
AGENTS = {
    "procedure": {...},
    "factcheck": {...},
    "myagent": {  # Add your agent here
        "name": "My Custom Agent",
        "description": "Does something useful",
        "class": MyAgentClass,
    },
}
```

3. Add a runner method:

```python
def run_my_agent(self, input_data, llm_provider="claude"):
    # Initialize and run your agent
    agent = MyAgentClass(primary_llm_provider=llm_provider)
    result = agent.analyze(input_data)

    # Save outputs
    files = self._save_my_agent_results(result)

    return result, files
```

4. Update the argument parser in `main()` to include your agent

## Troubleshooting

### "No such file or directory: pytest"

Run: `uv sync --extra dev`

### "Failed to build medical-reasoning-agent"

The package configuration has been updated. Try: `uv sync --reinstall`

### API Key Errors

Verify your API keys are set:
```bash
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
```

### Import Errors

Make sure you're in the project root directory and run:
```bash
uv sync
```

## Future Plans

- [ ] Make agents available as MCP (Model Context Protocol) servers
- [ ] Add web interface for interactive agent selection
- [ ] Support batch processing of multiple subjects
- [ ] Add agent comparison mode
- [ ] Create unified JSON schema for all agent outputs
- [ ] Add result visualization tools

## Contributing

When adding new agents:
1. Follow the existing agent structure
2. Implement consistent output formats
3. Add comprehensive tests
4. Update this usage guide
5. Add examples to the README

## License

Part of the medical reasoning agent research project.
