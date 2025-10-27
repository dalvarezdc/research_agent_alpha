# Medical Reasoning Agent 🧠⚕️

An AI agent that follows systematic medical analysis patterns, providing organ-focused reasoning for medical procedures with evidence-based recommendations.

## Overview

This agent replicates the analytical thinking pattern demonstrated in our conversation:
1. **Broad Analysis** → **Specific Focus** → **Critical Evaluation**
2. **Known Recommendations** vs **Potential Treatments** vs **Debunked Claims**
3. **Complete Reasoning Trace** with confidence scoring

## Key Features

- 🔍 **Systematic Medical Reasoning**: Follows structured analysis pipeline
- 🧭 **Organ-Focused Analysis**: Targets specific organ systems affected by procedures
- 🤖 **Multi-LLM Support**: Works with Claude, OpenAI, and local models (Ollama)
- 📊 **Evidence Classification**: Distinguishes between proven, potential, and debunked treatments
- 🔍 **Reasoning Transparency**: Full trace of AI thinking process
- ✅ **Quality Validation**: Comprehensive output validation and scoring
- 🧪 **Local Testing**: Complete testing framework for development

## Installation

### Prerequisites
- Python 3.12+
- UV package manager
- API keys for LLM providers (optional - can use local models)

### Setup

```bash
# Install dependencies with UV
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"

# Install optional dependencies
uv pip install -e ".[viz,notebook]"
```

### Environment Variables

Create a `.env` file:

```bash
# LLM API Keys (optional - for cloud providers)
ANTHROPIC_API_KEY=your_claude_key_here
OPENAI_API_KEY=your_openai_key_here

# Ollama (for local models)
OLLAMA_BASE_URL=http://localhost:11434
```

## Quick Start

### Basic Usage

```python
from medical_reasoning_agent import MedicalReasoningAgent, MedicalInput

# Create agent
agent = MedicalReasoningAgent(
    primary_llm_provider="claude",
    fallback_providers=["openai", "ollama"],
    enable_logging=True
)

# Define medical scenario
medical_input = MedicalInput(
    procedure="MRI Scanner",
    details="With gadolinium contrast",
    objectives=[
        "understand implications",
        "risks", 
        "post-procedure care",
        "organs affected",
        "organs at risk"
    ]
)

# Run analysis
result = agent.analyze_medical_procedure(medical_input)

# Display results
print(f"Procedure: {result.procedure_summary}")
print(f"Confidence: {result.confidence_score:.2f}")

for organ in result.organs_analyzed:
    print(f"\n🔍 {organ.organ_name.upper()}:")
    print(f"  Risk Level: {organ.risk_level}")
    print(f"  Known Recommendations: {organ.known_recommendations}")
    print(f"  Potential Recommendations: {organ.potential_recommendations}")
    print(f"  Debunked Claims: {organ.debunked_claims}")
```

### Local Testing

```bash
# Run single scenario with Claude
python local_runner.py --provider claude --scenario-name MRI_with_gadolinium

# Compare multiple LLM providers
python local_runner.py --compare

# Run all scenarios with custom output directory
python local_runner.py --output my_test_results --log-level DEBUG

# Run specific scenario only
python local_runner.py -n "CT_with_iodine" -p openai
```

## Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=medical_reasoning_agent --cov-report=html

# Test with different LLM providers
python local_runner.py --compare
```

**⚠️ Disclaimer**: This tool is for educational and research purposes only. Always consult qualified healthcare professionals for medical advice.
