"""
Scalable routing module for medical multi-agent CLI.

Routes user queries to the most appropriate specialized agent from a dynamic
list of available agents.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

# Import LLM utilities from shared integrations module
from llm_integrations import (
    LLMProvider,
    get_available_models,
    get_active_models,
    get_models_by_supplier,
    is_model_deprecated,
    call_model,
)

from check_llms import print_llm_status
from observability import setup_phoenix, get_tracer

# Load environment variables: .env.dev first (dev-specific), then .env (base)
try:
    from dotenv import load_dotenv as _load_dotenv
    import pathlib as _pathlib

    _repo_root = _pathlib.Path(__file__).parent
    _load_dotenv(_repo_root / ".env.dev", override=False)  # dev-specific vars
    _load_dotenv(_repo_root / ".env", override=False)       # base vars (don't overwrite)
except ImportError:
    pass  # python-dotenv not installed, rely on shell environment


# Default model for routing — grok-4.3 is the current xAI flagship
DEFAULT_ROUTING_MODEL = "grok-4.3"

# Maximum characters of document context passed to agents (prevents context overflow)
MAX_DOCUMENT_CONTEXT_CHARS = 100_000


@dataclass
class AgentSpec:
    """Specification for a routable agent."""
    id: str
    name: str
    description: str
    routing_notes: Optional[str] = None


# Define sample agents at the module level for clean importing
sample_agents = [
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


def call_llm(messages: list[dict[str, str]], model: str = DEFAULT_ROUTING_MODEL) -> str:
    """
    Call LLM for routing decision using the universal call_model function.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: LLM model identifier to use for the call

    Returns:
        LLM response string
    """
    return call_model(model, messages)


def route_agent(
    user_query: str,
    agents: list[AgentSpec],
    default_agent_id: str | None = None,
    model: str = DEFAULT_ROUTING_MODEL
) -> str:
    """
    Route a user query to exactly one agent from the provided list.

    Uses an LLM to analyze the query and select the most appropriate specialized
    agent based on agent descriptions and routing notes.

    Args:
        user_query: The user's input query
        agents: List of available agents to route to
        default_agent_id: Optional fallback agent id if routing fails
        model: LLM model to use for routing decision

    Returns:
        The id of the selected agent (guaranteed to be from the agents list)
    """
    if not agents:
        raise ValueError("agents list cannot be empty")

    # Build agent list for system prompt
    agent_sections = []
    for agent in agents:
        section = f"""Agent ID: {agent.id}
Name: {agent.name}
Description: {agent.description}"""
        if agent.routing_notes:
            section += f"\nRouting Notes: {agent.routing_notes}"
        agent_sections.append(section)

    agents_text = "\n\n".join(agent_sections)

    # Add scalability hint for large agent lists
    scalability_hint = ""
    if len(agents) >= 10:
        scalability_hint = "\nThink carefully about which agent is MOST specialized for this query."

    # Build system prompt
    system_prompt = f"""You are an intelligent router for a medical multi-agent system.

Your task is to analyze the user's query and select the ONE best agent to handle it.

Available agents:

{agents_text}

Instructions:
- Carefully read the user's query
- Consider which agent is most specialized and appropriate{scalability_hint}
- You MUST respond with EXACTLY ONE agent id from the list above
- Output ONLY the agent id, no explanations, no extra text
- The agent id must match exactly (case-sensitive)

Remember: Output only the agent id, nothing else."""

    # Call LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    response = call_llm(messages, model=model)

    # Normalize and validate response
    normalized_response = response.strip().lower()
    agent_id_map = {agent.id.lower(): agent.id for agent in agents}

    # Try to match normalized response
    if normalized_response in agent_id_map:
        return agent_id_map[normalized_response]

    # Fallback to default agent if provided and valid
    if default_agent_id:
        default_normalized = default_agent_id.lower()
        if default_normalized in agent_id_map:
            return agent_id_map[default_normalized]

    # Final fallback: return first agent
    return agents[0].id


def main():
    """Entry point for the medical-router console script."""
    import sys
    # Import the existing AgentOrchestrator that saves files
    from run_analysis import AgentOrchestrator
    from document_parser import parse_document, ParseStatus

    parser = argparse.ArgumentParser(description="Medical Multi-Agent Router (REPL)")
    parser.add_argument(
        "--check-llms",
        action="store_true",
        help="Print which LLM providers are configured and exit",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="List supported model identifiers and exit",
    )
    parser.add_argument(
        "--implementation",
        choices=["original", "langchain"],
        default="langchain",
        help="Agent implementation to use (default: langchain)",
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web research (enabled by default for LangChain implementation)",
    )
    args = parser.parse_args()

    # Start Phoenix observability (always-on, tolerates failure silently)
    _phoenix_url = setup_phoenix()

    if args.check_llms:
        print_llm_status(load_env=True)
        raise SystemExit(0)

    if args.models:
        available_models = get_available_models()
        print("\nAvailable model identifiers:")
        for model_name, provider in sorted(available_models.items()):
            print(f"  - {model_name} ({provider})")
        print()
        raise SystemExit(0)

    # NOTE: Mock implementation commented out - now using real LLM integration
    # Uncomment below to test without API calls:
    #
    # def mock_call_llm(messages: list[dict[str, str]], model: str = DEFAULT_ROUTING_MODEL) -> str:
    #     """Simple mock that returns first agent for testing."""
    #     query = messages[1]["content"].lower()
    #     if "medication" in query or "drug" in query or "prescription" in query:
    #         return "medication_agent"
    #     elif "procedure" in query or "surgery" in query or "treatment" in query:
    #         return "procedure_agent"
    #     elif "diagnosis" in query or "symptom" in query or "condition" in query:
    #         return "diagnostic_agent"
    #     else:
    #         return "general_agent"
    # globals()["call_llm"] = mock_call_llm

    implementation = args.implementation
    web_search_enabled = not args.no_web_search

    def _warn_langsmith() -> None:
        tracing_flag = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if tracing_flag in ("1", "true", "yes") and api_key:
            return
        print(
            "Warning: LangSmith tracing is not enabled. "
            "To enable, set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY. "
            "Optional: set LANGCHAIN_PROJECT."
        )

    # Using module-level sample_agents

    # Get selectable (non-deprecated) models. Deprecated models (release date
    # > 1 year ago) remain callable via the API but are hidden from the menu.
    available_models = list(get_active_models().keys())

    print("Medical Multi-Agent Router - Test REPL")
    print("=" * 50)

    # Log available LLM models grouped by supplier BEFORE accepting any input.
    print("\nAvailable LLM models by supplier:")
    _menu_index: dict[str, int] = {
        model: i for i, model in enumerate(available_models, 1)
    }
    for supplier, models in get_models_by_supplier().items():
        # Only show suppliers that have at least one active model
        active_for_supplier = [m for m in models if m in _menu_index]
        if not active_for_supplier:
            continue
        print(f"\n  {supplier}:")
        for model in active_for_supplier:
            default_marker = " (default)" if model == DEFAULT_ROUTING_MODEL else ""
            print(f"    {_menu_index[model]}. {model}{default_marker}")

    # Note any deprecated models that were hidden from selection
    _deprecated = [m for m in get_available_models() if is_model_deprecated(m)]
    if _deprecated:
        print(
            f"\n  ⚠️  {len(_deprecated)} model(s) deprecated (>1 year old) and hidden "
            f"from selection: {', '.join(sorted(_deprecated))}"
        )

    # Model selection
    print(f"\nSelect a model (1-{len(available_models)}) or press Enter for default [{DEFAULT_ROUTING_MODEL}]: ", end="")
    model_choice = input().strip()

    if model_choice and model_choice.isdigit():
        model_idx = int(model_choice) - 1
        if 0 <= model_idx < len(available_models):
            selected_model = available_models[model_idx]
        else:
            print(f"Invalid choice. Using default: {DEFAULT_ROUTING_MODEL}")
            selected_model = DEFAULT_ROUTING_MODEL
    else:
        selected_model = DEFAULT_ROUTING_MODEL

    # Vertex models require VERTEX_PROJECT — catch misconfiguration at selection time
    _vertex_providers = ("claude-vertex", "gemini-vertex", "claude-vertex-opus")
    _selected_provider = get_available_models().get(selected_model, "")
    if _selected_provider in _vertex_providers and not os.getenv("VERTEX_PROJECT"):
        print(
            f"\n⚠️  Model '{selected_model}' requires Vertex AI configuration.\n"
            f"   Set VERTEX_PROJECT in your .env.dev file (and optionally VERTEX_LOCATION).\n"
            f"   Falling back to default model: {DEFAULT_ROUTING_MODEL}\n"
        )
        selected_model = DEFAULT_ROUTING_MODEL

    print(f"\nUsing model: {selected_model}")
    print(f"Implementation: {implementation}")
    print(f"Web research: {'enabled' if web_search_enabled else 'disabled'}")
    print(f"Available agents: {', '.join(a.id for a in sample_agents)}")
    if _phoenix_url:
        print(f"Tracing (Phoenix): {_phoenix_url}")

    _warn_langsmith()

    # Initialize orchestrator (uses selected model through llm_provider param)
    orchestrator = AgentOrchestrator(output_dir="outputs")

    # Map model to provider name for orchestrator
    available_models_dict = get_available_models()
    llm_provider = available_models_dict.get(selected_model, "grok-4.3")

    print("\nCommands:")
    print("  - Type a query to route and execute it")
    print("  - '/models' to list available models")
    print("  - '/model <number>' to change model")
    print("  - '/impl <original|langchain>' to change implementation")
    print("  - '/web <on|off>' to toggle web research (on by default)")
    print("  - '/file <path>' to attach a document as context (PDF/Word/txt/md/rtf)")
    print("  - '/file' to show attachment status, '/file clear' to remove")
    print("  - 'quit' or 'exit' to stop\n")

    last_files = None  # Track last generated files
    attached_document_context: str | None = None  # sticky; cleared only by /file clear

    while True:
        try:
            query = input("Enter query: ").strip()

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not query:
                continue

            # Handle commands
            if query == "/models":
                print("\nAvailable models:")
                for i, model in enumerate(available_models, 1):
                    current_marker = " (current)" if model == selected_model else ""
                    print(f"  {i}. {model}{current_marker}")
                print()
                continue

            if query.startswith("/model "):
                parts = query.split()
                if len(parts) == 2 and parts[1].isdigit():
                    model_idx = int(parts[1]) - 1
                    if 0 <= model_idx < len(available_models):
                        selected_model = available_models[model_idx]
                        # Update llm_provider for new model
                        llm_provider = available_models_dict.get(selected_model, "grok-4.3")
                        print(f"→ Switched to model: {selected_model} (provider: {llm_provider})\n")
                    else:
                        print(f"Invalid model number. Use 1-{len(available_models)}\n")
                else:
                    print("Usage: /model <number>\n")
                continue

            if query.startswith("/impl "):
                parts = query.split()
                if len(parts) == 2 and parts[1] in ["original", "langchain"]:
                    implementation = parts[1]
                    print(f"→ Switched implementation to: {implementation}\n")
                else:
                    print("Usage: /impl original|langchain\n")
                continue

            if query.startswith("/web "):
                parts = query.split()
                if len(parts) == 2 and parts[1] in ["on", "off"]:
                    web_search_enabled = parts[1] == "on"
                    status = "enabled" if web_search_enabled else "disabled"
                    print(f"→ Web research {status}\n")
                else:
                    print("Usage: /web on|off\n")
                continue

            if query == "/file clear":
                attached_document_context = None
                print("→ Document context cleared.\n")
                continue

            if query == "/file":
                if attached_document_context:
                    print(f"→ Document attached: {len(attached_document_context):,} chars. Use '/file clear' to remove it.\n")
                else:
                    print("→ No document attached. Use '/file <path>' to attach one.\n")
                continue

            if query.startswith("/file "):
                file_path = query[len("/file "):].strip()
                if not file_path:
                    print("→ Usage: /file <path>\n")
                    continue
                try:
                    result = parse_document(file_path)
                except FileNotFoundError:
                    print(f"→ File not found: {file_path}\n")
                    continue

                if not result.ok:
                    print(f"→ Could not parse '{file_path}':")
                    for w in result.warnings:
                        print(f"   ⚠ {w}")
                    print()
                    continue

                # Truncation with overflow notification
                original_len = len(result.markdown)
                if original_len > MAX_DOCUMENT_CONTEXT_CHARS:
                    dropped = original_len - MAX_DOCUMENT_CONTEXT_CHARS
                    pct = dropped / original_len * 100
                    print(
                        f"⚠️  Document is {original_len:,} chars — exceeds the "
                        f"{MAX_DOCUMENT_CONTEXT_CHARS:,}-char context limit.\n"
                        f"   Truncated to {MAX_DOCUMENT_CONTEXT_CHARS:,} chars "
                        f"(dropped {dropped:,} chars, ~{pct:.1f}% of the document).\n"
                        f"   Only the first {MAX_DOCUMENT_CONTEXT_CHARS:,} chars will be used as context."
                    )
                    attached_document_context = result.markdown[:MAX_DOCUMENT_CONTEXT_CHARS]
                else:
                    attached_document_context = result.markdown

                # Build confirmation line with available metadata
                fmt = result.metadata.file_format
                pages = result.metadata.page_count
                page_str = f", {pages} page{'s' if pages != 1 else ''}" if pages else ""
                char_count = len(attached_document_context)

                # PARTIAL warning
                if result.status is ParseStatus.PARTIAL:
                    for w in result.warnings:
                        print(f"   ⚠ {w}")

                fname = os.path.basename(file_path)
                print(f"✓ Attached {fname} ({fmt}{page_str}, {char_count:,} chars). Stays attached until '/file clear'.\n")
                continue

            # Route and execute inside a tracing span
            tracer = get_tracer()
            with tracer.start_as_current_span("router.session") as session_span:
                session_span.set_attribute("query", query)
                session_span.set_attribute("model", selected_model)
                session_span.set_attribute("implementation", implementation)
                session_span.set_attribute("web_search_enabled", web_search_enabled)
                session_span.set_attribute("document_context_attached", attached_document_context is not None)

                # Route the query
                print(f"→ Routing query...")
                selected_agent_id = route_agent(
                    query,
                    sample_agents,
                    default_agent_id="general_agent",
                    model=selected_model
                )
                selected_agent = next(a for a in sample_agents if a.id == selected_agent_id)
                session_span.set_attribute("routed_to", selected_agent_id)

                print(f"→ Routed to: {selected_agent_id} ({selected_agent.name})")
                print(f"→ Executing {selected_agent.name}...")
                print()

                # Execute the appropriate analysis method based on agent
                try:
                    if selected_agent_id == "medication_agent":
                        result, files = orchestrator.run_medication_analyzer(
                            medication=query,
                            indication=None,
                            other_medications=None,
                            llm_provider=llm_provider,
                            timeout=300,
                            implementation=implementation,
                            enable_web_research=web_search_enabled,
                            document_context=attached_document_context or "",
                        )
                    elif selected_agent_id == "procedure_agent":
                        result, files = orchestrator.run_procedure_analyzer(
                            procedure=query,
                            details="User query via router",
                            llm_provider=llm_provider,
                            timeout=300,
                            implementation=implementation,
                            enable_web_research=web_search_enabled,
                            document_context=attached_document_context or "",
                        )
                    elif selected_agent_id == "diagnostic_agent":
                        result, files = orchestrator.run_diagnostic_analyzer(
                            query=query,
                            llm_provider=llm_provider,
                            timeout=300,
                            interactive=False,  # non-interactive in router mode
                            document_context=attached_document_context or "",
                        )
                    elif selected_agent_id == "general_agent":
                        result, files = orchestrator.run_fact_checker(
                            subject=query,
                            context="",
                            llm_provider=llm_provider,
                            timeout=300,
                            implementation=implementation,
                            enable_web_research=web_search_enabled,
                            document_context=attached_document_context or "",
                        )

                    last_files = files
                    session_span.set_attribute("output.files_count", len(files))

                    # Show generated files
                    print("\n" + "=" * 60)
                    print("📁 Generated Files:")
                    print("=" * 60)
                    for file_type, file_path in files.items():
                        print(f"✓ {file_type}: {file_path}")
                    print("=" * 60)
                    print()

                except Exception as e:
                    session_span.record_exception(e)
                    print(f"✗ Error during analysis: {e}")
                    import traceback
                    traceback.print_exc()
                    print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
