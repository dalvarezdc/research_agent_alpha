"""
Scalable routing module for medical multi-agent CLI.

Routes user queries to the most appropriate specialized agent from a dynamic
list of available agents.
"""

from dataclasses import dataclass
from typing import Optional

# Import LLM utilities from shared integrations module
from llm_integrations import LLMProvider, get_available_models, call_model


# Default model for routing
DEFAULT_ROUTING_MODEL = "grok-4-1-fast-non-reasoning-latest"


@dataclass
class AgentSpec:
    """Specification for a routable agent."""
    id: str
    name: str
    description: str
    routing_notes: Optional[str] = None


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


if __name__ == "__main__":
    import sys
    from orchestrator import AgentOrchestrator, AgentExecutionResult

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

    # Define sample agents
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

    # Get available models
    available_models = list(get_available_models().keys())

    print("Medical Multi-Agent Router - Test REPL")
    print("=" * 50)

    # Display available models
    print("\nAvailable LLM models:")
    for i, model in enumerate(available_models, 1):
        default_marker = " (default)" if model == DEFAULT_ROUTING_MODEL else ""
        print(f"  {i}. {model}{default_marker}")

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

    print(f"\nUsing model: {selected_model}")
    print(f"Available agents: {', '.join(a.id for a in sample_agents)}")

    # Initialize orchestrator with selected model
    orchestrator = AgentOrchestrator(llm_model=selected_model)

    print("\nCommands:")
    print("  - Type a query to route and execute it")
    print("  - '/models' to list available models")
    print("  - '/model <number>' to change model")
    print("  - 'full' to see complete analysis of last result")
    print("  - 'quit' or 'exit' to stop\n")

    last_result: Optional[AgentExecutionResult] = None

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
                        # Reinitialize orchestrator with new model
                        orchestrator = AgentOrchestrator(llm_model=selected_model)
                        print(f"→ Switched to model: {selected_model}\n")
                    else:
                        print(f"Invalid model number. Use 1-{len(available_models)}\n")
                else:
                    print("Usage: /model <number>\n")
                continue

            # Handle 'full' command to show complete analysis
            if query.lower() == "full":
                if last_result and last_result.success:
                    print("\n" + "=" * 60)
                    print(f"COMPLETE ANALYSIS: {last_result.agent_name}")
                    print("=" * 60)
                    print(f"\nQuery: {last_result.query}\n")
                    print(last_result.full_output)
                    print("\n" + "=" * 60 + "\n")
                else:
                    print("→ No previous result to display\n")
                continue

            # Route the query
            print(f"→ Routing query...")
            selected_agent_id = route_agent(
                query,
                sample_agents,
                default_agent_id="general_agent",
                model=selected_model
            )
            selected_agent = next(a for a in sample_agents if a.id == selected_agent_id)

            print(f"→ Routed to: {selected_agent_id} ({selected_agent.name})")
            print(f"→ Executing {selected_agent.name}...\n")

            # Execute the agent
            result = orchestrator.execute_agent(selected_agent_id, query)
            last_result = result

            if result.success:
                print(result.summary)
                print()
            else:
                print(f"✗ Error: {result.error_message}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}\n")
