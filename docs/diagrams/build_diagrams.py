#!/usr/bin/env python3
"""
Generates high-quality SVG diagrams corresponding to the Mermaid flowcharts in README.md.
Validates each SVG with xml.etree.ElementTree.
"""
import os
import xml.etree.ElementTree as ET

DIAGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))

def create_svg_header(width, height, title, subtitle=None):
    subtitle_elem = f'<text y="38" class="subtitle">{subtitle}</text>' if subtitle else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <!-- Card Shadow -->
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#000000" flood-opacity="0.35"/>
    </filter>

    <!-- Marker Arrows -->
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b"/>
    </marker>
    <marker id="arrow-dashed" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#94a3b8"/>
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8"/>
    </marker>
    <marker id="arrow-purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#a78bfa"/>
    </marker>
    <marker id="arrow-emerald" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#34d399"/>
    </marker>
    <marker id="arrow-amber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#fbbf24"/>
    </marker>
    <marker id="arrow-both-start" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 10 1 L 0 5 L 10 9 z" fill="#2dd4bf"/>
    </marker>
    <marker id="arrow-both-end" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#2dd4bf"/>
    </marker>
  </defs>

  <style>
    .bg {{ fill: #0b0f19; }}
    .border {{ stroke: #1e293b; stroke-width: 1.5; }}
    .title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 17px; font-weight: 700; fill: #f8fafc; letter-spacing: -0.01em; }}
    .subtitle {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11.5px; fill: #94a3b8; }}
    .node-title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 600; }}
    .node-desc {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; }}
    .node-code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 10.5px; }}
    .subgraph-title {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }}
    .edge-label {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 10.5px; font-weight: 500; fill: #cbd5e1; }}
    .edge-bg {{ fill: #1e293b; stroke: #334155; stroke-width: 1; rx: 4px; }}
    .flow-line {{ stroke: #64748b; stroke-width: 1.75; fill: none; }}
    .flow-dashed {{ stroke: #94a3b8; stroke-width: 1.5; stroke-dasharray: 4,4; fill: none; }}
  </style>

  <!-- Canvas Background -->
  <rect class="bg" width="{width}" height="{height}" rx="14"/>
  <rect class="border" fill="none" width="{width}" height="{height}" rx="14"/>

  <!-- Diagram Title Header -->
  <g transform="translate(30, 32)">
    <text class="title" y="16">{title}</text>
    {subtitle_elem}
  </g>
"""

def build_system_architecture_svg():
    width = 1120
    height = 940
    svg = create_svg_header(
        width, height,
        "System Architecture — Entry Points, Orchestrator &amp; Services",
        "Every entry point converges on the AgentOrchestrator to drive agents, validation, and multi-format outputs."
    )

    svg += """
  <!-- SUBGRAPH: ENTRY POINTS -->
  <g id="subgraph-clients">
    <rect x="30" y="75" width="670" height="145" rx="10" fill="#131d2e" stroke="#1e3a5f" stroke-width="1.2" stroke-dasharray="6,4"/>
    <rect x="42" y="65" width="115" height="20" rx="4" fill="#1e3a5f"/>
    <text x="50" y="79" class="subgraph-title" fill="#38bdf8">ENTRY POINTS</text>

    <!-- Node CLI -->
    <g id="node-cli" filter="url(#shadow)">
      <rect x="50" y="98" width="170" height="100" rx="8" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
      <text x="135" y="124" text-anchor="middle" class="node-title" fill="#e2e8f0">Direct CLI</text>
      <text x="135" y="148" text-anchor="middle" class="node-code" fill="#38bdf8">run_analysis.py</text>
      <text x="135" y="172" text-anchor="middle" class="node-desc" fill="#94a3b8">Explicit agent flag</text>
    </g>

    <!-- Node RT -->
    <g id="node-rt" filter="url(#shadow)">
      <rect x="245" y="98" width="185" height="100" rx="8" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
      <text x="337" y="124" text-anchor="middle" class="node-title" fill="#e2e8f0">Interactive Router</text>
      <text x="337" y="148" text-anchor="middle" class="node-code" fill="#38bdf8">router.py</text>
      <text x="337" y="172" text-anchor="middle" class="node-desc" fill="#94a3b8">Interactive REPL shell</text>
    </g>

    <!-- Node API -->
    <g id="node-api" filter="url(#shadow)">
      <rect x="455" y="98" width="225" height="100" rx="8" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>
      <text x="567" y="120" text-anchor="middle" class="node-title" fill="#e2e8f0">REST API (FastAPI)</text>
      <text x="567" y="140" text-anchor="middle" class="node-code" fill="#38bdf8">api.py</text>
      <text x="567" y="160" text-anchor="middle" class="node-desc" fill="#cbd5e1">/route · /analyze · /parse · /jobs</text>
      <text x="567" y="178" text-anchor="middle" class="node-desc" fill="#94a3b8">in-memory async job store</text>
    </g>
  </g>

  <!-- Node DP: document_parser -->
  <g id="node-dp" filter="url(#shadow)">
    <rect x="740" y="98" width="345" height="100" rx="8" fill="#1a2333" stroke="#0ea5e9" stroke-width="1.5"/>
    <text x="912" y="128" text-anchor="middle" class="node-title" fill="#38bdf8">📄 document_parser</text>
    <text x="912" y="152" text-anchor="middle" class="node-desc" fill="#e2e8f0">PDF · docx · txt · md · rtf · doc</text>
    <text x="912" y="174" text-anchor="middle" class="node-code" fill="#7dd3fc">→ structured markdown</text>
  </g>

  <!-- Node ROUTE: Router LLM -->
  <g id="node-route" filter="url(#shadow)">
    <polygon points="460,250 565,295 460,340 355,295" fill="#2d2212" stroke="#fbbf24" stroke-width="1.5"/>
    <text x="460" y="291" text-anchor="middle" class="node-title" fill="#fde68a">Router LLM</text>
    <text x="460" y="308" text-anchor="middle" class="node-code" fill="#fbbf24">route_agent()</text>
  </g>

  <!-- Node ORCH: AgentOrchestrator -->
  <g id="node-orch" filter="url(#shadow)">
    <rect x="300" y="390" width="320" height="75" rx="8" fill="#231d3d" stroke="#818cf8" stroke-width="1.75"/>
    <text x="460" y="420" text-anchor="middle" class="node-title" fill="#c7d2fe">AgentOrchestrator</text>
    <text x="460" y="442" text-anchor="middle" class="node-code" fill="#a5b4fc">run_analysis.py</text>
  </g>

  <!-- SUBGRAPH: SHARED SERVICES -->
  <g id="subgraph-services">
    <rect x="680" y="360" width="405" height="215" rx="10" fill="#0f2628" stroke="#145350" stroke-width="1.2" stroke-dasharray="6,4"/>
    <rect x="692" y="350" width="135" height="20" rx="4" fill="#145350"/>
    <text x="700" y="364" class="subgraph-title" fill="#2dd4bf">SHARED SERVICES</text>

    <!-- WEB -->
    <g id="node-web" filter="url(#shadow)">
      <rect x="705" y="385" width="355" height="48" rx="6" fill="#133838" stroke="#2dd4bf" stroke-width="1.2"/>
      <text x="720" y="407" class="node-title" fill="#99f6e4">🌐 Web Research</text>
      <text x="720" y="423" class="node-desc" fill="#94a3b8">Tavily → SerpAPI → DuckDuckGo priority chain</text>
    </g>

    <!-- COST -->
    <g id="node-cost" filter="url(#shadow)">
      <rect x="705" y="445" width="355" height="48" rx="6" fill="#133838" stroke="#2dd4bf" stroke-width="1.2"/>
      <text x="720" y="467" class="node-title" fill="#99f6e4">💰 CostTracker</text>
      <text x="720" y="483" class="node-desc" fill="#94a3b8">Per-phase tokens, cost sync &amp; limits</text>
    </g>

    <!-- TRACE -->
    <g id="node-trace" filter="url(#shadow)">
      <rect x="705" y="505" width="355" height="52" rx="6" fill="#133838" stroke="#2dd4bf" stroke-width="1.2"/>
      <text x="720" y="527" class="node-title" fill="#99f6e4">🔭 Observability &amp; Tracing</text>
      <text x="720" y="545" class="node-desc" fill="#94a3b8">Phoenix + OpenTelemetry (LangSmith optional)</text>
    </g>
  </g>

  <!-- Node AGENT: Agent LLM pipeline -->
  <g id="node-agent" filter="url(#shadow)">
    <rect x="300" y="500" width="320" height="75" rx="8" fill="#2e1a47" stroke="#a78bfa" stroke-width="1.75"/>
    <text x="460" y="529" text-anchor="middle" class="node-title" fill="#e9d5ff">🧠 Agent LLM Pipeline</text>
    <text x="460" y="552" text-anchor="middle" class="node-desc" fill="#c084fc">(black box — see Diagrams 2 and 3)</text>
  </g>

  <!-- Stage 4: Validation, Disclaimer & Cache -->
  <!-- Node REFVAL: Reference validation -->
  <g id="node-refval" filter="url(#shadow)">
    <rect x="300" y="615" width="320" height="65" rx="8" fill="#16273e" stroke="#38bdf8" stroke-width="1.5"/>
    <text x="460" y="642" text-anchor="middle" class="node-title" fill="#bae6fd">🔗 Reference Validation</text>
    <text x="460" y="662" text-anchor="middle" class="node-desc" fill="#7dd3fc">CitationURLCorrespondenceValidator</text>
  </g>

  <!-- Node RVC: SQLite Cache -->
  <g id="node-rvc" filter="url(#shadow)">
    <rect x="705" y="615" width="355" height="65" rx="8" fill="#162e2c" stroke="#2dd4bf" stroke-width="1.5"/>
    <text x="882" y="642" text-anchor="middle" class="node-title" fill="#99f6e4">🗄️ SQLite Reference Cache</text>
    <text x="882" y="662" text-anchor="middle" class="node-code" fill="#5eead4">cache/reference_validation.db (30d TTL)</text>
  </g>

  <!-- Node DISC: Hardcoded disclaimer -->
  <g id="node-disc" filter="url(#shadow)">
    <rect x="300" y="715" width="320" height="50" rx="8" fill="#3b111e" stroke="#f43f5e" stroke-width="1.5"/>
    <text x="460" y="745" text-anchor="middle" class="node-title" fill="#fecdd3">🛡️ Hardcoded Medical Disclaimer</text>
  </g>

  <!-- Stage 5: Outputs & Database -->
  <!-- Node FILES: outputs/ -->
  <g id="node-files" filter="url(#shadow)">
    <rect x="130" y="805" width="410" height="85" rx="8" fill="#0c2d25" stroke="#34d399" stroke-width="1.75"/>
    <text x="335" y="833" text-anchor="middle" class="node-title" fill="#a7f3d0">📁 Artifact Storage (outputs/)</text>
    <text x="335" y="855" text-anchor="middle" class="node-desc" fill="#6ee7b7">patient · practitioner · summary (.md &amp; .pdf)</text>
    <text x="335" y="873" text-anchor="middle" class="node-code" fill="#a7f3d0">result/session .json · cost_report.json · audit.json</text>
  </g>

  <!-- Node DB: SQLAlchemy Database -->
  <g id="node-db" filter="url(#shadow)">
    <rect x="630" y="805" width="430" height="85" rx="8" fill="#16223b" stroke="#60a5fa" stroke-width="1.5"/>
    <text x="845" y="833" text-anchor="middle" class="node-title" fill="#bfdbfe">🗃️ SQLAlchemy Database (SQLite / Postgres)</text>
    <text x="845" y="855" text-anchor="middle" class="node-desc" fill="#93c5fd">users · subjects · reports · report_files · patient_data</text>
    <text x="845" y="873" text-anchor="middle" class="node-desc" fill="#64748b">gated by DB_PERSISTENCE_ENABLED (best-effort)</text>
  </g>

  <!-- EDGES & CONNECTORS -->
  <!-- Clients -> DP (optional file) -->
  <path d="M 700 148 L 740 148" class="flow-line" marker-end="url(#arrow)"/>
  <rect x="696" y="131" width="42" height="15" class="edge-bg"/>
  <text x="717" y="142" text-anchor="middle" class="edge-label" font-size="9.5">file</text>

  <!-- RT -> ROUTE -->
  <path d="M 337 198 L 337 245 L 410 270" class="flow-line" marker-end="url(#arrow)"/>

  <!-- API -> ROUTE -->
  <path d="M 567 198 L 567 245 L 510 270" class="flow-line" marker-end="url(#arrow)"/>

  <!-- CLI -> ORCH (explicit agent) -->
  <path d="M 135 198 L 135 425 L 300 425" class="flow-line" marker-end="url(#arrow)"/>
  <rect x="142" y="340" width="80" height="18" class="edge-bg"/>
  <text x="182" y="353" text-anchor="middle" class="edge-label">explicit agent</text>

  <!-- DP -> ORCH (document context) -->
  <path d="M 912 198 L 912 280 L 640 280 L 640 405 L 620 405" class="flow-line" marker-end="url(#arrow)"/>
  <rect x="675" y="271" width="95" height="18" class="edge-bg"/>
  <text x="722" y="284" text-anchor="middle" class="edge-label">doc context</text>

  <!-- ROUTE -> ORCH (selected agent id) -->
  <path d="M 460 340 L 460 390" class="flow-line" marker-end="url(#arrow-amber)"/>
  <rect x="408" y="352" width="104" height="18" class="edge-bg"/>
  <text x="460" y="365" text-anchor="middle" class="edge-label" fill="#fde68a">selected agent</text>

  <!-- ORCH -> AGENT -->
  <path d="M 460 465 L 460 500" class="flow-line" marker-end="url(#arrow-purple)"/>

  <!-- Services <-> Pipeline -->
  <!-- WEB -.-> AGENT (web context) -->
  <path d="M 705 409 L 660 409 L 660 525 L 620 525" class="flow-dashed" marker-end="url(#arrow-dashed)"/>
  <rect x="625" y="475" width="70" height="17" class="edge-bg"/>
  <text x="660" y="487" text-anchor="middle" class="edge-label">web context</text>

  <!-- AGENT -.-> COST -->
  <path d="M 620 540 L 650 540 L 650 469 L 705 469" class="flow-dashed" marker-end="url(#arrow-dashed)"/>

  <!-- AGENT -.-> TRACE -->
  <path d="M 620 555 L 705 555" class="flow-dashed" marker-end="url(#arrow-dashed)"/>

  <!-- ROUTE -.-> TRACE -->
  <path d="M 545 315 L 670 315 L 670 515 L 705 515" class="flow-dashed" marker-end="url(#arrow-dashed)"/>

  <!-- AGENT -> REFVAL -->
  <path d="M 460 575 L 460 615" class="flow-line" marker-end="url(#arrow-blue)"/>

  <!-- REFVAL <-> RVC -->
  <path d="M 620 647 L 705 647" class="flow-line" marker-start="url(#arrow-both-start)" marker-end="url(#arrow-both-end)"/>

  <!-- REFVAL -> DISC -->
  <path d="M 460 680 L 460 715" class="flow-line" marker-end="url(#arrow)"/>

  <!-- DISC -> FILES -->
  <path d="M 460 765 L 460 785 L 335 785 L 335 805" class="flow-line" marker-end="url(#arrow-emerald)"/>

  <!-- ORCH -.-> DB (_persist_report_to_db) -->
  <path d="M 300 445 L 270 445 L 270 760 L 650 760 L 650 805" class="flow-dashed" marker-end="url(#arrow-dashed)"/>
  <rect x="420" y="751" width="180" height="18" class="edge-bg"/>
  <text x="510" y="764" text-anchor="middle" class="edge-label">_persist_report_to_db (best effort)</text>

  <!-- FILES -> DB -->
  <path d="M 540 847 L 630 847" class="flow-line" marker-end="url(#arrow)"/>

  <!-- FILES -.-> API (/outputs static mount) -->
  <path d="M 130 847 L 70 847 L 70 230 L 485 230 L 485 198" class="flow-dashed" marker-end="url(#arrow-dashed)"/>
  <rect x="18" y="530" width="105" height="18" class="edge-bg"/>
  <text x="70" y="543" text-anchor="middle" class="edge-label">/outputs mount</text>

</svg>"""
    return svg

def build_agent_pipeline_svg():
    width = 980
    height = 960
    svg = create_svg_header(
        width, height,
        "Agent Pipeline — Shared LLM Execution, Phases &amp; Layered Reporting",
        "The generic flow inside the agent black box, shared across procedure, medication, diagnostic, and fact-checker."
    )

    svg += """
  <!-- MAIN SEQUENTIAL NODES -->

  <!-- Node IN: Inputs -->
  <g id="node-in" filter="url(#shadow)">
    <rect x="250" y="80" width="380" height="65" rx="8" fill="#132338" stroke="#38bdf8" stroke-width="1.5"/>
    <text x="440" y="106" text-anchor="middle" class="node-title" fill="#e0f2fe">Subject + Context</text>
    <text x="440" y="128" text-anchor="middle" class="node-desc" fill="#7dd3fc">+ Web Research Context + Document Context</text>
  </g>

  <!-- Node BUILD: Build prompt -->
  <g id="node-build" filter="url(#shadow)">
    <rect x="270" y="175" width="340" height="55" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="1.5"/>
    <text x="440" y="200" text-anchor="middle" class="node-title" fill="#f1f5f9">Build Prompt</text>
    <text x="440" y="218" text-anchor="middle" class="node-desc" fill="#94a3b8">system + user templates</text>
  </g>

  <!-- Node OVR: _apply_provider_overrides -->
  <g id="node-ovr" filter="url(#shadow)">
    <rect x="270" y="260" width="340" height="55" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="1.5"/>
    <text x="440" y="284" text-anchor="middle" class="node-code" fill="#f8fafc">_apply_provider_overrides()</text>
    <text x="440" y="302" text-anchor="middle" class="node-desc" fill="#94a3b8">(Grok-specific injection &amp; token optimization)</text>
  </g>

  <!-- Node CALL: _call_llm -->
  <g id="node-call" filter="url(#shadow)">
    <rect x="230" y="345" width="420" height="70" rx="8" fill="#2d1b4e" stroke="#a78bfa" stroke-width="1.75"/>
    <text x="440" y="374" text-anchor="middle" class="node-title" fill="#f3e8ff">_call_llm() → Provider Adapter</text>
    <text x="440" y="396" text-anchor="middle" class="node-desc" fill="#d8b4fe">Claude · OpenAI · Grok · Gemini/Vertex · Ollama</text>
  </g>

  <!-- Node COST: CostTracker (Side node) -->
  <g id="node-cost" filter="url(#shadow)">
    <rect x="720" y="350" width="220" height="60" rx="8" fill="#0f2b26" stroke="#2dd4bf" stroke-width="1.5"/>
    <text x="830" y="376" text-anchor="middle" class="node-title" fill="#99f6e4">💰 CostTracker</text>
    <text x="830" y="395" text-anchor="middle" class="node-desc" fill="#5eead4">per-phase tokens &amp; cost</text>
  </g>

  <!-- Node PARSE: _parse_json -->
  <g id="node-parse" filter="url(#shadow)">
    <rect x="310" y="445" width="260" height="50" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="1.5"/>
    <text x="440" y="475" text-anchor="middle" class="node-code" fill="#cbd5e1">_parse_json()</text>
  </g>

  <!-- Node VALID: Pydantic model_validate -->
  <g id="node-valid" filter="url(#shadow)">
    <rect x="260" y="525" width="360" height="55" rx="8" fill="#231d3d" stroke="#818cf8" stroke-width="1.5"/>
    <text x="440" y="549" text-anchor="middle" class="node-title" fill="#e0e7ff">Pydantic model_validate()</text>
    <text x="440" y="567" text-anchor="middle" class="node-desc" fill="#a5b4fc">(fallback to empty model on failure)</text>
  </g>

  <!-- Node PHASES: Agent-specific phases -->
  <g id="node-phases" filter="url(#shadow)">
    <polygon points="440,610 650,650 440,690 230,650" fill="#312211" stroke="#fbbf24" stroke-width="1.5"/>
    <text x="440" y="646" text-anchor="middle" class="node-title" fill="#fde68a">Agent-Specific Phases</text>
    <text x="440" y="663" text-anchor="middle" class="node-desc" fill="#fef3c7">procedure · medication · diagnostic · fact-check</text>
  </g>

  <!-- Node REFS: PhaseResult.references -->
  <g id="node-refs" filter="url(#shadow)">
    <rect x="710" y="625" width="230" height="50" rx="8" fill="#14243b" stroke="#38bdf8" stroke-width="1.5"/>
    <text x="825" y="648" text-anchor="middle" class="node-title" fill="#bae6fd">PhaseResult.references</text>
    <text x="825" y="664" text-anchor="middle" class="node-desc" fill="#7dd3fc">Structured citation list per phase</text>
  </g>

  <!-- Node LAYER: Layered report helpers -->
  <g id="node-layer" filter="url(#shadow)">
    <rect x="250" y="725" width="380" height="60" rx="8" fill="#231d3d" stroke="#818cf8" stroke-width="1.75"/>
    <text x="440" y="751" text-anchor="middle" class="node-title" fill="#c7d2fe">Layered Report Helpers</text>
    <text x="440" y="771" text-anchor="middle" class="node-code" fill="#a5b4fc">_build_layered_report()</text>
  </g>

  <!-- SUBGRAPH: THREE REPORT LAYERS -->
  <g id="subgraph-layers">
    <rect x="40" y="820" width="900" height="110" rx="10" fill="#111827" stroke="#1f2937" stroke-width="1.2"/>

    <!-- L1 -->
    <g id="node-l1" filter="url(#shadow)">
      <rect x="60" y="835" width="260" height="80" rx="8" fill="#064e3b" stroke="#34d399" stroke-width="1.5"/>
      <text x="190" y="863" text-anchor="middle" class="node-title" fill="#a7f3d0">Layer 1 — Conclusions</text>
      <text x="190" y="885" text-anchor="middle" class="node-desc" fill="#6ee7b7">What readers value first</text>
      <text x="190" y="901" text-anchor="middle" class="node-desc" fill="#d1fae5">Plain language, inline [n] markers</text>
    </g>

    <!-- L2 -->
    <g id="node-l2" filter="url(#shadow)">
      <rect x="360" y="835" width="260" height="80" rx="8" fill="#0f2b38" stroke="#38bdf8" stroke-width="1.5"/>
      <text x="490" y="863" text-anchor="middle" class="node-title" fill="#bae6fd">Layer 2 — Reasoning</text>
      <text x="490" y="885" text-anchor="middle" class="node-desc" fill="#7dd3fc">The logic behind conclusions</text>
      <text x="490" y="901" text-anchor="middle" class="node-desc" fill="#e0f2fe">Organ &amp; mechanism analysis</text>
    </g>

    <!-- L3 -->
    <g id="node-l3" filter="url(#shadow)">
      <rect x="660" y="835" width="260" height="80" rx="8" fill="#2d1d42" stroke="#c084fc" stroke-width="1.5"/>
      <text x="790" y="863" text-anchor="middle" class="node-title" fill="#e9d5ff">Layer 3 — Statistical Appendix</text>
      <text x="790" y="885" text-anchor="middle" class="node-desc" fill="#d8b4fe">Deterministic in Python</text>
      <text x="790" y="901" text-anchor="middle" class="node-desc" fill="#f3e8ff">Lossless effect sizes &amp; CIs</text>
    </g>
  </g>

  <!-- CONNECTORS -->
  <!-- IN -> BUILD -->
  <path d="M 440 145 L 440 175" class="flow-line" marker-end="url(#arrow)"/>

  <!-- BUILD -> OVR -->
  <path d="M 440 230 L 440 260" class="flow-line" marker-end="url(#arrow)"/>

  <!-- OVR -> CALL -->
  <path d="M 440 315 L 440 345" class="flow-line" marker-end="url(#arrow-purple)"/>

  <!-- CALL -.-> COST -->
  <path d="M 650 380 L 720 380" class="flow-dashed" marker-end="url(#arrow-dashed)"/>
  <rect x="653" y="369" width="64" height="16" class="edge-bg"/>
  <text x="685" y="380" text-anchor="middle" class="edge-label">@track_cost</text>

  <!-- CALL -> PARSE -->
  <path d="M 440 415 L 440 445" class="flow-line" marker-end="url(#arrow)"/>

  <!-- PARSE -> VALID -->
  <path d="M 440 495 L 440 525" class="flow-line" marker-end="url(#arrow)"/>

  <!-- VALID -> PHASES -->
  <path d="M 440 580 L 440 610" class="flow-line" marker-end="url(#arrow-amber)"/>

  <!-- PHASES -> REFS (references per phase) -->
  <path d="M 650 650 L 710 650" class="flow-line" marker-end="url(#arrow-blue)"/>
  <rect x="652" y="632" width="56" height="15" class="edge-bg"/>
  <text x="680" y="643" text-anchor="middle" class="edge-label" font-size="9">refs per phase</text>

  <!-- PHASES -> LAYER -->
  <path d="M 440 690 L 440 725" class="flow-line" marker-end="url(#arrow)"/>

  <!-- LAYER -> L1, L2, L3 -->
  <path d="M 330 785 L 330 805 L 190 805 L 190 835" class="flow-line" marker-end="url(#arrow-emerald)"/>
  <path d="M 440 785 L 440 835" class="flow-line" marker-end="url(#arrow-blue)"/>
  <path d="M 550 785 L 550 805 L 790 805 L 790 835" class="flow-line" marker-end="url(#arrow-purple)"/>

</svg>"""
    return svg

def build_fact_checker_pipeline_svg():
    width = 1140
    height = 1320
    svg = create_svg_header(
        width, height,
        "Fact-Checker Pipeline — 5 Phases, Parallel Perspectives &amp; Layered Assembly",
        "Detailed multi-phase investigation: conflict scan, evidence audit, synthesis, 3 parallel LLMs, and lossless layered assembly."
    )

    svg += """
  <!-- TOP PHASES -->

  <!-- Node S: start_analysis -->
  <g id="node-s" filter="url(#shadow)">
    <rect x="420" y="75" width="300" height="50" rx="25" fill="#132338" stroke="#38bdf8" stroke-width="1.5"/>
    <text x="570" y="105" text-anchor="middle" class="node-title" fill="#bae6fd">start_analysis(subject)</text>
  </g>

  <!-- Node WC: Web Context -->
  <g id="node-wc" filter="url(#shadow)">
    <rect x="430" y="150" width="280" height="45" rx="8" fill="#0f2b26" stroke="#2dd4bf" stroke-width="1.5"/>
    <text x="570" y="177" text-anchor="middle" class="node-title" fill="#99f6e4">🌐 Build Web Context</text>
  </g>

  <!-- Phase 1: Conflict Scan -->
  <g id="node-p1" filter="url(#shadow)">
    <rect x="360" y="225" width="420" height="60" rx="8" fill="#1e293b" stroke="#60a5fa" stroke-width="1.5"/>
    <text x="570" y="250" text-anchor="middle" class="node-title" fill="#93c5fd">Phase 1 — Conflict Scan</text>
    <text x="570" y="269" text-anchor="middle" class="node-desc" fill="#bfdbfe">official vs counter-narrative + references</text>
  </g>

  <!-- User Gate U1 -->
  <g id="node-u1" filter="url(#shadow)">
    <polygon points="570,310 710,340 570,370 430,340" fill="#2d2212" stroke="#fbbf24" stroke-width="1.5"/>
    <text x="570" y="337" text-anchor="middle" class="node-title" fill="#fde68a">User Gate 1</text>
    <text x="570" y="352" text-anchor="middle" class="node-desc" fill="#fef3c7">Official / Independent / Both</text>
  </g>

  <!-- Phase 2: Evidence Audit -->
  <g id="node-p2" filter="url(#shadow)">
    <rect x="360" y="395" width="420" height="60" rx="8" fill="#1e293b" stroke="#60a5fa" stroke-width="1.5"/>
    <text x="570" y="420" text-anchor="middle" class="node-title" fill="#93c5fd">Phase 2 — Evidence Audit</text>
    <text x="570" y="439" text-anchor="middle" class="node-desc" fill="#bfdbfe">funding bias · methodology · recency</text>
  </g>

  <!-- User Gate U2 -->
  <g id="node-u2" filter="url(#shadow)">
    <polygon points="570,480 670,505 570,530 470,505" fill="#2d2212" stroke="#fbbf24" stroke-width="1.5"/>
    <text x="570" y="503" text-anchor="middle" class="node-title" fill="#fde68a">User Gate 2</text>
    <text x="570" y="517" text-anchor="middle" class="node-desc" fill="#fef3c7">Dig / Proceed</text>
  </g>

  <!-- Phase 3: Synthesis -->
  <g id="node-p3" filter="url(#shadow)">
    <rect x="360" y="555" width="420" height="60" rx="8" fill="#1e293b" stroke="#60a5fa" stroke-width="1.5"/>
    <text x="570" y="580" text-anchor="middle" class="node-title" fill="#93c5fd">Phase 3 — Synthesis</text>
    <text x="570" y="599" text-anchor="middle" class="node-desc" fill="#bfdbfe">biological truth · industry bias · grey zone</text>
  </g>

  <!-- User Gate U3 -->
  <g id="node-u3" filter="url(#shadow)">
    <polygon points="570,640 730,670 570,700 410,670" fill="#2d2212" stroke="#fbbf24" stroke-width="1.5"/>
    <text x="570" y="666" text-anchor="middle" class="node-title" fill="#fde68a">User Lens Selection</text>
    <text x="570" y="682" text-anchor="middle" class="node-desc" fill="#fef3c7">M (Mainstream) / N (Naturist) / B (Biohacker) / A (Balanced)</text>
  </g>

  <!-- SUBGRAPH: PHASE 4 (Parallel Perspectives) -->
  <g id="subgraph-p4">
    <rect x="50" y="730" width="1040" height="230" rx="12" fill="#151b2e" stroke="#2b3b64" stroke-width="1.5"/>
    <rect x="65" y="720" width="370" height="20" rx="4" fill="#2b3b64"/>
    <text x="75" y="734" class="subgraph-title" fill="#60a5fa">PHASE 4 — THREE PARALLEL PERSPECTIVE AGENTS</text>

    <!-- Node M: Mainstream -->
    <g id="node-m" filter="url(#shadow)">
      <rect x="80" y="755" width="290" height="90" rx="8" fill="#11293b" stroke="#38bdf8" stroke-width="1.5"/>
      <text x="225" y="785" text-anchor="middle" class="node-title" fill="#bae6fd">🏥 Mainstream LLM</text>
      <text x="225" y="809" text-anchor="middle" class="node-desc" fill="#e0f2fe">Established medical guidelines</text>
      <text x="225" y="827" text-anchor="middle" class="node-desc" fill="#7dd3fc">Clinical consensus &amp; standard of care</text>
    </g>

    <!-- Node N: Naturist -->
    <g id="node-n" filter="url(#shadow)">
      <rect x="425" y="755" width="290" height="90" rx="8" fill="#0d2b24" stroke="#34d399" stroke-width="1.5"/>
      <text x="570" y="785" text-anchor="middle" class="node-title" fill="#a7f3d0">🌿 Naturist LLM</text>
      <text x="570" y="809" text-anchor="middle" class="node-desc" fill="#d1fae5">Holistic &amp; evolutionary root causes</text>
      <text x="570" y="827" text-anchor="middle" class="node-desc" fill="#6ee7b7">Food, lifestyle &amp; low-intervention</text>
    </g>

    <!-- Node B: Biohacker -->
    <g id="node-b" filter="url(#shadow)">
      <rect x="770" y="755" width="290" height="90" rx="8" fill="#2d1c47" stroke="#c084fc" stroke-width="1.5"/>
      <text x="915" y="785" text-anchor="middle" class="node-title" fill="#e9d5ff">🚀 Biohacker LLM</text>
      <text x="915" y="809" text-anchor="middle" class="node-desc" fill="#f3e8ff">Emerging human optimization</text>
      <text x="915" y="827" text-anchor="middle" class="node-desc" fill="#d8b4fe">Biomarkers, off-label &amp; n=1 trials</text>
    </g>

    <!-- Node ASM: Assembler LLM -->
    <g id="node-asm" filter="url(#shadow)">
      <rect x="370" y="880" width="400" height="60" rx="8" fill="#231d3d" stroke="#818cf8" stroke-width="1.75"/>
      <text x="570" y="906" text-anchor="middle" class="node-title" fill="#c7d2fe">Assembler LLM (Merged Markdown)</text>
      <text x="570" y="926" text-anchor="middle" class="node-desc" fill="#a5b4fc">Integrated synthesis with 'Your Focus' section at top</text>
    </g>
  </g>

  <!-- Node SPLIT: Split body / references -->
  <g id="node-split" filter="url(#shadow)">
    <rect x="420" y="985" width="300" height="45" rx="8" fill="#1e293b" stroke="#64748b" stroke-width="1.5"/>
    <text x="570" y="1012" text-anchor="middle" class="node-title" fill="#e2e8f0">✂️ Split Body / References</text>
  </g>

  <!-- SUBGRAPH: PHASE 5 (Layered Assembly) -->
  <g id="subgraph-p5">
    <rect x="180" y="1055" width="780" height="135" rx="12" fill="#1c1630" stroke="#3b2b64" stroke-width="1.5"/>
    <rect x="195" y="1045" width="360" height="20" rx="4" fill="#3b2b64"/>
    <text x="205" y="1059" class="subgraph-title" fill="#c084fc">PHASE 5 — PROGRESSIVE DISCLOSURE ASSEMBLY</text>

    <!-- L12 -->
    <g id="node-l12" filter="url(#shadow)">
      <rect x="200" y="1075" width="340" height="50" rx="6" fill="#241e40" stroke="#a78bfa" stroke-width="1.2"/>
      <text x="370" y="1096" text-anchor="middle" class="node-title" fill="#ddd6fe">LLM: Layer 1 &amp; Layer 2</text>
      <text x="370" y="1113" text-anchor="middle" class="node-desc" fill="#c4b5fd">Conclusions &amp; Reasoning (plain language)</text>
    </g>

    <!-- L3f -->
    <g id="node-l3f" filter="url(#shadow)">
      <rect x="590" y="1075" width="350" height="50" rx="6" fill="#241e40" stroke="#a78bfa" stroke-width="1.2"/>
      <text x="765" y="1096" text-anchor="middle" class="node-title" fill="#ddd6fe">Deterministic Layer 3 Appendix</text>
      <text x="765" y="1113" text-anchor="middle" class="node-desc" fill="#c4b5fd">Python assembled, lossless quantitative evidence</text>
    </g>

    <!-- GUARD -->
    <g id="node-guard" filter="url(#shadow)">
      <rect x="360" y="1135" width="420" height="42" rx="6" fill="#3b111e" stroke="#f43f5e" stroke-width="1.2"/>
      <text x="570" y="1155" text-anchor="middle" class="node-title" fill="#fecdd3">🛡️ Verification Guard: _verify_no_silent_loss()</text>
      <text x="570" y="1169" text-anchor="middle" class="node-desc" fill="#fda4af">Audits &amp; verifies no key insights fail to survive</text>
    </g>
  </g>

  <!-- FINAL OUTPUT REPORTS -->
  <!-- PAT: Patient Report -->
  <g id="node-pat" filter="url(#shadow)">
    <rect x="180" y="1215" width="340" height="75" rx="8" fill="#0c2d25" stroke="#34d399" stroke-width="1.75"/>
    <text x="350" y="1243" text-anchor="middle" class="node-title" fill="#a7f3d0">Patient Report (Layers 1-2)</text>
    <text x="350" y="1263" text-anchor="middle" class="node-desc" fill="#6ee7b7">Layer 1: Conclusions · Layer 2: Reasoning</text>
    <text x="350" y="1279" text-anchor="middle" class="node-desc" fill="#d1fae5">+ Validated references re-attached</text>
  </g>

  <!-- PRAC: Practitioner Report -->
  <g id="node-prac" filter="url(#shadow)">
    <rect x="620" y="1215" width="340" height="75" rx="8" fill="#132338" stroke="#38bdf8" stroke-width="1.75"/>
    <text x="790" y="1243" text-anchor="middle" class="node-title" fill="#bae6fd">Practitioner Report (Layers 1-3)</text>
    <text x="790" y="1263" text-anchor="middle" class="node-desc" fill="#7dd3fc">Layers 1-2 + Layer 3 Statistical Appendix</text>
    <text x="790" y="1279" text-anchor="middle" class="node-desc" fill="#e0f2fe">+ Full verified citations &amp; methodology audit</text>
  </g>

  <!-- CONNECTORS -->
  <!-- S -> WC -->
  <path d="M 570 125 L 570 150" class="flow-line" marker-end="url(#arrow)"/>

  <!-- WC -> P1 -->
  <path d="M 570 195 L 570 225" class="flow-line" marker-end="url(#arrow-blue)"/>

  <!-- P1 -> U1 -->
  <path d="M 570 285 L 570 310" class="flow-line" marker-end="url(#arrow-amber)"/>

  <!-- U1 -> P2 -->
  <path d="M 570 370 L 570 395" class="flow-line" marker-end="url(#arrow)"/>

  <!-- P2 -> U2 -->
  <path d="M 570 455 L 570 480" class="flow-line" marker-end="url(#arrow-amber)"/>

  <!-- U2 -> P3 -->
  <path d="M 570 530 L 570 555" class="flow-line" marker-end="url(#arrow)"/>

  <!-- P3 -> U3 -->
  <path d="M 570 615 L 570 640" class="flow-line" marker-end="url(#arrow-amber)"/>

  <!-- U3 -> Phase 4 Parallel Agents -->
  <path d="M 570 700 L 570 715 L 225 715 L 225 755" class="flow-line" marker-end="url(#arrow-blue)"/>
  <path d="M 570 700 L 570 755" class="flow-line" marker-end="url(#arrow-emerald)"/>
  <path d="M 570 700 L 570 715 L 915 715 L 915 755" class="flow-line" marker-end="url(#arrow-purple)"/>

  <!-- Parallel Agents -> Assembler -->
  <path d="M 225 845 L 225 865 L 480 865 L 480 880" class="flow-line" marker-end="url(#arrow)"/>
  <path d="M 570 845 L 570 880" class="flow-line" marker-end="url(#arrow)"/>
  <path d="M 915 845 L 915 865 L 660 865 L 660 880" class="flow-line" marker-end="url(#arrow)"/>

  <!-- ASM -> SPLIT -->
  <path d="M 570 940 L 570 985" class="flow-line" marker-end="url(#arrow)"/>

  <!-- SPLIT -> Phase 5 -->
  <path d="M 570 1030 L 570 1055" class="flow-line" marker-end="url(#arrow-purple)"/>

  <!-- Phase 5 -> Outputs -->
  <path d="M 370 1177 L 370 1195 L 350 1195 L 350 1215" class="flow-line" marker-end="url(#arrow-emerald)"/>
  <path d="M 770 1177 L 770 1195 L 790 1195 L 790 1215" class="flow-line" marker-end="url(#arrow-blue)"/>

  <!-- SPLIT -.-> references re-attached verbatim to PAT & PRAC -->
  <path d="M 420 1007 L 130 1007 L 130 1252 L 180 1252" class="flow-dashed" marker-end="url(#arrow-dashed)"/>
  <rect x="80" y="1120" width="100" height="26" class="edge-bg"/>
  <text x="130" y="1132" text-anchor="middle" class="edge-label" font-size="9.5">references</text>
  <text x="130" y="1143" text-anchor="middle" class="edge-label" font-size="9.5">re-attached</text>

  <path d="M 720 1007 L 1010 1007 L 1010 1252 L 960 1252" class="flow-dashed" marker-end="url(#arrow-dashed)"/>
  <rect x="960" y="1120" width="100" height="26" class="edge-bg"/>
  <text x="1010" y="1132" text-anchor="middle" class="edge-label" font-size="9.5">references</text>
  <text x="1010" y="1143" text-anchor="middle" class="edge-label" font-size="9.5">re-attached</text>

</svg>"""
    return svg

def main():
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)

    diagrams = [
        ("system_architecture.svg", build_system_architecture_svg),
        ("agent_pipeline.svg", build_agent_pipeline_svg),
        ("fact_checker_pipeline.svg", build_fact_checker_pipeline_svg),
    ]

    for filename, builder in diagrams:
        svg_content = builder()
        # Verify valid XML
        ET.fromstring(svg_content)
        filepath = os.path.join(DIAGRAMS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"Generated and validated: {filepath}")

if __name__ == "__main__":
    main()
