"""
Shared utilities for LangChain-based agents.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from llm_integrations import TokenUsage, create_llm_manager

try:
    from langsmith import Client as LangSmithClient
except ImportError:  # pragma: no cover
    LangSmithClient = None

@dataclass
class LangChainAgentConfig:
    """Configuration shared across LangChain agents."""

    primary_llm_provider: str = "claude-sonnet"
    fallback_providers: list[str] = field(
        default_factory=lambda: ["claude-sonnet", "grok-4.3", "openai", "ollama"]
    )
    enable_logging: bool = True
    enable_reference_validation: bool = False
    enable_audit: bool = True
    enable_web_research: bool = False
    web_research_providers: list[str] = field(
        default_factory=lambda: ["tavily", "serpapi", "duckduckgo"]
    )
    web_research_max_results: int = 5


class LangChainAgentBase:
    """Base class for LangChain-based agents using shared LLM manager."""

    def __init__(self, config: LangChainAgentConfig):
        self.config = config
        self.provider_name = (config.primary_llm_provider or "").lower()
        self.enable_reference_validation = config.enable_reference_validation
        self.enable_audit = config.enable_audit
        self.reference_validator = None
        self.audit_events: list[dict[str, Any]] = []
        self.langsmith_client = None
        self.langsmith_project = os.getenv("LANGCHAIN_PROJECT") or "research-agent-alpha"
        self.enable_web_research = config.enable_web_research
        self.web_research = None
        self.web_context: str | None = None
        self.document_context: str | None = None

        if self.enable_reference_validation:
            try:
                from reference_validation import ReferenceValidator, ValidationConfig

                self.reference_validator = ReferenceValidator(
                    ValidationConfig(cache_backend="sqlite", min_credibility_score=70)
                )
            except ImportError:
                self.reference_validator = None

        try:
            self.llm_manager = create_llm_manager(
                primary_provider=config.primary_llm_provider,
                fallback_providers=config.fallback_providers,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize LLM manager: {exc}") from exc

        self.llm_provider = self.llm_manager.get_available_provider()
        if not self.llm_provider:
            raise RuntimeError("No LLM provider available for LangChain agents")

        self.total_token_usage = TokenUsage()
        self._initialize_langsmith()
        self._initialize_web_research()

    def _initialize_langsmith(self) -> None:
        if not self.enable_audit or LangSmithClient is None:
            return
        tracing_flag = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if tracing_flag not in ("1", "true", "yes") or not api_key:
            return
        try:
            self.langsmith_client = LangSmithClient()
        except Exception:
            self.langsmith_client = None

    def _initialize_web_research(self) -> None:
        if not self.enable_web_research:
            return
        try:
            from web_research import WebResearchClient

            self.web_research = WebResearchClient(
                providers=self.config.web_research_providers,
                max_results=self.config.web_research_max_results,
            )
        except Exception:
            self.web_research = None

    def _build_web_context(self, query: str) -> str:
        if not self.web_research:
            return ""
        results = self.web_research.search(query)
        if not results:
            return ""
        import logging as _logging
        provider_used = results[0].provider
        _logging.getLogger(__name__).info(
            "Web research: %d results from '%s' for query: %s",
            len(results),
            provider_used,
            query[:60],
        )
        lines = []
        for idx, item in enumerate(results, 1):
            lines.append(
                f"[{idx}] {item.title} ({item.source}) - {item.snippet} {item.url}".strip()
            )
        return "\n".join(lines)

    @staticmethod
    def _sanitize_prompt_input(val: Any) -> Any:
        """Sanitize input strings to prevent prompt injection and chat template delimiter collisions."""
        if not isinstance(val, str):
            return val
        # Strip system role override tokens and raw template control sequences
        sanitized = re.sub(r"<\|(?:im_start|im_end|endoftext|system|assistant|user)\|>", "", val, flags=re.IGNORECASE)
        sanitized = re.sub(r"\[\s*(?:SYSTEM|ASSISTANT|HUMAN|INSTRUCTION)\s*\]", "", sanitized, flags=re.IGNORECASE)
        return sanitized.strip()

    def _render_prompt(
        self, system_prompt: str, user_prompt: str, **kwargs: Any
    ) -> Tuple[str, str]:
        system_prompt, user_prompt = self._apply_provider_overrides(
            system_prompt, user_prompt
        )
        cleaned_kwargs = {
            k: self._sanitize_prompt_input(v) for k, v in kwargs.items()
        }
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt)]
        )
        messages = prompt.format_messages(**cleaned_kwargs)

        system_text = ""
        user_texts: list[str] = []
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            if msg_type == "system":
                system_text = msg.content
            else:
                user_texts.append(msg.content)

        return system_text, "\n".join(user_texts)

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        audit_step: str | None = None,
        **kwargs: Any,
    ) -> str:
        system_text, user_text = self._render_prompt(
            system_prompt, user_prompt, **kwargs
        )
        response, token_usage = self.llm_provider.generate_response(
            prompt=user_text, system_prompt=system_text
        )
        if token_usage:
            self.total_token_usage.add(token_usage)
        if self.enable_audit:
            self.audit_events.append(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "step": audit_step,
                    "system_prompt": system_text,
                    "user_prompt": user_text,
                    "response": response,
                    "token_usage": (
                        {
                            "input_tokens": token_usage.input_tokens,
                            "output_tokens": token_usage.output_tokens,
                            "total_tokens": token_usage.total_tokens,
                        }
                        if token_usage
                        else None
                    ),
                }
            )
            self._record_langsmith(audit_step, system_text, user_text, response)
        return response

    def _apply_provider_overrides(
        self, system_prompt: str, user_prompt: str
    ) -> Tuple[str, str]:
        if "grok" not in self.provider_name:
            return system_prompt, user_prompt

        grok_system = (
            "Grok output quality requirements: be exhaustive, avoid placeholders, "
            "and provide complete, clinically useful detail. "
            "If evidence is limited, label it as limited evidence but still provide guidance."
        )
        grok_user = (
            "Do NOT use 'N/A'. If unknown, use 'not established' and provide rationale. "
            "For narrative fields, write at least 2-3 sentences. "
            "Include numeric values (doses, timeframes, ranges) whenever possible."
        )
        return f"{system_prompt}\n\n{grok_system}", f"{user_prompt}\n\n{grok_user}"

    def _is_grok(self) -> bool:
        return "grok" in self.provider_name

    def _record_langsmith(
        self, audit_step: str | None, system_text: str, user_text: str, response: str
    ) -> None:
        if not self.langsmith_client:
            return
        try:
            self.langsmith_client.create_run(
                name=audit_step or "llm_call",
                run_type="llm",
                inputs={"system": system_text, "user": user_text},
                outputs={"response": response},
                project_name=self.langsmith_project,
            )
        except Exception:
            pass

    def _parse_json(self, text: str) -> Optional[Any]:
        # Attempt 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt 2: strip markdown code fence (```json ... ``` or ``` ... ```)
        stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Attempt 3: extract first JSON object or array via regex
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

    # ── Shared layered-report helpers ─────────────────────────────────────────
    # Progressive-disclosure report construction reused across all agents:
    #   Layer 1  Conclusions           (what readers value; first)
    #   Layer 2  Reasoning             (the logic behind the conclusions)
    #   Layer 3  Statistical Appendix  (precise numbers / grading; last, optional)
    # The appendix is assembled DETERMINISTICALLY so numbers are never dropped or
    # hallucinated. The patient variant omits Layer 3; the practitioner variant
    # includes it.

    def _build_statistical_appendix(
        self, sections: dict[str, list[str]]
    ) -> str:
        """
        Assemble a Statistical Appendix (Layer 3) deterministically from labelled
        groups of quantitative entries. No LLM involved.

        Args:
            sections: ordered mapping of ``section heading`` -> list of entry
                strings (effect sizes, CIs, p-values, evidence grading, etc.).

        Returns:
            Markdown for the appendix, or "" if there is nothing quantitative.
        """
        rendered: list[str] = []
        for heading, entries in sections.items():
            clean = [str(e).strip() for e in (entries or []) if str(e).strip()]
            if not clean:
                continue
            lines = "\n".join(f"- {e}" for e in clean)
            rendered.append(f"### {heading}\n{lines}")

        if not rendered:
            return ""

        header = (
            "## 📊 Statistical Appendix\n"
            "_For readers who want the precise numbers. Effect sizes, confidence "
            "intervals, p-values and evidence grading behind the conclusions above._\n"
        )
        return header + "\n\n" + "\n\n".join(rendered)

    def _build_layered_report(
        self,
        *,
        conclusions_and_reasoning: str,
        appendix: str,
    ) -> Tuple[str, str]:
        """
        Compose patient and practitioner variants of a layered report.

        Args:
            conclusions_and_reasoning: markdown for Layers 1-2 (already ordered
                Conclusions -> Reasoning), typically produced by
                ``_layer_plain_language`` or a deterministic builder.
            appendix: Layer 3 markdown from ``_build_statistical_appendix`` (may
                be "").

        Returns:
            ``(patient_output, practitioner_output)``. The patient variant omits
            the Statistical Appendix but points readers to the practitioner
            report when an appendix exists; the practitioner variant appends it.
        """
        patient = conclusions_and_reasoning
        practitioner = conclusions_and_reasoning
        if appendix:
            patient = (
                conclusions_and_reasoning
                + "\n\n---\n\n"
                + "_The precise statistics and evidence grading behind these "
                "conclusions are available in the detailed practitioner report._\n"
            )
            practitioner = conclusions_and_reasoning + "\n\n" + appendix
        return patient, practitioner

    def _verify_no_silent_loss(
        self,
        output: str,
        must_survive: list[str],
        *,
        audit_step: str = "layering_loss_check",
        fragment_words: int = 6,
    ) -> list[str]:
        """
        Best-effort check that each string in ``must_survive`` appears in
        ``output``. Missing items are logged and recorded as an audit event, but
        never raise — resilience over strictness.

        A leading word-fragment is used for the match so light paraphrasing does
        not trigger false positives.

        Returns the list of human-readable descriptions of missing items.
        """
        import logging

        def _norm(text: str) -> str:
            return " ".join(str(text).lower().split())

        haystack = _norm(output)
        missing: list[str] = []
        for item in must_survive:
            norm_item = _norm(item)
            if not norm_item:
                continue
            fragment = " ".join(norm_item.split()[:fragment_words])
            if fragment and fragment not in haystack:
                missing.append(str(item)[:80])

        if missing:
            logging.getLogger(__name__).warning(
                "Layering may have dropped %d key item(s): %s",
                len(missing),
                "; ".join(missing),
            )
            if getattr(self, "enable_audit", False):
                self.audit_events.append(
                    {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "step": audit_step,
                        "missing_items": missing,
                    }
                )
        return missing

    def _layer_plain_language(
        self,
        body: str,
        *,
        framing: str,
        audit_step: str = "layered_plain_language",
        extra_context: str = "",
    ) -> str:
        """
        One LLM call that rewrites content into plain-language Layers 1-2
        (Conclusions -> Reasoning), preserving inline ``[n]`` citation markers.
        Does NOT emit references or a statistical appendix (added separately).
        """
        system_prompt = (
            f"You are a medical writer producing a layered report for a general "
            f"audience. Use a {framing} "
            f"Write at a 6th grade reading level with short sentences and common words. "
            f"State conclusions FIRST, then explain the reasoning behind them so the "
            f"reader can understand the logic. "
            f"When you mention a statistic, describe it in plain language AND keep any "
            f"inline citation markers like [1], [2] exactly as they appear so claims "
            f"stay traceable. "
            f"Do NOT invent facts and do NOT drop any recommendation or key point "
            f"present in the content. "
            f"Do NOT include a References section — that is added separately. "
            f"Do NOT write a Statistical Appendix — that is added separately."
        )
        _extra_block = (
            "Additional context:\n{extra_context}\n\n" if extra_context else ""
        )
        user_prompt = (
            "Rewrite this medical content as a layered guide for a non-medical reader.\n\n"
            "Content:\n{body}\n\n"
            + _extra_block
            + "Structure the output with EXACTLY these two layers, in this order:\n\n"
            "# [topic from content]\n\n"
            "## ✅ Report Summary\n"
            "[The bottom line first: what the reader should take away, plus the top "
            "practical recommendations. Plain language.]\n\n"
            "## 🧠 The Reasoning\n"
            "[Explain the logic behind the conclusions. Keep inline citation markers.]\n\n"
            "Do NOT include a References section or a Statistical Appendix."
        )
        _kwargs: dict[str, Any] = dict(audit_step=audit_step, body=body)
        if extra_context:
            _kwargs["extra_context"] = extra_context
        return self._call_llm(system_prompt, user_prompt, **_kwargs)
