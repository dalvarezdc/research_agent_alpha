from typing import List, Optional

from pydantic import BaseModel, Field


class SymptomExtraction(BaseModel):
    """Structured extraction of symptoms and patient data from a medical query."""

    symptoms: List[str] = Field(
        description=(
            "Positive symptoms/findings in plain clinical language "
            "(e.g. 'knee pain', 'swelling after activity', 'fever'). "
            "Use free-form phrases — do not force a fixed vocabulary."
        )
    )
    negative_symptoms: List[str] = Field(
        default_factory=list,
        description="Symptoms or findings the patient explicitly denies",
    )
    duration: Optional[str] = Field(
        None, description="How long the symptoms have been present"
    )
    severity: Optional[str] = Field(
        None, description="Patient's description of symptom severity"
    )
    is_vague: bool = Field(
        description=(
            "True if the input is too generic (e.g. 'I feel sick') to form a "
            "useful differential without more detail"
        )
    )
    clarification_question: Optional[str] = Field(
        None,
        description="If is_vague is True, a polite question to gather more context",
    )
    clinical_context: Optional[str] = Field(
        None,
        description=(
            "Any other clinically relevant context mentioned (age, sex, trauma, "
            "comorbidities, medications, recent illness, occupation, etc.)"
        ),
    )


class ConditionCandidate(BaseModel):
    """One item on a differential diagnosis, with estimated likelihood and severity."""

    name: str = Field(description="Condition name in clinical language")
    probability: float = Field(
        description=(
            "Estimated relative likelihood among the listed candidates (0–1). "
            "Values across candidates should sum to approximately 1.0"
        ),
        ge=0.0,
        le=1.0,
    )
    severity: int = Field(
        description=(
            "Clinical seriousness if this condition is present (1=benign/self-limited, "
            "5=life- or limb-threatening emergency)"
        ),
        ge=1,
        le=5,
    )
    rationale: str = Field(
        description="Brief clinical reason this condition fits (or remains on the list)"
    )


class DifferentialAssessment(BaseModel):
    """LLM-generated differential grounded in the patient's actual presentation."""

    candidates: List[ConditionCandidate] = Field(
        description=(
            "Ranked differential (most likely first). Include 3–8 conditions that "
            "plausibly explain the presented symptoms. Always include important "
            "'cannot-miss' diagnoses even if less likely."
        )
    )
    differentiating_symptoms: List[str] = Field(
        default_factory=list,
        description=(
            "Additional history points or symptoms that would best narrow the "
            "differential (phrased as short clinical items, not full sentences)"
        ),
    )
    recommended_exams: List[str] = Field(
        default_factory=list,
        description=(
            "Focused exams or tests that would help confirm/refute top candidates "
            "(e.g. 'Lachman test', 'knee X-ray Ottawa rules', 'CRP/ESR')"
        ),
    )
    clinical_summary: str = Field(
        description=(
            "Short paragraph of clinical reasoning: what the presentation suggests, "
            "key uncertainties, and why top candidates are ordered as they are"
        )
    )


class DiagnosticReport(BaseModel):
    """The final structured diagnostic report."""

    top_5_candidates: List[str] = Field(
        description="The top 5 most likely or serious conditions"
    )
    most_probable: str = Field(
        description="The condition with the highest estimated probability"
    )
    most_serious: str = Field(
        description="The most severe condition that cannot yet be ruled out"
    )
    reasoning_summary: str = Field(
        description=(
            "Brief explanation of the diagnostic logic and how the presentation "
            "influenced the ranking"
        )
    )
    recommended_next_steps: List[str] = Field(
        description="Clinical next steps (e.g. see GP, go to ER, specific tests)"
    )
    suggested_agent: str = Field(
        description=(
            "The next agent to route to: 'medication_agent' or 'procedure_agent'"
        )
    )
    routing_rationale: str = Field(
        description="Why this specific agent is recommended for follow-up"
    )
    references: List[str] = Field(
        default_factory=list,
        description="APA 7 citations (with DOI/PMID/URL) supporting the reasoning",
    )
