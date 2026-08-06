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
            "Relative likelihood among the listed candidates only (0–1), not a "
            "calibrated population posterior. Values across candidates should "
            "sum to approximately 1.0 (host may renormalize)"
        ),
        ge=0.0,
        le=1.0,
    )
    severity: int = Field(
        description=(
            "Clinical seriousness if this condition is present (1=benign/self-limited, "
            "5=life- or limb-threatening emergency). Independent of probability — "
            "cannot-miss items may be high severity and low probability"
        ),
        ge=1,
        le=5,
    )
    rationale: str = Field(
        description=(
            "Brief clinical reason this condition fits the stated presentation "
            "(or remains on the cannot-miss list)"
        )
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
    """The final structured diagnostic report (Level 5)."""

    top_5_candidates: List[str] = Field(
        description="Up to five leading conditions from the ranked differential"
    )
    most_probable: str = Field(
        description=(
            "Condition with the highest relative likelihood among candidates "
            "(decision-support only — not a definitive diagnosis)"
        )
    )
    most_serious: str = Field(
        description=(
            "Highest-severity condition among top candidates that cannot yet "
            "be ruled out (cannot-miss / red-flag focus)"
        )
    )
    reasoning_summary: str = Field(
        description=(
            "Brief explanation of the diagnostic logic and how the presentation "
            "influenced the ranking"
        )
    )
    recommended_next_steps: List[str] = Field(
        description=(
            "Concrete clinical next steps (urgency, who to see, key exams/tests)"
        )
    )
    suggested_agent: str = Field(
        description=(
            "Follow-up specialist hint only: 'medication_agent' if next focus is "
            "mainly pharmacologic management, or 'procedure_agent' if "
            "interventional/procedural workup or treatment is central"
        )
    )
    routing_rationale: str = Field(
        description="Why this specific agent is recommended for follow-up"
    )
    references: List[str] = Field(
        default_factory=list,
        description=(
            "3–6 APA 7 citations (each with DOI, PMID, or URL) supporting "
            "reasoning for THIS presentation"
        ),
    )
