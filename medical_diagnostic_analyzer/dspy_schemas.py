from typing import List, Optional
from pydantic import BaseModel, Field

class SymptomExtraction(BaseModel):
    """Structured extraction of symptoms and patient data from a medical query."""
    symptoms: List[str] = Field(description="List of positive symptoms identified (e.g. ['fever', 'headache'])")
    negative_symptoms: List[str] = Field(default_factory=list, description="List of symptoms explicitly denied by the patient")
    duration: Optional[str] = Field(None, description="How long the symptoms have been present")
    severity: Optional[str] = Field(None, description="Patient's description of symptom severity")
    is_vague: bool = Field(description="True if the input is too generic (e.g. 'I feel sick') to make a diagnostic assessment")
    clarification_question: Optional[str] = Field(None, description="If is_vague is True, a polite question to gather more context")

class DiagnosticReport(BaseModel):
    """The final structured diagnostic report."""
    top_5_candidates: List[str] = Field(description="The top 5 most likely or serious conditions")
    most_probable: str = Field(description="The condition with the highest calculated probability")
    most_serious: str = Field(description="The most severe condition that cannot be ruled out")
    reasoning_summary: str = Field(description="Brief explanation of the diagnostic logic and how data influenced the scores")
    recommended_next_steps: List[str] = Field(description="Clinical next steps (e.g. see GP, go to ER)")
    suggested_agent: str = Field(description="The next agent to route to: 'medication_agent' or 'procedure_agent'")
    routing_rationale: str = Field(description="Why this specific agent is recommended for follow-up")
