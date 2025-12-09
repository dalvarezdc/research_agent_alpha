#!/usr/bin/env python3
"""
DSPy Structured Output Schemas
Pydantic models for enforcing structured LLM outputs using DSPy.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


# =============== Reference/Citation Schema ===============

class Reference(BaseModel):
    """Single academic or medical reference with full citation details"""
    authors: str = Field(description="Author names (e.g., 'Smith J, Jones A, et al.')")
    year: int = Field(description="Publication year")
    title: str = Field(description="Full title of the paper/guideline")
    source: str = Field(description="Journal name, organization, or source")
    doi: Optional[str] = Field(default=None, description="DOI if available")
    pmid: Optional[str] = Field(default=None, description="PubMed ID if available")
    url: Optional[str] = Field(default=None, description="Direct URL to the resource")
    reference_type: str = Field(
        default="peer_reviewed",
        description="Type: peer_reviewed, guideline, meta_analysis, review, case_study, clinical_trial"
    )
    relevance: str = Field(description="Why this reference is cited (1-2 sentences)")

    def to_apa_format(self) -> str:
        """Convert to APA 7 citation format"""
        citation = f"{self.authors} ({self.year}). {self.title}. {self.source}."
        if self.doi:
            citation += f" https://doi.org/{self.doi}"
        elif self.url:
            citation += f" {self.url}"
        if self.pmid:
            citation += f" PMID: {self.pmid}"
        return citation


class ReferencesCollection(BaseModel):
    """Collection of references from a specific analysis phase or section"""
    phase_name: str = Field(description="Name of the phase/section these references support")
    references: List[Reference] = Field(default_factory=list, description="List of references")

    def get_unique_references(self) -> List[Reference]:
        """Get unique references (deduplicate by DOI/PMID/title)"""
        seen = set()
        unique = []
        for ref in self.references:
            key = ref.doi or ref.pmid or ref.title.lower()
            if key not in seen:
                seen.add(key)
                unique.append(ref)
        return unique


# =============== Pharmacology Schema ===============

class PharmacologyData(BaseModel):
    """Structured pharmacology information"""
    drug_class: str = Field(description="Pharmacologic and therapeutic class")
    mechanism_of_action: str = Field(description="Detailed mechanism at molecular level")
    absorption: str = Field(description="Bioavailability, onset, peak concentration")
    distribution: str = Field(default="", description="Volume of distribution, protein binding")
    metabolism: str = Field(description="CYP enzymes, active metabolites")
    elimination: str = Field(description="Primary route, elimination half-life")
    half_life: str = Field(description="Elimination half-life value")
    approved_indications: List[str] = Field(description="FDA-approved indications")
    off_label_uses: List[str] = Field(default_factory=list)
    standard_dosing: str = Field(description="Standard adult dosing")
    dose_adjustments: Dict[str, str] = Field(default_factory=dict)


# =============== Interaction Schemas ===============

class InteractionDetail(BaseModel):
    """Single interaction detail"""
    interacting_agent: str
    severity: str = Field(description="severe, moderate, or minor")
    mechanism: str
    clinical_effect: str
    management: str
    time_separation: Optional[str] = None
    evidence_level: str = "moderate"


class DrugInteractionsData(BaseModel):
    """Structured drug interactions"""
    severe_interactions: List[InteractionDetail] = Field(default_factory=list)
    moderate_interactions: List[InteractionDetail] = Field(default_factory=list)
    minor_interactions: List[InteractionDetail] = Field(default_factory=list)


class FoodInteractionDetail(BaseModel):
    """Single food interaction"""
    food_or_beverage: str
    interaction_type: str
    mechanism: str
    clinical_impact: str
    management: str
    timing_guidance: Optional[str] = None


class FoodInteractionsData(BaseModel):
    """Structured food interactions"""
    foods_to_avoid: List[FoodInteractionDetail] = Field(default_factory=list)
    foods_that_help: List[FoodInteractionDetail] = Field(default_factory=list)
    alcohol_interaction: Optional[FoodInteractionDetail] = None
    timing_with_meals: str = ""


# =============== Safety Profile Schema ===============

class ContraindictionDetail(BaseModel):
    """Single contraindication"""
    condition: str
    severity: str
    reason: str
    alternative: str
    risk_if_ignored: str


class WarningSign(BaseModel):
    """Warning sign to monitor"""
    sign: str
    severity: str
    mechanism: str
    action: str
    timeframe: str


class SafetyProfileData(BaseModel):
    """Structured safety profile"""
    common_adverse_effects: List[str] = Field(default_factory=list)
    serious_adverse_effects: List[str] = Field(default_factory=list)
    black_box_warnings: List[str] = Field(default_factory=list)
    contraindications: List[ContraindictionDetail] = Field(default_factory=list)
    warning_signs: List[WarningSign] = Field(default_factory=list)


# =============== Recommendations Schema ===============

class EvidenceBasedRecommendation(BaseModel):
    """Evidence-based recommendation"""
    intervention: str
    rationale: str
    evidence_level: str
    implementation: str
    expected_outcome: str
    monitoring: str


class InvestigationalApproach(BaseModel):
    """Investigational approach"""
    intervention: str
    rationale: str
    evidence_level: str
    implementation: str
    dosing: Optional[str] = None
    limitations: str
    safety_profile: str


class DebunkedClaim(BaseModel):
    """Debunked treatment claim"""
    claim: str
    reason_debunked: str
    debunked_by: str
    evidence: str
    why_harmful: str
    common_misconception: str


class RecommendationsData(BaseModel):
    """Structured recommendations"""
    evidence_based: List[EvidenceBasedRecommendation] = Field(default_factory=list)
    investigational: List[InvestigationalApproach] = Field(default_factory=list)
    debunked: List[DebunkedClaim] = Field(default_factory=list)


# =============== Monitoring Requirements Schema ===============

class MonitoringParameter(BaseModel):
    """Monitoring parameter"""
    parameter: str
    frequency: str
    target_range: str
    rationale: str


class MonitoringData(BaseModel):
    """Structured monitoring requirements"""
    baseline_assessments: List[str] = Field(default_factory=list)
    routine_monitoring: List[MonitoringParameter] = Field(default_factory=list)
    symptom_monitoring: List[str] = Field(default_factory=list)
