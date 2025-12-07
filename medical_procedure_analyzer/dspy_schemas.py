#!/usr/bin/env python3
"""
DSPy Structured Output Schemas
Pydantic models for enforcing structured LLM outputs using DSPy.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


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
