#!/usr/bin/env python3
"""
Medical Data Module
Centralized medical knowledge database to eliminate code duplication.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class RecommendationItem:
    """Structure for medical recommendations"""
    intervention: str
    rationale: str
    evidence_level: str
    timing: str = "As clinically indicated"
    dosing: str = ""
    limitations: str = ""


@dataclass
class DebunkedClaim:
    """Structure for debunked medical claims"""
    claim: str
    reason_debunked: str
    debunked_by: str
    evidence: str
    why_harmful: str


@dataclass
class OrganRecommendations:
    """Complete recommendations for an organ system"""
    known_recommendations: List[RecommendationItem]
    potential_recommendations: List[RecommendationItem]
    debunked_claims: List[DebunkedClaim]


class MedicalDataRepository:
    """Centralized repository for medical knowledge"""
    
    # Medical recommendations database (extracted from duplicated code)
    ORGAN_RECOMMENDATIONS: Dict[str, OrganRecommendations] = {
        "kidneys": OrganRecommendations(
            known_recommendations=[
                RecommendationItem(
                    intervention="Adequate hydration pre/post procedure",
                    rationale="Increases urine flow rate and reduces concentration of contrast agent in tubules, minimizing direct nephrotoxic effects",
                    evidence_level="Strong - Multiple RCTs and guidelines (ESR, ACR)",
                    timing="500ml saline 1-2 hours before, continue 6-12 hours after"
                ),
                RecommendationItem(
                    intervention="Monitor kidney function in at-risk patients",
                    rationale="Early detection of contrast-induced nephropathy allows for prompt intervention and prevents progression to acute kidney injury",
                    evidence_level="Strong - Standard of care per nephrology guidelines",
                    timing="Baseline creatinine within 7 days, follow-up at 48-72 hours post-procedure"
                ),
                RecommendationItem(
                    intervention="Avoid NSAIDs 48-72 hours post-procedure",
                    rationale="NSAIDs reduce renal blood flow via prostaglandin inhibition, compounding contrast-induced vasoconstriction",
                    evidence_level="Strong - Consistent evidence across multiple studies",
                    timing="48-72 hours before and after contrast exposure"
                )
            ],
            potential_recommendations=[
                RecommendationItem(
                    intervention="N-Acetylcysteine supplementation",
                    rationale="Antioxidant properties may reduce oxidative stress and free radical damage from contrast agents. Acts as glutathione precursor.",
                    evidence_level="Mixed - Some positive RCTs but multiple negative studies and meta-analyses show conflicting results",
                    dosing="600mg orally twice daily for 2 days starting day before procedure",
                    limitations="2018 Cochrane review found no significant benefit; still used in some centers"
                ),
                RecommendationItem(
                    intervention="Magnesium support for kidney function",
                    rationale="Magnesium deficiency associated with increased nephrotoxicity; supplementation may maintain cellular energy and reduce calcium influx",
                    evidence_level="Limited - Small studies suggest benefit but larger RCTs needed",
                    dosing="Magnesium sulfate 3g in 250ml saline over 1 hour before procedure",
                    limitations="Mechanism unclear, optimal dosing not established"
                ),
                RecommendationItem(
                    intervention="Sodium bicarbonate pre-treatment",
                    rationale="Alkalinization of tubular fluid may reduce formation of reactive oxygen species and Tamm-Horsfall protein precipitation",
                    evidence_level="Moderate - Several positive studies but some negative trials",
                    dosing="154 mEq/L in D5W at 3ml/kg/hr for 1hr before, then 1ml/kg/hr for 6hrs after",
                    limitations="Not superior to saline in all studies; logistically complex"
                )
            ],
            debunked_claims=[
                DebunkedClaim(
                    claim="Kidney detox cleanses",
                    reason_debunked="No scientific evidence for enhanced elimination of contrast agents; may cause electrolyte imbalances and dehydration",
                    debunked_by="American Society of Nephrology, National Kidney Foundation",
                    evidence="Systematic reviews show no benefit and potential harm from commercial detox products",
                    why_harmful="Can lead to dehydration, electrolyte disturbances, and delayed medical care"
                ),
                DebunkedClaim(
                    claim="Herbal kidney flushes",
                    reason_debunked="No peer-reviewed evidence for gadolinium elimination; some herbs (aristolochia) are nephrotoxic",
                    debunked_by="FDA warnings, nephrology literature",
                    evidence="Case reports of acute kidney injury from herbal products",
                    why_harmful="Potential drug interactions and direct nephrotoxicity"
                ),
                DebunkedClaim(
                    claim="Juice cleanses for elimination",
                    reason_debunked="Contrast agents eliminated by glomerular filtration, not affected by dietary interventions",
                    debunked_by="Basic pharmacokinetic principles, radiology literature",
                    evidence="Gadolinium elimination follows first-order kinetics independent of diet",
                    why_harmful="May cause hypoglycemia, nutrient deficiencies, and false sense of protection"
                )
            ]
        ),
        
        "brain": OrganRecommendations(
            known_recommendations=[
                RecommendationItem(
                    intervention="No specific interventions required for healthy patients",
                    rationale="Gadolinium retention in brain has no proven clinical consequences in patients with normal kidney function",
                    evidence_level="Strong - Multiple safety studies and FDA review",
                    timing="Ongoing monitoring of safety data"
                )
            ],
            potential_recommendations=[
                RecommendationItem(
                    intervention="Minimize repeated exposures when possible",
                    rationale="Linear gadolinium agents show greater brain retention than macrocyclic agents; cumulative effects unknown",
                    evidence_level="Precautionary - Based on imaging studies showing retention",
                    dosing="Use lowest effective dose, prefer macrocyclic agents",
                    limitations="No proven clinical harm, may delay necessary imaging"
                )
            ],
            debunked_claims=[
                DebunkedClaim(
                    claim="Brain detox supplements",
                    reason_debunked="Blood-brain barrier prevents most oral supplements from accessing brain tissue; no evidence for gadolinium removal",
                    debunked_by="Neuropharmacology research, lack of clinical trials",
                    evidence="No peer-reviewed studies showing brain gadolinium reduction from supplements",
                    why_harmful="Expensive, false hope, may contain unlisted ingredients"
                ),
                DebunkedClaim(
                    claim="Chelation therapy for gadolinium removal",
                    reason_debunked="No evidence that chelating agents remove gadolinium from brain tissue; may be harmful",
                    debunked_by="FDA warnings, multiple medical societies including American College of Radiology",
                    evidence="EDTA and other chelators can cause kidney damage, electrolyte imbalances, and cardiac arrhythmias",
                    why_harmful="Serious adverse effects including death; no proven benefit for gadolinium removal"
                )
            ]
        ),
        
        "liver": OrganRecommendations(
            known_recommendations=[
                RecommendationItem(
                    intervention="Monitor liver function in patients with hepatic disease",
                    rationale="Severely impaired hepatic function may affect gadolinium elimination kinetics",
                    evidence_level="Moderate - Based on pharmacokinetic studies",
                    timing="Baseline and 48-72 hour follow-up in severe hepatic impairment"
                )
            ],
            potential_recommendations=[
                RecommendationItem(
                    intervention="Antioxidant support",
                    rationale="Theoretical benefit from reducing oxidative stress, though liver is not primary elimination route",
                    evidence_level="Very limited - Mostly preclinical data",
                    dosing="Various antioxidant combinations studied",
                    limitations="No specific studies with gadolinium contrast; unclear clinical relevance"
                ),
                RecommendationItem(
                    intervention="Milk thistle supplementation",
                    rationale="Silymarin may have hepatoprotective effects through antioxidant and anti-inflammatory mechanisms",
                    evidence_level="Limited - Some studies in other hepatotoxic contexts",
                    dosing="140-420mg daily of standardized silymarin extract",
                    limitations="No studies specific to contrast agents; variable product quality"
                )
            ],
            debunked_claims=[
                DebunkedClaim(
                    claim="Liver cleanses",
                    reason_debunked="Liver has natural detoxification processes; no evidence that commercial cleanses enhance gadolinium elimination",
                    debunked_by="American Liver Foundation, hepatology literature",
                    evidence="No scientific basis for enhanced liver 'cleansing' beyond normal physiology",
                    why_harmful="May cause diarrhea, electrolyte imbalances, and interfere with medications"
                ),
                DebunkedClaim(
                    claim="Coffee enemas",
                    reason_debunked="No mechanism for gadolinium elimination via colon; potentially dangerous procedure",
                    debunked_by="Multiple case reports of complications, FDA warnings",
                    evidence="Risk of electrolyte imbalances, infections, and rectal perforation",
                    why_harmful="Serious complications including death reported; no medical justification"
                )
            ]
        )
    }
    
    @classmethod
    def get_organ_recommendations(cls, organ: str) -> OrganRecommendations:
        """Get recommendations for a specific organ system"""
        return cls.ORGAN_RECOMMENDATIONS.get(organ, cls._get_default_recommendations(organ))
    
    @classmethod
    def _get_default_recommendations(cls, organ: str) -> OrganRecommendations:
        """Default recommendations for unknown organs"""
        return OrganRecommendations(
            known_recommendations=[
                RecommendationItem(
                    intervention="Consult healthcare provider",
                    rationale="Limited data available for this organ system with the specified procedure",
                    evidence_level="Expert opinion - Insufficient specific research",
                    timing="Before and after procedure as clinically indicated"
                )
            ],
            potential_recommendations=[],
            debunked_claims=[]
        )
    
    @classmethod
    def get_all_supported_organs(cls) -> List[str]:
        """Get list of all organs with available recommendations"""
        return list(cls.ORGAN_RECOMMENDATIONS.keys())
    
    @classmethod
    def convert_to_legacy_format(cls, organ: str) -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility"""
        recommendations = cls.get_organ_recommendations(organ)
        
        return {
            "known_recommendations": [
                {
                    "intervention": rec.intervention,
                    "rationale": rec.rationale,
                    "evidence_level": rec.evidence_level,
                    "timing": rec.timing
                }
                for rec in recommendations.known_recommendations
            ],
            "potential_recommendations": [
                {
                    "intervention": rec.intervention,
                    "rationale": rec.rationale,
                    "evidence_level": rec.evidence_level,
                    "dosing": rec.dosing,
                    "limitations": rec.limitations
                }
                for rec in recommendations.potential_recommendations
            ],
            "debunked_claims": [
                {
                    "claim": claim.claim,
                    "reason_debunked": claim.reason_debunked,
                    "debunked_by": claim.debunked_by,
                    "evidence": claim.evidence,
                    "why_harmful": claim.why_harmful
                }
                for claim in recommendations.debunked_claims
            ]
        }