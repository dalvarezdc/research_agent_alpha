#!/usr/bin/env python3
"""
Consolidated Medical Analyzer
Performs all analysis (organ identification, evidence gathering, recommendations) in ONE LLM call.
"""

from typing import Dict, List, Any, Optional
import json
import logging
from medical_reasoning_agent import MedicalInput


class ConsolidatedAnalyzer:
    """Single-call medical analyzer with structured JSON output"""

    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(__name__)
        self.procedure_overview_cache = {}  # Cache overviews to avoid duplicate calls

        # Comprehensive fallback database for all procedures
        self.fallback_db = {
            "radiotherapy": {
                "organs": [
                    {
                        "name": "skin",
                        "risk_level": "high",
                        "pathways": ["direct_radiation_exposure", "cellular_damage"],
                        "evidence": {
                            "risks": ["radiation_dermatitis", "skin_fibrosis", "hyperpigmentation"],
                            "protective_factors": ["moisturization", "avoiding_irritants", "proper_skincare"],
                            "quality": "strong"
                        },
                        "recommendations": {
                            "known": [
                                "Apply moisturizer regularly to irradiated skin",
                                "Avoid sun exposure on treated areas",
                                "Use mild, fragrance-free soap"
                            ],
                            "potential": [
                                "Hyaluronic acid-based creams may help with healing",
                                "Silver sulfadiazine for acute radiation dermatitis"
                            ],
                            "debunked": [
                                "Petroleum jelly prevents radiation dermatitis (no evidence)",
                                "Aloe vera significantly reduces skin damage (limited evidence)"
                            ]
                        }
                    },
                    {
                        "name": "bone_marrow",
                        "risk_level": "moderate",
                        "pathways": ["radiation_induced_myelosuppression"],
                        "evidence": {
                            "risks": ["decreased_blood_counts", "anemia", "thrombocytopenia"],
                            "protective_factors": ["limited_field_radiation", "dose_fractionation"],
                            "quality": "strong"
                        },
                        "recommendations": {
                            "known": [
                                "Monitor complete blood counts regularly",
                                "Adjust treatment if severe myelosuppression occurs",
                                "Consider growth factors if indicated"
                            ],
                            "potential": [
                                "Stem cell mobilization for severe cases",
                                "Prophylactic growth factors in high-risk patients"
                            ],
                            "debunked": [
                                "Routine prophylactic antibiotics prevent all infections (not recommended)"
                            ]
                        }
                    },
                    {
                        "name": "salivary_glands",
                        "risk_level": "high",
                        "pathways": ["radiation_induced_xerostomia"],
                        "evidence": {
                            "risks": ["dry_mouth", "difficulty_swallowing", "dental_decay"],
                            "protective_factors": ["intensity_modulated_radiation", "parotid_sparing"],
                            "quality": "strong"
                        },
                        "recommendations": {
                            "known": [
                                "Use saliva substitutes and stimulants",
                                "Maintain excellent oral hygiene",
                                "Regular dental follow-up with fluoride treatments"
                            ],
                            "potential": [
                                "Amifostine may provide radioprotection",
                                "Acupuncture for xerostomia management"
                            ],
                            "debunked": [
                                "Pilocarpine works for all patients (variable response)"
                            ]
                        }
                    }
                ]
            },
            "mri_scanner": {
                "organs": [
                    {
                        "name": "kidneys",
                        "risk_level": "moderate",
                        "pathways": ["glomerular_filtration", "contrast_excretion"],
                        "evidence": {
                            "risks": ["contrast_nephropathy", "nephrogenic_systemic_fibrosis"],
                            "protective_factors": ["adequate_hydration", "normal_kidney_function"],
                            "quality": "strong"
                        },
                        "recommendations": {
                            "known": [
                                "Ensure adequate hydration before and after procedure",
                                "Monitor kidney function with creatinine levels",
                                "Avoid nephrotoxic medications 48 hours before/after"
                            ],
                            "potential": [
                                "N-acetylcysteine for high-risk patients",
                                "Sodium bicarbonate hydration may provide additional protection"
                            ],
                            "debunked": [
                                "Furosemide prevents contrast nephropathy (no evidence)",
                                "Dopamine is protective (may be harmful)"
                            ]
                        }
                    }
                ]
            }
        }

    def analyze_procedure(self, medical_input: MedicalInput) -> Dict[str, Any]:
        """Perform complete analysis in one call with structured JSON output"""
        self.logger.info(f"Starting consolidated analysis for: {medical_input.procedure}")

        # Try LLM first
        if self.llm_manager:
            try:
                return self._llm_consolidated_analysis(medical_input)
            except Exception as e:
                self.logger.warning(f"LLM analysis failed: {e}, using fallback")

        # Fallback to database
        return self._fallback_analysis(medical_input)

    def _llm_consolidated_analysis(self, medical_input: MedicalInput) -> Dict[str, Any]:
        """Single LLM call for complete analysis with JSON output"""

        system_prompt = """You are a medical analysis AI. You must respond ONLY with valid JSON, no other text.
Analyze medical procedures systematically and provide evidence-based recommendations."""

        prompt = f"""Analyze this medical procedure and return a JSON object with the complete analysis.

PROCEDURE: {medical_input.procedure}
DETAILS: {medical_input.details}
OBJECTIVES: {', '.join(medical_input.objectives)}

Return ONLY valid JSON in this EXACT format (no markdown, no code blocks, just raw JSON):

{{
  "organs_analyzed": [
    {{
      "name": "organ_name",
      "risk_level": "low|moderate|high",
      "pathways": ["primary_biological_pathway", "secondary_pathway"],
      "evidence": {{
        "risks": ["risk_factor_1", "risk_factor_2", "risk_factor_3"],
        "protective_factors": ["protective_factor_1", "protective_factor_2"],
        "quality": "strong|moderate|limited"
      }},
      "recommendations": {{
        "known": ["evidence_based_recommendation_1", "evidence_based_recommendation_2"],
        "potential": ["investigational_treatment_1", "investigational_treatment_2"],
        "debunked": ["harmful_practice_1", "harmful_practice_2"]
      }}
    }}
  ],
  "confidence_score": 0.85
}}

REQUIREMENTS:
1. Identify 2-5 most affected organs
2. For radiation/radiotherapy: focus on skin, bone marrow, salivary glands, GI tract
3. For contrast procedures: focus on kidneys, brain
4. All recommendations must be specific and actionable
5. Known recommendations = proven, evidence-based only
6. Potential recommendations = investigational but promising
7. Debunked = proven harmful or ineffective
8. Return ONLY the JSON object, no other text"""

        response = self.llm_manager.medical_analysis_with_fallback(
            {
                "procedure": medical_input.procedure,
                "details": medical_input.details,
                "objectives": medical_input.objectives
            },
            "consolidated_analysis"
        )

        return self._parse_json_response(response.get("analysis", ""), medical_input)

    def _parse_json_response(self, response: str, medical_input: MedicalInput) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            # Try direct JSON parse
            result = json.loads(response)

            # Validate structure
            if "organs_analyzed" in result and isinstance(result["organs_analyzed"], list):
                self.logger.info(f"Successfully parsed JSON with {len(result['organs_analyzed'])} organs")
                return result
            else:
                self.logger.warning("JSON missing required fields, using fallback")
                return self._fallback_analysis(medical_input)

        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parse failed: {e}, trying to extract JSON")

            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    if "organs_analyzed" in result:
                        return result
                except json.JSONDecodeError:
                    pass

            # Try to find JSON object in response
            json_match = re.search(r'\{[^{}]*"organs_analyzed"[^{}]*\[.*?\][^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    return result
                except json.JSONDecodeError:
                    pass

            self.logger.warning("All JSON extraction attempts failed, using fallback")
            return self._fallback_analysis(medical_input)

    def _fallback_analysis(self, medical_input: MedicalInput) -> Dict[str, Any]:
        """Fallback to database when LLM is unavailable"""
        procedure_key = medical_input.procedure.lower().replace(" ", "_")

        # Try exact match
        if procedure_key in self.fallback_db:
            self.logger.info(f"Using fallback data for {procedure_key}")
            return {
                "organs_analyzed": self.fallback_db[procedure_key]["organs"],
                "confidence_score": 0.7
            }

        # Try partial matches
        for key, value in self.fallback_db.items():
            if key in procedure_key or procedure_key in key:
                self.logger.info(f"Using fallback data for similar procedure: {key}")
                return {
                    "organs_analyzed": value["organs"],
                    "confidence_score": 0.6
                }

        # Generic fallback
        self.logger.warning("No specific fallback data, using generic")
        return {
            "organs_analyzed": [
                {
                    "name": "general",
                    "risk_level": "moderate",
                    "pathways": ["consult_physician"],
                    "evidence": {
                        "risks": ["procedure_specific_risks"],
                        "protective_factors": ["medical_supervision"],
                        "quality": "limited"
                    },
                    "recommendations": {
                        "known": ["Consult healthcare provider before procedure"],
                        "potential": ["Follow standard medical protocols"],
                        "debunked": ["Avoid unproven treatments"]
                    }
                }
            ],
            "confidence_score": 0.4
        }

    def get_procedure_overview(self, medical_input: MedicalInput) -> Dict[str, str]:
        """Get procedure overview with description, conditions, and contraindications (separate API call)"""
        cache_key = medical_input.procedure.lower()

        # Check cache first
        if cache_key in self.procedure_overview_cache:
            self.logger.info(f"Using cached overview for {medical_input.procedure}")
            return self.procedure_overview_cache[cache_key]

        self.logger.info(f"Fetching procedure overview for: {medical_input.procedure}")

        # Try LLM first
        if self.llm_manager:
            try:
                overview = self._llm_procedure_overview(medical_input)
                self.procedure_overview_cache[cache_key] = overview
                return overview
            except Exception as e:
                self.logger.warning(f"LLM overview failed: {e}, using fallback")

        # Fallback to database
        return self._fallback_procedure_overview(medical_input)

    def _llm_procedure_overview(self, medical_input: MedicalInput) -> Dict[str, str]:
        """Get procedure overview from LLM with structured JSON output"""

        system_prompt = """You are a medical information AI. You must respond ONLY with valid JSON, no other text.
Provide concise, accurate medical procedure information."""

        prompt = f"""Provide a brief overview of this medical procedure in JSON format.

PROCEDURE: {medical_input.procedure}
DETAILS: {medical_input.details}

Return ONLY valid JSON in this EXACT format (no markdown, no code blocks, just raw JSON):

{{
  "description": "Brief description of the procedure in maximum 300 words. Explain what it is, how it works, and its general purpose.",
  "conditions_treated": "List the main medical conditions this procedure is used for in maximum 150 words. Be specific.",
  "contraindications": "List patients at special risk or who should not undergo this procedure in maximum 150 words. Include absolute and relative contraindications."
}}

REQUIREMENTS:
1. description: Maximum 100 words
2. conditions_treated: Maximum 50 words
3. contraindications: Maximum 50 words
4. Be medically accurate and evidence-based
5. Use clear, professional language
6. Return ONLY the JSON object, no other text"""

        response = self.llm_manager.medical_analysis_with_fallback(
            {
                "procedure": medical_input.procedure,
                "details": medical_input.details
            },
            "procedure_overview"
        )

        return self._parse_overview_json(response.get("analysis", ""), medical_input)

    def _parse_overview_json(self, response: str, medical_input: MedicalInput) -> Dict[str, str]:
        """Parse overview JSON from LLM response"""
        try:
            # Try direct JSON parse
            result = json.loads(response)

            # Validate structure
            required_fields = ["description", "conditions_treated", "contraindications"]
            if all(field in result for field in required_fields):
                self.logger.info("Successfully parsed overview JSON")
                return result
            else:
                self.logger.warning("JSON missing required fields, using fallback")
                return self._fallback_procedure_overview(medical_input)

        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parse failed: {e}, trying to extract JSON")

            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    required_fields = ["description", "conditions_treated", "contraindications"]
                    if all(field in result for field in required_fields):
                        return result
                except json.JSONDecodeError:
                    pass

            # Try to find JSON object in response
            json_match = re.search(r'\{[^{}]*"description"[^{}]*"conditions_treated"[^{}]*"contraindications"[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    return result
                except json.JSONDecodeError:
                    pass

            self.logger.warning("All JSON extraction attempts failed, using fallback")
            return self._fallback_procedure_overview(medical_input)

    def _fallback_procedure_overview(self, medical_input: MedicalInput) -> Dict[str, str]:
        """Fallback procedure overview from database"""
        procedure_key = medical_input.procedure.lower().replace(" ", "_")

        fallback_overviews = {
            "radiotherapy": {
                "description": "Radiotherapy (radiation therapy) uses high-energy radiation beams to damage cancer cell DNA, preventing their growth and division. It can be delivered externally via linear accelerators or internally through brachytherapy. Treatment is typically fractionated over several weeks to maximize cancer cell kill while allowing normal tissue recovery. Modern techniques like IMRT and IGRT improve precision and reduce side effects.",
                "conditions_treated": "Cancer treatment (primary or adjuvant), tumor shrinkage before surgery, palliative care for pain/symptoms, benign conditions like keloids and arteriovenous malformations in select cases.",
                "contraindications": "Pregnancy (absolute), previous radiation to same area (relative), active lupus or scleroderma, severe immunosuppression, life expectancy under 6 months for curative intent, inability to lie still during treatment."
            },
            "mri_scanner": {
                "description": "Magnetic Resonance Imaging (MRI) uses powerful magnetic fields and radio waves to create detailed images of internal body structures. The scanner generates a strong magnetic field that aligns hydrogen protons in the body, then uses radio pulses to disturb this alignment. As protons return to normal, they emit signals detected by the scanner. Different tissues emit different signals, creating contrast in the images.",
                "conditions_treated": "Brain and spinal cord disorders, joint injuries, organ abnormalities, cancer detection and staging, cardiovascular disease, unexplained pain investigation, infection localization.",
                "contraindications": "Cardiac pacemakers, cochlear implants, certain metal implants, claustrophobia (severe), pregnancy (first trimester), metallic foreign bodies in eyes, inability to remain still for 30-60 minutes."
            },
            "ct_scan": {
                "description": "Computed Tomography (CT) uses X-rays and computer processing to create detailed cross-sectional images of the body. An X-ray tube rotates around the patient, taking multiple images from different angles. Computer algorithms combine these images to produce 3D representations of internal structures with excellent spatial resolution.",
                "conditions_treated": "Trauma evaluation, cancer detection and monitoring, internal bleeding detection, bone fractures, lung disease, vascular disease, appendicitis and other acute conditions.",
                "contraindications": "Pregnancy (relative), contrast allergy for contrast-enhanced studies, severe renal impairment (for contrast CT), thyroid disease (for iodinated contrast), inability to cooperate with breath-holding."
            }
        }

        # Try exact match
        if procedure_key in fallback_overviews:
            self.logger.info(f"Using fallback overview for {procedure_key}")
            return fallback_overviews[procedure_key]

        # Try partial matches
        for key, value in fallback_overviews.items():
            if key in procedure_key or procedure_key in key:
                self.logger.info(f"Using fallback overview for similar procedure: {key}")
                return value

        # Generic fallback
        self.logger.warning("No specific overview data, using generic")
        return {
            "description": f"{medical_input.procedure} is a medical procedure used for diagnostic or therapeutic purposes. The specific mechanism and approach depend on the type of procedure and patient condition. Consult with healthcare providers for detailed information about this specific procedure.",
            "conditions_treated": "Various medical conditions depending on the specific procedure type and clinical indication. Consult with qualified healthcare providers for specific applications.",
            "contraindications": "Contraindications vary by procedure. Generally includes pregnancy concerns, allergy to procedure-specific agents, severe comorbidities, and patient-specific factors. Consult healthcare provider for comprehensive risk assessment."
        }
