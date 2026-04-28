import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

from llm_integrations import LLMManager, create_llm_manager, call_model
from .bayesian_engine import NaiveBayesDiagnosticEngine
from .dspy_schemas import SymptomExtraction, DiagnosticReport

class MedicalDiagnosticAgent:
    """
    A 5-level diagnostic agent that combines LLM NLP with Bayesian math.
    """
    
    def __init__(self,
                 primary_llm_provider: str = "claude",
                 fallback_providers: List[str] = None,
                 enable_logging: bool = True,
                 interactive: bool = True):
        self.interactive = interactive
        self.engine = NaiveBayesDiagnosticEngine()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO if enable_logging else logging.ERROR)
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM manager
        self.llm_provider_name = primary_llm_provider
        self.llm_manager = create_llm_manager(
            primary_provider=primary_llm_provider,
            fallback_providers=fallback_providers or ["openai"]
        )

    def run_diagnostic_pipeline(self, user_query: str) -> Dict[str, Any]:
        """
        Executes the 5-level diagnostic pipeline.
        """
        self.logger.info(f"Starting diagnostic pipeline for: {user_query}")
        
        # --- Level 1: Symptom Extraction ---
        extraction = self._level1_extract_symptoms(user_query)
        
        if extraction.is_vague and self.interactive:
            # Handle vague input interactively
            print(f"\n[Diagnostic Agent] {extraction.clarification_question}")
            new_input = input("Your response: ")
            # Re-extract with added context
            extraction = self._level1_extract_symptoms(f"{user_query}. Context: {new_input}")

        # --- Level 2: Initial Bayesian Scoring ---
        results = self.engine.calculate_probabilities(extraction.symptoms, extraction.negative_symptoms)
        
        # --- Level 3: Differentiating Questions & Exams ---
        diff_symptoms = self.engine.get_differentiating_symptoms(results, extraction.symptoms)
        recommended_exams = self.engine.get_recommended_exams(results)
        
        # Prepare the "Intervention" question
        exam_names = [e['name'] for e in recommended_exams]
        intervention_prompt = self._format_intervention_question(diff_symptoms, exam_names)
        
        # --- Level 4: Iterative Update (Interactive) ---
        if self.interactive and (diff_symptoms or recommended_exams):
            print(f"\n[Diagnostic Agent] {intervention_prompt}")
            print("Options:")
            print("  - Answer about symptoms (e.g., 'I have X but not Y')")
            print("  - Provide exam results (e.g., 'Positive Strep Test')")
            print("  - Press Enter to skip and generate report")
            
            user_response = input("Your response: ").strip()
            if user_response:
                # Update probabilities based on new info
                # Simple logic: re-extract or check for exam IDs
                new_extraction = self._level1_extract_symptoms(user_response)
                extraction.symptoms.extend(new_extraction.symptoms)
                extraction.negative_symptoms.extend(new_extraction.negative_symptoms)
                
                # Check for exam matches
                for exam in recommended_exams:
                    if exam['name'].lower() in user_response.lower() or exam['id'].lower() in user_response.lower():
                        is_pos = "positive" in user_response.lower() or "yes" in user_response.lower()
                        results = self.engine.update_with_exam_result(results, exam['id'], is_pos)
                
                # Re-calculate if symptoms changed
                results = self.engine.calculate_probabilities(extraction.symptoms, extraction.negative_symptoms)

        # --- Level 5: Final Report & Routing ---
        report = self._level5_generate_report(results, extraction)
        
        return {
            "extraction": extraction.model_dump(),
            "probabilities": results,
            "report": report.model_dump()
        }

    def _level1_extract_symptoms(self, query: str) -> SymptomExtraction:
        """Uses LLM to extract structured symptoms from free text."""
        system_prompt = f"""You are a medical NLP specialist. Extract symptoms from the user query.
Available symptoms in database: {', '.join(self.engine.all_symptoms)}
Return ONLY valid JSON matching the schema."""
        
        prompt = f"""User Query: {query}
Schema: {json.dumps(SymptomExtraction.model_json_schema())}
Only use symptoms from the available list if they match. If a symptom is mentioned as absent, put it in negative_symptoms."""
        
        # Use llm_manager to support provider aliases and fallbacks
        llm_provider = self.llm_manager.get_available_provider()
        if not llm_provider:
            self.logger.error("No LLM provider available for symptom extraction")
            return SymptomExtraction(symptoms=[], is_vague=True, clarification_question="I'm sorry, I'm having trouble connecting to my reasoning engine. Could you please try again later?")

        try:
            response, _ = llm_provider.generate_response(
                prompt=prompt,
                system_prompt=system_prompt
            )
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return SymptomExtraction(symptoms=[], is_vague=True, clarification_question="I encountered an error while processing your request. Could you please describe your symptoms again?")
        
        # Handle potential markdown formatting from LLM
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
            
        try:
            return SymptomExtraction.model_validate_json(response.strip())
        except Exception as e:
            self.logger.error(f"Failed to parse Level 1 response: {e}. Response was: {response}")
            # Fallback
            return SymptomExtraction(symptoms=[], is_vague=True, clarification_question="Could you describe your symptoms in more detail?")

    def _format_intervention_question(self, diff_symptoms: List[str], exams: List[str]) -> str:
        """Uses LLM to format a patient-friendly differentiating question."""
        prompt = f"""I have calculated that the following symptoms and exams would help differentiate the diagnosis:
Symptoms: {', '.join(diff_symptoms)}
Exams: {', '.join(exams)}

Create a single, polite, patient-friendly question asking if they have these symptoms or results."""
        
        llm_provider = self.llm_manager.get_available_provider()
        response, _ = llm_provider.generate_response(
            prompt=prompt,
            system_prompt="You are a helpful medical assistant."
        )
        return response.strip()

    def _level5_generate_report(self, results: List[Dict[str, Any]], extraction: SymptomExtraction) -> DiagnosticReport:
        """Uses LLM to generate the final empathetic report based on math data."""
        top_candidates = results[:5]
        most_probable = top_candidates[0]
        most_serious = max(top_candidates, key=lambda x: x['severity'])
        
        system_prompt = "You are a senior diagnostic physician. Generate a structured report based on the provided mathematical data."
        
        prompt = f"""
        Diagnostic Data:
        Top 5 Candidates: {json.dumps(top_candidates, indent=2)}
        Most Probable: {most_probable['name']} ({most_probable['probability']:.2%})
        Most Serious: {most_serious['name']} (Severity: {most_serious['severity']}/5)
        
        Extracted Symptoms: {', '.join(extraction.symptoms)}
        Duration: {extraction.duration}
        
        Generate a report following the schema:
        {json.dumps(DiagnosticReport.model_json_schema())}
        
        Ensure the 'suggested_agent' is 'medication_agent' if the solution is drug-based, or 'procedure_agent' if it requires interventional treatment.
        """
        
        llm_provider = self.llm_manager.get_available_provider()
        response, _ = llm_provider.generate_response(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
            
        try:
            return DiagnosticReport.model_validate_json(response.strip())
        except Exception as e:
            self.logger.error(f"Failed to parse Level 5 report: {e}")
            return DiagnosticReport(
                top_5_candidates=[c['name'] for c in top_candidates],
                most_probable=most_probable['name'],
                most_serious=most_serious['name'],
                reasoning_summary="Based on your symptoms and clinical probability models.",
                recommended_next_steps=["Consult with a healthcare professional."],
                suggested_agent="medication_agent",
                routing_rationale="General follow-up."
            )
