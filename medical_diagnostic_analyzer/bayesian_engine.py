import math
import json
import os
from typing import Dict, List, Any, Tuple, Optional

class NaiveBayesDiagnosticEngine:
    """
    A diagnostic engine that uses Naive Bayes for calculating disease probabilities.
    """
    
    def __init__(self, database_path: str = None):
        if database_path is None:
            database_path = os.path.join(os.path.dirname(__file__), 'symptom_database.json')
        
        with open(database_path, 'r') as f:
            data = json.load(f)
            self.diseases = data['diseases']
            self.exams = data['exams']
            
        # Extract all unique symptoms from the database
        self.all_symptoms = set()
        for disease in self.diseases:
            self.all_symptoms.update(disease['symptoms'].keys())
            
    def calculate_probabilities(self, reported_symptoms: List[str], negative_symptoms: List[str] = None) -> List[Dict[str, Any]]:
        """
        Calculates the posterior probability of each disease given the reported symptoms.
        
        P(Disease | Symptoms) = [P(Symptoms | Disease) * P(Disease)] / P(Symptoms)
        
        Since P(Symptoms) is constant across all diseases, we calculate the numerator and normalize.
        """
        if negative_symptoms is None:
            negative_symptoms = []
            
        results = []
        total_numerator = 0.0
        
        for disease in self.diseases:
            # Start with prior probability (prevalence)
            prior = disease['prevalence']
            
            # Calculate likelihood P(Symptoms | Disease)
            likelihood = 1.0
            
            # Process reported (positive) symptoms
            for symptom in reported_symptoms:
                if symptom in disease['symptoms']:
                    # Symptom is present in disease profile
                    likelihood *= disease['symptoms'][symptom]
                else:
                    # Symptom is NOT typically associated with this disease
                    # We use a small epsilon for penalty instead of zero to allow for outliers
                    likelihood *= 0.05 
            
            # Process negative symptoms (user explicitly stated they don't have these)
            for symptom in negative_symptoms:
                if symptom in disease['symptoms']:
                    # Disease typically HAS this symptom, but user doesn't
                    likelihood *= (1.0 - disease['symptoms'][symptom])
                else:
                    # Disease doesn't have it, and user doesn't have it. This is expected.
                    likelihood *= 0.95
            
            numerator = prior * likelihood
            results.append({
                "id": disease['id'],
                "name": disease['name'],
                "severity": disease['severity'],
                "raw_score": numerator
            })
            total_numerator += numerator
            
        # Normalize to get actual probabilities
        for res in results:
            if total_numerator > 0:
                res['probability'] = res['raw_score'] / total_numerator
            else:
                res['probability'] = 1.0 / len(self.diseases)
            del res['raw_score']
            
        # Sort by probability descending
        return sorted(results, key=lambda x: x['probability'], reverse=True)

    def get_differentiating_symptoms(self, top_candidates: List[Dict[str, Any]], reported_symptoms: List[str], limit: int = 3) -> List[str]:
        """
        Identifies symptoms that would most effectively differentiate between the top candidates.
        Uses a simplified Shannon Entropy approach (Information Gain).
        """
        candidate_ids = [c['id'] for c in top_candidates[:3]] # Focus on top 3
        candidate_profiles = [d for d in self.diseases if d['id'] in candidate_ids]
        
        # Symptoms not yet reported or ruled out
        remaining_symptoms = self.all_symptoms - set(reported_symptoms)
        
        scores = []
        for symptom in remaining_symptoms:
            # Calculate variance of this symptom across top candidates
            # High variance means the symptom is a good differentiator
            probs = []
            for profile in candidate_profiles:
                probs.append(profile['symptoms'].get(symptom, 0.05))
            
            mean = sum(probs) / len(probs)
            variance = sum((p - mean) ** 2 for p in probs) / len(probs)
            
            scores.append((symptom, variance))
            
        # Sort by variance descending
        sorted_symptoms = sorted(scores, key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_symptoms[:limit]]

    def get_recommended_exams(self, top_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Finds exams associated with the top candidate diseases.
        """
        candidate_ids = [c['id'] for c in top_candidates[:3]]
        recommended = []
        
        for exam in self.exams:
            # If the exam targets any of our top candidate diseases
            if any(target in candidate_ids for target in exam['target_diseases']):
                recommended.append(exam)
                
        return recommended

    def update_with_exam_result(self, current_probabilities: List[Dict[str, Any]], exam_id: str, result: bool) -> List[Dict[str, Any]]:
        """
        Updates probabilities based on an exam result.
        """
        exam = next((e for e in self.exams if e[ 'id'] == exam_id), None)
        if not exam:
            return current_probabilities
            
        targets = set(exam['target_diseases'])
        reliability = exam['reliability']
        
        updated = []
        total = 0.0
        
        for res in current_probabilities:
            prob = res['probability']
            is_target = res['id'] in targets
            
            if result: # Positive result
                if is_target:
                    # Exam is positive and disease is the target
                    prob *= reliability
                else:
                    # Exam is positive but disease is NOT the target
                    prob *= (1.0 - reliability)
            else: # Negative result
                if is_target:
                    # Exam is negative and disease is the target
                    prob *= (1.0 - reliability)
                else:
                    # Exam is negative and disease is NOT the target
                    prob *= reliability
                    
            res['probability'] = prob
            total += prob
            updated.append(res)
            
        # Re-normalize
        if total > 0:
            for res in updated:
                res['probability'] /= total
        
        return sorted(updated, key=lambda x: x['probability'], reverse=True)
