#!/usr/bin/env python3
"""
Output Validation and Quality Scoring Module
Evaluates medical reasoning agent outputs for accuracy, completeness, and safety.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from datetime import datetime

from medical_reasoning_agent import MedicalOutput, OrganAnalysis, ReasoningStep


class ValidationCategory(Enum):
    """Categories for validation checks"""
    MEDICAL_ACCURACY = "medical_accuracy"
    COMPLETENESS = "completeness"
    SAFETY = "safety"
    EVIDENCE_QUALITY = "evidence_quality"
    REASONING_LOGIC = "reasoning_logic"


class SeverityLevel(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"    # Could cause harm
    HIGH = "high"           # Major medical oversight
    MEDIUM = "medium"       # Important but not critical
    LOW = "low"            # Minor issue or improvement
    INFO = "info"          # Informational note


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    category: ValidationCategory
    severity: SeverityLevel
    message: str
    context: Dict[str, Any]
    suggested_fix: Optional[str] = None
    confidence: float = 0.8


@dataclass
class ValidationReport:
    """Complete validation report"""
    overall_score: float
    safety_score: float
    accuracy_score: float
    completeness_score: float
    issues: List[ValidationIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)


class MedicalKnowledgeValidator:
    """Validates medical content against known medical knowledge"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Medical knowledge base (simplified - would be much more comprehensive)
        self.contraindications = {
            "gadolinium": ["severe_kidney_disease", "pregnancy"],
            "iodine_contrast": ["iodine_allergy", "hyperthyroidism", "severe_kidney_disease"],
            "nsaids": ["kidney_disease", "heart_failure", "ulcers"]
        }
        
        self.debunked_treatments = {
            "kidney_cleanses", "detox_teas", "liver_flushes", 
            "colon_cleanses", "juice_cleanses", "chelation_therapy_for_general_detox"
        }
        
        self.evidence_based_treatments = {
            "hydration": {"kidney_protection": "strong_evidence"},
            "NAC": {"contrast_nephropathy": "moderate_evidence"},
            "magnesium": {"kidney_function": "limited_evidence"}
        }
        
        self.organ_pathways = {
            "kidneys": {
                "elimination_routes": ["glomerular_filtration", "tubular_secretion"],
                "risk_factors": ["ckd", "dehydration", "age", "diabetes"],
                "protective_measures": ["hydration", "avoid_nephrotoxins"]
            },
            "liver": {
                "elimination_routes": ["hepatic_metabolism", "biliary_excretion"],
                "risk_factors": ["cirrhosis", "hepatitis", "alcohol_use"],
                "protective_measures": ["avoid_hepatotoxins", "nutritional_support"]
            }
        }
    
    def validate_contraindications(self, medical_output: MedicalOutput) -> List[ValidationIssue]:
        """Check for missing contraindication warnings"""
        issues = []
        
        procedure_lower = medical_output.procedure_summary.lower()
        
        # Check for contrast-related contraindications
        if "gadolinium" in procedure_lower or "contrast" in procedure_lower:
            # Look for kidney disease warnings
            kidney_warnings = any(
                "kidney" in rec.lower() and ("disease" in rec.lower() or "impair" in rec.lower())
                for organ in medical_output.organs_analyzed
                for rec in organ.known_recommendations + organ.potential_recommendations
            )
            
            if not kidney_warnings:
                issues.append(ValidationIssue(
                    category=ValidationCategory.SAFETY,
                    severity=SeverityLevel.HIGH,
                    message="Missing contraindication warning for kidney disease with contrast agents",
                    context={"procedure": medical_output.procedure_summary},
                    suggested_fix="Add warning about contrast use in patients with severe kidney disease"
                ))
        
        return issues
    
    def validate_debunked_claims(self, medical_output: MedicalOutput) -> List[ValidationIssue]:
        """Check for inclusion of debunked medical claims"""
        issues = []
        
        for organ in medical_output.organs_analyzed:
            # Check if debunked treatments are incorrectly recommended
            all_recommendations = organ.known_recommendations + organ.potential_recommendations
            
            for recommendation in all_recommendations:
                rec_lower = recommendation.lower()
                for debunked in self.debunked_treatments:
                    if debunked.replace("_", " ") in rec_lower:
                        issues.append(ValidationIssue(
                            category=ValidationCategory.MEDICAL_ACCURACY,
                            severity=SeverityLevel.HIGH,
                            message=f"Debunked treatment '{debunked}' found in recommendations",
                            context={"organ": organ.organ_name, "recommendation": recommendation},
                            suggested_fix=f"Remove '{debunked}' and replace with evidence-based alternative"
                        ))
            
            # Check if debunked claims are properly identified
            if not organ.debunked_claims:
                issues.append(ValidationIssue(
                    category=ValidationCategory.COMPLETENESS,
                    severity=SeverityLevel.MEDIUM,
                    message=f"No debunked claims identified for {organ.organ_name}",
                    context={"organ": organ.organ_name},
                    suggested_fix="Add common debunked claims for this organ system"
                ))
        
        return issues
    
    def validate_evidence_quality(self, medical_output: MedicalOutput) -> List[ValidationIssue]:
        """Validate evidence quality classifications"""
        issues = []
        
        for organ in medical_output.organs_analyzed:
            # Check evidence quality consistency
            if organ.evidence_quality not in ["strong", "moderate", "limited", "poor"]:
                issues.append(ValidationIssue(
                    category=ValidationCategory.EVIDENCE_QUALITY,
                    severity=SeverityLevel.MEDIUM,
                    message=f"Invalid evidence quality classification: {organ.evidence_quality}",
                    context={"organ": organ.organ_name},
                    suggested_fix="Use standard evidence classification: strong/moderate/limited/poor"
                ))
            
            # Check if high-risk organs have appropriate evidence backing
            if organ.risk_level == "high" and organ.evidence_quality in ["poor", "limited"]:
                issues.append(ValidationIssue(
                    category=ValidationCategory.EVIDENCE_QUALITY,
                    severity=SeverityLevel.HIGH,
                    message=f"High-risk classification with poor evidence quality for {organ.organ_name}",
                    context={"organ": organ.organ_name, "risk_level": organ.risk_level},
                    suggested_fix="Either strengthen evidence or reconsider risk classification"
                ))
        
        return issues


class CompletenessValidator:
    """Validates completeness of medical analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.required_organs_by_procedure = {
            "mri_contrast": ["kidneys", "brain"],
            "ct_contrast": ["kidneys", "thyroid"],
            "cardiac_catheterization": ["heart", "kidneys", "blood_vessels"]
        }
    
    def validate_organ_coverage(self, medical_output: MedicalOutput) -> List[ValidationIssue]:
        """Check if all relevant organs are analyzed"""
        issues = []
        
        procedure_type = self._identify_procedure_type(medical_output.procedure_summary)
        required_organs = self.required_organs_by_procedure.get(procedure_type, [])
        
        analyzed_organs = {organ.organ_name.lower() for organ in medical_output.organs_analyzed}
        
        for required_organ in required_organs:
            if required_organ.lower() not in analyzed_organs:
                issues.append(ValidationIssue(
                    category=ValidationCategory.COMPLETENESS,
                    severity=SeverityLevel.HIGH,
                    message=f"Missing analysis for {required_organ}",
                    context={"procedure": medical_output.procedure_summary},
                    suggested_fix=f"Add comprehensive analysis for {required_organ}"
                ))
        
        return issues
    
    def validate_recommendation_completeness(self, medical_output: MedicalOutput) -> List[ValidationIssue]:
        """Check completeness of recommendations"""
        issues = []
        
        for organ in medical_output.organs_analyzed:
            # Check if high-risk organs have adequate recommendations
            if organ.at_risk and organ.risk_level in ["high", "moderate"]:
                total_recommendations = len(organ.known_recommendations) + len(organ.potential_recommendations)
                
                if total_recommendations < 2:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.COMPLETENESS,
                        severity=SeverityLevel.MEDIUM,
                        message=f"Insufficient recommendations for at-risk organ: {organ.organ_name}",
                        context={"organ": organ.organ_name, "risk_level": organ.risk_level},
                        suggested_fix="Provide more comprehensive recommendations for risk mitigation"
                    ))
            
            # Check for pathway information
            if organ.affected_by_procedure and not organ.pathways_involved:
                issues.append(ValidationIssue(
                    category=ValidationCategory.COMPLETENESS,
                    severity=SeverityLevel.MEDIUM,
                    message=f"Missing pathway information for {organ.organ_name}",
                    context={"organ": organ.organ_name},
                    suggested_fix="Add information about biological pathways involved"
                ))
        
        return issues
    
    def _identify_procedure_type(self, procedure_summary: str) -> str:
        """Identify procedure type from summary"""
        summary_lower = procedure_summary.lower()
        
        if "mri" in summary_lower and "contrast" in summary_lower:
            return "mri_contrast"
        elif "ct" in summary_lower and "contrast" in summary_lower:
            return "ct_contrast"
        elif "cardiac" in summary_lower or "catheter" in summary_lower:
            return "cardiac_catheterization"
        
        return "unknown"


class ReasoningValidator:
    """Validates reasoning logic and consistency"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_reasoning_flow(self, reasoning_trace: List[ReasoningStep]) -> List[ValidationIssue]:
        """Validate logical flow of reasoning"""
        issues = []
        
        # Check for required reasoning stages
        stages_present = {step.stage for step in reasoning_trace}
        required_stages = {
            "input_analysis", "organ_identification", 
            "evidence_gathering", "risk_assessment",
            "recommendation_synthesis", "critical_evaluation"
        }
        
        missing_stages = required_stages - {stage.value for stage in stages_present}
        
        for missing_stage in missing_stages:
            issues.append(ValidationIssue(
                category=ValidationCategory.REASONING_LOGIC,
                severity=SeverityLevel.HIGH,
                message=f"Missing reasoning stage: {missing_stage}",
                context={"present_stages": list(stages_present)},
                suggested_fix=f"Add {missing_stage} reasoning step"
            ))
        
        # Check confidence scores
        low_confidence_steps = [
            step for step in reasoning_trace 
            if step.confidence < 0.5
        ]
        
        if len(low_confidence_steps) > len(reasoning_trace) / 2:
            issues.append(ValidationIssue(
                category=ValidationCategory.REASONING_LOGIC,
                severity=SeverityLevel.MEDIUM,
                message="Multiple reasoning steps have low confidence",
                context={"low_confidence_count": len(low_confidence_steps)},
                suggested_fix="Review and strengthen reasoning with additional evidence"
            ))
        
        return issues
    
    def validate_consistency(self, medical_output: MedicalOutput) -> List[ValidationIssue]:
        """Check for internal consistency"""
        issues = []
        
        # Check risk level vs recommendations consistency
        for organ in medical_output.organs_analyzed:
            if organ.risk_level == "high" and len(organ.known_recommendations) == 0:
                issues.append(ValidationIssue(
                    category=ValidationCategory.REASONING_LOGIC,
                    severity=SeverityLevel.HIGH,
                    message=f"High risk classification but no known recommendations for {organ.organ_name}",
                    context={"organ": organ.organ_name},
                    suggested_fix="Provide evidence-based recommendations for high-risk organ"
                ))
            
            if not organ.at_risk and organ.risk_level in ["high", "moderate"]:
                issues.append(ValidationIssue(
                    category=ValidationCategory.REASONING_LOGIC,
                    severity=SeverityLevel.MEDIUM,
                    message=f"Inconsistent risk assessment for {organ.organ_name}",
                    context={"at_risk": organ.at_risk, "risk_level": organ.risk_level},
                    suggested_fix="Ensure consistency between at_risk flag and risk_level"
                ))
        
        return issues


class QualityScorer:
    """Calculates quality scores for medical analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_scores(self, medical_output: MedicalOutput, 
                        validation_issues: List[ValidationIssue]) -> Tuple[float, float, float, float]:
        """Calculate quality scores"""
        
        # Base scores
        safety_score = 1.0
        accuracy_score = 1.0
        completeness_score = 1.0
        
        # Deduct points based on issues
        for issue in validation_issues:
            deduction = self._get_deduction_for_issue(issue)
            
            if issue.category == ValidationCategory.SAFETY:
                safety_score -= deduction
            elif issue.category == ValidationCategory.MEDICAL_ACCURACY:
                accuracy_score -= deduction
            elif issue.category == ValidationCategory.COMPLETENESS:
                completeness_score -= deduction
        
        # Ensure scores don't go below 0
        safety_score = max(0.0, safety_score)
        accuracy_score = max(0.0, accuracy_score)
        completeness_score = max(0.0, completeness_score)
        
        # Calculate overall score (weighted average)
        overall_score = (
            safety_score * 0.4 +        # Safety is most important
            accuracy_score * 0.3 +      # Accuracy is crucial
            completeness_score * 0.3    # Completeness matters
        )
        
        return overall_score, safety_score, accuracy_score, completeness_score
    
    def _get_deduction_for_issue(self, issue: ValidationIssue) -> float:
        """Get point deduction for validation issue"""
        deductions = {
            SeverityLevel.CRITICAL: 0.5,
            SeverityLevel.HIGH: 0.3,
            SeverityLevel.MEDIUM: 0.15,
            SeverityLevel.LOW: 0.05,
            SeverityLevel.INFO: 0.01
        }
        
        return deductions.get(issue.severity, 0.1)


class OutputValidator:
    """Main output validation orchestrator"""
    
    def __init__(self):
        self.medical_validator = MedicalKnowledgeValidator()
        self.completeness_validator = CompletenessValidator()
        self.reasoning_validator = ReasoningValidator()
        self.quality_scorer = QualityScorer()
        self.logger = logging.getLogger(__name__)
    
    def validate_output(self, medical_output: MedicalOutput) -> ValidationReport:
        """Complete validation of medical output"""
        self.logger.info("Starting comprehensive output validation")
        
        all_issues = []
        
        # Medical knowledge validation
        all_issues.extend(self.medical_validator.validate_contraindications(medical_output))
        all_issues.extend(self.medical_validator.validate_debunked_claims(medical_output))
        all_issues.extend(self.medical_validator.validate_evidence_quality(medical_output))
        
        # Completeness validation
        all_issues.extend(self.completeness_validator.validate_organ_coverage(medical_output))
        all_issues.extend(self.completeness_validator.validate_recommendation_completeness(medical_output))
        
        # Reasoning validation
        all_issues.extend(self.reasoning_validator.validate_reasoning_flow(medical_output.reasoning_trace))
        all_issues.extend(self.reasoning_validator.validate_consistency(medical_output))
        
        # Calculate quality scores
        overall_score, safety_score, accuracy_score, completeness_score = \
            self.quality_scorer.calculate_scores(medical_output, all_issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues)
        
        report = ValidationReport(
            overall_score=overall_score,
            safety_score=safety_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            issues=all_issues,
            recommendations=recommendations
        )
        
        self.logger.info(f"Validation complete. Overall score: {overall_score:.2f}")
        return report
    
    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate actionable recommendations from validation issues"""
        recommendations = []
        
        # Group issues by category
        critical_issues = [i for i in issues if i.severity == SeverityLevel.CRITICAL]
        high_issues = [i for i in issues if i.severity == SeverityLevel.HIGH]
        
        if critical_issues:
            recommendations.append("🚨 CRITICAL: Address all critical safety issues immediately")
        
        if high_issues:
            recommendations.append("⚠️  HIGH PRIORITY: Review and fix high-severity issues")
        
        # Category-specific recommendations
        safety_issues = [i for i in issues if i.category == ValidationCategory.SAFETY]
        if safety_issues:
            recommendations.append("🛡️ Strengthen safety warnings and contraindication information")
        
        accuracy_issues = [i for i in issues if i.category == ValidationCategory.MEDICAL_ACCURACY]
        if accuracy_issues:
            recommendations.append("🎯 Verify medical accuracy against current clinical guidelines")
        
        completeness_issues = [i for i in issues if i.category == ValidationCategory.COMPLETENESS]
        if completeness_issues:
            recommendations.append("📋 Expand analysis to cover all relevant organ systems")
        
        return recommendations


# Utility functions for easy integration
def validate_medical_output(medical_output: MedicalOutput) -> ValidationReport:
    """Convenient function to validate medical output"""
    validator = OutputValidator()
    return validator.validate_output(medical_output)


def export_validation_report(report: ValidationReport, filepath: str):
    """Export validation report to JSON"""
    import json
    
    report_dict = {
        "scores": {
            "overall": report.overall_score,
            "safety": report.safety_score,
            "accuracy": report.accuracy_score,
            "completeness": report.completeness_score
        },
        "issues": [
            {
                "category": issue.category.value,
                "severity": issue.severity.value,
                "message": issue.message,
                "context": issue.context,
                "suggested_fix": issue.suggested_fix,
                "confidence": issue.confidence
            }
            for issue in report.issues
        ],
        "recommendations": report.recommendations,
        "validation_timestamp": report.validation_timestamp.isoformat(),
        "summary": {
            "total_issues": len(report.issues),
            "critical_issues": sum(1 for i in report.issues if i.severity == SeverityLevel.CRITICAL),
            "high_issues": sum(1 for i in report.issues if i.severity == SeverityLevel.HIGH),
            "medium_issues": sum(1 for i in report.issues if i.severity == SeverityLevel.MEDIUM)
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(report_dict, f, indent=2)


if __name__ == "__main__":
    # Example usage
    from medical_reasoning_agent import MedicalInput, MedicalReasoningAgent
    
    # Create sample input
    medical_input = MedicalInput(
        procedure="MRI Scanner",
        details="With gadolinium contrast",
        objectives=["safety assessment", "organ effects"]
    )
    
    # Create agent and run analysis
    agent = MedicalReasoningAgent(enable_logging=True)
    result = agent.analyze_medical_procedure(medical_input)
    
    # Validate output
    validation_report = validate_medical_output(result)
    
    print(f"Validation Results:")
    print(f"Overall Score: {validation_report.overall_score:.2f}")
    print(f"Safety Score: {validation_report.safety_score:.2f}")
    print(f"Total Issues: {len(validation_report.issues)}")
    
    # Export report
    export_validation_report(validation_report, "validation_report.json")