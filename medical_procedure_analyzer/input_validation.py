#!/usr/bin/env python3
"""
Input Validation Module
Provides validation and sanitization for user inputs to prevent errors and security issues.
"""

import re
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class InputType(Enum):
    """Types of inputs that can be validated"""
    MEDICAL_PROCEDURE = "medical_procedure"
    MEDICAL_ASPECT = "medical_aspect"
    PROVIDER_NAME = "provider_name"  
    FILE_PATH = "file_path"
    SCENARIO_NAME = "scenario_name"


@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    sanitized_input: str
    errors: List[str]
    warnings: List[str]


class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    # Allowed medical procedure patterns
    MEDICAL_PROCEDURE_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_()./,:]{1,200}$')
    
    # Allowed characters for medical aspects
    MEDICAL_ASPECT_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_]{1,100}$')
    
    # Provider names
    VALID_PROVIDERS = {'claude-sonnet', 'claude-opus', 'openai', 'ollama', 'grok-4-1-fast', 'grok-4-1-code', 'grok-4-1-reasoning'}
    
    # File path validation (basic security)
    SAFE_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_./]{1,500}$')
    
    # Dangerous patterns to reject
    DANGEROUS_PATTERNS = [
        re.compile(r'[<>"\']'),  # HTML/XML injection
        re.compile(r'javascript:', re.IGNORECASE),  # JS injection
        re.compile(r'data:', re.IGNORECASE),  # Data URLs
        re.compile(r'[;\|\&\$`]'),  # Command injection
        re.compile(r'\.\.[\\/]'),  # Path traversal
    ]
    
    @classmethod
    def validate_medical_procedure(cls, procedure: str) -> ValidationResult:
        """Validate and sanitize medical procedure name"""
        errors = []
        warnings = []
        
        # Basic checks
        if not procedure:
            return ValidationResult(False, "", ["Procedure name cannot be empty"], [])
        
        if not isinstance(procedure, str):
            return ValidationResult(False, "", ["Procedure name must be a string"], [])
        
        # Length check
        if len(procedure) > 200:
            errors.append("Procedure name too long (max 200 characters)")
        
        if len(procedure) < 2:
            errors.append("Procedure name too short (min 2 characters)")
        
        # Sanitize: remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', procedure.strip())
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.search(sanitized):
                errors.append("Procedure name contains potentially dangerous characters")
                break
        
        # Pattern matching
        if not cls.MEDICAL_PROCEDURE_PATTERN.match(sanitized):
            errors.append("Procedure name contains invalid characters")
        
        # Medical terminology validation
        if not cls._contains_medical_terms(sanitized):
            warnings.append("Procedure name may not be a valid medical term")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def validate_medical_aspects(cls, aspects: List[str]) -> ValidationResult:
        """Validate medical aspects list"""
        if not aspects:
            return ValidationResult(True, [], [], [])
        
        if not isinstance(aspects, list):
            return ValidationResult(False, [], ["Aspects must be a list"], [])
        
        errors = []
        warnings = []
        sanitized_aspects = []
        
        for aspect in aspects:
            if not isinstance(aspect, str):
                errors.append(f"Aspect must be string, got {type(aspect)}")
                continue
            
            # Sanitize aspect
            sanitized = re.sub(r'\s+', ' ', aspect.strip())
            
            if len(sanitized) == 0:
                warnings.append("Empty aspect ignored")
                continue
            
            if len(sanitized) > 100:
                errors.append(f"Aspect too long: {sanitized[:50]}...")
                continue
            
            # Check for dangerous patterns
            has_danger = False
            for pattern in cls.DANGEROUS_PATTERNS:
                if pattern.search(sanitized):
                    errors.append(f"Aspect contains dangerous characters: {sanitized}")
                    has_danger = True
                    break
            
            if not has_danger and cls.MEDICAL_ASPECT_PATTERN.match(sanitized):
                sanitized_aspects.append(sanitized)
            elif not has_danger:
                errors.append(f"Aspect contains invalid characters: {sanitized}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized_aspects,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def validate_provider_name(cls, provider: str) -> ValidationResult:
        """Validate LLM provider name"""
        if not provider:
            return ValidationResult(False, "", ["Provider name cannot be empty"], [])
        
        if not isinstance(provider, str):
            return ValidationResult(False, "", ["Provider name must be a string"], [])
        
        sanitized = provider.lower().strip()
        
        if sanitized not in cls.VALID_PROVIDERS:
            return ValidationResult(
                False, 
                sanitized, 
                [f"Invalid provider '{sanitized}'. Must be one of: {', '.join(cls.VALID_PROVIDERS)}"],
                []
            )
        
        return ValidationResult(True, sanitized, [], [])
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> ValidationResult:
        """Validate file path for security"""
        if not file_path:
            return ValidationResult(False, "", ["File path cannot be empty"], [])
        
        if not isinstance(file_path, str):
            return ValidationResult(False, "", ["File path must be a string"], [])
        
        errors = []
        warnings = []
        
        # Check for path traversal
        if '..' in file_path:
            errors.append("Path traversal detected")
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.search(file_path):
                errors.append("File path contains dangerous characters")
                break
        
        # Length check
        if len(file_path) > 500:
            errors.append("File path too long")
        
        # Basic pattern check
        if not cls.SAFE_PATH_PATTERN.match(file_path):
            errors.append("File path contains invalid characters")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=file_path.strip(),
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def validate_scenario_name(cls, scenario_name: str) -> ValidationResult:
        """Validate scenario name"""
        if not scenario_name:
            return ValidationResult(False, "", ["Scenario name cannot be empty"], [])
        
        if not isinstance(scenario_name, str):
            return ValidationResult(False, "", ["Scenario name must be a string"], [])
        
        # Sanitize
        sanitized = re.sub(r'[^\w\s\-_]', '', scenario_name.strip())
        sanitized = re.sub(r'\s+', '_', sanitized)
        
        errors = []
        warnings = []
        
        if len(sanitized) == 0:
            errors.append("Scenario name contains no valid characters")
        elif len(sanitized) > 100:
            errors.append("Scenario name too long (max 100 characters)")
        elif len(sanitized) < 2:
            errors.append("Scenario name too short (min 2 characters)")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def _contains_medical_terms(cls, text: str) -> bool:
        """Check if text contains common medical terminology"""
        medical_keywords = [
            'scan', 'procedure', 'test', 'examination', 'imaging', 'therapy', 'treatment',
            'surgery', 'biopsy', 'endoscopy', 'colonoscopy', 'mri', 'ct', 'ultrasound',
            'x-ray', 'mammography', 'angiography', 'catheter', 'injection', 'contrast',
            'diagnostic', 'screening', 'monitoring', 'evaluation', 'assessment'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in medical_keywords)
    
    @classmethod
    def validate_input(cls, input_value: Any, input_type: InputType) -> ValidationResult:
        """Generic validation dispatcher"""
        if input_type == InputType.MEDICAL_PROCEDURE:
            return cls.validate_medical_procedure(input_value)
        elif input_type == InputType.MEDICAL_ASPECT:
            return cls.validate_medical_aspects(input_value)
        elif input_type == InputType.PROVIDER_NAME:
            return cls.validate_provider_name(input_value)
        elif input_type == InputType.FILE_PATH:
            return cls.validate_file_path(input_value)
        elif input_type == InputType.SCENARIO_NAME:
            return cls.validate_scenario_name(input_value)
        else:
            return ValidationResult(False, input_value, ["Unknown input type"], [])


class SecureMedicalInput:
    """Wrapper for validated medical inputs"""
    
    def __init__(self, procedure: str, details: str = "", objectives: List[str] = None, patient_context: str = None):
        # Validate all inputs
        proc_result = InputValidator.validate_medical_procedure(procedure)
        if not proc_result.is_valid:
            raise ValidationError(f"Invalid procedure: {', '.join(proc_result.errors)}")
        
        details_result = InputValidator.validate_medical_procedure(details) if details else ValidationResult(True, "", [], [])
        if not details_result.is_valid:
            raise ValidationError(f"Invalid details: {', '.join(details_result.errors)}")
        
        if objectives:
            obj_result = InputValidator.validate_medical_aspects(objectives)
            if not obj_result.is_valid:
                raise ValidationError(f"Invalid objectives: {', '.join(obj_result.errors)}")
            self.objectives = tuple(obj_result.sanitized_input)
        else:
            self.objectives = tuple()
        
        self.procedure = proc_result.sanitized_input
        self.details = details_result.sanitized_input if details else ""
        self.patient_context = patient_context  # Could add validation here too
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            'procedure': self.procedure,
            'details': self.details,
            'objectives': list(self.objectives),
            'patient_context': self.patient_context
        }


# Usage examples and testing
if __name__ == "__main__":
    # Test medical procedure validation
    test_cases = [
        "Endoscopy",
        "MRI with gadolinium contrast",
        "CT Scan",
        "<script>alert('xss')</script>",
        "",
        "A" * 300,  # Too long
        "colonoscopy; rm -rf /",  # Command injection attempt
    ]
    
    print("Testing Medical Procedure Validation:")
    for test in test_cases:
        result = InputValidator.validate_medical_procedure(test)
        print(f"Input: '{test[:50]}{'...' if len(test) > 50 else ''}'")
        print(f"  Valid: {result.is_valid}")
        print(f"  Sanitized: '{result.sanitized_input}'")
        if result.errors:
            print(f"  Errors: {result.errors}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
        print()
    
    # Test secure medical input
    print("Testing SecureMedicalInput:")
    try:
        secure_input = SecureMedicalInput(
            procedure="Endoscopy",
            details="Upper gastrointestinal endoscopy",
            objectives=["risks", "preparation", "post-procedure care"]
        )
        print(f"✅ Secure input created: {secure_input.to_dict()}")
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    try:
        malicious_input = SecureMedicalInput(
            procedure="<script>alert('hack')</script>",
            details="Malicious input"
        )
    except ValidationError as e:
        print(f"✅ Malicious input blocked: {e}")