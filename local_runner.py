#!/usr/bin/env python3
"""
Local runner for testing the Medical Reasoning Agent
Allows testing different scenarios and LLM providers locally
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List
from pathlib import Path

from simple_medical_agent import SimpleMedicalAgent, create_simple_agent
from medical_reasoning_agent import MedicalInput
from llm_integrations import create_llm_manager, LLMConfig, LLMProvider
from web_research import WebResearchAgent


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('medical_agent.log')
        ]
    )


def create_dynamic_scenario(procedure_name: str, details: str = "") -> Dict[str, Any]:
    """Create a dynamic scenario from user-provided procedure name"""
    return {
        "name": procedure_name.replace(" ", "_"),
        "description": f"Analysis of {procedure_name} medical procedure",
        "input": {
            "procedure": procedure_name,
            "details": details or f"Standard {procedure_name} procedure",
            "objectives": [
                "understand implications",
                "risks and complications",
                "preparation requirements", 
                "organs affected by procedure",
                "post-procedure care",
                "contraindications"
            ]
        },
        "dynamic": True,  # Flag to indicate this is dynamically created
        "requires_web_research": True
    }

def load_test_scenarios(scenarios_file: str = "test_scenarios.json") -> List[Dict[str, Any]]:
    """Load test scenarios from JSON file"""
    scenarios_path = Path(scenarios_file)
    
    if not scenarios_path.exists():
        # Create default scenarios if file doesn't exist
        default_scenarios = [
            {
                "name": "MRI_with_gadolinium",
                "description": "MRI scan with gadolinium contrast agent",
                "input": {
                    "procedure": "MRI Scanner",
                    "details": "With gadolinium contrast",
                    "objectives": [
                        "understand implications",
                        "risks", 
                        "what to do after to prevent and reduce risks",
                        "organs affected by procedure",
                        "organs at risk"
                    ]
                },
                "expected_focus": ["kidneys", "brain"],
                "test_assertions": [
                    "Should identify kidney elimination pathway",
                    "Should recommend hydration",
                    "Should mention gadolinium retention risks"
                ]
            },
            {
                "name": "CT_with_iodine",
                "description": "CT scan with iodine contrast",
                "input": {
                    "procedure": "CT Scan",
                    "details": "With iodine contrast",
                    "objectives": [
                        "nephrotoxicity risks",
                        "thyroid effects", 
                        "pre and post procedure care"
                    ]
                },
                "expected_focus": ["kidneys", "thyroid"],
                "test_assertions": [
                    "Should warn about contrast-induced nephropathy",
                    "Should mention thyroid considerations",
                    "Should recommend kidney function monitoring"
                ]
            },
            {
                "name": "Cardiac_catheterization",
                "description": "Cardiac catheterization with contrast",
                "input": {
                    "procedure": "Cardiac Catheterization",
                    "details": "With iodine contrast for coronary angiography",
                    "objectives": [
                        "cardiac risks",
                        "kidney protection",
                        "post-procedure monitoring"
                    ]
                },
                "expected_focus": ["heart", "kidneys", "blood_vessels"],
                "test_assertions": [
                    "Should address cardiac risks",
                    "Should emphasize kidney protection",
                    "Should recommend cardiac monitoring"
                ]
            }
        ]
        
        with open(scenarios_path, 'w') as f:
            json.dump(default_scenarios, f, indent=2)
        
        print(f"Created default scenarios file: {scenarios_path}")
        return default_scenarios
    
    with open(scenarios_path, 'r') as f:
        return json.load(f)


def run_single_scenario(scenario: Dict[str, Any], agent: MedicalReasoningAgent, 
                       output_dir: Path) -> Dict[str, Any]:
    """Run a single test scenario"""
    print(f"\n{'='*60}")
    print(f"Running Scenario: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    print(f"{'='*60}")
    
    # Create medical input - convert objectives list to tuple
    input_data = scenario['input'].copy()
    input_data['objectives'] = tuple(input_data['objectives'])
    medical_input = MedicalInput(**input_data)
    
    try:
        # Run analysis
        result = agent.analyze_procedure(medical_input)
        
        # Create output directory for this scenario
        scenario_dir = output_dir / scenario['name']
        scenario_dir.mkdir(exist_ok=True)
        
        # Export detailed results
        results_file = scenario_dir / "analysis_result.json"
        with open(results_file, 'w') as f:
            json.dump({
                "scenario": scenario,
                "procedure_summary": result.procedure_summary,
                "organs_analyzed": [
                    {
                        "organ_name": organ.organ_name,
                        "affected_by_procedure": organ.affected_by_procedure,
                        "at_risk": organ.at_risk,
                        "risk_level": organ.risk_level,
                        "pathways_involved": organ.pathways_involved,
                        "known_recommendations": organ.known_recommendations,
                        "potential_recommendations": organ.potential_recommendations,
                        "debunked_claims": organ.debunked_claims,
                        "evidence_quality": organ.evidence_quality
                    }
                    for organ in result.organs_analyzed
                ],
                "general_recommendations": result.general_recommendations,
                "research_gaps": result.research_gaps,
                "confidence_score": result.confidence_score
            }, f, indent=2)
        
        # Export reasoning trace
        trace_file = scenario_dir / "reasoning_trace.json"
        agent.export_reasoning_trace(str(trace_file))
        
        # ADDITIONAL: Generate enhanced reports (keeping all existing outputs)
        try:
            from report_generator import generate_reports
            from validation_scoring import validate_medical_output
            
            print(f"Generating additional report formats...")
            
            # Validate the results
            validation_report = validate_medical_output(result)
            validation_dict = {
                "overall_score": validation_report.overall_score,
                "safety_score": validation_report.safety_score,
                "accuracy_score": validation_report.accuracy_score,
                "completeness_score": validation_report.completeness_score,
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "category": issue.category.value,
                        "message": issue.message,
                        "suggested_fix": issue.suggested_fix
                    }
                    for issue in validation_report.issues
                ]
            }
            
            # Generate additional report formats (supplementing existing files)
            additional_reports = generate_reports(
                result, 
                validation_dict, 
                str(scenario_dir),
                f"{scenario['name']}_report"
            )
            
            print(f"✅ Enhanced reports generated:")
            for report_type, path in additional_reports.items():
                print(f"    📄 {report_type}: {Path(path).name}")
                
        except Exception as e:
            print(f"⚠️  Enhanced reports not generated: {e}")
            print("    (Original outputs still available)")
        
        # Display key results
        print(f"\nProcedure: {result.procedure_summary}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Organs Analyzed: {len(result.organs_analyzed)}")
        
        for organ in result.organs_analyzed:
            print(f"\n  🔍 {organ.organ_name.upper()}:")
            print(f"    Risk Level: {organ.risk_level}")
            print(f"    Known Recommendations: {len(organ.known_recommendations)}")
            print(f"    Potential Recommendations: {len(organ.potential_recommendations)}")
            print(f"    Debunked Claims: {len(organ.debunked_claims)}")
        
        print(f"\nReasoning Stages: {len(result.reasoning_trace)}")
        for step in result.reasoning_trace:
            print(f"  • {step.stage.value}: {step.reasoning[:100]}...")
        
        print(f"\nResults saved to: {scenario_dir}")
        
        return {
            "scenario_name": scenario['name'],
            "status": "success",
            "confidence_score": result.confidence_score,
            "organs_count": len(result.organs_analyzed),
            "reasoning_steps": len(result.reasoning_trace),
            "output_dir": str(scenario_dir)
        }
        
    except Exception as e:
        error_msg = f"Error running scenario {scenario['name']}: {str(e)}"
        print(f"\n❌ {error_msg}")
        
        return {
            "scenario_name": scenario['name'],
            "status": "error",
            "error": error_msg
        }


def run_comparison_test(scenarios: List[Dict[str, Any]], llm_providers: List[str],
                       output_dir: Path):
    """Run comparison test across multiple LLM providers"""
    print(f"\n{'='*80}")
    print("RUNNING PROVIDER COMPARISON TEST")
    print(f"{'='*80}")
    
    comparison_results = {}
    
    for provider in llm_providers:
        print(f"\n🤖 Testing with {provider.upper()} provider...")
        
        try:
            # Create agent with specific provider
            agent = create_simple_agent(
                llm_provider=provider,
                enable_logging=True
            )
            
            provider_results = []
            
            for scenario in scenarios[:2]:  # Test with first 2 scenarios
                result = run_single_scenario(scenario, agent, output_dir / provider)
                provider_results.append(result)
            
            comparison_results[provider] = {
                "provider": provider,
                "scenarios_tested": len(provider_results),
                "success_rate": sum(1 for r in provider_results if r["status"] == "success") / len(provider_results),
                "avg_confidence": sum(r.get("confidence_score", 0) for r in provider_results if r["status"] == "success") / max(1, sum(1 for r in provider_results if r["status"] == "success")),
                "results": provider_results
            }
            
        except Exception as e:
            comparison_results[provider] = {
                "provider": provider,
                "error": str(e),
                "success_rate": 0.0
            }
    
    # Save comparison results
    comparison_file = output_dir / "provider_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Display comparison summary
    print(f"\n📊 PROVIDER COMPARISON SUMMARY:")
    print("-" * 50)
    for provider, results in comparison_results.items():
        if "error" not in results:
            print(f"{provider.upper():>12}: Success Rate: {results['success_rate']:.1%}, "
                  f"Avg Confidence: {results['avg_confidence']:.2f}")
        else:
            print(f"{provider.upper():>12}: ERROR - {results['error']}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Local Medical Reasoning Agent Runner")
    parser.add_argument("--scenarios", "-s", default="test_scenarios.json",
                       help="JSON file with test scenarios")
    parser.add_argument("--provider", "-p", default="claude",
                       choices=["claude", "openai", "ollama"],
                       help="Primary LLM provider to use")
    parser.add_argument("--output", "-o", default="test_outputs",
                       help="Output directory for results")
    parser.add_argument("--compare", "-c", action="store_true",
                       help="Run comparison across multiple providers")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--scenario-name", "-n", type=str,
                       help="Run only specific scenario by name")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("🧠 Medical Reasoning Agent - Local Runner")
    print(f"Output Directory: {output_dir.absolute()}")
    
    # Load scenarios
    scenarios = load_test_scenarios(args.scenarios)
    print(f"Loaded {len(scenarios)} test scenarios")
    
    # Filter specific scenario if requested OR create dynamic scenario
    if args.scenario_name:
        # First try to find existing scenario
        existing_scenarios = [s for s in scenarios if s['name'] == args.scenario_name]
        if existing_scenarios:
            scenarios = existing_scenarios
        else:
            # Create dynamic scenario from the name
            print(f"📋 Scenario '{args.scenario_name}' not found in predefined scenarios.")
            print(f"🔬 Creating dynamic scenario for medical procedure: {args.scenario_name}")
            
            # Validate the scenario name first
            from input_validation import InputValidator, ValidationError
            validation_result = InputValidator.validate_scenario_name(args.scenario_name)
            
            if not validation_result.is_valid:
                print(f"❌ Invalid scenario name: {', '.join(validation_result.errors)}")
                return
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    print(f"⚠️  Warning: {warning}")
            
            # Use sanitized name
            sanitized_scenario_name = validation_result.sanitized_input
            print(f"✅ Using sanitized scenario name: {sanitized_scenario_name}")
            
            # Initialize web research agent
            web_agent = WebResearchAgent()
            
            try:
                # Research the procedure to gather initial information
                print(f"🌐 Researching {sanitized_scenario_name} from medical literature...")
                research_results = web_agent.search_medical_procedure(sanitized_scenario_name)
                
                # Create enhanced scenario with research data
                details = f"Researched from {len(research_results['sources_consulted'])} authoritative sources"
                if research_results['organ_systems']:
                    details += f", affects: {', '.join(set(research_results['organ_systems']))}"
                
                dynamic_scenario = create_dynamic_scenario(sanitized_scenario_name, details)
                dynamic_scenario["research_data"] = research_results  # Include research data
                
                scenarios = [dynamic_scenario]
                
                print(f"✅ Dynamic scenario created successfully!")
                print(f"📊 Research confidence: {research_results['research_confidence']}")
                print(f"📚 Sources consulted: {', '.join(research_results['sources_consulted'])}")
                if research_results['organ_systems']:
                    print(f"🫀 Organs identified: {', '.join(set(research_results['organ_systems']))}")
                
            except Exception as e:
                print(f"⚠️  Web research failed: {str(e)}")
                print(f"🔄 Creating basic dynamic scenario without web research...")
                scenarios = [create_dynamic_scenario(sanitized_scenario_name)]
    
    if args.compare:
        # Run comparison test
        providers = ["claude", "openai", "ollama"]
        run_comparison_test(scenarios, providers, output_dir)
    else:
        # Run with single provider
        print(f"🤖 Using {args.provider.upper()} as primary LLM provider")
        
        agent = create_simple_agent(
            llm_provider=args.provider,
            enable_logging=True
        )
        
        # Run all scenarios
        results = []
        for scenario in scenarios:
            result = run_single_scenario(scenario, agent, output_dir)
            results.append(result)
        
        # Summary
        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"\n📈 SUMMARY:")
        print(f"Total Scenarios: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(results) - success_count}")
        print(f"Success Rate: {success_count/len(results):.1%}")
        
        if success_count > 0:
            avg_confidence = sum(r.get("confidence_score", 0) for r in results if r["status"] == "success") / success_count
            print(f"Average Confidence: {avg_confidence:.2f}")


if __name__ == "__main__":
    main()