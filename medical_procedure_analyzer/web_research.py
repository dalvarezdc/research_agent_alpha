#!/usr/bin/env python3
"""
Web Research Module
Provides capabilities for searching and retrieving medical information from authoritative sources using Tavily Search.
"""

import os
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import time
import json
import re
from .colored_logger import get_colored_logger

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


class WebResearchAgent:
    """Agent for researching medical information using Tavily Search"""
    
    # Source mapping for efficient URL classification
    SOURCE_MAPPING = {
        ("pubmed", "ncbi.nlm.nih.gov"): "PubMed/NCBI",
        ("nih.gov",): "NIH", 
        ("medlineplus",): "MedlinePlus",
        ("fda.gov",): "FDA",
        ("cdc.gov",): "CDC",
        ("cochrane",): "Cochrane Library",
        ("mayoclinic",): "Mayo Clinic",
        ("nhs.uk",): "NHS",
        ("who.int",): "WHO",
        ("jamanetwork",): "JAMA",
        ("nejm.org",): "NEJM",
        ("thelancet",): "The Lancet",
        ("bmj.com",): "BMJ",
        ("acr.org",): "ACR",
        ("rsna.org",): "RSNA"
    }
    
    # Pre-compiled regex patterns for efficient text processing
    ORGAN_PATTERNS = [
        (re.compile(r'\b(kidney|renal|nephro)\w*\b', re.IGNORECASE), 'kidneys'),
        (re.compile(r'\b(liver|hepatic|hepato)\w*\b', re.IGNORECASE), 'liver'),
        (re.compile(r'\b(brain|neural|neuro|cerebral)\w*\b', re.IGNORECASE), 'brain'),
        (re.compile(r'\b(heart|cardiac|cardio)\w*\b', re.IGNORECASE), 'heart'),
        (re.compile(r'\b(lung|pulmonary|respiratory)\w*\b', re.IGNORECASE), 'lungs'),
        (re.compile(r'\b(thyroid|endocrine)\w*\b', re.IGNORECASE), 'thyroid'),
        (re.compile(r'\b(stomach|gastric|gastro)\w*\b', re.IGNORECASE), 'stomach'),
        (re.compile(r'\b(intestin|bowel|colon)\w*\b', re.IGNORECASE), 'intestines'),
        (re.compile(r'\b(bladder|urinary)\w*\b', re.IGNORECASE), 'bladder'),
        (re.compile(r'\b(pancrea)\w*\b', re.IGNORECASE), 'pancreas')
    ]
    
    RISK_PATTERNS = [
        re.compile(r'risk(?:s)? of ([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'contraindicated in ([^.;,]{10,60})', re.IGNORECASE),
        re.compile(r'caution in ([^.;,]{10,60})', re.IGNORECASE),
        re.compile(r'adverse effect(?:s)? (?:include )?([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'complication(?:s)? (?:include )?([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'warning[:\s]+([^.;,]{10,80})', re.IGNORECASE)
    ]
    
    RECOMMENDATION_PATTERNS = [
        re.compile(r'recommend(?:ed|s|ation)?[:\s]+([^.;,]{10,100})', re.IGNORECASE),
        re.compile(r'should ([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'guidelines? suggest ([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'evidence supports ([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'best practice[:\s]+([^.;,]{10,80})', re.IGNORECASE)
    ]
    
    PREPARATION_PATTERNS = [
        re.compile(r'before (?:the )?procedure[:\s]+([^.;,]{10,100})', re.IGNORECASE),
        re.compile(r'preparation[:\s]+([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'pre-procedure[:\s]+([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'patient should ([^.;,]{10,80}) before', re.IGNORECASE),
        re.compile(r'prior to (?:the )?procedure[:\s]+([^.;,]{10,80})', re.IGNORECASE)
    ]
    
    POST_CARE_PATTERNS = [
        re.compile(r'after (?:the )?procedure[:\s]+([^.;,]{10,100})', re.IGNORECASE),
        re.compile(r'post-procedure[:\s]+([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'following (?:the )?procedure[:\s]+([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'monitor(?:ing)?[:\s]+([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'recovery[:\s]+([^.;,]{10,80})', re.IGNORECASE)
    ]
    
    CONTRAINDICATION_PATTERNS = [
        re.compile(r'contraindicated? in ([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'should not be used in ([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'avoid in ([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'warning[:\s]+([^.;,]{10,80})', re.IGNORECASE),
        re.compile(r'not recommended for ([^.;,]{10,80})', re.IGNORECASE)
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = get_colored_logger(__name__)
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        
        if TavilyClient is None:
            self.logger.web_research_disabled("Tavily client not available. Install with: pip install tavily-python")
            self.client = None
            return
        
        if not self.api_key:
            self.logger.web_research_disabled("No API key found. Set TAVILY_API_KEY environment variable")
            self.client = None
            return
        
        self.client = TavilyClient(api_key=self.api_key)
        self.logger.web_research_enabled()
        
    def search_medical_procedure(self, procedure: str, specific_aspects: List[str] = None) -> Dict[str, Any]:
        """
        Search for comprehensive information about a medical procedure using Tavily
        
        Args:
            procedure: Medical procedure name (e.g., "Endoscopy", "MRI with gadolinium")
            specific_aspects: Specific aspects to focus on (e.g., ["risks", "preparation", "organs affected"])
            
        Returns:
            Dictionary containing research results from authoritative sources
        """
        self.logger.info(f"Researching medical procedure: {procedure}")
        
        research_results = {
            "procedure": procedure,
            "timestamp": time.time(),
            "sources_consulted": [],
            "organ_systems": [],
            "risks_identified": [],
            "recommendations": {
                "evidence_based": [],
                "investigational": [],
                "contraindicated": []
            },
            "preparation_guidelines": [],
            "post_procedure_care": [],
            "contraindications": [],
            "evidence_quality": "moderate",
            "research_confidence": 0.0,
            "raw_results": []
        }
        
        # Generate search queries
        search_queries = self._generate_search_queries(procedure, specific_aspects)
        
        all_results = []
        
        # Check if Tavily is available
        if not self.client:
            self.logger.fallback_mode("Web Research", "Tavily unavailable - returning minimal research data")
            research_results["research_confidence"] = 0.1
            return research_results
        
        # Perform searches with Tavily
        for query in search_queries:
            try:
                self.logger.web_search_query(query)
                
                # Search with medical domain focus
                response = self.client.search(
                    query=query,
                    search_depth="advanced",
                    include_domains=[
                        "pubmed.ncbi.nlm.nih.gov",
                        "nih.gov",
                        "ncbi.nlm.nih.gov", 
                        "medlineplus.gov", 
                        "fda.gov",
                        "cdc.gov",
                        "cochranelibrary.com",
                        "uptodate.com",
                        "mayoclinic.org",
                        "nhs.uk",
                        "who.int",
                        "jamanetwork.com",
                        "nejm.org",
                        "thelancet.com",
                        "bmj.com",
                        "acr.org",
                        "rsna.org"
                    ],
                    max_results=15
                )
                
                if response and "results" in response:
                    all_results.extend(response["results"])
                    self.logger.info(f"Found {len(response['results'])} results for: {query}")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                self.logger.warning(f"Search failed for query '{query}': {str(e)}")
                continue
        
        # Process and integrate all results
        if all_results:
            research_results["raw_results"] = all_results
            self._process_tavily_results(research_results, all_results)
            research_results["research_confidence"] = self._calculate_confidence_from_tavily(research_results, all_results)
        else:
            self.logger.warning(f"No search results found for {procedure}")
            research_results["research_confidence"] = 0.1
        
        return research_results
    
    def _identify_source(self, url: str) -> str:
        """Efficiently identify source from URL using mapping"""
        url_lower = url.lower()
        
        for domains, source_name in self.SOURCE_MAPPING.items():
            if any(domain in url_lower for domain in domains):
                return source_name
        
        return "Medical Literature"
    
    def _generate_search_queries(self, procedure: str, aspects: List[str] = None) -> List[str]:
        """Generate targeted search queries for the procedure"""
        base_queries = [
            f"{procedure} medical procedure risks complications",
            f"{procedure} clinical guidelines preparation",
            f"{procedure} organ systems affected", 
            f"{procedure} post procedure care monitoring"
        ]
        
        if aspects:
            for aspect in aspects:
                base_queries.append(f"{procedure} {aspect} medical evidence")
        
        return base_queries[:4]  # Limit to avoid rate limits
    
    def _process_tavily_results(self, research_results: Dict[str, Any], tavily_results: List[Dict[str, Any]]):
        """Process and integrate Tavily search results into research data"""
        sources = set()
        
        for result in tavily_results:
            url = result.get("url", "")
            title = result.get("title", "")
            content = result.get("content", "")
            
            # Track sources using mapping
            sources.add(self._identify_source(url))
            
            # Extract information from content
            combined_text = f"{title} {content}".lower()
            
            # Extract organ systems
            organs = self._extract_organs_from_text(combined_text)
            research_results["organ_systems"].extend(organs)
            
            # Extract risks
            risks = self._extract_risks_from_text(combined_text)
            research_results["risks_identified"].extend(risks)
            
            # Extract recommendations
            recommendations = self._extract_recommendations_from_text(combined_text)
            
            # Categorize recommendations based on source authority
            if any(domain in url for domain in ["pubmed", "cochrane", "fda"]):
                research_results["recommendations"]["evidence_based"].extend(recommendations)
            else:
                research_results["recommendations"]["investigational"].extend(recommendations)
            
            # Extract preparation guidelines
            prep_guidelines = self._extract_preparation_from_text(combined_text)
            research_results["preparation_guidelines"].extend(prep_guidelines)
            
            # Extract post-procedure care
            post_care = self._extract_post_care_from_text(combined_text)
            research_results["post_procedure_care"].extend(post_care)
            
            # Extract contraindications
            contraindications = self._extract_contraindications_from_text(combined_text)
            research_results["contraindications"].extend(contraindications)
        
        # Remove duplicates and update sources
        research_results["sources_consulted"] = list(sources)
        research_results["organ_systems"] = list(set(research_results["organ_systems"]))
        research_results["risks_identified"] = list(set(research_results["risks_identified"]))
        
        # Limit results to prevent overwhelming output
        for key in ["preparation_guidelines", "post_procedure_care", "contraindications"]:
            research_results[key] = list(set(research_results[key]))[:5]
        
        for rec_type in research_results["recommendations"]:
            research_results["recommendations"][rec_type] = list(set(research_results["recommendations"][rec_type]))[:5]
    
    def _extract_organs_from_text(self, text: str) -> List[str]:
        """Extract organ systems mentioned in medical text using pre-compiled patterns"""
        organs = []
        for pattern, organ_name in self.ORGAN_PATTERNS:
            if pattern.search(text):
                organs.append(organ_name)
        
        return organs
    
    def _extract_risks_from_text(self, text: str) -> List[str]:
        """Extract risk factors from medical text using pre-compiled patterns"""
        risks = []
        for pattern in self.RISK_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if len(match.strip()) > 5:  # Filter out very short matches
                    risks.append(match.strip())
        
        return risks[:10]  # Limit to top 10 risks
    
    def _extract_recommendations_from_text(self, text: str) -> List[str]:
        """Extract recommendations from medical text using pre-compiled patterns"""
        recommendations = []
        for pattern in self.RECOMMENDATION_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if len(match.strip()) > 5:
                    recommendations.append(match.strip())
        
        return recommendations[:10]
    
    def _extract_preparation_from_text(self, text: str) -> List[str]:
        """Extract preparation guidelines from text using pre-compiled patterns"""
        preparations = []
        for pattern in self.PREPARATION_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if len(match.strip()) > 5:
                    preparations.append(match.strip())
        
        return preparations
    
    def _extract_post_care_from_text(self, text: str) -> List[str]:
        """Extract post-procedure care from text using pre-compiled patterns"""
        post_care = []
        for pattern in self.POST_CARE_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if len(match.strip()) > 5:
                    post_care.append(match.strip())
        
        return post_care
    
    def _extract_contraindications_from_text(self, text: str) -> List[str]:
        """Extract contraindications from text using pre-compiled patterns"""
        contraindications = []
        for pattern in self.CONTRAINDICATION_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if len(match.strip()) > 5:
                    contraindications.append(match.strip())
        
        return contraindications
    
    def _calculate_confidence_from_tavily(self, research_results: Dict[str, Any], tavily_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on Tavily search results"""
        base_confidence = 0.3
        
        # Increase confidence based on number of results
        results_count = len(tavily_results)
        confidence_bonus = min(results_count * 0.05, 0.3)
        
        # Increase confidence based on authoritative sources
        authoritative_sources = {"PubMed/NCBI", "NIH", "FDA", "CDC", "Cochrane Library", "NHS", "WHO", "JAMA", "NEJM", "The Lancet", "BMJ"}
        sources_found = set(research_results["sources_consulted"])
        authority_bonus = len(sources_found.intersection(authoritative_sources)) * 0.08
        
        # Increase confidence if we found comprehensive information
        if research_results["organ_systems"]:
            confidence_bonus += 0.1
        if research_results["risks_identified"]:
            confidence_bonus += 0.1
        if research_results["recommendations"]["evidence_based"]:
            confidence_bonus += 0.15
        
        final_confidence = min(base_confidence + confidence_bonus + authority_bonus, 0.95)
        return round(final_confidence, 2)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Check if API key is available
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  TAVILY_API_KEY environment variable not set")
        print("To test, set your Tavily API key: export TAVILY_API_KEY='your-key-here'")
        exit(1)
    
    try:
        agent = WebResearchAgent()
        
        # Test with endoscopy
        results = agent.search_medical_procedure(
            "Endoscopy", 
            ["risks", "preparation", "organs affected", "post-procedure care"]
        )
        
        print(f"\n🔬 Research Results for Endoscopy:")
        print(f"📚 Sources consulted: {results['sources_consulted']}")
        print(f"🫀 Organs identified: {results['organ_systems']}")
        print(f"⚠️  Risks found: {len(results['risks_identified'])}")
        print(f"💡 Evidence-based recommendations: {len(results['recommendations']['evidence_based'])}")
        print(f"📊 Confidence: {results['research_confidence']}")
        
        # Print a few examples
        if results['risks_identified']:
            print(f"\nExample risks:")
            for risk in results['risks_identified'][:3]:
                print(f"  • {risk}")
        
        if results['recommendations']['evidence_based']:
            print(f"\nExample recommendations:")
            for rec in results['recommendations']['evidence_based'][:3]:
                print(f"  • {rec}")
                
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")