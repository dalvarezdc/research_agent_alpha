#!/usr/bin/env python3
"""
Colored Logger Module
Provides enhanced logging with colors for different components and states.
"""

import sys
import logging

# Color codes for enhanced logging
ENABLE_COLOR = sys.stdout.isatty()
GREEN = "\033[32m" if ENABLE_COLOR else ""
BLUE = "\033[34m" if ENABLE_COLOR else ""
ORANGE = "\033[38;5;208m" if ENABLE_COLOR else ""
RED = "\033[31m" if ENABLE_COLOR else ""
YELLOW = "\033[38;5;226m" if ENABLE_COLOR else ""
PURPLE = "\033[35m" if ENABLE_COLOR else ""
CYAN = "\033[36m" if ENABLE_COLOR else ""
RESET = "\033[0m" if ENABLE_COLOR else ""


class ColoredLogger:
    """Enhanced logger with color coding for medical analysis components"""
    
    def __init__(self, name: str, enable_logging: bool = True):
        self.logger = logging.getLogger(name)
        if not enable_logging:
            self.logger.disabled = True
    
    # LLM Status Logging
    def llm_enabled(self, provider: str):
        """Log successful LLM initialization"""
        emoji = "🤖"
        message = f"{emoji} {GREEN}LLM integration enabled{RESET} - using {BLUE}{provider}{RESET} for enhanced analysis"
        self.logger.info(message)
    
    def llm_failed(self, provider: str, error: str):
        """Log LLM initialization failure"""
        emoji = "🔴"
        message = f"{emoji} {RED}LLM {provider} failed{RESET}: {error}"
        self.logger.error(message)
    
    def llm_offline_mode(self):
        """Log offline mode operation"""
        emoji = "🔧"
        message = f"{emoji} {ORANGE}Running in offline mode{RESET} - using built-in medical knowledge base"
        self.logger.info(message)
    
    # Web Research Status Logging
    def web_research_enabled(self):
        """Log successful web research initialization"""
        emoji = "🔍"
        message = f"{emoji} {GREEN}Tavily web research enabled{RESET} - will search authoritative medical sources"
        self.logger.info(message)
    
    def web_research_disabled(self, reason: str):
        """Log web research disabled"""
        emoji = "🔍"
        message = f"{emoji} {YELLOW}Web research disabled{RESET}: {reason}"
        self.logger.warning(message)
    
    def web_search_query(self, query: str):
        """Log web search queries"""
        emoji = "🔍"
        message = f"{emoji} {CYAN}Searching medical literature{RESET}: {query}"
        self.logger.info(message)
    
    # Analysis Pipeline Logging
    def analysis_start(self, procedure: str):
        """Log analysis start"""
        emoji = "🧠"
        message = f"{emoji} {GREEN}Starting analysis{RESET} of {BLUE}{procedure}{RESET}"
        self.logger.info(message)
    
    def analysis_stage(self, stage: str, description: str):
        """Log analysis pipeline stages"""
        emoji = "⚙️"
        message = f"{emoji} {PURPLE}[{stage}]{RESET} {description}"
        self.logger.info(message)
    
    def organs_identified(self, organs: list):
        """Log identified organs"""
        emoji = "🫀"
        organ_list = f"{GREEN}{', '.join(organs)}{RESET}"
        message = f"{emoji} Organs identified: {organ_list}"
        self.logger.info(message)
    
    def evidence_gathered(self, organ: str, quality: str):
        """Log evidence gathering"""
        emoji = "📚"
        quality_color = GREEN if quality == "strong" else ORANGE if quality == "moderate" else YELLOW
        message = f"{emoji} Evidence for {BLUE}{organ}{RESET}: {quality_color}{quality}{RESET} quality"
        self.logger.info(message)
    
    def recommendations_generated(self, organ: str, count: int):
        """Log recommendation generation"""
        emoji = "💡"
        message = f"{emoji} Generated {GREEN}{count}{RESET} recommendations for {BLUE}{organ}{RESET}"
        self.logger.info(message)
    
    # File Output Logging
    def file_saved(self, file_type: str, path: str):
        """Log file saves with full paths"""
        emoji_map = {
            "reasoning_trace": "📄",
            "analysis_result": "📊", 
            "summary_report": "📝",
            "comprehensive_report": "📋"
        }
        emoji = emoji_map.get(file_type, "📁")
        message = f"{emoji} {GREEN}{file_type.replace('_', ' ').title()}{RESET} saved to: {CYAN}{path}{RESET}"
        self.logger.info(message)
    
    # Error and Warning Logging
    def validation_error(self, error: str):
        """Log validation errors"""
        emoji = "❌"
        message = f"{emoji} {RED}Validation error{RESET}: {error}"
        self.logger.error(message)
    
    def fallback_mode(self, component: str, reason: str):
        """Log fallback to offline mode"""
        emoji = "🔄"
        message = f"{emoji} {YELLOW}{component} fallback{RESET}: {reason}"
        self.logger.warning(message)
    
    def analysis_complete(self, confidence: float, organs_count: int):
        """Log analysis completion"""
        emoji = "✅"
        confidence_color = GREEN if confidence >= 0.7 else ORANGE if confidence >= 0.5 else YELLOW
        message = f"{emoji} {GREEN}Analysis complete{RESET}! Confidence: {confidence_color}{confidence:.2f}{RESET}, Organs: {GREEN}{organs_count}{RESET}"
        self.logger.info(message)
    
    # Provider-specific authentication logging
    def provider_auth_success(self, provider: str):
        """Log successful provider authentication"""
        emoji = "🔐"
        message = f"{emoji} {GREEN}{provider} authenticated{RESET} successfully"
        self.logger.info(message)
    
    def provider_auth_failed(self, provider: str, reason: str):
        """Log provider authentication failure"""
        emoji = "🔴"
        message = f"{emoji} {RED}{provider} authentication failed{RESET}: {reason}"
        self.logger.error(message)
    
    def provider_unavailable(self, provider: str):
        """Log provider unavailable"""
        emoji = "⚠️"
        message = f"{emoji} {YELLOW}{provider} unavailable{RESET} - not installed or configured"
        self.logger.warning(message)
    
    # Generic logging methods with colors
    def info(self, message: str, component: str = None):
        """Generic info logging"""
        if component:
            message = f"{GREEN}[{component}]{RESET} {message}"
        self.logger.info(message)
    
    def warning(self, message: str, component: str = None):
        """Generic warning logging"""
        if component:
            message = f"{YELLOW}[{component}]{RESET} {message}"
        self.logger.warning(message)
    
    def error(self, message: str, component: str = None):
        """Generic error logging"""
        if component:
            message = f"{RED}[{component}]{RESET} {message}"
        self.logger.error(message)


# Convenience function to create colored logger
def get_colored_logger(name: str, enable_logging: bool = True) -> ColoredLogger:
    """Create a colored logger instance"""
    return ColoredLogger(name, enable_logging)


# Test the colored logger
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    logger = get_colored_logger(__name__)
    
    print("Testing colored logger:")
    logger.llm_enabled("claude")
    logger.llm_failed("openai", "API key not found")
    logger.llm_offline_mode()
    logger.web_research_enabled()
    logger.web_research_disabled("No API key")
    logger.analysis_start("MRI Scanner")
    logger.organs_identified(["kidneys", "brain"])
    logger.evidence_gathered("kidneys", "strong")
    logger.recommendations_generated("kidneys", 5)
    logger.file_saved("summary_report", "/path/to/report.md")
    logger.analysis_complete(0.85, 2)
    logger.provider_auth_success("Claude")
    logger.provider_auth_failed("OpenAI", "Invalid API key")