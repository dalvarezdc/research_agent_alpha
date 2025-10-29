#!/usr/bin/env python3
"""
LLM Integration Module
Supports multiple LLM providers with fallback mechanisms using LangChain and DSPy.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from abc import ABC, abstractmethod
import os
from dataclasses import dataclass
import logging
from enum import Enum

import dspy
from pydantic import BaseModel

# Import TokenUsage from medical_reasoning_agent
from medical_reasoning_agent import TokenUsage

# Backwards compatible imports for LangChain
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    try:
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        # Fallback for very old versions
        from langchain.schema.messages import HumanMessage, SystemMessage

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    try:
        from langchain.chat_models import ChatAnthropic
    except ImportError:
        ChatAnthropic = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        ChatOpenAI = None

try:
    from langchain_community.llms import Ollama
except ImportError:
    try:
        from langchain.llms import Ollama
    except ImportError:
        Ollama = None


class LLMProvider(Enum):
    """Supported LLM providers"""
    CLAUDE = "claude"
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 3000
    timeout: int = 60


class MedicalQuerySignature(dspy.Signature):
    """DSPy signature for medical reasoning queries"""
    medical_input = dspy.InputField(desc="Medical procedure and context information")
    reasoning_stage = dspy.InputField(desc="Current stage of medical reasoning")
    context = dspy.InputField(desc="Additional context and previous reasoning steps")
    
    analysis = dspy.OutputField(desc="Detailed medical analysis for this reasoning stage")
    confidence = dspy.OutputField(desc="Confidence score (0.0-1.0) for the analysis")
    sources_needed = dspy.OutputField(desc="Additional sources or information needed")


class LLMInterface(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, TokenUsage]:
        """Generate response from LLM - returns (response, token_usage)"""
        pass

    @abstractmethod
    def medical_analysis(self, medical_input: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Specialized medical analysis method"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM provider is available"""
        pass


class ClaudeLLM(LLMInterface):
    """Claude (Anthropic) LLM implementation"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if ChatAnthropic is None:
            raise ImportError("ChatAnthropic not available. Install langchain-anthropic: pip install langchain-anthropic")
        
        try:
            self.client = ChatAnthropic(
                anthropic_api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"),
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Claude client: {e}")
            self.client = None
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, TokenUsage]:
        """Generate response using Claude"""
        if self.client is None:
            raise RuntimeError("Claude client not initialized")

        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            response = self.client.invoke(messages)

            # Extract token usage from response
            token_usage = TokenUsage()
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage.input_tokens = response.usage_metadata.get('input_tokens', 0)
                token_usage.output_tokens = response.usage_metadata.get('output_tokens', 0)
                token_usage.total_tokens = token_usage.input_tokens + token_usage.output_tokens
            elif hasattr(response, 'response_metadata') and response.response_metadata:
                usage = response.response_metadata.get('usage', {})
                token_usage.input_tokens = usage.get('input_tokens', 0)
                token_usage.output_tokens = usage.get('output_tokens', 0)
                token_usage.total_tokens = token_usage.input_tokens + token_usage.output_tokens

            return response.content, token_usage

        except Exception as e:
            self.logger.error(f"Claude API error: {str(e)}")
            raise
    
    def medical_analysis(self, medical_input: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Specialized medical analysis using Claude"""
        system_prompt = """You are a medical reasoning AI that provides systematic analysis
        of medical procedures. Focus on evidence-based recommendations and clearly distinguish
        between proven interventions, potential treatments, and debunked claims."""

        prompt = f"""
        Medical Input: {medical_input}
        Reasoning Stage: {stage}

        Provide analysis in this exact format:
        - Analysis: [detailed analysis]
        - Confidence: [0.0-1.0]
        - Sources Needed: [list of additional sources needed]
        """

        response, token_usage = self.generate_response(prompt, system_prompt)

        # Parse response (simplified - would need more robust parsing)
        return {
            "analysis": response,
            "confidence": 0.8,  # Would extract from response
            "sources_needed": [],
            "token_usage": token_usage
        }

    def is_available(self) -> bool:
        """Check if Claude is available"""
        if self.client is None:
            return False
        try:
            test_response, _ = self.generate_response("Test", "Respond with 'OK'")
            return "OK" in test_response
        except Exception:
            return False


class OpenAILLM(LLMInterface):
    """OpenAI LLM implementation"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if ChatOpenAI is None:
            raise ImportError("ChatOpenAI not available. Install langchain-openai: pip install langchain-openai")
        
        try:
            self.client = ChatOpenAI(
                openai_api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, TokenUsage]:
        """Generate response using OpenAI"""
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")

        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            response = self.client.invoke(messages)

            # Extract token usage from response
            token_usage = TokenUsage()
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage.input_tokens = response.usage_metadata.get('prompt_tokens', 0)
                token_usage.output_tokens = response.usage_metadata.get('completion_tokens', 0)
                token_usage.total_tokens = response.usage_metadata.get('total_tokens', 0)
            elif hasattr(response, 'response_metadata') and response.response_metadata:
                usage = response.response_metadata.get('token_usage', {})
                token_usage.input_tokens = usage.get('prompt_tokens', 0)
                token_usage.output_tokens = usage.get('completion_tokens', 0)
                token_usage.total_tokens = usage.get('total_tokens', 0)

            return response.content, token_usage

        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise

    def medical_analysis(self, medical_input: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Specialized medical analysis using OpenAI"""
        system_prompt = """You are a medical reasoning AI that provides systematic analysis
        of medical procedures with focus on organ-specific effects and evidence-based recommendations."""

        prompt = f"""
        Analyze this medical procedure:
        Input: {medical_input}
        Stage: {stage}

        Provide structured analysis with confidence scores.
        """

        response, token_usage = self.generate_response(prompt, system_prompt)

        return {
            "analysis": response,
            "confidence": 0.75,
            "sources_needed": [],
            "token_usage": token_usage
        }

    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        if self.client is None:
            return False
        try:
            test_response, _ = self.generate_response("Test", "Respond with 'OK'")
            return "OK" in test_response
        except Exception:
            return False


class OllamaLLM(LLMInterface):
    """Ollama local LLM implementation"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if Ollama is None:
            raise ImportError("Ollama not available. Install langchain-community: pip install langchain-community")
        
        try:
            self.client = Ollama(
                model=config.model,
                base_url=config.base_url or "http://localhost:11434",
                temperature=config.temperature
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama client: {e}")
            self.client = None
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, TokenUsage]:
        """Generate response using Ollama"""
        if self.client is None:
            raise RuntimeError("Ollama client not initialized")

        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = self.client.invoke(full_prompt)

            # Ollama may not provide token counts, so we estimate
            token_usage = TokenUsage()
            # Simple estimation: ~4 chars per token
            token_usage.input_tokens = len(full_prompt) // 4
            token_usage.output_tokens = len(response) // 4
            token_usage.total_tokens = token_usage.input_tokens + token_usage.output_tokens

            return response, token_usage

        except Exception as e:
            self.logger.error(f"Ollama error: {str(e)}")
            raise

    def medical_analysis(self, medical_input: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Medical analysis using local Ollama model"""
        prompt = f"""
        Medical Analysis Task:
        Input: {medical_input}
        Stage: {stage}

        Provide evidence-based medical analysis.
        """

        response, token_usage = self.generate_response(prompt)

        return {
            "analysis": response,
            "confidence": 0.7,  # Lower confidence for local models
            "sources_needed": [],
            "token_usage": token_usage
        }

    def is_available(self) -> bool:
        """Check if Ollama is available"""
        if self.client is None:
            return False
        try:
            test_response, _ = self.generate_response("Test: respond with OK")
            return len(test_response) > 0
        except Exception:
            return False


class LLMManager:
    """Manages multiple LLM providers with fallback mechanisms"""

    def __init__(self, configs: List[LLMConfig]):
        self.configs = configs
        self.providers: Dict[LLMProvider, LLMInterface] = {}
        self.current_provider: Optional[LLMProvider] = None
        self.logger = logging.getLogger(__name__)
        self.token_usage = TokenUsage()  # Track total token usage

        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured LLM providers"""
        for config in self.configs:
            try:
                if config.provider == LLMProvider.CLAUDE:
                    self.providers[config.provider] = ClaudeLLM(config)
                elif config.provider == LLMProvider.OPENAI:
                    self.providers[config.provider] = OpenAILLM(config)
                elif config.provider == LLMProvider.OLLAMA:
                    self.providers[config.provider] = OllamaLLM(config)
                
                self.logger.info(f"Initialized {config.provider.value} provider")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {config.provider.value}: {str(e)}")
    
    def get_available_provider(self) -> Optional[LLMInterface]:
        """Get first available LLM provider"""
        for provider_type, provider in self.providers.items():
            if provider.is_available():
                self.current_provider = provider_type
                self.logger.info(f"Using {provider_type.value} provider")
                return provider
        
        self.logger.error("No LLM providers available")
        return None
    
    def medical_analysis_with_fallback(self, medical_input: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Perform medical analysis with automatic fallback"""
        for provider_type, provider in self.providers.items():
            try:
                if provider.is_available():
                    self.logger.info(f"Attempting analysis with {provider_type.value}")
                    result = provider.medical_analysis(medical_input, stage)
                    result["provider_used"] = provider_type.value

                    # Accumulate token usage
                    if "token_usage" in result and result["token_usage"]:
                        self.token_usage.add(result["token_usage"])

                    return result

            except Exception as e:
                self.logger.warning(f"{provider_type.value} failed: {str(e)}, trying next provider")
                continue

        raise RuntimeError("All LLM providers failed")

    def get_token_usage(self) -> TokenUsage:
        """Get accumulated token usage"""
        return self.token_usage

    def reset_token_usage(self):
        """Reset token usage counter"""
        self.token_usage = TokenUsage()
    
    def setup_dspy_integration(self):
        """Setup DSPy with current LLM provider"""
        provider = self.get_available_provider()
        if not provider:
            raise RuntimeError("No LLM provider available for DSPy")
        
        # Configure DSPy with the available provider
        if self.current_provider == LLMProvider.CLAUDE:
            # DSPy Claude integration would go here
            pass
        elif self.current_provider == LLMProvider.OPENAI:
            dspy.configure(lm=dspy.OpenAI(model=self.configs[0].model))
        
        self.logger.info(f"DSPy configured with {self.current_provider.value}")


# Factory function for easy setup
def create_llm_manager(primary_provider: str = "claude", 
                      fallback_providers: List[str] = None) -> LLMManager:
    """Create LLM manager with default configurations"""
    
    if fallback_providers is None:
        fallback_providers = ["openai", "ollama"]
    
    configs = []
    
    # Primary provider
    if primary_provider == "claude":
        configs.append(LLMConfig(
            provider=LLMProvider.CLAUDE,
            model="claude-sonnet-4-5-20250929",
            temperature=0.1
        ))
    
    # Fallback providers
    for provider in fallback_providers:
        if provider == "openai":
            configs.append(LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                temperature=0.1
            ))
        elif provider == "ollama":
            configs.append(LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="llama2:13b",
                base_url="http://localhost:11434",
                temperature=0.1
            ))
    
    return LLMManager(configs)