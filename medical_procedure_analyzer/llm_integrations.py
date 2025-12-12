#!/usr/bin/env python3
"""
LLM Integration Module
Supports multiple LLM providers with fallback mechanisms using LangChain and DSPy.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from abc import ABC, abstractmethod
import os
from dataclasses import dataclass
import logging
from enum import Enum
import time
from functools import wraps

import dspy
from pydantic import BaseModel

# Import TokenUsage from medical_reasoning_agent
from .medical_reasoning_agent import TokenUsage


def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Don't retry on final attempt
                    if attempt == max_retries:
                        break

                    # Log the retry attempt
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    # Wait before retrying
                    time.sleep(delay)
                    delay *= backoff_factor

            # All retries exhausted, raise the last exception
            raise last_exception

        return wrapper
    return decorator

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

try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user as xai_user
except ImportError:
    XAIClient = None
    xai_user = None


class LLMProvider(Enum):
    """Supported LLM providers"""
    CLAUDE_SONNET = "claude-sonnet"
    CLAUDE_OPUS = "claude-opus"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GROK_41_FAST = "grok-4-1-fast"
    GROK_41_CODE = "grok-4-1-code"
    GROK_41_REASONING = "grok-4-1-reasoning"


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096  # Increased for detailed responses
    timeout: int = 300  # 5 minutes for complex medical analysis


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
            self.logger.warning(f"Failed to initialize Claude client (fallback provider): {e}")
            self.client = None

    @retry_with_backoff(max_retries=1, initial_delay=1.0, backoff_factor=2.0)
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, TokenUsage]:
        """Generate response using Claude"""
        if self.client is None:
            raise RuntimeError("Claude client not initialized")

        try:
            # Record model usage for cost tracking
            try:
                from cost_tracker import record_model_usage
                record_model_usage(self.config.model)
            except:
                pass  # Cost tracking optional

            # Build professional system prompt
            base_system_prompt = "You are a professional assistant. Respond in a formal, concise, and objective manner without humor or casual language."
            if system_prompt:
                full_system_prompt = f"{base_system_prompt}\n\n{system_prompt}"
            else:
                full_system_prompt = base_system_prompt

            # Prefix user message with professional tone instruction
            professional_prompt = f"Please answer this in a professional tone: {prompt}"

            messages = []
            messages.append(SystemMessage(content=full_system_prompt))
            messages.append(HumanMessage(content=professional_prompt))

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
            self.logger.warning(f"Failed to initialize OpenAI client (fallback provider): {e}")
            self.client = None

    @retry_with_backoff(max_retries=1, initial_delay=1.0, backoff_factor=2.0)
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, TokenUsage]:
        """Generate response using OpenAI"""
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")

        try:
            # Record model usage for cost tracking
            try:
                from cost_tracker import record_model_usage
                record_model_usage(self.config.model)
            except:
                pass  # Cost tracking optional

            # Build professional system prompt
            base_system_prompt = "You are a professional assistant. Respond in a formal, concise, and objective manner without humor or casual language."
            if system_prompt:
                full_system_prompt = f"{base_system_prompt}\n\n{system_prompt}"
            else:
                full_system_prompt = base_system_prompt

            # Prefix user message with professional tone instruction
            professional_prompt = f"Please answer this in a professional tone: {prompt}"

            messages = []
            messages.append(SystemMessage(content=full_system_prompt))
            messages.append(HumanMessage(content=professional_prompt))

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
            self.logger.warning(f"Failed to initialize Ollama client (fallback provider): {e}")
            self.client = None

    @retry_with_backoff(max_retries=1, initial_delay=1.0, backoff_factor=2.0)
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, TokenUsage]:
        """Generate response using Ollama"""
        if self.client is None:
            raise RuntimeError("Ollama client not initialized")

        try:
            # Record model usage for cost tracking
            try:
                from cost_tracker import record_model_usage
                record_model_usage(self.config.model)
            except:
                pass  # Cost tracking optional

            # Build professional system prompt
            base_system_prompt = "You are a professional assistant. Respond in a formal, concise, and objective manner without humor or casual language."
            if system_prompt:
                full_system_prompt = f"{base_system_prompt}\n\n{system_prompt}"
            else:
                full_system_prompt = base_system_prompt

            # Prefix user message with professional tone instruction
            professional_prompt = f"Please answer this in a professional tone: {prompt}"

            full_prompt = f"{full_system_prompt}\n\n{professional_prompt}"

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


class XaiLLM(LLMInterface):
    """xAI Grok LLM implementation"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if XAIClient is None:
            raise ImportError("XAIClient not available. Install xai-sdk: pip install xai-sdk")

        try:
            self.client = XAIClient(
                api_key=config.api_key or os.getenv("GROK_API_KEY"),
                timeout=config.timeout
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize xAI client (fallback provider): {e}")
            self.client = None

    @retry_with_backoff(max_retries=1, initial_delay=1.0, backoff_factor=2.0)
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, TokenUsage]:
        """Generate response using xAI Grok"""
        if self.client is None:
            raise RuntimeError("xAI client not initialized")

        try:
            # Record model usage for cost tracking
            try:
                from cost_tracker import record_model_usage
                record_model_usage(self.config.model)
            except:
                pass  # Cost tracking optional

            # Create chat session
            chat = self.client.chat.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            # Build professional system prompt
            base_system_prompt = "You are a professional assistant. Respond in a formal, concise, and objective manner without humor or casual language."
            if system_prompt:
                full_system_prompt = f"{base_system_prompt}\n\n{system_prompt}"
            else:
                full_system_prompt = base_system_prompt

            # Prefix user message with professional tone instruction
            professional_prompt = f"Please answer this in a professional tone: {prompt}"

            # Build the full prompt with system message
            full_prompt = f"{full_system_prompt}\n\n{professional_prompt}"

            # Append user message and sample response
            chat.append(xai_user(full_prompt))
            response = chat.sample()

            # Extract token usage from response
            token_usage = TokenUsage()
            if hasattr(response, 'usage') and response.usage:
                # Try dictionary access first, then attribute access
                if isinstance(response.usage, dict):
                    token_usage.input_tokens = response.usage.get('prompt_tokens', 0)
                    token_usage.output_tokens = response.usage.get('completion_tokens', 0)
                    token_usage.total_tokens = response.usage.get('total_tokens', 0)
                else:
                    # Access as object attributes
                    token_usage.input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    token_usage.output_tokens = getattr(response.usage, 'completion_tokens', 0)
                    token_usage.total_tokens = getattr(response.usage, 'total_tokens', 0)

                # Calculate total if not provided
                if not token_usage.total_tokens:
                    token_usage.total_tokens = token_usage.input_tokens + token_usage.output_tokens

            # Get the text content from response
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                content = response.message.content
            elif hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)

            return content, token_usage

        except Exception as e:
            self.logger.error(f"xAI API error: {str(e)}")
            raise

    def medical_analysis(self, medical_input: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Specialized medical analysis using xAI Grok"""
        system_prompt = """You are a medical reasoning AI that provides systematic analysis
        of medical procedures with focus on evidence-based recommendations and detailed analysis."""

        prompt = f"""
        Analyze this medical procedure:
        Input: {medical_input}
        Stage: {stage}

        Provide structured analysis with confidence scores and evidence-based recommendations.
        """

        response, token_usage = self.generate_response(prompt, system_prompt)

        return {
            "analysis": response,
            "confidence": 0.8,
            "sources_needed": [],
            "token_usage": token_usage
        }

    def is_available(self) -> bool:
        """Check if xAI is available"""
        if self.client is None:
            return False
        try:
            test_response, _ = self.generate_response("Test", "Respond with 'OK'")
            return "OK" in test_response or len(test_response) > 0
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
                if config.provider in [LLMProvider.CLAUDE_SONNET, LLMProvider.CLAUDE_OPUS]:
                    self.providers[config.provider] = ClaudeLLM(config)
                elif config.provider == LLMProvider.OPENAI:
                    self.providers[config.provider] = OpenAILLM(config)
                elif config.provider == LLMProvider.OLLAMA:
                    self.providers[config.provider] = OllamaLLM(config)
                elif config.provider in [LLMProvider.GROK_41_FAST, LLMProvider.GROK_41_CODE, LLMProvider.GROK_41_REASONING]:
                    self.providers[config.provider] = XaiLLM(config)

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
        if self.current_provider in [LLMProvider.CLAUDE_SONNET, LLMProvider.CLAUDE_OPUS]:
            # Configure DSPy with Claude/Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")

            # DSPy 3.x uses different import for Anthropic
            lm = dspy.Claude(
                model=self.configs[0].model,
                api_key=api_key,
                temperature=self.configs[0].temperature,
                max_tokens=self.configs[0].max_tokens
            )
            dspy.settings.configure(lm=lm)

        elif self.current_provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            lm = dspy.OpenAI(
                model=self.configs[0].model,
                api_key=api_key,
                temperature=self.configs[0].temperature,
                max_tokens=self.configs[0].max_tokens
            )
            dspy.settings.configure(lm=lm)

        self.logger.info(f"DSPy configured with {self.current_provider.value}")
        return True


# Factory function for easy setup
def create_llm_manager(primary_provider: str = "claude-sonnet",
                      fallback_providers: List[str] = None) -> LLMManager:
    """Create LLM manager with default configurations"""

    if fallback_providers is None:
        fallback_providers = ["openai", "ollama"]
    
    configs = []
    
    # Primary provider
    if primary_provider == "claude-sonnet":
        configs.append(LLMConfig(
            provider=LLMProvider.CLAUDE_SONNET,
            model="claude-sonnet-4-5-20250929",
            temperature=0.1
        ))
    elif primary_provider == "claude-opus":
        configs.append(LLMConfig(
            provider=LLMProvider.CLAUDE_OPUS,
            model="claude-opus-4-5-20251101",
            temperature=0.1
        ))
    elif primary_provider == "grok-4-1-fast":
        configs.append(LLMConfig(
            provider=LLMProvider.GROK_41_FAST,
            model="grok-4-1-fast-non-reasoning-latest",
            temperature=0.1
        ))
    elif primary_provider == "grok-4-1-code":
        configs.append(LLMConfig(
            provider=LLMProvider.GROK_41_CODE,
            model="grok-code-fast",
            temperature=0.1
        ))
    elif primary_provider == "grok-4-1-reasoning":
        configs.append(LLMConfig(
            provider=LLMProvider.GROK_41_REASONING,
            model="grok-4-1-fast-reasoning-latest",
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
        elif provider == "claude-sonnet":
            configs.append(LLMConfig(
                provider=LLMProvider.CLAUDE_SONNET,
                model="claude-sonnet-4-5-20250929",
                temperature=0.1
            ))
        elif provider == "claude-opus":
            configs.append(LLMConfig(
                provider=LLMProvider.CLAUDE_OPUS,
                model="claude-opus-4-5-20251101",
                temperature=0.1
            ))
        elif provider == "grok-4-1-fast":
            configs.append(LLMConfig(
                provider=LLMProvider.GROK_41_FAST,
                model="grok-4-1-fast-non-reasoning-latest",
                temperature=0.1
            ))
        elif provider == "grok-4-1-code":
            configs.append(LLMConfig(
                provider=LLMProvider.GROK_41_CODE,
                model="grok-code-fast",
                temperature=0.1
            ))
        elif provider == "grok-4-1-reasoning":
            configs.append(LLMConfig(
                provider=LLMProvider.GROK_41_REASONING,
                model="grok-4-1-fast-reasoning-latest",
                temperature=0.1
            ))

    return LLMManager(configs)