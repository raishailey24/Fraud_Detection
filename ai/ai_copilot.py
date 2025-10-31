"""
AI Copilot orchestrator that manages AI provider selection and operations.
"""
from typing import Dict, Any, Optional
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .base_provider import BaseAIProvider
from config import Config
import streamlit as st


class AICopilot:
    """Orchestrates AI operations using configured provider."""
    
    def __init__(self):
        """Initialize AI Copilot with configured provider."""
        self.provider: Optional[BaseAIProvider] = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the appropriate AI provider based on configuration."""
        self.primary_provider = None
        self.fallback_provider = None
        
        if Config.AI_PROVIDER == "openai":
            self.primary_provider = OpenAIProvider(
                api_key=Config.OPENAI_API_KEY,
                model=Config.OPENAI_MODEL
            )
            # Set up Anthropic as fallback if available
            if Config.ANTHROPIC_API_KEY:
                self.fallback_provider = AnthropicProvider(
                    api_key=Config.ANTHROPIC_API_KEY,
                    model=Config.ANTHROPIC_MODEL
                )
        elif Config.AI_PROVIDER == "anthropic":
            self.primary_provider = AnthropicProvider(
                api_key=Config.ANTHROPIC_API_KEY,
                model=Config.ANTHROPIC_MODEL
            )
            # Set up OpenAI as fallback if available
            if Config.OPENAI_API_KEY:
                self.fallback_provider = OpenAIProvider(
                    api_key=Config.OPENAI_API_KEY,
                    model=Config.OPENAI_MODEL
                )
        else:
            raise ValueError(f"Unsupported AI provider: {Config.AI_PROVIDER}")
        
        self.provider = self.primary_provider
    
    def is_available(self) -> bool:
        """Check if AI copilot is available."""
        return self.provider is not None and self.provider.is_available()
    
    def get_provider_name(self) -> str:
        """Get the name of the current provider."""
        if isinstance(self.provider, OpenAIProvider):
            return f"OpenAI ({self.provider.model})"
        elif isinstance(self.provider, AnthropicProvider):
            return f"Anthropic ({self.provider.model})"
        return "Unknown"
    
    def _generate_with_fallback(self, method_name: str, *args, **kwargs) -> str:
        """
        Generate response with automatic fallback to secondary provider if primary fails.
        
        Args:
            method_name: Name of the method to call on the provider
            *args, **kwargs: Arguments to pass to the method
            
        Returns:
            Generated response text
        """
        if not self.is_available():
            return "AI Copilot is not available. Please configure API keys in .env file."
        
        try:
            # Try primary provider
            method = getattr(self.provider, method_name)
            response = method(*args, **kwargs)
            
            # Check if response indicates an overload error
            if "overloaded" in response.lower() and self.fallback_provider and self.fallback_provider.is_available():
                st.info(f"🔄 Primary provider ({self.get_provider_name()}) is overloaded. Trying fallback provider...")
                
                # Switch to fallback provider
                original_provider = self.provider
                self.provider = self.fallback_provider
                
                try:
                    fallback_method = getattr(self.fallback_provider, method_name)
                    fallback_response = fallback_method(*args, **kwargs)
                    
                    # If fallback succeeds, show success message
                    if not ("overloaded" in fallback_response.lower() or "error" in fallback_response.lower()):
                        fallback_name = "OpenAI" if isinstance(self.fallback_provider, OpenAIProvider) else "Anthropic"
                        st.success(f"✅ Successfully switched to {fallback_name} provider")
                        return fallback_response
                    else:
                        # Both providers failed, restore original and return error
                        self.provider = original_provider
                        return response
                except Exception:
                    # Fallback failed, restore original provider
                    self.provider = original_provider
                    return response
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_executive_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Generate executive summary from metrics.
        
        Args:
            metrics: Aggregated metrics dictionary (no PII)
            
        Returns:
            Executive summary text
        """
        return self._generate_with_fallback("generate_executive_summary", metrics)
    
    def generate_what_if_analysis(self, metrics: Dict[str, Any], scenario: str) -> str:
        """
        Generate what-if scenario analysis.
        
        Args:
            metrics: Aggregated metrics dictionary
            scenario: Scenario description
            
        Returns:
            What-if analysis text
        """
        return self._generate_with_fallback("generate_what_if_analysis", metrics, scenario)
    
    def generate_detection_rules(self, metrics: Dict[str, Any]) -> str:
        """
        Generate fraud detection rule proposals.
        
        Args:
            metrics: Aggregated metrics dictionary
            
        Returns:
            Detection rules text
        """
        return self._generate_with_fallback("generate_detection_rules", metrics)
    
    def answer_custom_query(self, metrics: Dict[str, Any], query: str) -> str:
        """
        Answer custom user query.
        
        Args:
            metrics: Aggregated metrics dictionary
            query: User's question
            
        Returns:
            Answer text
        """
        return self._generate_with_fallback("answer_custom_query", metrics, query)
