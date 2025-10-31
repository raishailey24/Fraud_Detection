"""
Anthropic (Claude) provider implementation.
"""
from anthropic import Anthropic
from .base_provider import BaseAIProvider
import streamlit as st
import time
import random


class AnthropicProvider(BaseAIProvider):
    """Anthropic (Claude) API provider."""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model: Model to use (default: claude-3-haiku-20240307 - cheapest model)
        """
        super().__init__(api_key, model)
        self.client = None
        if api_key:
            try:
                self.client = Anthropic(api_key=api_key)
            except Exception as e:
                st.error(f"Failed to initialize Anthropic client: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if Anthropic is properly configured."""
        return self.client is not None and self.api_key is not None
    
    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate a response using Anthropic API with retry logic.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated response text
        """
        if not self.is_available():
            return "Error: Anthropic is not properly configured. Please check your API key."
        
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": 1500,
                    "temperature": 0.3,
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                if system_prompt:
                    kwargs["system"] = system_prompt
                
                response = self.client.messages.create(**kwargs)
                
                return response.content[0].text
                
            except Exception as e:
                error_str = str(e)
                
                # Handle specific error types
                if "overloaded" in error_str.lower() or "529" in error_str:
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        st.warning(f"⚠️ Anthropic servers are overloaded. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        return self._handle_overload_error()
                
                elif "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                        st.warning(f"⚠️ Rate limit exceeded. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        return "⚠️ **Rate Limit Exceeded**\n\nThe Anthropic API rate limit has been exceeded. Please wait a few minutes before trying again, or consider upgrading your API plan for higher limits."
                
                elif "authentication" in error_str.lower() or "401" in error_str:
                    return "❌ **Authentication Error**\n\nYour Anthropic API key appears to be invalid or expired. Please check your `.env` file and ensure you have a valid API key."
                
                elif "not_found_error" in error_str.lower() or "404" in error_str:
                    return f"❌ **Model Not Found**\n\nThe model `{self.model}` is not available in your API account. \n\n**Available Claude models (verified working):**\n• claude-3-opus-20240229 (most powerful, default)\n• claude-3-sonnet-20240229 (balanced)\n• claude-3-haiku-20240307 (fastest)\n\n**Note:** Claude 3.5 Sonnet may not be available in all API accounts.\n\n**To fix this:**\n1. Update your `.env` file with: `ANTHROPIC_MODEL=claude-3-opus-20240229`\n2. Or remove the ANTHROPIC_MODEL line to use the default (Claude 3 Opus)\n3. Click the 🔄 Refresh button in the AI Copilot panel"
                
                else:
                    # For other errors, don't retry
                    error_msg = f"Anthropic API error: {error_str}"
                    st.error(error_msg)
                    return f"❌ **API Error**\n\nUnexpected error occurred: {error_str}\n\nPlease try again or check your API configuration."
        
        return self._handle_overload_error()
    
    def _handle_overload_error(self) -> str:
        """Handle overload error with helpful message."""
        return """⚠️ **Anthropic Servers Overloaded**

The Anthropic API servers are currently experiencing high load and are temporarily unavailable.

**What you can do:**
• **Wait and retry**: Server load typically decreases within 5-15 minutes
• **Try OpenAI**: Switch to OpenAI provider in your config if you have an OpenAI API key
• **Check status**: Visit https://status.anthropic.com for real-time server status

**To switch to OpenAI:**
1. Set `AI_PROVIDER=openai` in your `.env` file
2. Add your OpenAI API key as `OPENAI_API_KEY=your_key_here`
3. Restart the application

This is a temporary issue and should resolve shortly."""
