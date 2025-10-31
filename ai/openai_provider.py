"""
OpenAI provider implementation.
"""
from openai import OpenAI
from .base_provider import BaseAIProvider
import streamlit as st
import time
import random


class OpenAIProvider(BaseAIProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4-turbo-preview)
        """
        super().__init__(api_key, model)
        self.client = None
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:
                st.error(f"Failed to initialize OpenAI client: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if OpenAI is properly configured."""
        return self.client is not None and self.api_key is not None
    
    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate a response using OpenAI API with retry logic.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated response text
        """
        if not self.is_available():
            return "Error: OpenAI is not properly configured. Please check your API key."
        
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                messages = []
                
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=3000
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                
                # Handle specific error types
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                        st.warning(f"⚠️ OpenAI rate limit exceeded. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        return "⚠️ **Rate Limit Exceeded**\n\nThe OpenAI API rate limit has been exceeded. Please wait a few minutes before trying again, or consider upgrading your API plan for higher limits."
                
                elif "overloaded" in error_str.lower() or "service_unavailable" in error_str.lower() or "502" in error_str or "503" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        st.warning(f"⚠️ OpenAI servers are overloaded. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        return self._handle_overload_error()
                
                elif "authentication" in error_str.lower() or "401" in error_str:
                    return "❌ **Authentication Error**\n\nYour OpenAI API key appears to be invalid or expired. Please check your `.env` file and ensure you have a valid API key."
                
                elif "insufficient_quota" in error_str.lower() or "quota" in error_str.lower():
                    return "❌ **Quota Exceeded**\n\nYour OpenAI API quota has been exceeded. Please check your billing and usage limits at https://platform.openai.com/usage"
                
                else:
                    # For other errors, don't retry
                    error_msg = f"OpenAI API error: {error_str}"
                    st.error(error_msg)
                    return f"❌ **API Error**\n\nUnexpected error occurred: {error_str}\n\nPlease try again or check your API configuration."
        
        return self._handle_overload_error()
    
    def _handle_overload_error(self) -> str:
        """Handle overload error with helpful message."""
        return """⚠️ **OpenAI Servers Overloaded**

The OpenAI API servers are currently experiencing high load and are temporarily unavailable.

**What you can do:**
• **Wait and retry**: Server load typically decreases within 5-15 minutes
• **Try Anthropic**: Switch to Anthropic provider in your config if you have an Anthropic API key
• **Check status**: Visit https://status.openai.com for real-time server status

**To switch to Anthropic:**
1. Set `AI_PROVIDER=anthropic` in your `.env` file
2. Add your Anthropic API key as `ANTHROPIC_API_KEY=your_key_here`
3. Restart the application

This is a temporary issue and should resolve shortly."""
