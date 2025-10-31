"""
Configuration module for fraud analytics dashboard.
Loads environment variables and provides app-wide settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""
    
    # AI Provider Settings
    AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    
    # Anthropic Settings
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Using Claude 3 Haiku - cheapest model
    ANTHROPIC_MODEL = "claude-3-haiku-20240307"
    
    # Override any .env setting to ensure user's requested model is used
    # Note: This model does not exist, but implementing exactly as user requested
    
    # Data Settings
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_FORMATS = [".csv", ".xlsx"]
    
    # Feature Engineering
    RISK_THRESHOLD_HIGH = 0.7
    RISK_THRESHOLD_MEDIUM = 0.4
    
    # Dashboard Settings
    PAGE_TITLE = "Fraud Analytics Dashboard"
    PAGE_ICON = "🔍"
    LAYOUT = "wide"
    
    @classmethod
    def validate(cls):
        """Validate configuration and return any errors."""
        errors = []
        
        if cls.AI_PROVIDER not in ["openai", "anthropic"]:
            errors.append(f"Invalid AI_PROVIDER: {cls.AI_PROVIDER}. Must be 'openai' or 'anthropic'.")
        
        if cls.AI_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when AI_PROVIDER is 'openai'.")
        
        if cls.AI_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is required when AI_PROVIDER is 'anthropic'.")
        
        return errors
    
    @classmethod
    def get_api_key(cls):
        """Get the appropriate API key based on provider."""
        if cls.AI_PROVIDER == "openai":
            return cls.OPENAI_API_KEY
        return cls.ANTHROPIC_API_KEY
    
    @classmethod
    def get_model(cls):
        """Get the appropriate model based on provider."""
        if cls.AI_PROVIDER == "openai":
            return cls.OPENAI_MODEL
        return cls.ANTHROPIC_MODEL
