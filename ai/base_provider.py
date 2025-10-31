"""
Base class for AI providers.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAIProvider(ABC):
    """Abstract base class for AI providers."""
    
    def __init__(self, api_key: str, model: str):
        """
        Initialize AI provider.
        
        Args:
            api_key: API key for the provider
            model: Model identifier to use
        """
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate a response from the AI model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is properly configured and available.
        
        Returns:
            True if available, False otherwise
        """
        pass
    
    def generate_executive_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Generate an executive summary from aggregated metrics.
        
        Args:
            metrics: Dictionary of aggregated metrics (no PII)
            
        Returns:
            Executive summary text
        """
        system_prompt = """You are a fraud analytics expert. Write clearly with proper word spacing."""
        
        prompt = f"""Based on these fraud analytics metrics, write a simple executive summary:

{self._format_metrics(metrics)}

EXECUTIVE SUMMARY

Key Findings:
- [Write 3 key findings in simple language]

What This Means:
[Explain in plain English]

Recommendations:
1. [First recommendation]
2. [Second recommendation] 
3. [Third recommendation]"""
        
        return self.generate_response(prompt, system_prompt)
    
    @staticmethod
    def _fix_formatting(text: str) -> str:
        """Fix common formatting issues in AI responses using multiple libraries."""
        import re
        import unicodedata
        from typing import List, Tuple
        
        # Step 1: Normalize unicode and remove special characters
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Step 2: Fix common concatenation patterns
        concatenation_fixes = [
            # Number + text concatenations
            (r'(\$?\d+(?:,\d{3})*(?:\.\d+)?),withatotal', r'\1, with a total'),
            (r'(\$?\d+(?:,\d{3})*(?:\.\d+)?),whiletheaverage', r'\1, while the average'),
            (r'(\$?\d+(?:,\d{3})*(?:\.\d+)?),with', r'\1, with'),
            (r'(\$?\d+(?:,\d{3})*(?:\.\d+)?),while', r'\1, while'),
            (r'(\$?\d+(?:,\d{3})*(?:\.\d+)?),and', r'\1, and'),
            
            # Word concatenations
            (r'withatotalfraudamountof', r'with a total fraud amount of'),
            (r'whiletheaveragefraudamountwas', r'while the average fraud amount was'),
            (r'andtheaverage', r'and the average'),
            (r'ofthetotal', r'of the total'),
            (r'wasthe', r'was the'),
            (r'ofthe', r'of the'),
            (r'inthe', r'in the'),
            (r'forthe', r'for the'),
            (r'tothe', r'to the'),
            
            # Number + word concatenations
            (r'(\d+)fraud', r'\1 fraud'),
            (r'fraud(\d+)', r'fraud \1'),
            (r'(\d+)amount', r'\1 amount'),
            (r'amount(\d+)', r'amount \1'),
            (r'(\d+)transactions', r'\1 transactions'),
            (r'transactions(\d+)', r'transactions \1'),
            (r'was(\d+)', r'was \1'),
            (r'of(\d+)', r'of \1'),
            (r'(\d+)cases', r'\1 cases'),
            (r'cases(\d+)', r'cases \1'),
        ]
        
        # Apply concatenation fixes
        for pattern, replacement in concatenation_fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Step 3: Fix spacing issues around punctuation
        text = re.sub(r'\s*,\s*', ', ', text)  # Fix comma spacing
        text = re.sub(r'\s*\.\s*', '. ', text)  # Fix period spacing
        text = re.sub(r'\s*:\s*', ': ', text)   # Fix colon spacing
        
        # Step 4: Fix multiple spaces and clean up
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)  # Add space after punctuation
        
        # Step 5: Fix line breaks and formatting
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean up multiple line breaks
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
        
        # Step 6: Remove any formatting instructions that leaked through
        formatting_patterns = [
            r'FORMATTING CHECKLIST.*?(?=\n\n|\Z)',
            r'VERIFY BEFORE RESPONDING.*?(?=\n\n|\Z)',
            r'REMEMBER:.*?(?=\n\n|\Z)',
            r'Double-check.*?(?=\n\n|\Z)',
        ]
        
        for pattern in formatting_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        return text.strip()
    
    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from AI response text."""
        import re
        
        # Try to find JSON object in the response
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            return match.group(0)
        
        # If no JSON found, return the original text
        return text
    
    def generate_what_if_analysis(self, metrics: Dict[str, Any], scenario: str) -> str:
        """
        Generate what-if scenario analysis.
        
        Args:
            metrics: Dictionary of aggregated metrics
            scenario: Scenario description from user
            
        Returns:
            What-if analysis text
        """
        system_prompt = """You are a fraud analytics expert. Write clearly with proper word spacing."""
        
        prompt = f"""Based on these fraud analytics metrics, analyze this scenario: {scenario}

{self._format_metrics(metrics)}

SCENARIO ANALYSIS

Expected Impact:
[Explain what would happen in simple terms]

Potential Risks:
- [Risk 1]
- [Risk 2]
- [Risk 3]

Opportunities:
- [Opportunity 1]
- [Opportunity 2]
- [Opportunity 3]

What We Should Do:
1. [Immediate action]
2. [Short-term action]
3. [Long-term action]

Key Things to Watch:
- [KPI 1 to monitor]
- [KPI 2 to monitor]
- [KPI 3 to monitor]"""
        
        return self.generate_response(prompt, system_prompt)
    
    def generate_detection_rules(self, metrics: Dict[str, Any]) -> str:
        """
        Generate fraud detection rule proposals.
        
        Args:
            metrics: Dictionary of aggregated metrics
            
        Returns:
            Detection rules text
        """
        system_prompt = """You are a fraud detection expert. Write clearly with proper word spacing."""
        
        prompt = f"""Based on these fraud analytics metrics, propose 5 fraud detection rules:

{self._format_metrics(metrics)}

FRAUD DETECTION RULES

Rule 1: [Rule Name]
- Condition: [When to trigger]
- Risk Level: [High/Medium/Low]
- Why: [Simple explanation]
- False Positives: [Expected rate]

Rule 2: [Rule Name]
- Condition: [When to trigger]
- Risk Level: [High/Medium/Low]
- Why: [Simple explanation]
- False Positives: [Expected rate]

Rule 3: [Rule Name]
- Condition: [When to trigger]
- Risk Level: [High/Medium/Low]
- Why: [Simple explanation]
- False Positives: [Expected rate]

Rule 4: [Rule Name]
- Condition: [When to trigger]
- Risk Level: [High/Medium/Low]
- Why: [Simple explanation]
- False Positives: [Expected rate]

Rule 5: [Rule Name]
- Condition: [When to trigger]
- Risk Level: [High/Medium/Low]
- Why: [Simple explanation]
- False Positives: [Expected rate]

Implementation Plan:
1. [First step]
2. [Second step]
3. [Third step]"""
        
        return self.generate_response(prompt, system_prompt)
    
    def answer_custom_query(self, metrics: Dict[str, Any], query: str) -> str:
        """
        Answer a custom user query about the data.
        
        Args:
            metrics: Dictionary of aggregated metrics
            query: User's question
            
        Returns:
            Answer text
        """
        system_prompt = """You are a fraud analytics expert. Write clearly with proper word spacing."""
        
        prompt = f"""Based on this fraud analytics data, answer the user's question:

{self._format_metrics(metrics)}

USER QUESTION: {query}

ANSWER

Direct Answer:
[Answer the question in simple terms]

Supporting Data:
[What data supports this answer]

What This Means:
[Explain in plain English]

Additional Insights:
[Any other relevant observations]

Limitations:
[If there are any data limitations]"""
        
        return self.generate_response(prompt, system_prompt)
    
    @staticmethod
    def _format_metrics(metrics: Dict[str, Any]) -> str:
        """Format metrics dictionary as readable text with proper formatting."""
        lines = []
        
        def format_number(value):
            """Format numbers with proper separators and precision."""
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    if value < 1:
                        return f"{value:.4f}"
                    elif value < 100:
                        return f"{value:.2f}"
                    else:
                        return f"{value:,.2f}"
                else:
                    return f"{value:,}"
            return str(value)
        
        def format_key(key):
            """Format key names to be more readable."""
            return key.replace("_", " ").title()
        
        for key, value in metrics.items():
            formatted_key = format_key(key)
            
            if isinstance(value, dict):
                lines.append(f"\n{formatted_key}:")
                for sub_key, sub_value in value.items():
                    formatted_sub_key = format_key(sub_key)
                    formatted_value = format_number(sub_value)
                    lines.append(f"  • {formatted_sub_key}: {formatted_value}")
            else:
                formatted_value = format_number(value)
                lines.append(f"{formatted_key}: {formatted_value}")
        
        return "\n".join(lines)
