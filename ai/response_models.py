"""
Pydantic models for structured AI responses.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import re


class ExecutiveSummaryResponse(BaseModel):
    """Structured model for executive summary responses."""
    
    key_findings: List[str] = Field(
        ..., 
        min_items=3, 
        max_items=5,
        description="Key findings in simple language"
    )
    
    what_this_means: str = Field(
        ..., 
        min_length=50,
        description="Plain English explanation of what the data tells us"
    )
    
    recommendations: List[str] = Field(
        ..., 
        min_items=3, 
        max_items=5,
        description="Actionable recommendations in simple language"
    )
    
    @validator('key_findings', 'recommendations', each_item=True)
    def validate_text_formatting(cls, v):
        """Ensure proper text formatting."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        
        # Check for concatenated words
        problematic_patterns = [
            r'\d+,withatotal',
            r'\d+,whiletheaverage', 
            r'withatotalfraudamountof',
            r'whiletheaveragefraudamountwas',
            r'andtheaverage'
        ]
        
        for pattern in problematic_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Text contains concatenated words: {v}")
        
        return v.strip()
    
    @validator('what_this_means')
    def validate_explanation(cls, v):
        """Validate the explanation text."""
        if not v or not v.strip():
            raise ValueError("Explanation cannot be empty")
        
        # Check for technical jargon that should be simplified
        technical_terms = ['fraud rate', 'risk score', 'detection threshold']
        simple_alternatives = ['percentage of fake transactions', 'danger level', 'alert setting']
        
        # This is just a warning, not a hard validation
        return v.strip()
    
    def to_formatted_string(self) -> str:
        """Convert to properly formatted string output."""
        output = ["EXECUTIVE SUMMARY", ""]
        
        output.append("Key Findings:")
        for i, finding in enumerate(self.key_findings, 1):
            output.append(f"- {finding}")
        
        output.append("")
        output.append("What This Means:")
        output.append(self.what_this_means)
        
        output.append("")
        output.append("Recommendations:")
        for i, rec in enumerate(self.recommendations, 1):
            output.append(f"{i}. {rec}")
        
        return "\n".join(output)


class DetectionRuleResponse(BaseModel):
    """Structured model for fraud detection rules."""
    
    rule_name: str = Field(..., min_length=5, description="Descriptive rule name")
    condition: str = Field(..., min_length=10, description="Rule condition in plain English")
    risk_level: str = Field(..., regex=r'^(High|Medium|Low)$', description="Risk level")
    rationale: str = Field(..., min_length=20, description="Why this rule makes sense")
    expected_false_positive_rate: str = Field(..., description="Expected false positive rate")
    
    @validator('condition', 'rationale')
    def validate_plain_english(cls, v):
        """Ensure text is in plain English."""
        # Check for overly technical language
        if len(v.split()) < 5:
            raise ValueError("Description too short - needs more explanation")
        return v.strip()


class WhatIfAnalysisResponse(BaseModel):
    """Structured model for what-if analysis responses."""
    
    scenario_impact: str = Field(..., min_length=50, description="Impact description")
    risks: List[str] = Field(..., min_items=2, max_items=4, description="Potential risks")
    opportunities: List[str] = Field(..., min_items=2, max_items=4, description="Potential opportunities")
    recommendations: List[str] = Field(..., min_items=3, max_items=5, description="What to do")
    
    def to_formatted_string(self) -> str:
        """Convert to properly formatted string output."""
        output = ["SCENARIO ANALYSIS", ""]
        
        output.append("Expected Impact:")
        output.append(self.scenario_impact)
        
        output.append("")
        output.append("Potential Risks:")
        for risk in self.risks:
            output.append(f"- {risk}")
        
        output.append("")
        output.append("Opportunities:")
        for opp in self.opportunities:
            output.append(f"- {opp}")
        
        output.append("")
        output.append("What We Should Do:")
        for i, rec in enumerate(self.recommendations, 1):
            output.append(f"{i}. {rec}")
        
        return "\n".join(output)
