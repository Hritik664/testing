#!/usr/bin/env python3
"""
Structured Output System for Financial RAG
Provides formatted responses that can be consumed by other agents.
"""

import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

from config import Config
from utils.logger import logger

class ResponseType(Enum):
    """Types of structured responses."""
    FINANCIAL_METRIC = "financial_metric"
    COMPARISON = "comparison"
    TREND = "trend"
    CAPACITY = "capacity"
    COST = "cost"
    REVENUE = "revenue"
    GENERAL = "general"
    NOT_FOUND = "not_found"

@dataclass
class FinancialMetric:
    """Structured financial metric data."""
    metric_name: str
    value: Union[str, float, int]
    unit: Optional[str] = None
    time_period: Optional[str] = None
    source: Optional[str] = None
    confidence: float = 0.0
    page_reference: Optional[str] = None

@dataclass
class StructuredResponse:
    """Structured response container."""
    query: str
    response_type: ResponseType
    answer: str
    metrics: List[FinancialMetric]
    confidence: float
    sources: List[str]
    timestamp: str
    processing_time: float
    raw_response: str

class ResponseParser:
    """Parse and structure responses from LLM outputs."""
    
    def __init__(self):
        self.financial_patterns = {
            'revenue': r'(?:revenue|sales|income)\s*(?:was|is|of)?\s*([\d,]+(?:\.[\d]+)?)\s*(?:cr|crore|million|billion|%)?',
            'cost': r'(?:cost|expense|expenditure)\s*(?:was|is|of)?\s*([\d,]+(?:\.[\d]+)?)\s*(?:cr|crore|million|billion|%)?',
            'capacity': r'(?:capacity|production)\s*(?:was|is|of)?\s*([\d,]+(?:\.[\d]+)?)\s*(?:mnT|million|billion)?',
            'volume': r'(?:volume|sales\s+volume)\s*(?:was|is|of)?\s*([\d,]+(?:\.[\d]+)?)\s*(?:mnT|million|billion)?',
            'growth': r'(?:growth|increase|decrease)\s*(?:was|is|of)?\s*([+-]?[\d,]+(?:\.[\d]+)?)\s*%',
            'capex': r'(?:capex|capital\s+expenditure)\s*(?:was|is|of)?\s*([\d,]+(?:\.[\d]+)?)\s*(?:cr|crore|million|billion)?',
            'percentage': r'([\d,]+(?:\.[\d]+)?)\s*%',
            'currency': r'([\d,]+(?:\.[\d]+)?)\s*(?:cr|crore|million|billion|USD|INR)',
        }
        
        self.time_patterns = [
            r'Q[1-4]FY\d{4}',
            r'FY\d{4}',
            r'Q[1-4]\s+\d{4}',
            r'\d{4}',
        ]
    
    def parse_response(self, query: str, raw_response: str, sources: List[str], processing_time: float) -> StructuredResponse:
        """Parse raw response into structured format."""
        
        try:
            # Determine response type
            response_type = self._classify_response_type(query, raw_response)
            
            # Extract metrics
            metrics = self._extract_metrics(raw_response)
            
            # Calculate confidence
            confidence = self._calculate_confidence(raw_response, metrics)
            
            # Create structured response
            structured_response = StructuredResponse(
                query=query,
                response_type=response_type,
                answer=self._clean_answer(raw_response),
                metrics=metrics,
                confidence=confidence,
                sources=sources,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time,
                raw_response=raw_response
            )
            
            return structured_response
            
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return self._create_error_response(query, raw_response, sources, processing_time)
    
    def _classify_response_type(self, query: str, response: str) -> ResponseType:
        """Classify the type of response based on query and content."""
        query_lower = query.lower()
        response_lower = response.lower()
        
        if any(word in query_lower for word in ['revenue', 'sales', 'income']):
            return ResponseType.REVENUE
        elif any(word in query_lower for word in ['cost', 'expense', 'expenditure']):
            return ResponseType.COST
        elif any(word in query_lower for word in ['capacity', 'production']):
            return ResponseType.CAPACITY
        elif any(word in query_lower for word in ['growth', 'increase', 'decrease']):
            return ResponseType.TREND
        elif any(word in query_lower for word in ['compare', 'vs', 'versus']):
            return ResponseType.COMPARISON
        elif any(word in query_lower for word in ['not available', 'not found', 'information not']):
            return ResponseType.NOT_FOUND
        else:
            return ResponseType.GENERAL
    
    def _extract_metrics(self, response: str) -> List[FinancialMetric]:
        """Extract financial metrics from response."""
        metrics = []
        
        for metric_type, pattern in self.financial_patterns.items():
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                unit = self._extract_unit(match.group(0))
                time_period = self._extract_time_period(response)
                
                metric = FinancialMetric(
                    metric_name=metric_type,
                    value=self._normalize_value(value),
                    unit=unit,
                    time_period=time_period,
                    confidence=0.8
                )
                metrics.append(metric)
        
        return metrics
    
    def _extract_unit(self, text: str) -> Optional[str]:
        """Extract unit from text."""
        units = ['cr', 'crore', 'million', 'billion', 'mnT', '%', 'USD', 'INR']
        for unit in units:
            if unit.lower() in text.lower():
                return unit
        return None
    
    def _extract_time_period(self, text: str) -> Optional[str]:
        """Extract time period from text."""
        for pattern in self.time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None
    
    def _normalize_value(self, value: str) -> Union[str, float, int]:
        """Normalize value to appropriate type."""
        try:
            # Remove commas and convert to float
            clean_value = value.replace(',', '')
            if '.' in clean_value:
                return float(clean_value)
            else:
                return int(clean_value)
        except:
            return value
    
    def _calculate_confidence(self, response: str, metrics: List[FinancialMetric]) -> float:
        """Calculate confidence score for the response."""
        base_confidence = 0.5
        
        # Increase confidence for specific metrics found
        if metrics:
            base_confidence += 0.2
        
        # Increase confidence for time periods mentioned
        if any(self._extract_time_period(response) for _ in [1]):
            base_confidence += 0.1
        
        # Decrease confidence for uncertainty indicators
        uncertainty_words = ['approximately', 'around', 'about', 'roughly', 'estimate']
        if any(word in response.lower() for word in uncertainty_words):
            base_confidence -= 0.1
        
        # Decrease confidence for not found responses
        if 'not available' in response.lower() or 'not found' in response.lower():
            base_confidence -= 0.2
        
        return max(0.0, min(1.0, base_confidence))
    
    def _clean_answer(self, response: str) -> str:
        """Clean and format the answer."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', response.strip())
        
        # Ensure proper sentence ending
        if not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        return cleaned
    
    def _create_error_response(self, query: str, raw_response: str, sources: List[str], processing_time: float) -> StructuredResponse:
        """Create error response when parsing fails."""
        return StructuredResponse(
            query=query,
            response_type=ResponseType.NOT_FOUND,
            answer="Failed to parse response structure.",
            metrics=[],
            confidence=0.0,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            raw_response=raw_response
        )

class StructuredOutputFormatter:
    """Format structured responses for different output types."""
    
    def __init__(self):
        self.parser = ResponseParser()
    
    def format_for_agent(self, structured_response: StructuredResponse) -> Dict[str, Any]:
        """Format response for agent consumption."""
        return {
            "status": "success" if structured_response.confidence > 0.3 else "low_confidence",
            "query": structured_response.query,
            "response_type": structured_response.response_type.value,
            "answer": structured_response.answer,
            "metrics": [asdict(metric) for metric in structured_response.metrics],
            "confidence": structured_response.confidence,
            "sources": structured_response.sources,
            "timestamp": structured_response.timestamp,
            "processing_time": structured_response.processing_time,
            "metadata": {
                "total_metrics": len(structured_response.metrics),
                "has_sources": len(structured_response.sources) > 0,
                "response_quality": "high" if structured_response.confidence > 0.7 else "medium" if structured_response.confidence > 0.4 else "low"
            }
        }
    
    def format_for_json(self, structured_response: StructuredResponse) -> str:
        """Format response as JSON string."""
        return json.dumps(self.format_for_agent(structured_response), indent=2)
    
    def format_for_csv(self, structured_response: StructuredResponse) -> str:
        """Format response as CSV string."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['Query', 'Response Type', 'Answer', 'Confidence', 'Processing Time'])
        
        # Data
        writer.writerow([
            structured_response.query,
            structured_response.response_type.value,
            structured_response.answer,
            structured_response.confidence,
            structured_response.processing_time
        ])
        
        return output.getvalue()
    
    def format_for_markdown(self, structured_response: StructuredResponse) -> str:
        """Format response as Markdown."""
        md = f"# Query Response\n\n"
        md += f"**Query:** {structured_response.query}\n\n"
        md += f"**Response Type:** {structured_response.response_type.value}\n\n"
        md += f"**Answer:** {structured_response.answer}\n\n"
        md += f"**Confidence:** {structured_response.confidence:.2f}\n\n"
        
        if structured_response.metrics:
            md += "## Metrics\n\n"
            for metric in structured_response.metrics:
                md += f"- **{metric.metric_name}:** {metric.value}"
                if metric.unit:
                    md += f" {metric.unit}"
                if metric.time_period:
                    md += f" ({metric.time_period})"
                md += "\n"
        
        if structured_response.sources:
            md += "\n## Sources\n\n"
            for source in structured_response.sources:
                md += f"- {source}\n"
        
        return md

# Global instances
response_parser = ResponseParser()
output_formatter = StructuredOutputFormatter() 