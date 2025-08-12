#!/usr/bin/env python3
"""
Test script for enhanced retriever functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.enhanced_retriever import EnhancedRetriever, FinancialQueryAnalyzer
from langchain_core.retrievers import BaseRetriever

def test_enhanced_retriever():
    """Test that EnhancedRetriever properly inherits from BaseRetriever."""
    print("🧪 Testing EnhancedRetriever...")
    
    # Test inheritance
    retriever = EnhancedRetriever(None, None)
    assert isinstance(retriever, BaseRetriever), "EnhancedRetriever should inherit from BaseRetriever"
    print("✅ EnhancedRetriever properly inherits from BaseRetriever")
    
    # Test query analyzer
    analyzer = FinancialQueryAnalyzer()
    query = "What was the revenue in Q1 2025?"
    analysis = analyzer.analyze_query(query)
    
    assert 'query_type' in analysis, "Query analysis should include query_type"
    assert 'financial_entities' in analysis, "Query analysis should include financial_entities"
    print("✅ FinancialQueryAnalyzer works correctly")
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    test_enhanced_retriever() 