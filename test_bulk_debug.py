#!/usr/bin/env python3
"""
Debug script to test bulk processing step by step.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.structured_output import response_parser
from utils.enhanced_retriever import answer_enhancer

def test_bulk_debug():
    """Test the bulk processing components step by step."""
    
    print("🧪 Testing Bulk Processing Components...")
    print("=" * 60)
    
    # Test case
    query = "What was the revenue in Q1FY26?"
    answer = "The revenue in Q1 FY26 was $7,421 million, a year-on-year decline of 1.1%."
    sources = ["Conference Call Transcript (page 3)", "Investor Presentation (page 15)"]
    processing_time = 2.5
    
    print(f"1. Original Answer: {answer}")
    print()
    
    # Test answer enhancement
    print("2. Testing Answer Enhancement...")
    try:
        # Mock documents for testing
        class MockDocument:
            def __init__(self, content, source, page):
                self.page_content = content
                self.metadata = {'source': source, 'page': page}
        
        mock_docs = [
            MockDocument("Revenue in Q1FY26 was $7,421 million", "Conference Call Transcript", 3),
            MockDocument("Q1FY26 revenue declined by 1.1%", "Investor Presentation", 15)
        ]
        
        enhanced_answer = answer_enhancer.enhance_answer(answer, mock_docs, query)
        print(f"✅ Enhanced Answer: {enhanced_answer}")
        print()
        
    except Exception as e:
        print(f"❌ Answer enhancement failed: {str(e)}")
        return
    
    # Test structured response parsing
    print("3. Testing Structured Response Parsing...")
    try:
        structured_response = response_parser.parse_response(query, enhanced_answer, sources, processing_time)
        
        if structured_response and structured_response.metrics:
            print(f"✅ Parsing successful! Found {len(structured_response.metrics)} metrics")
            for i, metric in enumerate(structured_response.metrics):
                print(f"   Metric {i+1}: {metric.metric_name} = {metric.value} {metric.unit} ({metric.time_period})")
        else:
            print(f"⚠️ Parsing completed but no metrics found")
            print(f"   Response: {structured_response}")
        
        print()
        
    except Exception as e:
        print(f"❌ Structured response parsing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Test QueryResult creation
    print("4. Testing QueryResult Creation...")
    try:
        class QueryResult:
            def __init__(self, answer, structured_response=None):
                self.answer = answer
                self.structured_response = structured_response
        
        query_result = QueryResult(enhanced_answer, structured_response)
        
        print(f"✅ QueryResult created successfully")
        print(f"   Has answer: {hasattr(query_result, 'answer')}")
        print(f"   Has structured_response: {hasattr(query_result, 'structured_response')}")
        print(f"   Answer: {query_result.answer[:100]}...")
        print(f"   Structured Response: {query_result.structured_response is not None}")
        
        if query_result.structured_response:
            print(f"   Metrics in structured response: {len(query_result.structured_response.metrics)}")
        
    except Exception as e:
        print(f"❌ QueryResult creation failed: {str(e)}")
        return
    
    print("\n" + "=" * 60)
    print("🎯 Debug test completed!")

if __name__ == "__main__":
    test_bulk_debug()
