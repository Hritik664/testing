#!/usr/bin/env python3
"""
Simple test script to verify structured output parsing works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.structured_output import response_parser

def test_parser():
    """Test the structured output parser with a simple example."""
    
    print("🧪 Testing Structured Output Parser...")
    print("=" * 50)
    
    # Test case
    query = "What was the revenue in Q1FY26?"
    answer = "The revenue in Q1 FY26 was $7,421 million, a year-on-year decline of 1.1%."
    sources = ["Conference Call Transcript (page 3)", "Investor Presentation (page 15)"]
    processing_time = 2.5
    
    try:
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print(f"Sources: {sources}")
        print(f"Processing Time: {processing_time}s")
        print()
        
        # Parse the response
        structured_response = response_parser.parse_response(query, answer, sources, processing_time)
        
        print("✅ Parsing successful!")
        print(f"Response Type: {structured_response.response_type.value}")
        print(f"Confidence: {structured_response.confidence:.2f}")
        print(f"Metrics Found: {len(structured_response.metrics)}")
        
        # Show metrics
        for i, metric in enumerate(structured_response.metrics):
            print(f"\nMetric {i+1}:")
            print(f"  Name: '{metric.metric_name}'")
            print(f"  Value: '{metric.value}'")
            print(f"  Unit: '{metric.unit}'")
            print(f"  Time Period: '{metric.time_period}'")
            print(f"  Source: '{metric.source}'")
            print(f"  Confidence: {metric.confidence:.2f}")
        
        print("\n" + "=" * 50)
        print("🎯 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parser()
