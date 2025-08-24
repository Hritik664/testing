#!/usr/bin/env python3
"""
Test script to verify structured output parsing for bulk questions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.structured_output import response_parser

def test_structured_parsing():
    """Test structured output parsing with sample financial questions."""
    
    print("🧪 Testing Structured Output Parsing for Bulk Questions...")
    print("=" * 60)
    
    # Sample questions and expected answers
    test_cases = [
        {
            "question": "What was the revenue in Q1FY26?",
            "answer": "The revenue in Q1FY26 was Rs 2,500 Cr, representing a growth of 15% compared to Q1FY25."
        },
        {
            "question": "What is the current capacity utilization?",
            "answer": "The current capacity utilization is 85% with a total capacity of 50 mnT."
        },
        {
            "question": "What was the EBITDA margin?",
            "answer": "The EBITDA margin for the quarter was 22.5%."
        },
        {
            "question": "What is the capex guidance for FY26?",
            "answer": "The capex guidance for FY26 is Rs 3,000 Cr."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['question']}")
        print(f"Answer: {test_case['answer']}")
        
        try:
            # Parse the response
            structured_response = response_parser.parse_response(
                test_case['question'], 
                test_case['answer']
            )
            
            print(f"✅ Parsed successfully!")
            print(f"   Response Type: {structured_response.response_type.value}")
            print(f"   Confidence: {structured_response.confidence:.2f}")
            print(f"   Metrics Found: {len(structured_response.metrics)}")
            
            # Show extracted metrics
            for j, metric in enumerate(structured_response.metrics):
                print(f"   Metric {j+1}:")
                print(f"     Name: {metric.metric_name}")
                print(f"     Value: {metric.value}")
                print(f"     Unit: {metric.unit}")
                print(f"     Period: {metric.period}")
                
        except Exception as e:
            print(f"❌ Failed to parse: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎯 Test completed! Check if metrics are being extracted correctly.")

if __name__ == "__main__":
    test_structured_parsing()
