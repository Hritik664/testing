#!/usr/bin/env python3
"""
Test script for the Structured Output System
Demonstrates how the system parses and formats responses for agent consumption.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.structured_output import response_parser, output_formatter, ResponseType

def test_structured_output():
    """Test the structured output system with sample responses."""
    
    print("🧪 Testing Structured Output System")
    print("=" * 50)
    
    # Sample queries and responses
    test_cases = [
        {
            "query": "What was the sales volume in Q1FY25?",
            "response": "Based on the provided documents, the sales volume in Q1FY25 was 7.4 mnT. This represents a significant increase from the previous quarter.",
            "expected_type": ResponseType.CAPACITY
        },
        {
            "query": "What was the revenue growth in Q1FY26?",
            "response": "The revenue growth in Q1FY26 was -1.1% in USD terms. This decline was primarily due to market conditions.",
            "expected_type": ResponseType.TREND
        },
        {
            "query": "What is the current capacity?",
            "response": "The current capacity is 49.5 mnT. This includes both existing facilities and recent expansions.",
            "expected_type": ResponseType.CAPACITY
        },
        {
            "query": "What was the cost per ton in Q1FY26?",
            "response": "The cost per ton in Q1FY26 was 3932. This represents an increase from the previous quarter.",
            "expected_type": ResponseType.COST
        },
        {
            "query": "What is the capex for new expansion?",
            "response": "The capex for new expansion is 3287 cr. This investment is expected to be completed by FY26.",
            "expected_type": ResponseType.COST
        },
        {
            "query": "Information not available",
            "response": "Information not available in the provided documents. The requested data was not found in the available sources.",
            "expected_type": ResponseType.NOT_FOUND
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['query']}")
        print("-" * 40)
        
        # Parse response
        structured_response = response_parser.parse_response(
            test_case['query'],
            test_case['response'],
            ["test_document.pdf - Investor Presentation (Page 5)"],
            2.5
        )
        
        # Display results
        print(f"✅ Response Type: {structured_response.response_type.value}")
        print(f"✅ Confidence: {structured_response.confidence:.2f}")
        print(f"✅ Answer: {structured_response.answer}")
        
        if structured_response.metrics:
            print("📊 Extracted Metrics:")
            for metric in structured_response.metrics:
                metric_text = f"  - {metric.metric_name}: {metric.value}"
                if metric.unit:
                    metric_text += f" {metric.unit}"
                if metric.time_period:
                    metric_text += f" ({metric.time_period})"
                print(metric_text)
        
        # Test different output formats
        print("\n📤 Output Formats:")
        
        # JSON format
        json_output = output_formatter.format_for_json(structured_response)
        print(f"  JSON: {len(json_output)} characters")
        
        # Markdown format
        md_output = output_formatter.format_for_markdown(structured_response)
        print(f"  Markdown: {len(md_output)} characters")
        
        # CSV format
        csv_output = output_formatter.format_for_csv(structured_response)
        print(f"  CSV: {len(csv_output)} characters")
        
        # Agent format
        agent_output = output_formatter.format_for_agent(structured_response)
        print(f"  Agent Format: {len(str(agent_output))} characters")
        
        # Verify expected type
        if structured_response.response_type == test_case['expected_type']:
            print(f"✅ Type classification correct")
        else:
            print(f"⚠️ Type classification mismatch: expected {test_case['expected_type'].value}, got {structured_response.response_type.value}")

def test_agent_consumption():
    """Test how the structured output can be consumed by another agent."""
    
    print("\n🤖 Testing Agent Consumption")
    print("=" * 50)
    
    # Simulate a query and response
    query = "What was the sales volume in Q1FY25?"
    response = "The sales volume in Q1FY25 was 7.4 mnT according to the investor presentation."
    
    # Parse into structured format
    structured_response = response_parser.parse_response(
        query,
        response,
        ["investor_presentation.pdf - Q1FY25 Results (Page 3)"],
        1.8
    )
    
    # Format for agent consumption
    agent_data = output_formatter.format_for_agent(structured_response)
    
    print("📊 Agent-Ready Data Structure:")
    print(f"  Status: {agent_data['status']}")
    print(f"  Response Type: {agent_data['response_type']}")
    print(f"  Confidence: {agent_data['confidence']}")
    print(f"  Processing Time: {agent_data['processing_time']}s")
    print(f"  Total Metrics: {agent_data['metadata']['total_metrics']}")
    print(f"  Response Quality: {agent_data['metadata']['response_quality']}")
    
    print("\n📋 Extracted Metrics for Agent:")
    for metric in agent_data['metrics']:
        print(f"  - {metric['metric_name']}: {metric['value']} {metric.get('unit', '')} ({metric.get('time_period', 'N/A')})")
    
    print("\n🔗 Sources for Verification:")
    for source in agent_data['sources']:
        print(f"  - {source}")
    
    # Demonstrate how an agent might use this data
    print("\n🤖 Agent Usage Example:")
    if agent_data['status'] == 'success' and agent_data['confidence'] > 0.7:
        print("  ✅ High confidence response - can be used directly")
        if agent_data['metrics']:
            print("  📊 Metrics available for further processing")
    elif agent_data['confidence'] > 0.4:
        print("  ⚠️ Medium confidence - may need verification")
    else:
        print("  ❌ Low confidence - should request clarification")

def test_error_handling():
    """Test error handling in the structured output system."""
    
    print("\n🚨 Testing Error Handling")
    print("=" * 50)
    
    # Test with malformed response
    try:
        structured_response = response_parser.parse_response(
            "Test query",
            "",  # Empty response
            [],
            0.0
        )
        print(f"✅ Handled empty response: {structured_response.response_type.value}")
    except Exception as e:
        print(f"❌ Failed to handle empty response: {e}")
    
    # Test with invalid data
    try:
        structured_response = response_parser.parse_response(
            "Test query",
            "Invalid response with no metrics",
            ["test.pdf"],
            1.0
        )
        print(f"✅ Handled invalid response: {structured_response.confidence:.2f}")
    except Exception as e:
        print(f"❌ Failed to handle invalid response: {e}")

if __name__ == "__main__":
    test_structured_output()
    test_agent_consumption()
    test_error_handling()
    
    print("\n🎉 All tests completed!")
    print("\n📝 Usage Instructions:")
    print("1. Run the structured RAG app: streamlit run ui/structured_rag_app.py")
    print("2. Upload documents and ask questions")
    print("3. Choose output format (JSON, Markdown, CSV, or Agent Format)")
    print("4. Download structured responses for agent consumption") 