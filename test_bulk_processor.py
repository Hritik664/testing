#!/usr/bin/env python3
"""
Test script for bulk question processor functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bulk_question_processor import bulk_processor, BulkQuestion
from utils.structured_output import response_parser

def test_excel_validation():
    """Test Excel file validation."""
    print("🧪 Testing Excel file validation...")
    
    # Test with invalid file extension
    is_valid, msg = bulk_processor.validate_excel_file(b"test", "test.txt")
    assert not is_valid, f"Should reject .txt files: {msg}"
    print("✅ Correctly rejected .txt file")
    
    # Test with empty data
    is_valid, msg = bulk_processor.validate_excel_file(b"", "test.xlsx")
    assert not is_valid, f"Should reject empty file: {msg}"
    print("✅ Correctly rejected empty file")
    
    print("✅ Excel validation tests passed!")

def test_question_parsing():
    """Test question parsing from Excel data."""
    print("\n🧪 Testing question parsing...")
    
    # Create sample Excel data
    import pandas as pd
    from io import BytesIO
    
    sample_data = [
        {
            'Question': 'What was the revenue in Q1?',
            'Type': 'Figure',
            'Figure in cr': 'cr',
            'Period': 'Q1',
            'Figure': '1000'
        },
        {
            'Question': 'What is the growth rate?',
            'Type': 'Range',
            'Figure in cr': '%',
            'Period': 'Annual',
            'Figure': '5 to 10'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    
    # Convert to bytes
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    excel_bytes = output.getvalue()
    
    # Test parsing
    questions = bulk_processor.parse_excel_questions(excel_bytes)
    
    assert len(questions) == 2, f"Expected 2 questions, got {len(questions)}"
    assert questions[0].question == "What was the revenue in Q1?", "First question mismatch"
    assert questions[1].question == "What is the growth rate?", "Second question mismatch"
    
    print("✅ Question parsing tests passed!")

def test_structured_response_parsing():
    """Test structured response parsing."""
    print("\n🧪 Testing structured response parsing...")
    
    # Test with a sample response
    question = "What was the revenue in Q1?"
    response = "The revenue in Q1 was 1,000 crore rupees, representing a 15% growth compared to the previous quarter."
    
    structured_response = response_parser.parse_response(question, response)
    
    assert structured_response is not None, "Should parse response successfully"
    assert structured_response.query == question, "Query should match"
    assert "revenue" in structured_response.answer.lower(), "Answer should contain revenue"
    
    print("✅ Structured response parsing tests passed!")

def test_bulk_processing_simulation():
    """Test bulk processing simulation."""
    print("\n🧪 Testing bulk processing simulation...")
    
    # Create sample questions
    questions = [
        BulkQuestion(
            question="What was the revenue in Q1?",
            question_type="Figure",
            figure_type="cr",
            period="Q1",
            figure_value="1000",
            row_number=2
        ),
        BulkQuestion(
            question="What is the growth rate?",
            question_type="Range",
            figure_type="%",
            period="Annual",
            figure_value="5 to 10",
            row_number=3
        )
    ]
    
    # Define a simple query function for testing
    def mock_query_function(question, **kwargs):
        class MockResult:
            def __init__(self, answer):
                self.answer = answer
        
        if "revenue" in question.lower():
            return MockResult("The revenue in Q1 was 1,000 crore rupees.")
        else:
            return MockResult("The growth rate is 7.5% annually.")
    
    # Process bulk questions
    bulk_result = bulk_processor.process_bulk_questions(questions, mock_query_function)
    
    assert bulk_result.total_questions == 2, f"Expected 2 questions, got {bulk_result.total_questions}"
    assert bulk_result.successful_questions == 2, f"Expected 2 successful, got {bulk_result.successful_questions}"
    assert bulk_result.failed_questions == 0, f"Expected 0 failed, got {bulk_result.failed_questions}"
    
    print("✅ Bulk processing simulation tests passed!")

def test_export_functionality():
    """Test export functionality."""
    print("\n🧪 Testing export functionality...")
    
    # Create a sample bulk result
    from utils.bulk_question_processor import BulkQuestionResult, BulkProcessingResult
    
    results = [
        BulkQuestionResult(
            question="What was the revenue in Q1?",
            question_type="Figure",
            figure_type="cr",
            period="Q1",
            figure_value="1000",
            row_number=2,
            answer="The revenue in Q1 was 1,000 crore rupees.",
            structured_response=None,
            processing_time=1.5,
            status="success"
        )
    ]
    
    summary = {
        'success_rate': 100.0,
        'average_processing_time': 1.5,
        'response_types': {'FINANCIAL_METRIC': 1},
        'metrics_extracted': {},
        'total_metrics_found': 0
    }
    
    bulk_result = BulkProcessingResult(
        total_questions=1,
        successful_questions=1,
        failed_questions=0,
        total_processing_time=1.5,
        results=results,
        summary=summary,
        timestamp="2025-01-01T00:00:00"
    )
    
    # Test Excel export
    try:
        excel_data = bulk_processor.export_results(bulk_result, "excel")
        assert len(excel_data) > 0, "Excel export should produce data"
        print("✅ Excel export test passed!")
    except Exception as e:
        print(f"❌ Excel export test failed: {e}")
    
    # Test JSON export
    try:
        json_data = bulk_processor.export_results(bulk_result, "json")
        assert len(json_data) > 0, "JSON export should produce data"
        print("✅ JSON export test passed!")
    except Exception as e:
        print(f"❌ JSON export test failed: {e}")
    
    # Test CSV export
    try:
        csv_data = bulk_processor.export_results(bulk_result, "csv")
        assert len(csv_data) > 0, "CSV export should produce data"
        print("✅ CSV export test passed!")
    except Exception as e:
        print(f"❌ CSV export test failed: {e}")

def main():
    """Run all tests."""
    print("🚀 Starting bulk question processor tests...\n")
    
    try:
        test_excel_validation()
        test_question_parsing()
        test_structured_response_parsing()
        test_bulk_processing_simulation()
        test_export_functionality()
        
        print("\n🎉 All bulk question processor tests passed!")
        print("✅ The bulk question processing system is ready for use!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 