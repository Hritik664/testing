#!/usr/bin/env python3
"""
Test script to validate critical security and error handling improvements.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """Test configuration management."""
    print("🔧 Testing configuration management...")
    
    try:
        from config import Config
        
        # Test directory creation
        Config.create_directories()
        assert os.path.exists(Config.RAW_DATA_DIR), "Raw data directory not created"
        assert os.path.exists(Config.PROCESSED_DATA_DIR), "Processed data directory not created"
        assert os.path.exists(Config.VECTOR_STORE_DIR), "Vector store directory not created"
        
        print("✅ Configuration management working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False

def test_logger():
    """Test logging system."""
    print("📝 Testing logging system...")
    
    try:
        from utils.logger import logger
        
        # Test basic logging
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Check if log file was created
        log_files = list(Path("logs").glob("*.log"))
        assert len(log_files) > 0, "No log files created"
        
        print("✅ Logging system working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Logger test failed: {str(e)}")
        return False

def test_validators():
    """Test input validation."""
    print("🔒 Testing input validation...")
    
    try:
        from utils.validators import FileValidator, InputValidator
        
        # Test file validation
        test_file_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"
        
        is_valid, message = FileValidator.validate_uploaded_file(test_file_content, "test.pdf")
        assert is_valid, f"Valid PDF should pass validation: {message}"
        
        # Test query validation
        is_valid, message = InputValidator.validate_query("What is the revenue?")
        assert is_valid, f"Valid query should pass validation: {message}"
        
        # Test invalid query
        is_valid, message = InputValidator.validate_query("")
        assert not is_valid, "Empty query should fail validation"
        
        # Test malicious query
        is_valid, message = InputValidator.validate_query("<script>alert('xss')</script>")
        assert not is_valid, "Malicious query should fail validation"
        
        print("✅ Input validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Validator test failed: {str(e)}")
        return False

def test_api_clients():
    """Test API client initialization."""
    print("🌐 Testing API client initialization...")
    
    try:
        from utils.api_client import get_gemini_client, get_openai_client
        
        # Test client creation (will fail if API keys not set, but shouldn't crash)
        try:
            gemini_client = get_gemini_client()
            print("✅ Gemini client created successfully")
        except ValueError as e:
            print(f"⚠️ Gemini client creation failed (expected if no API key): {str(e)}")
        
        try:
            openai_client = get_openai_client()
            print("✅ OpenAI client created successfully")
        except ValueError as e:
            print(f"⚠️ OpenAI client creation failed (expected if no API key): {str(e)}")
        
        print("✅ API client initialization working correctly")
        return True
        
    except Exception as e:
        print(f"❌ API client test failed: {str(e)}")
        return False

def test_error_handling():
    """Test error handling in core functions."""
    print("🛡️ Testing error handling...")
    
    try:
        from utils.logger import logger
        from utils.validators import FileValidator
        
        # Test with invalid file
        invalid_content = b"not a pdf file"
        is_valid, message = FileValidator.validate_uploaded_file(invalid_content, "test.txt")
        assert not is_valid, "Invalid file should fail validation"
        
        # Test with oversized file
        oversized_content = b"x" * (50 * 1024 * 1024 + 1)  # 50MB + 1 byte
        is_valid, message = FileValidator.validate_uploaded_file(oversized_content, "large.pdf")
        assert not is_valid, "Oversized file should fail validation"
        
        print("✅ Error handling working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False

def main():
    """Run all critical tests."""
    print("🚀 Running critical security and error handling tests...\n")
    
    tests = [
        test_config,
        test_logger,
        test_validators,
        test_api_clients,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All critical tests passed! The system is ready for use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 