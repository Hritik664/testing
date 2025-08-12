import os
import hashlib
from typing import List, Tuple, Optional
from pathlib import Path
import fitz  # PyMuPDF
from config import Config
from utils.logger import logger

class FileValidator:
    """File validation utilities for security and data integrity."""
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Validate file extension against allowed types."""
        file_ext = Path(filename).suffix.lower()
        return file_ext in Config.ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_file_size(file_bytes: bytes) -> bool:
        """Validate file size against maximum allowed size."""
        file_size = len(file_bytes)
        return file_size <= Config.MAX_FILE_SIZE
    
    @staticmethod
    def validate_pdf_content(file_bytes: bytes) -> Tuple[bool, str]:
        """Validate that the file is actually a valid PDF."""
        try:
            # Try to open with PyMuPDF to validate PDF structure
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page_count = len(doc)
            doc.close()
            
            if page_count == 0:
                return False, "PDF file contains no pages"
            
            return True, f"Valid PDF with {page_count} pages"
            
        except Exception as e:
            return False, f"Invalid PDF file: {str(e)}"
    
    @staticmethod
    def calculate_file_hash(file_bytes: bytes) -> str:
        """Calculate MD5 hash of file content."""
        return hashlib.md5(file_bytes).hexdigest()
    
    @staticmethod
    def validate_uploaded_file(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
        """Comprehensive file validation for uploads."""
        # Check file extension
        if not FileValidator.validate_file_extension(filename):
            return False, f"File type not allowed. Allowed types: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        
        # Check file size
        if not FileValidator.validate_file_size(file_bytes):
            max_size_mb = Config.MAX_FILE_SIZE // (1024 * 1024)
            return False, f"File too large. Maximum size: {max_size_mb}MB"
        
        # Validate PDF content
        is_valid_pdf, pdf_message = FileValidator.validate_pdf_content(file_bytes)
        if not is_valid_pdf:
            return False, pdf_message
        
        logger.info(f"File validation passed: {filename}")
        return True, "File validation passed"

class InputValidator:
    """Input validation utilities for user queries and API inputs."""
    
    @staticmethod
    def validate_query(query: str) -> Tuple[bool, str]:
        """Validate user query input."""
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query.strip()) < 3:
            return False, "Query must be at least 3 characters long"
        
        if len(query) > 2000:
            return False, "Query too long. Maximum 2000 characters allowed"
        
        # Check for potentially malicious content
        dangerous_patterns = [
            "javascript:", "data:", "vbscript:", "onload=", "onerror=",
            "<script", "</script>", "eval(", "exec(", "import "
        ]
        
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if pattern in query_lower:
                return False, "Query contains potentially unsafe content"
        
        return True, "Query validation passed"
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input to prevent injection attacks."""
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&']
        sanitized = text
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000] + "..."
        
        return sanitized.strip()

class PathValidator:
    """Path validation utilities for file operations."""
    
    @staticmethod
    def validate_directory_path(path: str) -> bool:
        """Validate that a directory path is safe and exists."""
        try:
            # Check if path is within allowed directories
            allowed_dirs = [
                Config.RAW_DATA_DIR,
                Config.PROCESSED_DATA_DIR,
                Config.VECTOR_STORE_DIR
            ]
            
            path_obj = Path(path).resolve()
            for allowed_dir in allowed_dirs:
                allowed_path = Path(allowed_dir).resolve()
                if path_obj.is_relative_to(allowed_path):
                    return True
            
            return False
            
        except Exception:
            return False
    
    @staticmethod
    def ensure_safe_filename(filename: str) -> str:
        """Ensure filename is safe for file system operations."""
        # Remove dangerous characters
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        safe_filename = filename
        for char in dangerous_chars:
            safe_filename = safe_filename.replace(char, '_')
        
        # Limit length
        if len(safe_filename) > 255:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:255-len(ext)] + ext
        
        return safe_filename 