import logging
import os
from datetime import datetime
from typing import Optional
from config import Config

class Logger:
    """Centralized logging system for the Financial RAG application."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """Initialize the logger with proper configuration."""
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("financial_rag")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
            
            # File handler
            file_handler = logging.FileHandler(
                f"logs/financial_rag_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: Optional[Exception] = None):
        """Log error message with optional exception info."""
        if exc_info:
            self.logger.error(message, exc_info=True)
        else:
            self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def log_api_call(self, api_name: str, success: bool, response_time: float = None):
        """Log API call details."""
        status = "SUCCESS" if success else "FAILED"
        time_info = f" ({response_time:.2f}s)" if response_time else ""
        self.info(f"API Call - {api_name}: {status}{time_info}")
    
    def log_file_processing(self, filename: str, operation: str, success: bool):
        """Log file processing operations."""
        status = "SUCCESS" if success else "FAILED"
        self.info(f"File Processing - {filename} ({operation}): {status}")
    
    def log_query(self, query: str, model_used: str, success: bool):
        """Log user queries."""
        status = "SUCCESS" if success else "FAILED"
        # Truncate long queries for logging
        truncated_query = query[:100] + "..." if len(query) > 100 else query
        self.info(f"Query - Model: {model_used}, Status: {status}, Query: {truncated_query}")

# Global logger instance
logger = Logger() 