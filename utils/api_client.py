import time
import requests
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
import threading

from config import Config
from utils.logger import logger

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def can_make_request(self, api_name: str) -> bool:
        """Check if a request can be made without exceeding rate limit."""
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.time_window)
            
            # Clean old requests
            self.requests[api_name] = [
                req_time for req_time in self.requests[api_name]
                if req_time > cutoff
            ]
            
            # Check if we can make a new request
            if len(self.requests[api_name]) < self.max_requests:
                self.requests[api_name].append(now)
                return True
            
            return False
    
    def wait_if_needed(self, api_name: str) -> float:
        """Wait if rate limit is exceeded and return wait time."""
        while not self.can_make_request(api_name):
            time.sleep(1)
        
        return 0.0

class APIClient:
    """Centralized API client with error handling and rate limiting."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(Config.MAX_REQUESTS_PER_MINUTE)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Financial-RAG/1.0'
        })
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling."""
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {str(e)}")
            raise
    
    def _log_api_call(self, api_name: str, success: bool, response_time: float = None):
        """Log API call details."""
        logger.log_api_call(api_name, success, response_time)

class GeminiClient(APIClient):
    """Gemini API client with proper error handling."""
    
    def __init__(self):
        super().__init__()
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
    
    def query(self, prompt: str) -> str:
        """Query Gemini API with proper error handling and rate limiting."""
        api_name = "Gemini"
        
        # Rate limiting
        wait_time = self.rate_limiter.wait_if_needed(api_name)
        if wait_time > 0:
            logger.warning(f"Rate limited, waited {wait_time:.2f}s")
        
        start_time = time.time()
        
        try:
            headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": Config.GEMINI_API_KEY
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ]
            }
            
            response = self._make_request(
                "POST",
                Config.GEMINI_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            result = response.json()
            response_time = time.time() - start_time
            
            # Extract response text
            try:
                response_text = result['candidates'][0]['content']['parts'][0]['text']
                self._log_api_call(api_name, True, response_time)
                return response_text
                
            except (KeyError, IndexError) as e:
                error_msg = f"Failed to parse Gemini response: {str(e)}"
                logger.error(error_msg)
                self._log_api_call(api_name, False, response_time)
                return f"❌ Gemini API Error: {error_msg}"
                
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Gemini API request failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._log_api_call(api_name, False, response_time)
            return f"❌ Gemini API Error: {error_msg}"

class OpenAIClient(APIClient):
    """OpenAI API client with proper error handling."""
    
    def __init__(self):
        super().__init__()
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        
        # Import here to avoid circular imports
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
    
    def query(self, prompt: str) -> str:
        """Query OpenAI API with proper error handling."""
        api_name = "OpenAI"
        
        # Rate limiting
        wait_time = self.rate_limiter.wait_if_needed(api_name)
        if wait_time > 0:
            logger.warning(f"Rate limited, waited {wait_time:.2f}s")
        
        start_time = time.time()
        
        try:
            response = self.llm.invoke(prompt)
            response_time = time.time() - start_time
            
            self._log_api_call(api_name, True, response_time)
            return response.content
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"OpenAI API request failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._log_api_call(api_name, False, response_time)
            return f"❌ OpenAI API Error: {error_msg}"

# Global API clients
gemini_client = None
openai_client = None

def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client singleton."""
    global gemini_client
    if gemini_client is None:
        gemini_client = GeminiClient()
    return gemini_client

def get_openai_client() -> OpenAIClient:
    """Get or create OpenAI client singleton."""
    global openai_client
    if openai_client is None:
        openai_client = OpenAIClient()
    return openai_client 