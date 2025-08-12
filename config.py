import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Config:
    """Centralized configuration management for the Financial RAG system."""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    # API URLs
    GEMINI_API_URL: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    # File paths
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    VECTOR_STORE_DIR: str = "embeddings/vector_store"
    MODELS_DIR: str = "models/local_llms"
    
    # Embedding model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM settings
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE: float = 0.0
    
    # RAG settings
    CHUNK_SIZE: int = 800  # Optimized for better semantic chunks
    CHUNK_OVERLAP: int = 200  # Increased overlap for better context
    RETRIEVER_K: int = 8  # Increased for better retrieval
    MIN_TEXT_LENGTH: int = 50
    
    # Performance settings
    MAX_WORKERS: int = 4  # Parallel processing workers
    CACHE_ENABLED: bool = True
    GPU_ENABLED: bool = False  # Set to True if GPU available
    BATCH_SIZE: int = 32  # Embedding batch size
    
    # File upload settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {".pdf"}
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = 60
    
    @classmethod
    def validate_api_keys(cls) -> bool:
        """Validate that required API keys are present."""
        missing_keys = []
        
        if not cls.OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        if not cls.GEMINI_API_KEY:
            missing_keys.append("GEMINI_API_KEY")
            
        if missing_keys:
            print(f"❌ Missing required API keys: {', '.join(missing_keys)}")
            print("Please set these environment variables in your .env file")
            return False
            
        return True
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.VECTOR_STORE_DIR,
            cls.MODELS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_embedding_model_name(cls) -> str:
        """Get the embedding model name with validation."""
        if not cls.EMBEDDING_MODEL:
            raise ValueError("Embedding model not configured")
        return cls.EMBEDDING_MODEL 