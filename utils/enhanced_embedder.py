import os
import json
from typing import List, Dict, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import Config
from utils.logger import logger
from pathlib import Path
import hashlib

class SemanticChunker:
    """Advanced semantic chunking for better document segmentation."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def _extract_financial_entities(self, text: str) -> List[str]:
        """Extract financial entities from text."""
        import re
        
        # Financial patterns
        patterns = [
            r'\$[\d,]+(?:\.\d{2})?',  # Dollar amounts
            r'[\d,]+(?:\.\d{2})?%',   # Percentages
            r'Q[1-4]\s+\d{4}',        # Quarters
            r'FY\s+\d{4}',             # Fiscal years
            r'[\d,]+(?:\.\d{2})?\s*(?:million|billion|trillion)',  # Large numbers
            r'(?:revenue|profit|earnings|income|expense|cost|margin)',  # Financial terms
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _calculate_semantic_similarity(self, chunk1: str, chunk2: str) -> float:
        """Calculate semantic similarity between chunks."""
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            vectors = vectorizer.fit_transform([chunk1, chunk2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def _merge_similar_chunks(self, chunks: List[str]) -> List[str]:
        """Merge semantically similar chunks."""
        if len(chunks) <= 1:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for i in range(1, len(chunks)):
            similarity = self._calculate_semantic_similarity(current_chunk, chunks[i])
            
            if similarity > 0.7 and len(current_chunk + " " + chunks[i]) <= self.chunk_size * 1.5:
                current_chunk += " " + chunks[i]
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunks[i]
        
        merged_chunks.append(current_chunk)
        return merged_chunks
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Document]:
        """Create semantic chunks from text."""
        # Basic chunking
        chunks = self.splitter.split_text(text)
        
        # Merge similar chunks
        merged_chunks = self._merge_similar_chunks(chunks)
        
        # Create documents with enhanced metadata
        documents = []
        for i, chunk in enumerate(merged_chunks):
            # Extract financial entities
            entities = self._extract_financial_entities(chunk)
            
            # Enhanced metadata
            enhanced_metadata = {
                **metadata,
                "chunk_id": i,
                "chunk_size": len(chunk),
                "word_count": len(chunk.split()),
                "financial_entities": entities,
                "entity_count": len(entities),
                "has_numbers": any(char.isdigit() for char in chunk),
                "has_percentages": "%" in chunk,
                "has_currency": "$" in chunk or "USD" in chunk.upper(),
            }
            
            documents.append(Document(
                page_content=chunk,
                metadata=enhanced_metadata
            ))
        
        return documents

class EnhancedEmbedder:
    """High-performance embedding system with semantic chunking and caching."""
    
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Config.get_embedding_model_name(),
            model_kwargs={'device': 'cuda' if Config.GPU_ENABLED else 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': Config.BATCH_SIZE}
        )
        self.chunker = SemanticChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.cache_dir = Path("cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_embedding_cache_key(self, documents: List[Document]) -> str:
        """Generate cache key for embeddings."""
        content_hash = hashlib.md5()
        for doc in documents:
            content_hash.update(doc.page_content.encode())
        return content_hash.hexdigest()
    
    def _load_embeddings_from_cache(self, cache_key: str) -> Optional[List[Document]]:
        """Load embeddings from cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    return [Document(**doc_data) for doc_data in cached_data]
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
        return None
    
    def _save_embeddings_to_cache(self, cache_key: str, documents: List[Document]):
        """Save embeddings to cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            cached_data = [doc.dict() for doc in documents]
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def process_documents(self, processed_data: List[Dict]) -> List[Document]:
        """Process documents with semantic chunking and caching."""
        all_documents = []
        
        for data in tqdm(processed_data, desc="Processing documents"):
            try:
                # Create base metadata
                base_metadata = {
                    "filename": data["filename"],
                    "source": data["source"],
                    "total_pages": data["total_pages"],
                    "total_words": data["total_words"],
                    "processed_at": data["processed_at"]
                }
                
                # Process each page
                for page in data["pages"]:
                    if len(page["text"].strip()) < Config.MIN_TEXT_LENGTH:
                        continue
                    
                    # Create page-specific metadata
                    page_metadata = {
                        **base_metadata,
                        "page": page["page"],
                        "page_word_count": page["word_count"]
                    }
                    
                    # Create semantic chunks
                    chunks = self.chunker.chunk_text(page["text"], page_metadata)
                    all_documents.extend(chunks)
                
            except Exception as e:
                logger.error(f"Failed to process document {data.get('filename', 'unknown')}: {str(e)}")
                continue
        
        return all_documents
    
    def embed_and_store(self, documents: List[Document], force_reload: bool = False):
        """Embed documents and store in vector database with caching."""
        if not documents:
            logger.warning("No documents to embed")
            return
        
        # Generate cache key
        cache_key = self._get_embedding_cache_key(documents)
        
        # Check cache if not forcing reload
        if not force_reload:
            cached_docs = self._load_embeddings_from_cache(cache_key)
            if cached_docs:
                logger.info(f"Loaded {len(cached_docs)} documents from embedding cache")
                documents = cached_docs
        
        try:
            # Clean metadata
            documents = filter_complex_metadata(documents)
            
            # Create or update vector store
            if os.path.exists(Config.VECTOR_STORE_DIR) and not force_reload:
                # Update existing store
                vectordb = Chroma(
                    persist_directory=Config.VECTOR_STORE_DIR,
                    embedding_function=self.embedding_model,
                    client_settings=Settings(
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
                vectordb.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to existing vector store")
            else:
                # Create new store
                vectordb = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    persist_directory=Config.VECTOR_STORE_DIR,
                    client_settings=Settings(
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
                logger.info(f"Created new vector store with {len(documents)} documents")
            
            # Save to cache
            self._save_embeddings_to_cache(cache_key, documents)
            
            logger.log_file_processing("embeddings", "embed_and_store", True)
            print(f"✅ Stored {len(documents)} embeddings to {Config.VECTOR_STORE_DIR}")
            
            return vectordb
            
        except Exception as e:
            logger.error(f"Failed to embed and store documents: {str(e)}", exc_info=True)
            print(f"❌ Failed to embed documents: {str(e)}")
            raise

# Global embedder instance
enhanced_embedder = EnhancedEmbedder() 