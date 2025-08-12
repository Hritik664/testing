#!/usr/bin/env python3
"""
Simple test to verify ChromaDB is working correctly.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings
from langchain.schema import Document

def test_chroma():
    """Test ChromaDB functionality."""
    
    print("🧪 Testing ChromaDB...")
    
    # Create test documents
    test_docs = [
        Document(
            page_content="Sales volume in Q1FY25 was 7.4 mnT",
            metadata={"source": "test", "filename": "test.pdf", "page": 1}
        ),
        Document(
            page_content="Revenue growth in Q1FY26 was 15%",
            metadata={"source": "test", "filename": "test.pdf", "page": 2}
        )
    ]
    
    try:
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name=Config.get_embedding_model_name(),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        print("✅ Embedding model loaded")
        
        # Create vector store
        vectordb = Chroma.from_documents(
            documents=test_docs,
            embedding=embedding_model,
            persist_directory=Config.VECTOR_STORE_DIR,
            client_settings=Settings(anonymized_telemetry=False)
        )
        
        print("✅ Vector store created")
        
        # Test retrieval
        results = vectordb.similarity_search("sales volume", k=1)
        print(f"✅ Retrieval test passed: {len(results)} results")
        
        # Check if files exist
        vectordb_path = Path(Config.VECTOR_STORE_DIR)
        chroma_files = list(vectordb_path.glob("*.sqlite3"))
        embedding_dirs = [d for d in vectordb_path.iterdir() if d.is_dir()]
        
        print(f"📁 ChromaDB files: {len(chroma_files)}")
        print(f"📁 Embedding directories: {len(embedding_dirs)}")
        
        if chroma_files:
            print("✅ ChromaDB database file exists")
        else:
            print("❌ ChromaDB database file missing")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_chroma() 