#!/usr/bin/env python3
"""
Script to check and fix vector database issues.
"""

import os
import shutil
from pathlib import Path
from config import Config

def check_vector_database():
    """Check the status of the vector database."""
    
    print("🔍 Checking vector database...")
    
    vectordb_path = Path(Config.VECTOR_STORE_DIR)
    
    if not vectordb_path.exists():
        print("❌ Vector database directory does not exist")
        return False
    
    # Check for ChromaDB files
    chroma_files = list(vectordb_path.glob("*.sqlite3"))
    if not chroma_files:
        print("❌ No ChromaDB database file found")
        return False
    
    # Check for embedding directories
    embedding_dirs = [d for d in vectordb_path.iterdir() if d.is_dir() and d.name != "__pycache__"]
    if not embedding_dirs:
        print("❌ No embedding directories found")
        return False
    
    print(f"✅ Vector database exists with {len(embedding_dirs)} embedding directories")
    return True

def reset_vector_database():
    """Reset the vector database by removing and recreating it."""
    
    print("🔄 Resetting vector database...")
    
    vectordb_path = Path(Config.VECTOR_STORE_DIR)
    
    if vectordb_path.exists():
        try:
            shutil.rmtree(vectordb_path)
            print("✅ Removed existing vector database")
        except Exception as e:
            print(f"❌ Failed to remove vector database: {e}")
            return False
    
    # Recreate directory
    vectordb_path.mkdir(parents=True, exist_ok=True)
    print("✅ Created new vector database directory")
    
    return True

def rebuild_vector_database():
    """Rebuild the vector database from processed documents."""
    
    print("🔨 Rebuilding vector database...")
    
    try:
        from utils.enhanced_embedder import enhanced_embedder
        import json
        from pathlib import Path
        
        # Load processed documents
        processed_dir = Path(Config.PROCESSED_DATA_DIR)
        if not processed_dir.exists():
            print("❌ No processed documents directory found")
            return False
        
        json_files = list(processed_dir.glob("*.json"))
        if not json_files:
            print("❌ No processed documents found")
            return False
        
        print(f"📁 Found {len(json_files)} processed documents")
        
        # Load processed data
        processed_data = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    processed_data.append(data)
            except Exception as e:
                print(f"❌ Failed to load {json_file.name}: {e}")
        
        if not processed_data:
            print("❌ No valid processed data found")
            return False
        
        # Create documents and embed
        documents = enhanced_embedder.process_documents(processed_data)
        enhanced_embedder.embed_and_store(documents, force_reload=True)
        
        print(f"✅ Successfully rebuilt vector database with {len(documents)} documents")
        return True
        
    except Exception as e:
        print(f"❌ Failed to rebuild vector database: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            check_vector_database()
        elif command == "reset":
            reset_vector_database()
        elif command == "rebuild":
            rebuild_vector_database()
        else:
            print("Usage: python check_vector_db.py [check|reset|rebuild]")
    else:
        # Default: check status
        if not check_vector_database():
            print("\n🔧 Vector database has issues. Run 'python check_vector_db.py reset' to fix.")
        else:
            print("\n✅ Vector database is healthy!") 