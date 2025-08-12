#!/usr/bin/env python3
"""
Rebuild vector database from existing processed documents
"""

import sys
import os
from pathlib import Path
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.logger import logger
from utils.enhanced_embedder import EnhancedEmbedder

def rebuild_vector_database():
    """Rebuild the vector database from processed documents"""
    
    print("🔨 Rebuilding vector database from processed documents...")
    
    # Initialize embedder
    embedder = EnhancedEmbedder()
    
    # Load processed documents
    processed_dir = Path(Config.PROCESSED_DATA_DIR)
    if not processed_dir.exists():
        print("❌ No processed documents directory found")
        return
    
    processed_files = list(processed_dir.glob("*.json"))
    if not processed_files:
        print("❌ No processed documents found")
        return
    
    print(f"📁 Found {len(processed_files)} processed documents")
    
    # Load all processed data
    all_processed_data = []
    for json_file in processed_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_processed_data.append(data)
                print(f"✅ Loaded: {data.get('filename', json_file.name)}")
        except Exception as e:
            print(f"❌ Failed to load {json_file.name}: {e}")
    
    if not all_processed_data:
        print("❌ No valid processed data found")
        return
    
    # Create documents for embedding
    print("📝 Creating documents for embedding...")
    documents = embedder.process_documents(all_processed_data)
    
    if not documents:
        print("❌ No documents created for embedding")
        return
    
    print(f"📄 Created {len(documents)} document chunks")
    
    # Clear existing vector database and create new one
    print("🗑️ Clearing existing vector database...")
    vector_store_dir = Path(Config.VECTOR_STORE_DIR)
    if vector_store_dir.exists():
        import shutil
        shutil.rmtree(vector_store_dir)
        print("✅ Cleared existing vector database")
    
    # Embed and store documents
    print("🔄 Embedding and storing documents...")
    try:
        vectordb = embedder.embed_and_store(documents, force_reload=True)
        
        # Verify the vector database
        collections = vectordb._client.list_collections()
        if collections:
            collection = vectordb._client.get_collection(name=collections[0].name)
            count = collection.count()
            print(f"✅ Successfully created vector database with {count} documents")
            
            # Test a sample query
            print("🧪 Testing vector database with sample query...")
            test_results = vectordb.similarity_search("revenue", k=3)
            print(f"✅ Test query returned {len(test_results)} results")
            
            return True
        else:
            print("❌ No collections found in vector database")
            return False
            
    except Exception as e:
        print(f"❌ Failed to embed and store documents: {e}")
        logger.error(f"Failed to rebuild vector database: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = rebuild_vector_database()
    if success:
        print("\n🎉 Vector database rebuilt successfully!")
        print("💡 You can now restart the Streamlit app and use existing documents.")
    else:
        print("\n❌ Failed to rebuild vector database")
        print("💡 Try uploading the PDFs again through the Streamlit app.")
