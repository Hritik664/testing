#!/usr/bin/env python3
"""
Fix vector database persistence issue
"""

import sys
import os
from pathlib import Path
import json
import shutil

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

def fix_vector_database():
    """Fix the vector database by rebuilding from processed documents"""
    
    print("🔧 Fixing vector database...")
    
    # Check processed documents
    processed_dir = Path(Config.PROCESSED_DATA_DIR)
    processed_files = list(processed_dir.glob("*.json")) if processed_dir.exists() else []
    
    print(f"📁 Found {len(processed_files)} processed documents:")
    for f in processed_files:
        print(f"   - {f.name}")
    
    if not processed_files:
        print("❌ No processed documents found. Please upload PDFs first.")
        return False
    
    # Clear vector database directory
    vector_dir = Path(Config.VECTOR_STORE_DIR)
    if vector_dir.exists():
        print("🗑️ Clearing existing vector database...")
        shutil.rmtree(vector_dir)
    
    vector_dir.mkdir(parents=True, exist_ok=True)
    print("✅ Created fresh vector database directory")
    
    # Load and process documents
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain.schema import Document
        from chromadb.config import Settings
        
        print("📝 Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=Config.get_embedding_model_name(),
            model_kwargs={'device': 'cuda' if Config.GPU_ENABLED else 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': Config.BATCH_SIZE}
        )
        
        # Load all processed documents
        all_documents = []
        for json_file in processed_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            filename = data.get('filename', json_file.stem)
            pages = data.get('pages', [])
            
            print(f"📄 Processing {filename} ({len(pages)} pages)...")
            
            for page in pages:
                page_num = page.get('page', page.get('page_number', 0))
                content = page.get('text', page.get('content', '')).strip()
                
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': filename,
                            'page': page_num,
                            'chunk_id': f"{filename}_page_{page_num}"
                        }
                    )
                    all_documents.append(doc)
        
        print(f"📚 Created {len(all_documents)} document chunks")
        
        if not all_documents:
            print("❌ No valid document content found")
            return False
        
        # Create vector database
        print("🔄 Creating vector database...")
        vectordb = Chroma.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            persist_directory=str(vector_dir),
            client_settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Verify the database
        collections = vectordb._client.list_collections()
        if collections:
            collection = vectordb._client.get_collection(name=collections[0].name)
            count = collection.count()
            print(f"✅ Vector database created with {count} documents")
            
            # Test query
            print("🧪 Testing with sample query...")
            results = vectordb.similarity_search("revenue", k=3)
            print(f"✅ Test query returned {len(results)} results")
            
            if results:
                print(f"   Sample result: {results[0].page_content[:100]}...")
            
            return True
        else:
            print("❌ No collections found in vector database")
            return False
            
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_vector_database()
    if success:
        print("\n🎉 Vector database fixed successfully!")
        print("💡 Now restart the Streamlit app and try querying existing documents.")
    else:
        print("\n❌ Failed to fix vector database")
        print("💡 You may need to re-upload your PDF documents.")
