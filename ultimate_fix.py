#!/usr/bin/env python3
"""
Ultimate fix for vector database - handles all JSON structures
"""

import sys
import os
import json
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

def ultimate_fix():
    """Ultimate fix that handles all possible JSON structures"""
    
    print("🔧 Ultimate Vector Database Fix")
    print("=" * 50)
    
    # Step 1: Check processed documents
    processed_dir = Path(Config.PROCESSED_DATA_DIR)
    if not processed_dir.exists():
        print("❌ No processed documents directory found")
        return False
    
    json_files = list(processed_dir.glob("*.json"))
    if not json_files:
        print("❌ No processed JSON files found")
        return False
    
    print(f"📁 Found {len(json_files)} processed files:")
    for f in json_files:
        print(f"   - {f.name}")
    
    # Step 2: Analyze JSON structure
    print(f"\n🔍 Analyzing JSON structure...")
    all_documents = []
    
    for json_file in json_files:
        print(f"\n📄 Processing: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filename = data.get('filename', json_file.stem)
            pages = data.get('pages', [])
            
            print(f"   Filename: {filename}")
            print(f"   Pages: {len(pages)}")
            
            # Check different possible structures
            valid_pages = 0
            for i, page in enumerate(pages):
                # Try different field combinations
                page_num = (page.get('page') or 
                           page.get('page_number') or 
                           page.get('page_num') or 
                           i + 1)
                
                content = (page.get('text') or 
                          page.get('content') or 
                          page.get('page_content') or 
                          '')
                
                if isinstance(content, str) and content.strip():
                    valid_pages += 1
                    
                    # Create document
                    from langchain.schema import Document
                    doc = Document(
                        page_content=content.strip(),
                        metadata={
                            'source': filename,
                            'page': page_num,
                            'chunk_id': f"{filename}_page_{page_num}",
                            'word_count': page.get('word_count', len(content.split()))
                        }
                    )
                    all_documents.append(doc)
            
            print(f"   Valid pages with content: {valid_pages}")
            
        except Exception as e:
            print(f"   ❌ Error processing {json_file.name}: {e}")
    
    if not all_documents:
        print(f"\n❌ No valid documents found across all JSON files")
        print("💡 The JSON files might be corrupted or have unexpected structure")
        return False
    
    print(f"\n📚 Total documents created: {len(all_documents)}")
    
    # Step 3: Clear and recreate vector database
    print(f"\n🗑️ Clearing vector database...")
    vector_dir = Path(Config.VECTOR_STORE_DIR)
    if vector_dir.exists():
        shutil.rmtree(vector_dir)
    vector_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 4: Create vector database
    try:
        print(f"🔄 Creating vector database...")
        
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from chromadb.config import Settings
        
        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name=Config.get_embedding_model_name(),
            model_kwargs={'device': 'cuda' if Config.GPU_ENABLED else 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': Config.BATCH_SIZE}
        )
        
        # Create vector database
        vectordb = Chroma.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            persist_directory=str(vector_dir),
            client_settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Verify
        collections = vectordb._client.list_collections()
        if collections:
            collection = vectordb._client.get_collection(name=collections[0].name)
            count = collection.count()
            print(f"✅ Vector database created with {count} documents")
            
            # Test query
            print(f"🧪 Testing with sample queries...")
            test_queries = ["revenue", "sales", "cost", "capacity", "guidance"]
            
            for query in test_queries[:3]:
                try:
                    results = vectordb.similarity_search(query, k=2)
                    print(f"   Query '{query}': {len(results)} results")
                    if results:
                        print(f"      Sample: {results[0].page_content[:80]}...")
                except Exception as e:
                    print(f"   Query '{query}': Error - {e}")
            
            return True
        else:
            print("❌ No collections created")
            return False
            
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Ultimate Vector Database Fix...")
    success = ultimate_fix()
    
    if success:
        print("\n" + "="*50)
        print("🎉 SUCCESS! Vector database has been rebuilt!")
        print("🚀 Now restart your Streamlit app:")
        print("   streamlit run ui/structured_rag_app.py")
        print("\n💡 Your existing documents should now work properly!")
    else:
        print("\n" + "="*50)
        print("❌ FAILED to rebuild vector database")
        print("💡 Please try re-uploading your PDF documents through the Streamlit app")
        print("   The JSON files might be corrupted or have unexpected structure")
