#!/usr/bin/env python3
"""
PDF Processing Diagnostic Script
-------------------------------
This script specifically tests PDF processing to identify why the app
stops after PDF upload on the other laptop.
"""

import sys
import os
import traceback
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pdf_reading():
    """Test basic PDF reading capabilities."""
    print("🔍 Testing PDF Reading...")
    
    try:
        import pypdf
        print(f"✅ PyPDF version: {pypdf.__version__}")
        
        # Check for PDF files
        raw_data_path = "data/raw"
        if not os.path.exists(raw_data_path):
            print(f"❌ Raw data directory not found: {raw_data_path}")
            return False
        
        pdf_files = [f for f in os.listdir(raw_data_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"❌ No PDF files found in {raw_data_path}")
            return False
        
        print(f"✅ Found {len(pdf_files)} PDF files")
        
        # Test reading the first PDF
        test_pdf = os.path.join(raw_data_path, pdf_files[0])
        print(f"📄 Testing with: {pdf_files[0]}")
        
        with open(test_pdf, 'rb') as f:
            reader = pypdf.PdfReader(f)
            pages = len(reader.pages)
            print(f"✅ Successfully read PDF with {pages} pages")
            
            # Test text extraction from first page
            if pages > 0:
                first_page = reader.pages[0]
                text = first_page.extract_text()
                print(f"✅ Extracted {len(text)} characters from first page")
                
                if len(text.strip()) == 0:
                    print("⚠️  Warning: First page appears to have no text (might be image-based)")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF reading failed: {e}")
        traceback.print_exc()
        return False

def test_pdf_processor():
    """Test the PDF processor module."""
    print("\n🔍 Testing PDF Processor...")
    
    try:
        from utils.optimized_pdf_processor import pdf_processor
        print("✅ PDF processor module imported successfully")
        
        # Check for PDF files
        raw_data_path = "data/raw"
        pdf_files = [f for f in os.listdir(raw_data_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print("❌ No PDF files available for testing")
            return False
        
        # Test with the first PDF
        test_pdf = os.path.join(raw_data_path, pdf_files[0])
        print(f"📄 Processing: {pdf_files[0]}")
        
        start_time = time.time()
        result = pdf_processor(test_pdf)
        processing_time = time.time() - start_time
        
        print(f"✅ PDF processing completed in {processing_time:.2f} seconds")
        print(f"✅ Generated {len(result) if result else 0} chunks")
        
        if result and len(result) > 0:
            # Show sample of first chunk
            first_chunk = result[0]
            print(f"✅ First chunk length: {len(first_chunk.page_content)} characters")
            print(f"✅ First chunk metadata: {first_chunk.metadata}")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        traceback.print_exc()
        return False

def test_embedding_generation():
    """Test embedding generation with processed chunks."""
    print("\n🔍 Testing Embedding Generation...")
    
    try:
        from utils.enhanced_embedder import enhanced_embedder
        from utils.optimized_pdf_processor import pdf_processor
        from config import Config
        
        print("✅ Embedder module imported successfully")
        
        # Get a sample PDF
        raw_data_path = "data/raw"
        pdf_files = [f for f in os.listdir(raw_data_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print("❌ No PDF files available for testing")
            return False
        
        # Process PDF to get chunks
        test_pdf = os.path.join(raw_data_path, pdf_files[0])
        chunks = pdf_processor(test_pdf)
        
        if not chunks:
            print("❌ No chunks generated from PDF")
            return False
        
        print(f"📄 Testing embeddings with {len(chunks)} chunks")
        
        # Test embedding generation
        start_time = time.time()
        embeddings = enhanced_embedder(chunks)
        embedding_time = time.time() - start_time
        
        print(f"✅ Embedding generation completed in {embedding_time:.2f} seconds")
        print(f"✅ Generated {len(embeddings) if embeddings else 0} embeddings")
        
        if embeddings and len(embeddings) > 0:
            print(f"✅ First embedding dimension: {len(embeddings[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        traceback.print_exc()
        return False

def test_vector_store():
    """Test vector store operations."""
    print("\n🔍 Testing Vector Store...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        vector_store_path = "embeddings/vector_store"
        
        # Check if vector store exists
        if not os.path.exists(vector_store_path):
            print(f"❌ Vector store directory not found: {vector_store_path}")
            return False
        
        print("✅ Vector store directory exists")
        
        # Initialize client
        client = chromadb.PersistentClient(
            path=vector_store_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        print("✅ ChromaDB client initialized")
        
        # Check collections
        collections = client.list_collections()
        print(f"✅ Found {len(collections)} collections")
        
        for collection in collections:
            try:
                count = collection.count()
                print(f"   📊 Collection '{collection.name}': {count} documents")
            except Exception as e:
                print(f"   ⚠️  Error counting collection '{collection.name}': {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage during processing."""
    print("\n🔍 Testing Memory Usage...")
    
    try:
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"✅ Initial memory usage: {initial_memory:.2f} MB")
        
        # Test PDF processing with memory monitoring
        from utils.optimized_pdf_processor import pdf_processor
        
        raw_data_path = "data/raw"
        pdf_files = [f for f in os.listdir(raw_data_path) if f.lower().endswith('.pdf')]
        
        if pdf_files:
            test_pdf = os.path.join(raw_data_path, pdf_files[0])
            
            # Monitor memory during processing
            chunks = pdf_processor(test_pdf)
            after_processing = process.memory_info().rss / 1024 / 1024
            print(f"✅ Memory after PDF processing: {after_processing:.2f} MB")
            print(f"✅ Memory increase: {after_processing - initial_memory:.2f} MB")
            
            # Test embedding generation
            from utils.enhanced_embedder import enhanced_embedder
            embeddings = enhanced_embedder(chunks)
            after_embedding = process.memory_info().rss / 1024 / 1024
            print(f"✅ Memory after embedding: {after_embedding:.2f} MB")
            print(f"✅ Total memory increase: {after_embedding - initial_memory:.2f} MB")
            
            # Clean up
            del chunks, embeddings
            gc.collect()
            after_cleanup = process.memory_info().rss / 1024 / 1024
            print(f"✅ Memory after cleanup: {after_cleanup:.2f} MB")
        
        return True
        
    except ImportError:
        print("⚠️  psutil not available - skipping memory test")
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_file_permissions():
    """Test file permissions for critical directories."""
    print("\n🔍 Testing File Permissions...")
    
    critical_dirs = [
        'data/raw',
        'data/processed', 
        'embeddings/vector_store',
        'logs'
    ]
    
    all_good = True
    
    for dir_path in critical_dirs:
        if os.path.exists(dir_path):
            readable = os.access(dir_path, os.R_OK)
            writable = os.access(dir_path, os.W_OK)
            
            status = "✅" if readable and writable else "❌"
            print(f"{status} {dir_path}: R={readable}, W={writable}")
            
            if not (readable and writable):
                all_good = False
        else:
            print(f"⚠️  {dir_path}: Directory does not exist")
    
    return all_good

def run_pdf_diagnostic():
    """Run all PDF-related diagnostic tests."""
    print("🚀 Starting PDF Processing Diagnostic...\n")
    
    tests = [
        ("File Permissions", test_file_permissions),
        ("PDF Reading", test_pdf_reading),
        ("PDF Processor", test_pdf_processor),
        ("Embedding Generation", test_embedding_generation),
        ("Vector Store", test_vector_store),
        ("Memory Usage", test_memory_usage)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PDF processing should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\n🔍 Common issues to check on the other laptop:")
        print("   - Missing Python packages")
        print("   - Insufficient memory")
        print("   - File permission issues")
        print("   - Corrupted PDF files")
        print("   - Network connectivity issues")
    
    return results

if __name__ == "__main__":
    run_pdf_diagnostic()
