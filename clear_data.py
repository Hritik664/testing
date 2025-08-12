#!/usr/bin/env python3
"""
Script to clear all stored data from the Financial RAG system.
"""

import os
import shutil
from pathlib import Path

def clear_all_data():
    """Clear all stored data including vector database, processed files, and cache."""
    
    print("🧹 Clearing all stored data...")
    
    # Clear vector database
    vector_store_path = Path("embeddings/vector_store")
    if vector_store_path.exists():
        shutil.rmtree(vector_store_path)
        print("✅ Cleared vector database")
    
    # Clear processed data
    processed_data_path = Path("data/processed")
    if processed_data_path.exists():
        for file in processed_data_path.glob("*.json"):
            file.unlink()
        print("✅ Cleared processed data files")
    
    # Clear cache
    cache_path = Path("cache")
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print("✅ Cleared cache")
    
    # Recreate necessary directories
    vector_store_path.mkdir(parents=True, exist_ok=True)
    processed_data_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print("🎉 All data cleared successfully!")
    print("📁 Directories recreated for fresh start")

def clear_vector_database_only():
    """Clear only the vector database (keeps processed files and cache)."""
    
    print("🗄️ Clearing vector database only...")
    
    vector_store_path = Path("embeddings/vector_store")
    if vector_store_path.exists():
        shutil.rmtree(vector_store_path)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        print("✅ Vector database cleared")
    else:
        print("ℹ️ Vector database already empty")

def clear_cache_only():
    """Clear only the cache (keeps vector database and processed files)."""
    
    print("🗑️ Clearing cache only...")
    
    cache_path = Path("cache")
    if cache_path.exists():
        shutil.rmtree(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        print("✅ Cache cleared")
    else:
        print("ℹ️ Cache already empty")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        option = sys.argv[1].lower()
        if option == "vector":
            clear_vector_database_only()
        elif option == "cache":
            clear_cache_only()
        elif option == "all":
            clear_all_data()
        else:
            print("Usage: python clear_data.py [all|vector|cache]")
            print("  all: Clear everything (vector DB, processed files, cache)")
            print("  vector: Clear only vector database")
            print("  cache: Clear only cache")
    else:
        # Default: clear all data
        clear_all_data() 