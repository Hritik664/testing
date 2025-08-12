#!/usr/bin/env python3
"""
Script to fix processed files by converting cached files to processed directory.
"""

import json
import shutil
from pathlib import Path
from config import Config

def fix_processed_files():
    """Convert cached PDF processing files to processed directory files."""
    
    print("🔧 Fixing processed files...")
    
    # Create processed directory if it doesn't exist
    processed_dir = Path(Config.PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Check cache directory
    cache_dir = Path("cache/pdf_processing")
    if not cache_dir.exists():
        print("ℹ️ No cache directory found")
        return
    
    # Find cached files
    cached_files = list(cache_dir.glob("*.json"))
    
    if not cached_files:
        print("ℹ️ No cached files found")
        return
    
    print(f"📁 Found {len(cached_files)} cached files")
    
    # Convert each cached file
    for cached_file in cached_files:
        try:
            # Read cached data
            with open(cached_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract filename from cached data
            filename = data.get('filename', cached_file.stem.split('_')[0])
            
            # Create processed file path
            processed_file = processed_dir / f"{filename}.json"
            
            # Save to processed directory
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Converted: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to convert {cached_file.name}: {e}")
    
    print(f"🎉 Fixed {len(cached_files)} processed files!")

if __name__ == "__main__":
    fix_processed_files() 