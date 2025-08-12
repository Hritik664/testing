#!/usr/bin/env python3
"""
Test script to check JSON content structure
"""

import json
from pathlib import Path

def test_json_content():
    processed_dir = Path("data/processed")
    json_files = list(processed_dir.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files:")
    
    for json_file in json_files[:1]:  # Test first file only
        print(f"\n📁 Testing: {json_file.name}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Filename: {data.get('filename')}")
        print(f"Source: {data.get('source')}")
        
        pages = data.get('pages', [])
        print(f"Total pages: {len(pages)}")
        
        valid_pages = 0
        total_content_length = 0
        
        for i, page in enumerate(pages[:3]):  # Check first 3 pages
            page_num = page.get('page', page.get('page_number', 'unknown'))
            content = page.get('text', page.get('content', ''))
            word_count = page.get('word_count', 0)
            
            print(f"\nPage {page_num}:")
            print(f"  Word count: {word_count}")
            print(f"  Content length: {len(content)}")
            print(f"  Content preview: {content[:100]}...")
            
            if content.strip():
                valid_pages += 1
                total_content_length += len(content)
        
        print(f"\nSummary for {json_file.name}:")
        print(f"  Valid pages with content: {valid_pages}")
        print(f"  Total content length: {total_content_length}")

if __name__ == "__main__":
    test_json_content()
