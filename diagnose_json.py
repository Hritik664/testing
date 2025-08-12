#!/usr/bin/env python3
"""
Diagnose JSON file structure
"""

import json
from pathlib import Path

def diagnose_json_files():
    """Show the exact structure of JSON files"""
    
    processed_dir = Path("data/processed")
    json_files = list(processed_dir.glob("*.json"))
    
    print("🔍 JSON File Structure Diagnosis")
    print("=" * 50)
    
    for json_file in json_files:
        print(f"\n📁 File: {json_file.name}")
        print("-" * 30)
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Show top-level keys
            print(f"Top-level keys: {list(data.keys())}")
            
            # Check filename
            if 'filename' in data:
                print(f"Filename: {data['filename']}")
            
            # Check pages structure
            if 'pages' in data:
                pages = data['pages']
                print(f"Number of pages: {len(pages)}")
                
                if pages:
                    # Show structure of first page
                    first_page = pages[0]
                    print(f"First page keys: {list(first_page.keys())}")
                    
                    # Check for content fields
                    content_field = None
                    content = ""
                    
                    if 'text' in first_page:
                        content_field = 'text'
                        content = first_page['text']
                    elif 'content' in first_page:
                        content_field = 'content'
                        content = first_page['content']
                    elif 'page_content' in first_page:
                        content_field = 'page_content'
                        content = first_page['page_content']
                    
                    if content_field:
                        print(f"Content field: '{content_field}'")
                        print(f"Content length: {len(content)}")
                        print(f"Content preview: {content[:100]}...")
                    else:
                        print("❌ No content field found!")
                        print(f"Available fields: {list(first_page.keys())}")
                    
                    # Count pages with content
                    pages_with_content = 0
                    total_content_length = 0
                    
                    for page in pages:
                        page_content = (page.get('text', '') or 
                                      page.get('content', '') or 
                                      page.get('page_content', ''))
                        
                        if page_content and page_content.strip():
                            pages_with_content += 1
                            total_content_length += len(page_content)
                    
                    print(f"Pages with content: {pages_with_content}/{len(pages)}")
                    print(f"Total content length: {total_content_length}")
            
        except Exception as e:
            print(f"❌ Error reading {json_file.name}: {e}")

if __name__ == "__main__":
    diagnose_json_files()
