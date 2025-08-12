#!/usr/bin/env python3
"""
Quick fix to rebuild vector database from processed documents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from fix_vector_db import fix_vector_database
        print("🔧 Running vector database fix...")
        success = fix_vector_database()
        
        if success:
            print("\n✅ SUCCESS! Vector database has been rebuilt.")
            print("🚀 Now you can restart the Streamlit app:")
            print("   streamlit run ui/structured_rag_app.py")
            print("\n💡 Your existing documents should now work properly!")
        else:
            print("\n❌ Failed to rebuild vector database.")
            print("💡 You may need to re-upload your PDF documents.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Please try re-uploading your PDF documents.")

if __name__ == "__main__":
    main()
