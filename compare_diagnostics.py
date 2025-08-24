#!/usr/bin/env python3
"""
Diagnostic Comparison Script
----------------------------
This script compares diagnostic results from two different laptops
to identify differences that might cause issues.
"""

import json
import sys
from typing import Dict, Any, List

def load_diagnostic_file(file_path: str) -> Dict[str, Any]:
    """Load diagnostic results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return {}

def compare_system_info(laptop1: Dict, laptop2: Dict) -> None:
    """Compare system information between laptops."""
    print("🖥️  SYSTEM INFORMATION COMPARISON")
    print("=" * 50)
    
    if not laptop1 or not laptop2:
        print("❌ Missing system information from one or both laptops")
        return
    
    sys1 = laptop1.get('system_info', {})
    sys2 = laptop2.get('system_info', {})
    
    # Compare key system attributes
    key_attrs = ['platform', 'python_version', 'memory_total_gb', 'memory_available_gb']
    
    for attr in key_attrs:
        val1 = sys1.get(attr, 'N/A')
        val2 = sys2.get(attr, 'N/A')
        
        if val1 != val2:
            print(f"⚠️  {attr}:")
            print(f"   Laptop 1: {val1}")
            print(f"   Laptop 2: {val2}")
        else:
            print(f"✅ {attr}: {val1}")

def compare_packages(laptop1: Dict, laptop2: Dict) -> None:
    """Compare Python package versions between laptops."""
    print("\n📦 PYTHON PACKAGES COMPARISON")
    print("=" * 50)
    
    if not laptop1 or not laptop2:
        print("❌ Missing package information from one or both laptops")
        return
    
    pkg1 = laptop1.get('python_packages', {})
    pkg2 = laptop2.get('python_packages', {})
    
    # Get all unique package names
    all_packages = set(pkg1.keys()) | set(pkg2.keys())
    
    differences = []
    missing_packages = []
    
    for package in sorted(all_packages):
        val1 = pkg1.get(package, 'NOT INSTALLED')
        val2 = pkg2.get(package, 'NOT INSTALLED')
        
        if val1 != val2:
            differences.append((package, val1, val2))
            
            # Check if one is missing
            if 'NOT INSTALLED' in str(val1) or 'ERROR' in str(val1):
                missing_packages.append((package, 'Laptop 1', val1))
            if 'NOT INSTALLED' in str(val2) or 'ERROR' in str(val2):
                missing_packages.append((package, 'Laptop 2', val2))
    
    if differences:
        print("⚠️  Package differences found:")
        for package, val1, val2 in differences:
            print(f"\n📦 {package}:")
            print(f"   Laptop 1: {val1}")
            print(f"   Laptop 2: {val2}")
    else:
        print("✅ All packages have same versions")
    
    if missing_packages:
        print("\n❌ Missing or problematic packages:")
        for package, laptop, status in missing_packages:
            print(f"   {laptop}: {package} - {status}")

def compare_embedding_system(laptop1: Dict, laptop2: Dict) -> None:
    """Compare embedding system status between laptops."""
    print("\n🔥 EMBEDDING SYSTEM COMPARISON")
    print("=" * 50)
    
    if not laptop1 or not laptop2:
        print("❌ Missing embedding information from one or both laptops")
        return
    
    emb1 = laptop1.get('embedding_system', {})
    emb2 = laptop2.get('embedding_system', {})
    
    # Compare torch information
    torch1 = emb1.get('torch', {})
    torch2 = emb2.get('torch', {})
    
    if isinstance(torch1, dict) and isinstance(torch2, dict):
        print("🔥 PyTorch Comparison:")
        print(f"   Laptop 1: v{torch1.get('version', 'Unknown')} - CUDA: {torch1.get('cuda_available', False)}")
        print(f"   Laptop 2: v{torch2.get('version', 'Unknown')} - CUDA: {torch2.get('cuda_available', False)}")
    
    # Compare sentence transformer
    st1 = emb1.get('sentence_transformer', {})
    st2 = emb2.get('sentence_transformer', {})
    
    if isinstance(st1, dict) and isinstance(st2, dict):
        print("\n🤖 Sentence Transformer:")
        print(f"   Laptop 1: {st1.get('model_name', 'Unknown')} on {st1.get('device', 'Unknown')}")
        print(f"   Laptop 2: {st2.get('model_name', 'Unknown')} on {st2.get('device', 'Unknown')}")
    
    # Compare embedding tests
    test1 = emb1.get('embedding_test', {})
    test2 = emb2.get('embedding_test', {})
    
    if isinstance(test1, dict) and isinstance(test2, dict):
        print("\n🧪 Embedding Test Results:")
        print(f"   Laptop 1: {'✅ Success' if test1.get('success', False) else '❌ Failed'}")
        print(f"   Laptop 2: {'✅ Success' if test2.get('success', False) else '❌ Failed'}")

def compare_vector_database(laptop1: Dict, laptop2: Dict) -> None:
    """Compare vector database status between laptops."""
    print("\n🗄️  VECTOR DATABASE COMPARISON")
    print("=" * 50)
    
    if not laptop1 or not laptop2:
        print("❌ Missing vector database information from one or both laptops")
        return
    
    vec1 = laptop1.get('vector_database', {})
    vec2 = laptop2.get('vector_database', {})
    
    # Compare client status
    client1 = vec1.get('chromadb_client', {})
    client2 = vec2.get('chromadb_client', {})
    
    print("🗄️  ChromaDB Client:")
    print(f"   Laptop 1: {'✅ Success' if isinstance(client1, dict) and client1.get('success', False) else '❌ Failed'}")
    print(f"   Laptop 2: {'✅ Success' if isinstance(client2, dict) and client2.get('success', False) else '❌ Failed'}")
    
    # Compare collections
    coll1 = vec1.get('collections', {})
    coll2 = vec2.get('collections', {})
    
    if isinstance(coll1, dict) and isinstance(coll2, dict):
        print(f"\n📊 Collections:")
        print(f"   Laptop 1: {coll1.get('count', 0)} collections")
        print(f"   Laptop 2: {coll2.get('count', 0)} collections")
    
    # Compare document counts
    docs1 = vec1.get('total_documents', 0)
    docs2 = vec2.get('total_documents', 0)
    
    print(f"\n📄 Total Documents:")
    print(f"   Laptop 1: {docs1}")
    print(f"   Laptop 2: {docs2}")

def compare_pdf_processing(laptop1: Dict, laptop2: Dict) -> None:
    """Compare PDF processing capabilities between laptops."""
    print("\n📄 PDF PROCESSING COMPARISON")
    print("=" * 50)
    
    if not laptop1 or not laptop2:
        print("❌ Missing PDF processing information from one or both laptops")
        return
    
    pdf1 = laptop1.get('pdf_processing', {})
    pdf2 = laptop2.get('pdf_processing', {})
    
    # Compare PyPDF
    pypdf1 = pdf1.get('pypdf', {})
    pypdf2 = pdf2.get('pypdf', {})
    
    if isinstance(pypdf1, dict) and isinstance(pypdf2, dict):
        print("📄 PyPDF:")
        print(f"   Laptop 1: v{pypdf1.get('version', 'Unknown')} - {'✅ Available' if pypdf1.get('available', False) else '❌ Not Available'}")
        print(f"   Laptop 2: v{pypdf2.get('version', 'Unknown')} - {'✅ Available' if pypdf2.get('available', False) else '❌ Not Available'}")
    
    # Compare available PDFs
    avail1 = pdf1.get('available_pdfs', {})
    avail2 = pdf2.get('available_pdfs', {})
    
    if isinstance(avail1, dict) and isinstance(avail2, dict):
        print(f"\n📁 Available PDFs:")
        print(f"   Laptop 1: {avail1.get('count', 0)} files")
        print(f"   Laptop 2: {avail2.get('count', 0)} files")
    
    # Compare processing tests
    proc1 = pdf1.get('pdf_processor_test', {})
    proc2 = pdf2.get('pdf_processor_test', {})
    
    if isinstance(proc1, dict) and isinstance(proc2, dict):
        print("\n⚙️  PDF Processor Test:")
        print(f"   Laptop 1: {'✅ Success' if proc1.get('success', False) else '❌ Failed'}")
        print(f"   Laptop 2: {'✅ Success' if proc2.get('success', False) else '❌ Failed'}")

def compare_environment(laptop1: Dict, laptop2: Dict) -> None:
    """Compare environment variables and configuration between laptops."""
    print("\n🔧 ENVIRONMENT COMPARISON")
    print("=" * 50)
    
    if not laptop1 or not laptop2:
        print("❌ Missing environment information from one or both laptops")
        return
    
    env1 = laptop1.get('environment_variables', {})
    env2 = laptop2.get('environment_variables', {})
    
    # Compare config values
    config1 = env1.get('config_values', {})
    config2 = env2.get('config_values', {})
    
    if isinstance(config1, dict) and isinstance(config2, dict):
        print("⚙️  Configuration Values:")
        key_configs = ['EMBEDDING_MODEL', 'GPU_ENABLED', 'BATCH_SIZE', 'CHUNK_SIZE']
        
        for key in key_configs:
            val1 = config1.get(key, 'N/A')
            val2 = config2.get(key, 'N/A')
            
            if val1 != val2:
                print(f"   ⚠️  {key}: {val1} vs {val2}")
            else:
                print(f"   ✅ {key}: {val1}")

def generate_recommendations(laptop1: Dict, laptop2: Dict) -> None:
    """Generate recommendations based on differences found."""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = []
    
    # Check for missing packages
    pkg1 = laptop1.get('python_packages', {})
    pkg2 = laptop2.get('python_packages', {})
    
    missing_in_2 = [pkg for pkg, info in pkg2.items() if 'NOT INSTALLED' in str(info) or 'ERROR' in str(info)]
    if missing_in_2:
        recommendations.append(f"📦 Install missing packages on Laptop 2: {', '.join(missing_in_2)}")
    
    # Check for memory issues
    sys1 = laptop1.get('system_info', {})
    sys2 = laptop2.get('system_info', {})
    
    if 'memory_available_gb' in sys2:
        available_memory = sys2.get('memory_available_gb', 0)
        if available_memory < 2.0:
            recommendations.append(f"💾 Low memory on Laptop 2: {available_memory}GB available. Consider closing other applications.")
    
    # Check for embedding issues
    emb2 = laptop2.get('embedding_system', {})
    if isinstance(emb2.get('embedding_test', {}), dict) and not emb2['embedding_test'].get('success', False):
        recommendations.append("🔥 Embedding system failing on Laptop 2. Check PyTorch installation and model downloads.")
    
    # Check for vector database issues
    vec2 = laptop2.get('vector_database', {})
    if isinstance(vec2.get('chromadb_client', {}), dict) and not vec2['chromadb_client'].get('success', False):
        recommendations.append("🗄️  Vector database failing on Laptop 2. Check file permissions and ChromaDB installation.")
    
    # Check for PDF processing issues
    pdf2 = laptop2.get('pdf_processing', {})
    if isinstance(pdf2.get('pdf_processor_test', {}), dict) and not pdf2['pdf_processor_test'].get('success', False):
        recommendations.append("📄 PDF processing failing on Laptop 2. Check PyPDF installation and file permissions.")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✅ No specific issues identified. Both laptops appear to have similar configurations.")

def compare_diagnostics(file1: str, file2: str) -> None:
    """Compare diagnostic results from two laptops."""
    print("🔍 COMPARING DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"Laptop 1: {file1}")
    print(f"Laptop 2: {file2}")
    print("=" * 60)
    
    # Load diagnostic files
    laptop1 = load_diagnostic_file(file1)
    laptop2 = load_diagnostic_file(file2)
    
    if not laptop1 or not laptop2:
        print("❌ Failed to load one or both diagnostic files")
        return
    
    # Run comparisons
    compare_system_info(laptop1, laptop2)
    compare_packages(laptop1, laptop2)
    compare_embedding_system(laptop1, laptop2)
    compare_vector_database(laptop1, laptop2)
    compare_pdf_processing(laptop1, laptop2)
    compare_environment(laptop1, laptop2)
    generate_recommendations(laptop1, laptop2)
    
    print("\n🎯 Next Steps:")
    print("1. Run the PDF processing diagnostic on the problematic laptop:")
    print("   python pdf_processing_diagnostic.py")
    print("2. Check the logs directory for any error messages")
    print("3. Try running the app with verbose logging enabled")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_diagnostics.py <laptop1_results.json> <laptop2_results.json>")
        print("\nExample:")
        print("python compare_diagnostics.py diagnostic_results_20241201_143022.json diagnostic_results_20241201_150045.json")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    compare_diagnostics(file1, file2)
