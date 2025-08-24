#!/usr/bin/env python3
"""
Comprehensive Diagnostic Script
-------------------------------
This script performs a thorough check of all system components to identify
why the RAG app works on one laptop but not another.
"""

import sys
import os
import platform
import subprocess
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_command(cmd: str) -> Dict[str, Any]:
    """Run a command and return results."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Command timed out',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def check_system_info() -> Dict[str, Any]:
    """Check basic system information."""
    print("🔍 Checking System Information...")
    
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'architecture': platform.architecture(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'node': platform.node(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version()
    }
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        info['memory_total_gb'] = round(memory.total / (1024**3), 2)
        info['memory_available_gb'] = round(memory.available / (1024**3), 2)
        info['memory_percent'] = memory.percent
    except ImportError:
        info['memory_info'] = "psutil not available"
    
    return info

def check_python_packages() -> Dict[str, Any]:
    """Check installed Python packages and versions."""
    print("🔍 Checking Python Packages...")
    
    packages = {}
    
    # Check critical packages
    critical_packages = [
        'torch', 'transformers', 'sentence-transformers', 'langchain',
        'langchain_huggingface', 'chromadb', 'streamlit', 'pandas',
        'numpy', 'pypdf', 'python-dotenv', 'requests'
    ]
    
    for package in critical_packages:
        try:
            if package == 'torch':
                import torch
                packages[package] = {
                    'version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
                }
            elif package == 'transformers':
                import transformers
                packages[package] = transformers.__version__
            elif package == 'sentence-transformers':
                import sentence_transformers
                packages[package] = sentence_transformers.__version__
            elif package == 'langchain':
                import langchain
                packages[package] = langchain.__version__
            elif package == 'langchain_huggingface':
                import langchain_huggingface
                packages[package] = langchain_huggingface.__version__
            elif package == 'chromadb':
                import chromadb
                packages[package] = chromadb.__version__
            elif package == 'streamlit':
                import streamlit
                packages[package] = streamlit.__version__
            elif package == 'pandas':
                import pandas
                packages[package] = pandas.__version__
            elif package == 'numpy':
                import numpy
                packages[package] = numpy.__version__
            elif package == 'pypdf':
                import pypdf
                packages[package] = pypdf.__version__
            elif package == 'python-dotenv':
                import dotenv
                packages[package] = dotenv.__version__
            elif package == 'requests':
                import requests
                packages[package] = requests.__version__
        except ImportError as e:
            packages[package] = f"NOT INSTALLED: {e}"
        except Exception as e:
            packages[package] = f"ERROR: {e}"
    
    return packages

def check_project_structure() -> Dict[str, Any]:
    """Check project directory structure and files."""
    print("🔍 Checking Project Structure...")
    
    structure = {
        'current_directory': os.getcwd(),
        'project_root': os.path.dirname(os.path.abspath(__file__)),
        'directories': {},
        'critical_files': {},
        'data_files': {}
    }
    
    # Check critical directories
    critical_dirs = [
        'data/raw', 'data/processed', 'embeddings/vector_store',
        'utils', 'ui', 'models/local_llms', 'logs'
    ]
    
    for dir_path in critical_dirs:
        full_path = os.path.join(os.getcwd(), dir_path)
        structure['directories'][dir_path] = {
            'exists': os.path.exists(full_path),
            'is_dir': os.path.isdir(full_path) if os.path.exists(full_path) else False,
            'readable': os.access(full_path, os.R_OK) if os.path.exists(full_path) else False,
            'writable': os.access(full_path, os.W_OK) if os.path.exists(full_path) else False
        }
    
    # Check critical files
    critical_files = [
        'config.py', 'requirements.txt', '.env',
        'ui/structured_rag_app.py', 'utils/logger.py'
    ]
    
    for file_path in critical_files:
        full_path = os.path.join(os.getcwd(), file_path)
        structure['critical_files'][file_path] = {
            'exists': os.path.exists(full_path),
            'is_file': os.path.isfile(full_path) if os.path.exists(full_path) else False,
            'readable': os.access(full_path, os.R_OK) if os.path.exists(full_path) else False,
            'size': os.path.getsize(full_path) if os.path.exists(full_path) else 0
        }
    
    # Check data files
    data_dirs = ['data/raw', 'data/processed', 'embeddings/vector_store']
    for data_dir in data_dirs:
        full_path = os.path.join(os.getcwd(), data_dir)
        if os.path.exists(full_path):
            try:
                files = os.listdir(full_path)
                structure['data_files'][data_dir] = {
                    'file_count': len(files),
                    'files': files[:10]  # First 10 files only
                }
            except Exception as e:
                structure['data_files'][data_dir] = f"ERROR: {e}"
    
    return structure

def check_environment_variables() -> Dict[str, Any]:
    """Check environment variables and configuration."""
    print("🔍 Checking Environment Variables...")
    
    env_info = {
        'env_file_exists': os.path.exists('.env'),
        'environment_variables': {},
        'config_values': {}
    }
    
    # Check critical environment variables
    critical_env_vars = [
        'OPENAI_API_KEY', 'GEMINI_API_KEY', 'PYTHONPATH',
        'CUDA_VISIBLE_DEVICES', 'TOKENIZERS_PARALLELISM'
    ]
    
    for var in critical_env_vars:
        value = os.getenv(var)
        if value:
            # Mask API keys for security
            if 'API_KEY' in var:
                env_info['environment_variables'][var] = f"SET ({len(value)} chars)"
            else:
                env_info['environment_variables'][var] = value
        else:
            env_info['environment_variables'][var] = "NOT SET"
    
    # Check config values
    try:
        from config import Config
        env_info['config_values'] = {
            'EMBEDDING_MODEL': Config.EMBEDDING_MODEL,
            'GPU_ENABLED': Config.GPU_ENABLED,
            'BATCH_SIZE': Config.BATCH_SIZE,
            'CHUNK_SIZE': Config.CHUNK_SIZE,
            'CHUNK_OVERLAP': Config.CHUNK_OVERLAP,
            'MAX_WORKERS': Config.MAX_WORKERS
        }
    except Exception as e:
        env_info['config_values'] = f"ERROR: {e}"
    
    return env_info

def check_embedding_system() -> Dict[str, Any]:
    """Check embedding system specifically."""
    print("🔍 Checking Embedding System...")
    
    embedding_info = {}
    
    try:
        import torch
        embedding_info['torch'] = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
        }
    except Exception as e:
        embedding_info['torch'] = f"ERROR: {e}"
    
    try:
        from sentence_transformers import SentenceTransformer
        from config import Config
        
        model_name = Config.EMBEDDING_MODEL
        device = 'cuda' if Config.GPU_ENABLED and torch.cuda.is_available() else 'cpu'
        
        # Test model loading
        model = SentenceTransformer(model_name, device=device)
        embedding_info['sentence_transformer'] = {
            'model_name': model_name,
            'device': str(model.device),
            'max_seq_length': model.max_seq_length,
            'embedding_dimension': model.get_sentence_embedding_dimension()
        }
        
        # Test embedding generation
        test_texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(test_texts)
        embedding_info['embedding_test'] = {
            'success': True,
            'input_count': len(test_texts),
            'output_shape': embeddings.shape,
            'output_type': str(type(embeddings))
        }
        
    except Exception as e:
        embedding_info['sentence_transformer'] = f"ERROR: {e}"
        embedding_info['embedding_test'] = f"ERROR: {e}"
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import Config
        
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Force CPU for testing
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 1}
        )
        
        # Test LangChain embeddings
        test_texts = ["Test document 1", "Test document 2"]
        langchain_embeddings = embeddings.embed_documents(test_texts)
        
        embedding_info['langchain_embeddings'] = {
            'success': True,
            'input_count': len(test_texts),
            'output_count': len(langchain_embeddings),
            'embedding_dimension': len(langchain_embeddings[0]) if langchain_embeddings else 0
        }
        
    except Exception as e:
        embedding_info['langchain_embeddings'] = f"ERROR: {e}"
    
    return embedding_info

def check_vector_database() -> Dict[str, Any]:
    """Check vector database setup."""
    print("🔍 Checking Vector Database...")
    
    vector_info = {}
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Check if vector store directory exists
        vector_store_path = "embeddings/vector_store"
        vector_info['vector_store_path'] = {
            'exists': os.path.exists(vector_store_path),
            'is_dir': os.path.isdir(vector_store_path) if os.path.exists(vector_store_path) else False,
            'readable': os.access(vector_store_path, os.R_OK) if os.path.exists(vector_store_path) else False,
            'writable': os.access(vector_store_path, os.W_OK) if os.path.exists(vector_store_path) else False
        }
        
        # Try to initialize ChromaDB
        client = chromadb.PersistentClient(
            path=vector_store_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        vector_info['chromadb_client'] = {
            'success': True,
            'client_type': type(client).__name__
        }
        
        # Check collections
        collections = client.list_collections()
        vector_info['collections'] = {
            'count': len(collections),
            'names': [col.name for col in collections]
        }
        
        # Check if there are any documents
        total_docs = 0
        for collection in collections:
            try:
                count = collection.count()
                total_docs += count
            except:
                pass
        
        vector_info['total_documents'] = total_docs
        
    except Exception as e:
        vector_info['chromadb_client'] = f"ERROR: {e}"
        vector_info['collections'] = f"ERROR: {e}"
        vector_info['total_documents'] = f"ERROR: {e}"
    
    return vector_info

def check_pdf_processing() -> Dict[str, Any]:
    """Check PDF processing capabilities."""
    print("🔍 Checking PDF Processing...")
    
    pdf_info = {}
    
    try:
        import pypdf
        pdf_info['pypdf'] = {
            'version': pypdf.__version__,
            'available': True
        }
    except Exception as e:
        pdf_info['pypdf'] = f"ERROR: {e}"
    
    # Check if there are any PDF files to test with
    raw_data_path = "data/raw"
    if os.path.exists(raw_data_path):
        pdf_files = [f for f in os.listdir(raw_data_path) if f.lower().endswith('.pdf')]
        pdf_info['available_pdfs'] = {
            'count': len(pdf_files),
            'files': pdf_files[:5]  # First 5 files
        }
        
        # Try to process the first PDF if available
        if pdf_files:
            try:
                from utils.optimized_pdf_processor import pdf_processor
                test_pdf_path = os.path.join(raw_data_path, pdf_files[0])
                
                # Test basic PDF reading
                with open(test_pdf_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    pdf_info['pdf_reading_test'] = {
                        'success': True,
                        'pages': len(reader.pages),
                        'file_size_mb': round(os.path.getsize(test_pdf_path) / (1024*1024), 2)
                    }
                
                # Test PDF processor
                result = pdf_processor(test_pdf_path)
                pdf_info['pdf_processor_test'] = {
                    'success': True,
                    'chunks': len(result) if result else 0,
                    'result_type': type(result).__name__
                }
                
            except Exception as e:
                pdf_info['pdf_reading_test'] = f"ERROR: {e}"
                pdf_info['pdf_processor_test'] = f"ERROR: {e}"
    
    return pdf_info

def check_streamlit_app() -> Dict[str, Any]:
    """Check Streamlit app components."""
    print("🔍 Checking Streamlit App...")
    
    streamlit_info = {}
    
    try:
        import streamlit as st
        streamlit_info['streamlit'] = {
            'version': st.__version__,
            'available': True
        }
    except Exception as e:
        streamlit_info['streamlit'] = f"ERROR: {e}"
    
    # Check if the main app file can be imported
    try:
        sys.path.append('ui')
        from structured_rag_app import load_existing_documents
        streamlit_info['app_import'] = {
            'success': True,
            'functions_available': ['load_existing_documents']
        }
    except Exception as e:
        streamlit_info['app_import'] = f"ERROR: {e}"
    
    return streamlit_info

def check_network_connectivity() -> Dict[str, Any]:
    """Check network connectivity for API calls."""
    print("🔍 Checking Network Connectivity...")
    
    network_info = {}
    
    try:
        import requests
        
        # Test basic internet connectivity
        response = requests.get('https://httpbin.org/get', timeout=10)
        network_info['internet_connectivity'] = {
            'success': response.status_code == 200,
            'status_code': response.status_code
        }
        
        # Test HuggingFace connectivity
        response = requests.get('https://huggingface.co/api/models', timeout=10)
        network_info['huggingface_connectivity'] = {
            'success': response.status_code == 200,
            'status_code': response.status_code
        }
        
    except Exception as e:
        network_info['internet_connectivity'] = f"ERROR: {e}"
        network_info['huggingface_connectivity'] = f"ERROR: {e}"
    
    return network_info

def run_comprehensive_diagnostic():
    """Run all diagnostic checks."""
    print("🚀 Starting Comprehensive Diagnostic...\n")
    
    diagnostic_results = {
        'timestamp': str(pd.Timestamp.now()),
        'system_info': check_system_info(),
        'python_packages': check_python_packages(),
        'project_structure': check_project_structure(),
        'environment_variables': check_environment_variables(),
        'embedding_system': check_embedding_system(),
        'vector_database': check_vector_database(),
        'pdf_processing': check_pdf_processing(),
        'streamlit_app': check_streamlit_app(),
        'network_connectivity': check_network_connectivity()
    }
    
    # Save results to file
    output_file = f"diagnostic_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(diagnostic_results, f, indent=2, default=str)
    
    print(f"\n✅ Diagnostic completed! Results saved to: {output_file}")
    
    # Print summary
    print("\n📊 DIAGNOSTIC SUMMARY:")
    print("=" * 50)
    
    # System summary
    sys_info = diagnostic_results['system_info']
    print(f"🖥️  System: {sys_info['platform']}")
    print(f"🐍 Python: {sys_info['python_version'].split()[0]}")
    if 'memory_total_gb' in sys_info:
        print(f"💾 Memory: {sys_info['memory_total_gb']}GB total, {sys_info['memory_available_gb']}GB available")
    
    # Package summary
    packages = diagnostic_results['python_packages']
    critical_errors = [pkg for pkg, info in packages.items() if isinstance(info, str) and 'ERROR' in info]
    if critical_errors:
        print(f"❌ Critical package errors: {', '.join(critical_errors)}")
    else:
        print("✅ All critical packages installed")
    
    # Embedding summary
    embedding = diagnostic_results['embedding_system']
    if 'torch' in embedding and isinstance(embedding['torch'], dict):
        torch_info = embedding['torch']
        print(f"🔥 PyTorch: {torch_info.get('version', 'Unknown')} - CUDA: {torch_info.get('cuda_available', False)}")
    
    # Vector DB summary
    vector_db = diagnostic_results['vector_database']
    if 'total_documents' in vector_db and isinstance(vector_db['total_documents'], int):
        print(f"🗄️  Vector DB: {vector_db['total_documents']} documents")
    
    # PDF processing summary
    pdf_info = diagnostic_results['pdf_processing']
    if 'available_pdfs' in pdf_info and isinstance(pdf_info['available_pdfs'], dict):
        print(f"📄 PDF files: {pdf_info['available_pdfs']['count']} available")
    
    return diagnostic_results

if __name__ == "__main__":
    try:
        import pandas as pd
        results = run_comprehensive_diagnostic()
        print("\n🎯 Run this script on both laptops and compare the JSON output files!")
        print("🔍 Look for differences in:")
        print("   - Python package versions")
        print("   - System memory and resources")
        print("   - Environment variables")
        print("   - File permissions")
        print("   - Network connectivity")
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        traceback.print_exc()
