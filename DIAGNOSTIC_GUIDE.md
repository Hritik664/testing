# Diagnostic Guide: Troubleshooting RAG App Issues

This guide will help you identify why the Financial RAG app works on one laptop but stops after PDF processing on another laptop.

## 🚀 Quick Start

### Step 1: Run Basic Embedding Check
First, run the existing embedding check on both laptops:

```bash
python check_embedder.py
```

This should show similar results on both laptops. If there are differences, note them.

### Step 2: Run Comprehensive Diagnostic
Run the comprehensive diagnostic on both laptops:

```bash
python comprehensive_diagnostic.py
```

This will create a JSON file with detailed system information. Save these files with descriptive names like:
- `diagnostic_working_laptop.json`
- `diagnostic_problematic_laptop.json`

### Step 3: Run PDF Processing Diagnostic
Run the focused PDF diagnostic on the problematic laptop:

```bash
python pdf_processing_diagnostic.py
```

This will specifically test the PDF processing pipeline that's likely causing the issue.

### Step 4: Compare Results
Compare the diagnostic results from both laptops:

```bash
python compare_diagnostics.py diagnostic_working_laptop.json diagnostic_problematic_laptop.json
```

## 🔍 What to Look For

### Common Issues That Cause App to Stop After PDF Processing:

1. **Memory Issues**
   - Insufficient RAM (less than 4GB available)
   - Memory leaks during PDF processing
   - Large PDF files overwhelming the system

2. **Package Version Differences**
   - Different versions of PyTorch, transformers, or sentence-transformers
   - Missing or corrupted packages
   - Incompatible package combinations

3. **File Permission Issues**
   - Cannot write to `data/processed` directory
   - Cannot write to `embeddings/vector_store` directory
   - Cannot create log files

4. **PDF Processing Issues**
   - Corrupted PDF files
   - PDF files that are too large
   - PDF files with complex layouts or images
   - PyPDF version incompatibilities

5. **Embedding Model Issues**
   - Model download failures
   - Insufficient disk space for model cache
   - Network connectivity issues to HuggingFace

6. **Vector Database Issues**
   - ChromaDB corruption
   - Insufficient disk space
   - Permission issues with database files

## 🛠️ Troubleshooting Steps

### If PDF Processing Diagnostic Fails:

1. **Check Memory Usage**
   ```bash
   # On Windows
   tasklist /FI "IMAGENAME eq python.exe"
   
   # On Linux/Mac
   ps aux | grep python
   ```

2. **Check Disk Space**
   ```bash
   # On Windows
   dir
   
   # On Linux/Mac
   df -h
   ```

3. **Check File Permissions**
   ```bash
   # On Windows
   icacls data\processed
   icacls embeddings\vector_store
   
   # On Linux/Mac
   ls -la data/processed
   ls -la embeddings/vector_store
   ```

4. **Check Logs**
   ```bash
   # Look for recent error messages
   tail -n 50 logs/financial_rag_*.log
   ```

### If Embedding Generation Fails:

1. **Check PyTorch Installation**
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

2. **Check Model Download**
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
   ```

3. **Check Network Connectivity**
   ```python
   import requests
   response = requests.get('https://huggingface.co/api/models')
   print(f"Status: {response.status_code}")
   ```

### If Vector Database Fails:

1. **Reset Vector Database**
   ```bash
   # Backup existing data first
   cp -r embeddings/vector_store embeddings/vector_store_backup
   
   # Remove and recreate
   rm -rf embeddings/vector_store
   mkdir -p embeddings/vector_store
   ```

2. **Check ChromaDB Installation**
   ```python
   import chromadb
   print(f"ChromaDB version: {chromadb.__version__}")
   ```

## 📋 Diagnostic Checklist

Before running diagnostics, ensure:

- [ ] Both laptops have the same project files
- [ ] Both laptops have the same `.env` file
- [ ] Both laptops have the same Python version
- [ ] Both laptops have the same virtual environment activated
- [ ] Both laptops have the same PDF files in `data/raw`

## 🎯 Specific Solutions

### For Memory Issues:
- Reduce `BATCH_SIZE` in `config.py` from 32 to 16 or 8
- Reduce `MAX_WORKERS` from 4 to 2
- Close other applications to free up memory

### For Package Issues:
- Reinstall packages: `pip install -r requirements.txt --force-reinstall`
- Update packages: `pip install --upgrade torch transformers sentence-transformers`

### For Permission Issues:
- Run as administrator (Windows) or with sudo (Linux)
- Check folder permissions and ownership
- Ensure antivirus isn't blocking file operations

### For PDF Issues:
- Try with a smaller, simpler PDF file first
- Check if PDF is password-protected or corrupted
- Try different PDF files to isolate the issue

## 📞 Getting Help

If you're still having issues after running these diagnostics:

1. **Collect Information:**
   - Run all diagnostic scripts
   - Save the JSON output files
   - Note any error messages from the logs
   - Take screenshots of any error dialogs

2. **Share Results:**
   - The diagnostic JSON files
   - The PDF processing diagnostic output
   - Any error messages from the logs
   - System specifications of both laptops

3. **Test with Minimal Setup:**
   - Try with just one small PDF file
   - Try with default configuration settings
   - Try on a fresh virtual environment

## 🔧 Advanced Debugging

### Enable Verbose Logging:
Add this to your `.env` file:
```
LOG_LEVEL=DEBUG
```

### Monitor System Resources:
Use Task Manager (Windows) or htop (Linux) to monitor:
- CPU usage
- Memory usage
- Disk I/O
- Network activity

### Check for Timeouts:
The app might be timing out during long operations. Check if:
- PDF processing takes more than 5 minutes
- Embedding generation takes more than 10 minutes
- Vector database operations are slow

This diagnostic approach should help you identify the root cause of why the app works on one laptop but not the other.
