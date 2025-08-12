#!/bin/bash

echo "📁 Creating project folder structure..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p embeddings/vector_store
mkdir -p models/local_llms
mkdir -p ui
mkdir -p utils

echo "📦 Writing requirements.txt..."

cat <<EOL > requirements.txt
openai
langchain
streamlit
tqdm
pymupdf
faiss-cpu
chromadb
unstructured
sentence-transformers
python-dotenv
PyPDF2
EOL

echo "✅ Setting up virtual environment..."
python -m venv venv
./venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ RAG project initialized."
