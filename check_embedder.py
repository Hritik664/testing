"""
check_embedder.py
-----------------
Standalone script to verify HuggingFace Embeddings setup
and device (CPU/GPU) availability.
"""

import torch
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

def check_embeddings():
    print("\n🔍 Checking Embedding Setup...\n")

    # 1. Check Torch GPU availability
    print(f"✅ Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   🔹 CUDA Device Count: {torch.cuda.device_count()}")
        print(f"   🔹 Current Device: {torch.cuda.current_device()}")
        print(f"   🔹 Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU (no GPU detected).")

    # 2. Get Config values
    model_name = Config.get_embedding_model_name()
    device = 'cuda' if Config.GPU_ENABLED and torch.cuda.is_available() else 'cpu'
    batch_size = Config.BATCH_SIZE

    print(f"\n✅ Config says:")
    print(f"   🔹 Embedding Model: {model_name}")
    print(f"   🔹 Device (from Config): {device}")
    print(f"   🔹 Batch Size: {batch_size}")

    # 3. Load HuggingFace model directly (SentenceTransformer)
    try:
        model = SentenceTransformer(model_name, device=device)
        print(f"\n✅ Successfully loaded SentenceTransformer model: {model_name}")
        print(f"   🔹 Model running on: {model.device}")
    except Exception as e:
        print(f"\n❌ Failed to load SentenceTransformer model: {e}")

    # 4. Test LangChain HuggingFaceEmbeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': batch_size}
        )
        print(f"\n✅ HuggingFaceEmbeddings initialized with {model_name}")
    except Exception as e:
        print(f"\n❌ Failed to initialize HuggingFaceEmbeddings: {e}")

if __name__ == "__main__":
    check_embeddings()
