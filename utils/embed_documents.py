import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores.utils import filter_complex_metadata
from chromadb.config import Settings

from config import Config
from utils.logger import logger

# Create directories
Config.create_directories()

# ✅ Use a performant embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=Config.get_embedding_model_name()
)

def load_documents():
    docs = []
    try:
        for fname in os.listdir(Config.PROCESSED_DATA_DIR):
            if fname.endswith(".json"):
                fpath = os.path.join(Config.PROCESSED_DATA_DIR, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    parsed = json.load(f)
                    for page in parsed["pages"]:
                        text = page["text"].strip()
                        if len(text) > Config.MIN_TEXT_LENGTH:  # ignore blank/short pages
                            docs.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "filename": parsed["filename"],
                                        "source": parsed["source"],
                                        "page": page["page"],
                                    }
                                )
                            )
        
        logger.info(f"Loaded {len(docs)} documents from {Config.PROCESSED_DATA_DIR}")
        return docs
        
    except Exception as e:
        logger.error(f"Failed to load documents: {str(e)}", exc_info=True)
        raise

def embed_and_store():
    try:
        print("📦 Loading documents...")
        documents = load_documents()
        print(f"🧠 Embedding {len(documents)} documents...")

        # Clean metadata to avoid issues with ChromaDB
        documents = filter_complex_metadata(documents)

        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=Config.VECTOR_STORE_DIR,
            client_settings=Settings(anonymized_telemetry=False)
        )

        vectordb.persist()
        logger.log_file_processing("embeddings", "embed_and_store", True)
        print(f"✅ Stored {len(documents)} embeddings to {Config.VECTOR_STORE_DIR}")
        
    except Exception as e:
        logger.error(f"Failed to embed and store documents: {str(e)}", exc_info=True)
        print(f"❌ Failed to embed documents: {str(e)}")
        raise

if __name__ == "__main__":
    embed_and_store()
