import streamlit as st 
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import hashlib
from chromadb.config import Settings

# Import our new infrastructure
from config import Config
from utils.logger import logger
from utils.validators import FileValidator, InputValidator
from utils.api_client import get_gemini_client, get_openai_client

# Validate configuration
if not Config.validate_api_keys():
    st.error("❌ Missing API keys. Please check your .env file.")
    st.stop()

# Create directories
Config.create_directories()

# 📦 Load embeddings & vector DB
try:
    embedding_model = HuggingFaceEmbeddings(model_name=Config.get_embedding_model_name())

    chroma_settings = Settings(
        persist_directory=Config.VECTOR_STORE_DIR,
        anonymized_telemetry=False,
        is_persistent=True
    )

    vectordb = Chroma(
        persist_directory=Config.VECTOR_STORE_DIR,
        embedding_function=embedding_model,
        client_settings=chroma_settings
    )
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": Config.RETRIEVER_K})
    
    logger.info("Vector database loaded successfully")
    
except Exception as e:
    logger.error(f"Failed to load vector database: {str(e)}", exc_info=True)
    st.error(f"❌ Failed to load vector database: {str(e)}")
    st.stop()

# 🧠 LLM Setup
try:
    openai_client = get_openai_client()
    gemini_client = get_gemini_client()
    
    # Create QA chain for OpenAI
    from langchain_openai import ChatOpenAI
    openai_llm = ChatOpenAI(
        model=Config.OPENAI_MODEL, 
        temperature=Config.OPENAI_TEMPERATURE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=openai_llm, 
        retriever=retriever, 
        return_source_documents=True
    )
    
    logger.info("LLM clients initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize LLM clients: {str(e)}", exc_info=True)
    st.error(f"❌ Failed to initialize LLM clients: {str(e)}")
    st.stop()

# 🧮 File hash utility
def get_file_hash(file_bytes):
    return FileValidator.calculate_file_hash(file_bytes)

# 🌐 UI Config
st.set_page_config(page_title="📄 Financial Document Q&A")
st.title("💰 Ask Questions About Investor Documents")

# 📄 Upload PDFs
st.subheader("📂 Upload Investor Documents (PDF)")
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("📁 Processing and embedding PDFs..."):
        try:
            # Get hashes of already stored documents
            existing_docs = vectordb.get()
            existing_hashes = set()
            for metadata in existing_docs['metadatas']:
                if isinstance(metadata, dict) and "hash" in metadata:
                    existing_hashes.add(metadata["hash"])

            new_chunks = []
            processed_files = 0
            
            for uploaded_file in uploaded_files:
                try:
                    file_bytes = uploaded_file.read()
                    
                    # Validate uploaded file
                    is_valid, validation_message = FileValidator.validate_uploaded_file(file_bytes, uploaded_file.name)
                    if not is_valid:
                        st.error(f"❌ {uploaded_file.name}: {validation_message}")
                        logger.error(f"File validation failed: {uploaded_file.name} - {validation_message}")
                        continue
                    
                    file_hash = get_file_hash(file_bytes)

                    if file_hash in existing_hashes:
                        st.info(f"🔁 Skipped already embedded: {uploaded_file.name}")
                        logger.info(f"Skipped already embedded file: {uploaded_file.name}")
                        continue

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        loader = PyPDFLoader(tmp.name)
                        docs = loader.load()

                        for doc in docs:
                            doc.metadata["source"] = uploaded_file.name
                            doc.metadata["hash"] = file_hash

                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=Config.CHUNK_SIZE, 
                            chunk_overlap=Config.CHUNK_OVERLAP
                        )
                        chunks = splitter.split_documents(docs)
                        new_chunks.extend(chunks)
                        processed_files += 1
                        
                        logger.log_file_processing(uploaded_file.name, "upload_and_embed", True)

            if new_chunks:
                vectordb.add_documents(new_chunks)
                st.success(f"✅ {len(new_chunks)} new chunks embedded from {processed_files} files!")
                logger.info(f"Successfully embedded {len(new_chunks)} chunks from {processed_files} files")
            else:
                st.info("ℹ️ No new documents to embed")
                
        except Exception as e:
            error_msg = f"Failed to process uploaded files: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(f"❌ {error_msg}")

# 🤖 LLM Choice
model_choice = st.radio("🧠 Choose Model:", ["OpenAI GPT-3.5", "Gemini 2.0 (Free)"])

# ❓ Question
question = st.text_input("🔍 Enter your financial question:")

if question:
    # Validate query input
    is_valid_query, validation_message = InputValidator.validate_query(question)
    if not is_valid_query:
        st.error(f"❌ Invalid query: {validation_message}")
        st.stop()
    
    # Sanitize query
    sanitized_question = InputValidator.sanitize_text(question)
    
    with st.spinner("🔎 Searching and answering..."):
        try:
            if model_choice == "OpenAI GPT-3.5":
                result = qa_chain({"query": sanitized_question})
                st.subheader("📬 Answer")
                st.write(result["result"])

                st.subheader("📚 Source Snippets")
                for doc in result["source_documents"]:
                    st.markdown(f"- *{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})*")
                    st.code(doc.page_content[:500] + "...")
                
                logger.log_query(sanitized_question, "OpenAI GPT-3.5", True)
                
            else:
                docs = retriever.get_relevant_documents(sanitized_question)
                context = "\n\n".join([doc.page_content for doc in docs[:6]])
                prompt = f"""
You are a financial assistant specialized in analyzing investor presentations.

Based only on the following document snippets, answer the question in a factual and numeric way. 
Avoid assumptions. Quote values exactly if present.

--- SNIPPETS ---
{context}

--- QUESTION ---
{sanitized_question}

--- ANSWER ---
"""
                answer = gemini_client.query(prompt)
                st.subheader("📬 Answer")
                st.write(answer)

                st.subheader("📚 Source Snippets")
                for doc in docs:
                    st.markdown(f"- *{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})*")
                    st.code(doc.page_content[:500] + "...")
                
                logger.log_query(sanitized_question, "Gemini 2.0", True)
                
        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            logger.log_query(sanitized_question, model_choice, False)
            st.error(f"❌ {error_msg}") 