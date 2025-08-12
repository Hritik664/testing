import streamlit as st 
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import hashlib
from chromadb.config import Settings
import time

# Import our new infrastructure
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.logger import logger
from utils.validators import FileValidator, InputValidator
from utils.api_client import get_gemini_client, get_openai_client
from utils.optimized_pdf_processor import pdf_processor
from utils.enhanced_embedder import enhanced_embedder
from utils.enhanced_retriever import EnhancedRetriever, answer_enhancer

# Validate configuration
if not Config.validate_api_keys():
    st.error("❌ Missing API keys. Please check your .env file.")
    st.stop()

# Create directories
Config.create_directories()

# 📦 Load embeddings & vector DB
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name=Config.get_embedding_model_name(),
        model_kwargs={'device': 'cuda' if Config.GPU_ENABLED else 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': Config.BATCH_SIZE}
    )

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
    
    # Create enhanced retriever
    base_retriever = vectordb.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": Config.RETRIEVER_K}
    )
    enhanced_retriever = EnhancedRetriever(vectordb, base_retriever)
    
    logger.info("Vector database and enhanced retriever loaded successfully")
    
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
        retriever=enhanced_retriever,
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
st.set_page_config(
    page_title="📄 Financial Document Q&A - Optimized",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Performance settings
    st.subheader("Performance")
    use_cache = st.checkbox("Enable Caching", value=Config.CACHE_ENABLED)
    use_enhanced_retrieval = st.checkbox("Enhanced Retrieval", value=True)
    
    # Model settings
    st.subheader("Model Settings")
    model_choice = st.radio("🧠 Choose Model:", ["OpenAI GPT-3.5", "Gemini 2.0 (Free)"])
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    k_value = st.slider("Number of documents to retrieve", 4, 16, 8)
    
    # Show system info
    st.subheader("System Info")
    st.info(f"Chunk Size: {Config.CHUNK_SIZE}")
    st.info(f"Max Workers: {Config.MAX_WORKERS}")
    st.info(f"GPU Enabled: {Config.GPU_ENABLED}")

# Main content
st.title("💰 Financial Document Q&A - Optimized")
st.markdown("**Enhanced with parallel processing, semantic chunking, and improved accuracy**")

# 📄 Upload PDFs
st.subheader("📂 Upload Investor Documents (PDF)")
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("📁 Processing and embedding PDFs..."):
        try:
            start_time = time.time()
            
            # Prepare files for processing
            files_data = []
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                
                # Validate uploaded file
                is_valid, validation_message = FileValidator.validate_uploaded_file(file_bytes, uploaded_file.name)
                if not is_valid:
                    st.error(f"❌ {uploaded_file.name}: {validation_message}")
                    logger.error(f"File validation failed: {uploaded_file.name} - {validation_message}")
                    continue
                
                files_data.append((file_bytes, uploaded_file.name))
            
            if not files_data:
                st.warning("No valid files to process")
                st.stop()
            
            # Process PDFs in parallel
            processed_data = pdf_processor.process_multiple_pdfs(files_data)
            
            if processed_data:
                # Create documents with semantic chunking
                documents = enhanced_embedder.process_documents(processed_data)
                
                # Embed and store
                enhanced_embedder.embed_and_store(documents, force_reload=False)
                
                processing_time = time.time() - start_time
                st.success(f"✅ Processed {len(processed_data)} files in {processing_time:.2f}s!")
                logger.info(f"Successfully processed {len(processed_data)} files in {processing_time:.2f}s")
                
                # Show processing stats
                total_pages = sum(data.get('total_pages', 0) for data in processed_data)
                total_words = sum(data.get('total_words', 0) for data in processed_data)
                st.info(f"📊 Stats: {total_pages} pages, {total_words:,} words processed")
                
            else:
                st.warning("No documents were processed successfully")
                
        except Exception as e:
            error_msg = f"Failed to process uploaded files: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(f"❌ {error_msg}")

# ❓ Question
st.subheader("🔍 Ask Your Financial Question")
question = st.text_input("Enter your financial question:", placeholder="e.g., What was the revenue in Q1 2025?")

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
            start_time = time.time()
            
            if model_choice == "OpenAI GPT-3.5":
                # Use enhanced retriever for better results
                if use_enhanced_retrieval:
                    docs = enhanced_retriever.get_diverse_documents(sanitized_question, k_value)
                else:
                    docs = base_retriever.get_relevant_documents(sanitized_question, k=k_value)
                
                # Create context from documents
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Create enhanced prompt
                enhanced_prompt = f"""
You are a financial analyst assistant. Answer the following question based ONLY on the provided document snippets.

IMPORTANT GUIDELINES:
- Quote exact numbers and percentages when available
- Mention specific time periods (quarters, years) when relevant
- Cite the source document when possible
- If information is not available in the snippets, say so clearly
- Be precise with financial figures and dates

DOCUMENT SNIPPETS:
{context}

QUESTION: {sanitized_question}

ANSWER:
"""
                
                result = openai_llm.invoke(enhanced_prompt)
                answer = result.content
                
                # Enhance answer with additional context
                enhanced_answer = answer_enhancer.enhance_answer(answer, docs, sanitized_question)
                
                st.subheader("📬 Answer")
                st.write(enhanced_answer)
                
                # Show source documents
                st.subheader("📚 Source Documents")
                for i, doc in enumerate(docs):
                    with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})"):
                        st.write(doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content)
                
                logger.log_query(sanitized_question, "OpenAI GPT-3.5", True)
                
            else:
                # Use enhanced retriever for Gemini as well
                if use_enhanced_retrieval:
                    docs = enhanced_retriever.get_diverse_documents(sanitized_question, k_value)
                else:
                    docs = base_retriever.get_relevant_documents(sanitized_question, k=k_value)
                
                context = "\n\n".join([doc.page_content for doc in docs])
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
                
                # Enhance answer
                enhanced_answer = answer_enhancer.enhance_answer(answer, docs, sanitized_question)
                
                st.subheader("📬 Answer")
                st.write(enhanced_answer)

                # Show source documents
                st.subheader("📚 Source Documents")
                for i, doc in enumerate(docs):
                    with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})"):
                        st.write(doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content)
                
                logger.log_query(sanitized_question, "Gemini 2.0", True)
            
            # Show performance metrics
            response_time = time.time() - start_time
            st.info(f"⏱️ Response time: {response_time:.2f}s")
            
            # Show retrieval stats
            if docs:
                sources = set(doc.metadata.get('source', 'Unknown') for doc in docs)
                st.info(f"📊 Retrieved {len(docs)} documents from {len(sources)} sources")
                
        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            logger.error(error_msg, exc_info=True)
            logger.log_query(sanitized_question, model_choice, False)
            st.error(f"❌ {error_msg}")

# Footer with performance tips
st.markdown("---")
st.markdown("""
### 🚀 Performance Tips:
- **Caching**: Enable caching for faster repeated queries
- **Enhanced Retrieval**: Uses semantic analysis for better document selection
- **Parallel Processing**: PDFs are processed in parallel for faster uploads
- **Semantic Chunking**: Documents are split intelligently for better context
""") 