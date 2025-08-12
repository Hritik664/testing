#!/usr/bin/env python3
"""
Improved Financial RAG Application with Enhanced Prompts and Document Selection
"""

import streamlit as st
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
import time
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Financial RAG - Enhanced",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize configuration
Config.create_directories()

def get_file_hash(file_bytes):
    """Get hash of uploaded file for caching."""
    import hashlib
    return hashlib.md5(file_bytes).hexdigest()

def load_existing_documents():
    """Load list of existing processed documents."""
    processed_dir = Path(Config.PROCESSED_DATA_DIR)
    if not processed_dir.exists():
        return []
    
    documents = []
    for json_file in processed_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.append({
                    'filename': data.get('filename', json_file.stem),
                    'source': data.get('source', 'Unknown'),
                    'total_pages': data.get('total_pages', 0),
                    'total_words': data.get('total_words', 0),
                    'processed_at': data.get('processed_at', 'Unknown'),
                    'file_path': str(json_file)
                })
        except Exception as e:
            logger.error(f"Failed to load document info from {json_file}: {e}")
    
    return documents

def main():
    st.title("📊 Enhanced Financial RAG System")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "🤖 Choose LLM Model",
        ["OpenAI GPT-3.5", "Google Gemini 2.0"],
        help="Select the language model for generating answers"
    )
    
    # Performance settings
    st.sidebar.subheader("🚀 Performance Settings")
    use_caching = st.sidebar.checkbox("Enable Caching", value=True, help="Cache processed documents for faster loading")
    use_enhanced_retrieval = st.sidebar.checkbox("Enhanced Retrieval", value=True, help="Use advanced document selection")
    k_value = st.sidebar.slider("Number of Documents (K)", min_value=3, max_value=12, value=6, help="Number of documents to retrieve")
    
    # Document selection
    st.sidebar.subheader("📚 Document Selection")
    existing_docs = load_existing_documents()
    selected_docs = []  # Initialize here
    
    if existing_docs:
        st.sidebar.write("**Existing Documents:**")
        
        # Group by source
        sources = {}
        for doc in existing_docs:
            source = doc['source']
            if source not in sources:
                sources[source] = []
            sources[source].append(doc)
        
        for source, docs in sources.items():
            with st.sidebar.expander(f"📁 {source} ({len(docs)} files)"):
                for doc in docs:
                    if st.checkbox(
                        f"📄 {doc['filename']} ({doc['total_pages']} pages)",
                        key=f"doc_{doc['filename']}"
                    ):
                        selected_docs.append(doc)
        
        if selected_docs:
            st.sidebar.success(f"✅ {len(selected_docs)} documents selected")
        else:
            st.sidebar.warning("⚠️ No documents selected")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload New Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload financial documents (earnings calls, presentations, reports)"
        )
        
        if uploaded_files:
            # Check if files have been processed already
            uploaded_filenames = [file.name for file in uploaded_files]
            processed_key = f"processed_{hash(tuple(uploaded_filenames))}"
            
            if processed_key not in st.session_state:
                st.session_state[processed_key] = False
            
            if not st.session_state[processed_key]:
                # Process files automatically when uploaded
                with st.spinner("Processing documents..."):
                    try:
                        # Process uploaded files
                        files_data = [(file.read(), file.name) for file in uploaded_files]
                        processed_data = pdf_processor.process_multiple_pdfs(files_data)
                        
                        if processed_data:
                            # Save processed data to JSON files
                            for data in processed_data:
                                filename = data['filename']
                                json_path = Path(Config.PROCESSED_DATA_DIR) / f"{filename}.json"
                                with open(json_path, 'w', encoding='utf-8') as f:
                                    json.dump(data, f, indent=2, ensure_ascii=False)
                            
                            # Create documents for embedding
                            documents = enhanced_embedder.process_documents(processed_data)
                            
                            # Embed and store
                            enhanced_embedder.embed_and_store(documents, force_reload=False)
                            
                            st.success(f"✅ Successfully processed {len(processed_data)} documents!")
                            st.balloons()
                            
                            # Mark as processed
                            st.session_state[processed_key] = True
                            
                            # Refresh the page to show new documents
                            st.rerun()
                        else:
                            st.error("❌ No documents were processed successfully")
                            
                    except Exception as e:
                        error_msg = f"Failed to process documents: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        st.error(f"❌ {error_msg}")
            else:
                st.success("✅ Documents already processed!")
                
                # Add a button to reprocess if needed
                if st.button("🔄 Reprocess Documents"):
                    st.session_state[processed_key] = False
                    st.rerun()
    
    with col2:
        st.header("📊 System Status")
        
        # Check vector database
        vectordb_path = Path(Config.VECTOR_STORE_DIR)
        if vectordb_path.exists() and any(vectordb_path.iterdir()):
            st.success("✅ Vector database ready")
        else:
            st.warning("⚠️ Vector database empty")
        
        # Check processed documents
        processed_dir = Path(Config.PROCESSED_DATA_DIR)
        if processed_dir.exists():
            json_files = list(processed_dir.glob("*.json"))
            if json_files:
                st.success(f"✅ {len(json_files)} processed documents")
            else:
                st.warning("⚠️ No processed documents")
        
        # Check cache
        cache_dir = Path("cache")
        if cache_dir.exists() and any(cache_dir.iterdir()):
            st.info("💾 Cache available")
        else:
            st.info("💾 No cache")
    
    st.markdown("---")
    
    # Query section
    st.header("🔍 Ask Questions")
    
    # Query input
    question = st.text_area(
        "Enter your financial question:",
        placeholder="e.g., What was the revenue growth in Q1FY26? What is the current capacity?",
        height=100,
        help="Ask specific questions about financial metrics, performance, or company data"
    )
    
    # Query options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_type = st.selectbox(
            "🔎 Search Type",
            ["Standard", "Enhanced", "Diverse"],
            help="Standard: Basic retrieval\nEnhanced: Better relevance\nDiverse: Multiple sources"
        )
    
    with col2:
        include_sources = st.checkbox("Include Sources", value=True, help="Show source documents")
    
    with col3:
        if st.button("🚀 Get Answer", type="primary"):
            if not question.strip():
                st.warning("⚠️ Please enter a question")
            else:
                process_query(question, model_choice, search_type, k_value, include_sources, selected_docs)

def process_query(question, model_choice, search_type, k_value, include_sources, selected_docs):
    """Process the user query with enhanced prompting."""
    
    try:
        # Validate input
        is_valid, error_msg = InputValidator.validate_query(question)
        if not is_valid:
            st.error(f"❌ Invalid query: {error_msg}")
            return
        
        sanitized_question = InputValidator.sanitize_text(question)
        
        # Initialize LLM clients
        try:
            if model_choice == "OpenAI GPT-3.5":
                openai_llm = get_openai_client().get_llm()
            else:
                gemini_client = get_gemini_client()
        except Exception as e:
            st.error(f"❌ Failed to initialize LLM clients: {str(e)}")
            return
        
        # Initialize vector database and retriever
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            from chromadb.config import Settings
            from langchain.chains import RetrievalQA
            
            # Load embedding model
            embedding_model = HuggingFaceEmbeddings(
                model_name=Config.get_embedding_model_name(),
                model_kwargs={'device': 'cuda' if Config.GPU_ENABLED else 'cpu'},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': Config.BATCH_SIZE}
            )
            
            # Load vector database
            vectordb = Chroma(
                persist_directory=Config.VECTOR_STORE_DIR,
                embedding_function=embedding_model,
                client_settings=Settings(anonymized_telemetry=False)
            )
            
            # Create retrievers
            base_retriever = vectordb.as_retriever(search_kwargs={"k": k_value})
            enhanced_retriever = EnhancedRetriever(vectordb, base_retriever)
            
            logger.info("Vector database and enhanced retriever loaded successfully")
            
        except Exception as e:
            st.error(f"❌ Failed to load vector database: {str(e)}")
            return
        
        # Process query
        start_time = time.time()
        
        # Select retrieval method
        if search_type == "Enhanced":
            docs = enhanced_retriever.get_relevant_documents(sanitized_question, k=k_value)
        elif search_type == "Diverse":
            docs = enhanced_retriever.get_diverse_documents(sanitized_question, k=k_value)
        else:
            docs = base_retriever.get_relevant_documents(sanitized_question, k=k_value)
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Enhanced prompts for better financial analysis
        if model_choice == "OpenAI GPT-3.5":
            enhanced_prompt = f"""
You are an expert financial analyst assistant. Answer the following question based ONLY on the provided document snippets.

CRITICAL INSTRUCTIONS:
1. Quote EXACT numbers, percentages, and figures when available
2. Include specific time periods (Q1FY26, Q2FY25, etc.) in your answer
3. If the information is not in the snippets, say "Information not available in the provided documents"
4. Be precise with financial metrics and avoid approximations
5. If multiple values exist, mention the context (e.g., "Q1FY26 revenue was $X, while Q2FY25 was $Y")
6. For capacity, costs, capex, and other specific metrics, provide the exact values

DOCUMENT SNIPPETS:
{context}

QUESTION: {sanitized_question}

Provide a clear, factual answer with specific numbers and time periods:
"""
            
            result = openai_llm.invoke(enhanced_prompt)
            answer = result.content
            
        else:  # Gemini
            prompt = f"""
You are a financial analyst expert. Answer the question based ONLY on the provided document snippets.

IMPORTANT RULES:
- Quote exact numbers, percentages, and financial figures
- Include specific quarters/years (Q1FY26, FY25, etc.)
- If information is missing, say "Information not available in the provided documents"
- Be precise with metrics like capacity, costs, capex, revenue, etc.
- Don't make assumptions or approximations

DOCUMENT CONTENT:
{context}

QUESTION: {sanitized_question}

Provide a precise answer with exact figures and time periods:
"""
            
            answer = gemini_client.query(prompt)
        
        # Enhance answer with additional context
        enhanced_answer = answer_enhancer.enhance_answer(answer, docs, sanitized_question)
        
        # Display results
        st.subheader("📬 Answer")
        st.write(enhanced_answer)
        
        # Show source documents if requested
        if include_sources and docs:
            st.subheader("📚 Source Documents")
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', '?')
                filename = doc.metadata.get('filename', 'Unknown')
                
                with st.expander(f"📄 {filename} - {source} (Page {page})"):
                    st.write(doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content)
        
        # Performance metrics
        response_time = time.time() - start_time
        st.info(f"⏱️ Response time: {response_time:.2f}s | 📊 Retrieved {len(docs)} documents")
        
        # Log successful query
        logger.log_query(sanitized_question, model_choice, True)
        
    except Exception as e:
        error_msg = f"Failed to process query: {str(e)}"
        logger.error(error_msg, exc_info=True)
        logger.log_query(question, model_choice, False)
        st.error(f"❌ {error_msg}")

if __name__ == "__main__":
    main() 