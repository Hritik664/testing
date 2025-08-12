#!/usr/bin/env python3
"""
Structured Financial RAG Application
Provides formatted responses that can be consumed by other agents.
"""

import streamlit as st
import sys
import os
import json
import time
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.logger import logger
from utils.validators import FileValidator, InputValidator
from utils.api_client import get_gemini_client, get_openai_client
from utils.optimized_pdf_processor import pdf_processor
from utils.enhanced_embedder import enhanced_embedder
from utils.enhanced_retriever import EnhancedRetriever, answer_enhancer
from utils.structured_output import response_parser, output_formatter, ResponseType
from utils.bulk_question_processor import bulk_processor

# Page configuration
st.set_page_config(
    page_title="Structured Financial RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize configuration
Config.create_directories()

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
    st.title("📊 Structured Financial RAG System")
    st.markdown("**Agent-Ready Responses with Structured Output**")
    st.markdown("---")
    
    # Check vector database status
    vectordb_path = Path(Config.VECTOR_STORE_DIR)
    vectordb_status = "empty"
    doc_count = 0
    
    if vectordb_path.exists() and any(vectordb_path.iterdir()):
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            from chromadb.config import Settings
            
            embedding_model = HuggingFaceEmbeddings(
                model_name=Config.get_embedding_model_name(),
                model_kwargs={'device': 'cuda' if Config.GPU_ENABLED else 'cpu'},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': Config.BATCH_SIZE}
            )
            
            vectordb = Chroma(
                persist_directory=Config.VECTOR_STORE_DIR,
                embedding_function=embedding_model,
                client_settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            collections = vectordb._client.list_collections()
            if collections:
                collection = vectordb._client.get_collection(name=collections[0].name)
                doc_count = collection.count()
                if doc_count > 0:
                    vectordb_status = "ready"
                else:
                    vectordb_status = "empty"
            else:
                vectordb_status = "no_collections"
        except Exception as e:
            logger.error(f"Error checking vector database status: {e}")
            vectordb_status = "error"
    
    # Show vector database status
    if vectordb_status == "ready":
        st.success(f"✅ Vector database ready with {doc_count} documents")
    elif vectordb_status == "empty":
        st.warning("⚠️ Vector database is empty. Please upload documents to get started.")
    elif vectordb_status == "no_collections":
        st.warning("⚠️ Vector database has no collections. Please upload documents to get started.")
    elif vectordb_status == "error":
        st.error("❌ Error accessing vector database. Please check the system.")
    else:
        st.info("ℹ️ No vector database found. Please upload documents to get started.")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "🤖 Choose LLM Model",
        ["Google Gemini 2.0", "OpenAI GPT-3.5"],
        help="Select the language model for generating answers"
    )
    
    # Output format selection
    output_format = st.sidebar.selectbox(
        "📤 Output Format",
        ["Markdown", "JSON", "CSV", "Agent Format"],
        help="Format for structured output"
    )
    
    # Performance settings
    st.sidebar.subheader("🚀 Performance Settings")
    k_value = st.sidebar.slider("Number of Documents (K)", min_value=3, max_value=12, value=6)
    include_sources = st.sidebar.checkbox("Include Sources", value=True)
    
    # Document selection
    st.sidebar.subheader("📚 Document Selection")
    existing_docs = load_existing_documents()
    selected_docs = []
    
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
            uploaded_filenames = [file.name for file in uploaded_files]
            processed_key = f"processed_{hash(tuple(uploaded_filenames))}"
            
            if processed_key not in st.session_state:
                st.session_state[processed_key] = False
            
            if not st.session_state[processed_key]:
                with st.spinner("Processing documents..."):
                    try:
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
                            vectordb = enhanced_embedder.embed_and_store(documents, force_reload=False)
                            
                            st.success(f"✅ Successfully processed {len(processed_data)} documents!")
                            st.balloons()
                            
                            st.session_state[processed_key] = True
                            st.rerun()
                        else:
                            st.error("❌ No documents were processed successfully")
                            
                    except Exception as e:
                        error_msg = f"Failed to process documents: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        st.error(f"❌ {error_msg}")
            else:
                st.success("✅ Documents already processed!")
                
                if st.button("🔄 Reprocess Documents", key="reprocess_documents"):
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
    
    st.markdown("---")
    
    # Bulk Question Processing Section
    st.header("📋 Bulk Question Processing")
    
    # Create tabs for single query and bulk processing
    tab1, tab2 = st.tabs(["🔍 Single Query", "📊 Bulk Questions"])
    
    with tab1:
        # Enhanced Question Input Section
        st.markdown("### 🔍 **Ask Your Financial Question**")
        
        # Sample questions for inspiration
        with st.expander("💡 **Sample Questions**", expanded=False):
            sample_questions = [
                "What was the revenue growth in Q1FY26?",
                "What is the current capacity utilization?",
                "What was the sales volume in Q1FY25?",
                "What is the company's guidance on capex?",
                "What was the EBITDA margin in the last quarter?",
                "What are the key cost factors affecting profitability?"
            ]
            
            cols = st.columns(2)
            for i, sample in enumerate(sample_questions):
                with cols[i % 2]:
                    if st.button(f"📝 {sample}", key=f"sample_{i}", help="Click to use this question"):
                        st.session_state.sample_question = sample
                        st.rerun()
        
        # Query input with better styling
        question = st.text_area(
            "**Your Question:**",
            value=st.session_state.get('sample_question', ''),
            placeholder="💭 Ask about revenue, costs, capacity, growth, margins, guidance, or any financial metric...",
            height=120,
            help="💡 Be specific with time periods (Q1FY26, FY25, etc.) and metrics for better results",
            key="main_question_input"
        )
        
        # Clear sample question after use
        if 'sample_question' in st.session_state:
            del st.session_state.sample_question
        
        # Enhanced Query Options
        st.markdown("#### ⚙️ **Query Settings**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_type = st.selectbox(
                "🔎 **Search Method**",
                ["Enhanced", "Standard", "Diverse"],
                key="main_search_type",
                help="• Enhanced: AI-powered relevance ranking\n• Standard: Basic similarity search\n• Diverse: Multiple source perspectives"
            )
        
        with col2:
            st.metric("📚 Selected Documents", len(selected_docs), help="Number of documents to search in")
        
        with col3:
            st.metric("🎯 Retrieval Count", k_value, help="Number of document chunks to retrieve")
        
        # Action buttons with better styling
        st.markdown("---")
        
        button_col1, button_col2, button_col3 = st.columns([2, 1, 1])
        
        with button_col1:
            if st.button(
                "🚀 **Get Structured Answer**", 
                type="primary", 
                key="main_get_answer",
                help="Analyze your question and provide a detailed, structured response",
                use_container_width=True
            ):
                if not question.strip():
                    st.error("⚠️ Please enter a question to get started!")
                elif not selected_docs:
                    st.warning("⚠️ Please select at least one document from the sidebar")
                else:
                    with st.spinner("🔄 Processing your question..."):
                        process_structured_query(question, model_choice, search_type, k_value, include_sources, selected_docs, output_format)
        
        with button_col2:
            if st.button("🔄 **Clear Question**", key="clear_question", use_container_width=True):
                st.session_state.main_question_input = ""
                st.rerun()
        
        with button_col3:
            if st.button("📋 **History**", key="main_view_history", use_container_width=True):
                show_response_history()
    
    with tab2:
        st.subheader("📊 Process Questions from Excel")
        
        # Excel file upload
        uploaded_excel = st.file_uploader(
            "Upload Excel file with questions",
            type=['xlsx', 'xls'],
            help="Upload an Excel file with columns: Question, Type, Figure in cr, Period, Figure"
        )
        
        if uploaded_excel:
            # Validate Excel file
            excel_bytes = uploaded_excel.read()
            is_valid, validation_msg = bulk_processor.validate_excel_file(excel_bytes, uploaded_excel.name)
            
            if not is_valid:
                st.error(f"❌ {validation_msg}")
            else:
                st.success(f"✅ {validation_msg}")
                
                # Show preview of questions
                try:
                    questions = bulk_processor.parse_excel_questions(excel_bytes)
                    st.info(f"📋 Found {len(questions)} questions in the Excel file")
                    
                    # Show first few questions as preview
                    if questions:
                        with st.expander("👀 Preview Questions"):
                            preview_df = pd.DataFrame([
                                {
                                    'Row': q.row_number,
                                    'Question': q.question[:100] + "..." if len(q.question) > 100 else q.question,
                                    'Type': q.question_type,
                                    'Figure Type': q.figure_type,
                                    'Period': q.period,
                                    'Expected Figure': q.figure_value
                                }
                                for q in questions[:5]  # Show first 5 questions
                            ])
                            st.dataframe(preview_df, use_container_width=True)
                            
                            if len(questions) > 5:
                                st.info(f"... and {len(questions) - 5} more questions")
                    
                    # Bulk processing options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        bulk_search_type = st.selectbox(
                            "🔎 Search Type",
                            ["Standard", "Enhanced", "Diverse"],
                            key="bulk_search_type",
                            help="Search strategy for bulk processing"
                        )
                    
                    with col2:
                        bulk_output_format = st.selectbox(
                            "📤 Output Format",
                            ["Excel", "JSON", "CSV"],
                            key="bulk_output_format",
                            help="Format for bulk results export"
                        )
                    
                    with col3:
                        if st.button("🚀 Process All Questions", type="primary", key="bulk_process_questions"):
                            process_bulk_questions(questions, model_choice, bulk_search_type, k_value, include_sources, selected_docs, bulk_output_format)
                
                except Exception as e:
                    st.error(f"❌ Failed to parse Excel file: {str(e)}")

def process_structured_query(question, model_choice, search_type, k_value, include_sources, selected_docs, output_format):
    """Process the user query and return structured output."""
    
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
            import os
            
            # Check if vector database exists
            vectordb_path = Path(Config.VECTOR_STORE_DIR)
            if not vectordb_path.exists():
                st.error("❌ Vector database directory does not exist. Please upload and process documents first.")
                return
            
            # Check if vector database has data
            if not any(vectordb_path.iterdir()):
                st.error("❌ Vector database is empty. Please upload and process documents first.")
                return
            
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
                client_settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Check if vector database has collections
            collections = vectordb._client.list_collections()
            if not collections:
                st.error("❌ Vector database has no collections. Please upload and process documents first.")
                return
            
            # Get collection count
            collection = vectordb._client.get_collection(name=collections[0].name)
            count = collection.count()
            if count == 0:
                st.error("❌ Vector database collection is empty. Please upload and process documents first.")
                return
            
            logger.info(f"Vector database loaded successfully with {count} documents")
            
            # Create retrievers
            base_retriever = vectordb.as_retriever(search_kwargs={"k": k_value})
            enhanced_retriever = EnhancedRetriever(vectordb, base_retriever)
            
            logger.info("Enhanced retriever initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to load vector database: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(f"❌ {error_msg}")
            st.info("💡 Try uploading and processing documents again, or check the system status in the sidebar.")
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
        
        # Enhanced prompts for structured output
        if model_choice == "OpenAI GPT-3.5":
            enhanced_prompt = f"""
You are an expert financial analyst assistant. Answer the following question based ONLY on the provided document snippets.

CRITICAL INSTRUCTIONS FOR STRUCTURED OUTPUT:
1. Quote EXACT numbers, percentages, and figures when available
2. Include specific time periods (Q1FY26, Q2FY25, etc.) in your answer
3. If the information is not in the snippets, say "Information not available in the provided documents"
4. Be precise with financial metrics and avoid approximations
5. Format your response clearly with specific values and units
6. For capacity, costs, capex, and other specific metrics, provide the exact values

DOCUMENT SNIPPETS:
{context}

QUESTION: {sanitized_question}

Provide a clear, factual answer with specific numbers and time periods:
"""
            
            result = openai_llm.invoke(enhanced_prompt)
            raw_answer = result.content
            
        else:  # Gemini
            prompt = f"""
You are a financial analyst expert. Answer the question based ONLY on the provided document snippets.

IMPORTANT RULES FOR STRUCTURED OUTPUT:
- Quote exact numbers, percentages, and financial figures
- Include specific quarters/years (Q1FY26, FY25, etc.)
- If information is missing, say "Information not available in the provided documents"
- Be precise with metrics like capacity, costs, capex, revenue, etc.
- Don't make assumptions or approximations
- Format response clearly with specific values

DOCUMENT CONTENT:
{context}

QUESTION: {sanitized_question}

Provide a precise answer with exact figures and time periods:
"""
            
            raw_answer = gemini_client.query(prompt)
        
        # Enhance answer with additional context
        enhanced_answer = answer_enhancer.enhance_answer(raw_answer, docs, sanitized_question)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Extract source information
        sources = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            filename = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', '?')
            sources.append(f"{filename} - {source} (Page {page})")
        
        # Parse into structured format
        try:
            structured_response = response_parser.parse_response(
                sanitized_question, 
                enhanced_answer, 
                sources, 
                processing_time
            )
            
        except Exception as e:
            st.error(f"❌ Error parsing structured response: {str(e)}")
            logger.error(f"Structured response parsing error: {e}", exc_info=True)
            
            # Fallback: show raw answer
            st.subheader("📬 Answer")
            with st.container():
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                    <p style="margin: 0; font-size: 16px; line-height: 1.6;">{enhanced_answer}</p>
                </div>
                """, unsafe_allow_html=True)
            st.info(f"⏱️ Response time: {processing_time:.2f}s | 📊 Retrieved {len(docs)} documents")
            return
        
        # Enhanced Answer Display
        st.markdown("---")
        
        # Success notification
        st.success(f"✅ **Answer Generated Successfully!** Found relevant information from {len(docs)} document sources.")
        
        # Main Answer Section
        st.markdown("### 💡 **Answer**")
        
        # Display the main answer in a beautiful card
        answer_text = structured_response.answer if hasattr(structured_response, 'answer') else enhanced_answer
        
        # Clean the answer text to prevent HTML issues
        import html
        clean_answer = html.escape(str(answer_text))
        
        confidence_color = "#28a745" if structured_response.confidence > 0.7 else "#ffc107" if structured_response.confidence > 0.4 else "#dc3545"
        
        # Display answer using native Streamlit components (avoid raw HTML rendering)
        st.write(answer_text)
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.metric(label="Confidence", value=f"{structured_response.confidence:.0%}")
        with meta_col2:
            st.caption(f"⏱️ {processing_time:.2f}s · 📊 {len(docs)} documents")
        
        # Key Metrics Section
        if structured_response.metrics and len(structured_response.metrics) > 0:
            st.markdown("### 📊 **Key Financial Metrics**")
            
            # Filter out invalid metrics
            valid_metrics = []
            for metric in structured_response.metrics:
                if hasattr(metric, 'metric_name') and hasattr(metric, 'value'):
                    # Skip metrics with generic names or empty values
                    if (metric.metric_name and 
                        metric.metric_name.lower() not in ['currency', 'type', 'format'] and
                        metric.value and str(metric.value).strip()):
                        valid_metrics.append(metric)
            
            if valid_metrics:
                # Create columns for metrics (max 3 per row)
                num_cols = min(len(valid_metrics), 3)
                metric_cols = st.columns(num_cols)
                
                for i, metric in enumerate(valid_metrics):
                    col_idx = i % num_cols
                    with metric_cols[col_idx]:
                        # Clean and format metric data
                        metric_name = str(metric.metric_name).title()
                        value_text = str(metric.value).strip()
                        
                        # Format unit text
                        unit_text = ""
                        if hasattr(metric, 'unit') and metric.unit:
                            unit = str(metric.unit).strip()
                            if unit and unit not in value_text:
                                unit_text = f" {unit}"
                        
                        # Format period text
                        period_text = ""
                        if hasattr(metric, 'time_period') and metric.time_period:
                            period = str(metric.time_period).strip()
                            if period:
                                period_text = f" ({period})"
                        
                        # Display metric
                        display_value = f"{value_text}{unit_text}"
                        help_text = f"Time Period: {period_text}" if period_text else None
                        
                        st.metric(
                            label=metric_name,
                            value=display_value,
                            help=help_text
                        )
            else:
                st.info("📊 No specific financial metrics extracted from this response")
        
        # Response Type and Classification
        col1, col2 = st.columns(2)
        with col1:
            response_type = structured_response.response_type.value if hasattr(structured_response, 'response_type') else "General"
            st.info(f"📝 **Response Type:** {response_type.title()}")
        
        with col2:
            query_type = getattr(structured_response, 'query_type', 'Financial Query')
            st.info(f"🏷️ **Query Category:** {query_type}")
        
        # Structured Output Formats (Collapsible)
        with st.expander("📋 **View in Different Formats**", expanded=False):
            format_tab1, format_tab2, format_tab3, format_tab4 = st.tabs(["JSON", "Markdown", "CSV", "Agent Format"])
            
            try:
                with format_tab1:
                    json_output = output_formatter.format_for_json(structured_response)
                    st.json(json.loads(json_output))
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_output,
                        file_name=f"response_{int(time.time())}.json",
                        mime="application/json"
                    )
                
                with format_tab2:
                    md_output = output_formatter.format_for_markdown(structured_response)
                    st.markdown(md_output)
                    st.download_button(
                        label="📥 Download Markdown",
                        data=md_output,
                        file_name=f"response_{int(time.time())}.md",
                        mime="text/markdown"
                    )
                
                with format_tab3:
                    csv_output = output_formatter.format_for_csv(structured_response)
                    st.code(csv_output, language="csv")
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_output,
                        file_name=f"response_{int(time.time())}.csv",
                        mime="text/csv"
                    )
                
                with format_tab4:
                    agent_output = output_formatter.format_for_agent(structured_response)
                    st.json(agent_output)
                    st.download_button(
                        label="📥 Download Agent Format",
                        data=json.dumps(agent_output, indent=2),
                        file_name=f"agent_response_{int(time.time())}.json",
                        mime="application/json"
                    )
            
            except Exception as e:
                st.error(f"❌ Error formatting structured output: {str(e)}")
                st.write("**Raw Answer:**")
                st.write(enhanced_answer)
        
        # Source Documents Section
        if include_sources and docs:
            with st.expander(f"📚 **Source Documents** ({len(docs)} references)", expanded=False):
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', '?')
                    filename = doc.metadata.get('filename', 'Unknown')
                    
                    # Clean content to prevent HTML issues
                    content = str(doc.page_content)
                    clean_content = html.escape(content[:800] + "..." if len(content) > 800 else content)
                    
                    # Create a nice card for each source
                    st.markdown(f"""
                    <div style="
                        background-color: #f8f9fa;
                        border: 1px solid #dee2e6;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 15px 0;
                        border-left: 5px solid #007bff;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <div style="
                            display: flex;
                            align-items: center;
                            margin-bottom: 12px;
                        ">
                            <h6 style="
                                margin: 0;
                                color: #495057;
                                font-weight: 600;
                                font-size: 16px;
                            ">
                                📄 Source {i}: {html.escape(str(filename))}
                            </h6>
                        </div>
                        <div style="
                            margin-bottom: 15px;
                            padding: 8px 12px;
                            background-color: #e3f2fd;
                            border-radius: 6px;
                            font-size: 14px;
                            color: #1565c0;
                        ">
                            📍 {html.escape(str(source))} • Page {html.escape(str(page))}
                        </div>
                        <div style="
                            background-color: white;
                            padding: 15px;
                            border-radius: 8px;
                            border: 1px solid #e9ecef;
                            font-size: 14px;
                            line-height: 1.6;
                            max-height: 250px;
                            overflow-y: auto;
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            white-space: pre-wrap;
                        ">
                            {clean_content}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        # Store in session state for history
        if 'response_history' not in st.session_state:
            st.session_state.response_history = []
        
        st.session_state.response_history.append({
            'timestamp': structured_response.timestamp,
            'query': structured_response.query,
            'response_type': structured_response.response_type.value,
            'confidence': structured_response.confidence,
            'processing_time': structured_response.processing_time
        })
        
        # Log successful query
        logger.log_query(sanitized_question, model_choice, True)
        
    except Exception as e:
        error_msg = f"Failed to process query: {str(e)}"
        logger.error(error_msg, exc_info=True)
        logger.log_query(question, model_choice, False)
        st.error(f"❌ {error_msg}")


def process_bulk_questions(questions, model_choice, search_type, k_value, include_sources, selected_docs, output_format):
    """Process multiple questions from Excel file."""
    
    try:
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
                client_settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Initialize base and enhanced retrievers
            base_retriever = vectordb.as_retriever(search_kwargs={"k": k_value})
            enhanced_retriever = EnhancedRetriever(vectordb, base_retriever)
            
        except Exception as e:
            st.error(f"❌ Failed to initialize vector database: {str(e)}")
            return
        
        # Define query function for bulk processing
        def query_function(question, **kwargs):
            try:
                # Validate and sanitize question
                is_valid, error_msg = InputValidator.validate_query(question)
                if not is_valid:
                    raise ValueError(f"Invalid query: {error_msg}")
                
                sanitized_question = InputValidator.sanitize_text(question)
                
                # Retrieve relevant documents
                if search_type == "Standard":
                    docs = base_retriever.get_relevant_documents(sanitized_question, k=k_value)
                elif search_type == "Enhanced":
                    docs = enhanced_retriever.get_relevant_documents(sanitized_question, k=k_value)
                    # Apply reranking
                    docs = enhanced_retriever.rerank_documents(sanitized_question, docs)
                else:  # Diverse
                    docs = enhanced_retriever.get_diverse_documents(sanitized_question, k=k_value)
                
                # Prepare context
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Generate answer using LLM
                if model_choice == "OpenAI GPT-3.5":
                    prompt = f"""You are a financial analyst assistant. Answer the following question based on the provided context.

Context:
{context}

Question: {sanitized_question}

Instructions:
1. Provide a clear, concise answer based on the context
2. If the information is not available in the context, say "Information not found in the provided documents"
3. Use specific numbers and metrics when available
4. Be precise and factual

Answer:"""
                    
                    response = openai_llm.invoke(prompt)
                    answer = response.content if hasattr(response, 'content') else str(response)
                    
                else:  # Gemini
                    prompt = f"""You are a financial analyst assistant. Answer the following question based on the provided context.

Context:
{context}

Question: {sanitized_question}

Instructions:
1. Provide a clear, concise answer based on the context
2. If the information is not available in the context, say "Information not found in the provided documents"
3. Use specific numbers and metrics when available
4. Be precise and factual

Answer:"""
                    
                    answer = gemini_client.query(prompt)
                
                # Enhance answer
                enhanced_answer = answer_enhancer.enhance_answer(answer, docs, sanitized_question)
                
                # Create a simple result object
                class QueryResult:
                    def __init__(self, answer):
                        self.answer = answer
                
                return QueryResult(enhanced_answer)
                
            except Exception as e:
                logger.error(f"Error in query function: {str(e)}")
                raise
        
        # Process bulk questions
        with st.spinner(f"Processing {len(questions)} questions..."):
            bulk_result = bulk_processor.process_bulk_questions(
                questions, 
                query_function,
                model_choice=model_choice,
                search_type=search_type,
                k_value=k_value
            )
        
        # Display results
        st.success(f"✅ Bulk processing completed!")
        
        # Show summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Questions", bulk_result.total_questions)
        with col2:
            st.metric("Successful", bulk_result.successful_questions)
        with col3:
            st.metric("Failed", bulk_result.failed_questions)
        with col4:
            st.metric("Success Rate", f"{bulk_result.summary['success_rate']:.1f}%")
        
        # Show processing time
        st.info(f"⏱️ Total processing time: {bulk_result.total_processing_time:.2f}s | Average per question: {bulk_result.summary['average_processing_time']:.2f}s")
        
        # Show detailed results
        with st.expander("📊 Detailed Results"):
            results_data = []
            for result in bulk_result.results:
                row = {
                    'Row': result.row_number,
                    'Question': result.question[:80] + "..." if len(result.question) > 80 else result.question,
                    'Status': result.status,
                    'Answer': result.answer[:100] + "..." if len(result.answer) > 100 else result.answer,
                    'Processing Time (s)': round(result.processing_time, 2),
                    'Error': result.error_message or ''
                }
                
                if result.structured_response:
                    row.update({
                        'Response Type': result.structured_response.response_type.value,
                        'Confidence': f"{result.structured_response.confidence:.2f}",
                        'Metrics Found': len(result.structured_response.metrics)
                    })
                else:
                    row.update({
                        'Response Type': '',
                        'Confidence': '',
                        'Metrics Found': 0
                    })
                
                results_data.append(row)
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
        
        # Export results
        st.subheader("📤 Export Results")
        
        try:
            export_data = bulk_processor.export_results(bulk_result, output_format.lower())
            
            # Determine file extension and MIME type
            if output_format.lower() == "excel":
                file_extension = "xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif output_format.lower() == "json":
                file_extension = "json"
                mime_type = "application/json"
            else:  # CSV
                file_extension = "csv"
                mime_type = "text/csv"
            
            # Create download button
            st.download_button(
                label=f"📥 Download {output_format.upper()} Results",
                data=export_data,
                file_name=f"bulk_questions_results_{int(time.time())}.{file_extension}",
                mime=mime_type
            )
            
        except Exception as e:
            st.error(f"❌ Failed to export results: {str(e)}")
        
        # Log bulk processing
        logger.info(f"Bulk processing completed: {bulk_result.successful_questions}/{bulk_result.total_questions} successful")
        
    except Exception as e:
        error_msg = f"Failed to process bulk questions: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")


def show_response_history():
    """Display response history."""
    if 'response_history' not in st.session_state or not st.session_state.response_history:
        st.info("📋 No response history available")
        return
    
    st.subheader("📋 Response History")
    
    history_data = st.session_state.response_history[-10:]  # Show last 10 responses
    
    for i, response in enumerate(reversed(history_data)):
        with st.expander(f"Query {len(history_data) - i}: {response['query'][:50]}..."):
            st.write(f"**Response Type:** {response['response_type']}")
            st.write(f"**Confidence:** {response['confidence']:.2f}")
            st.write(f"**Processing Time:** {response['processing_time']:.2f}s")
            st.write(f"**Timestamp:** {response['timestamp']}")

if __name__ == "__main__":
    main() 