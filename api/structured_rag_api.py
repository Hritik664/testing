#!/usr/bin/env python3
"""
FastAPI endpoint for Structured Financial RAG
Provides REST API for agent consumption of structured responses.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dataclasses import asdict
import uvicorn
import time
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.logger import logger
from utils.validators import InputValidator
from utils.api_client import get_gemini_client, get_openai_client
from utils.optimized_pdf_processor import pdf_processor
from utils.enhanced_embedder import enhanced_embedder
from utils.enhanced_retriever import EnhancedRetriever, answer_enhancer
from utils.structured_output import response_parser, output_formatter
from utils.bulk_question_processor import bulk_processor

# Initialize FastAPI app
app = FastAPI(
    title="Structured Financial RAG API",
    description="API for structured financial document analysis and querying",
    version="1.0.0"
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    model: str = "gemini"  # "openai" or "gemini"
    search_type: str = "enhanced"  # "standard", "enhanced", "diverse"
    k_value: int = 6
    include_sources: bool = True
    output_format: str = "agent"  # "json", "markdown", "csv", "agent"

class QueryResponse(BaseModel):
    status: str
    query: str
    response_type: str
    answer: str
    metrics: List[Dict[str, Any]]
    confidence: float
    sources: List[str]
    timestamp: str
    processing_time: float
    metadata: Dict[str, Any]

class DocumentUploadResponse(BaseModel):
    status: str
    message: str
    documents_processed: int
    embeddings_created: int

class SystemStatusResponse(BaseModel):
    status: str
    vector_database_ready: bool
    documents_available: int
    cache_status: str

# Global variables for caching
_vectordb = None
_retriever = None
_embedding_model = None

def initialize_components():
    """Initialize vector database and retriever components."""
    global _vectordb, _retriever, _embedding_model
    
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from chromadb.config import Settings
        
        if _embedding_model is None:
            _embedding_model = HuggingFaceEmbeddings(
                model_name=Config.get_embedding_model_name(),
                model_kwargs={'device': 'cuda' if Config.GPU_ENABLED else 'cpu'},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': Config.BATCH_SIZE}
            )
        
        if _vectordb is None:
            _vectordb = Chroma(
                persist_directory=Config.VECTOR_STORE_DIR,
                embedding_function=_embedding_model,
                client_settings=Settings(anonymized_telemetry=False)
            )
        
        if _retriever is None:
            base_retriever = _vectordb.as_retriever(search_kwargs={"k": 6})
            _retriever = EnhancedRetriever(_vectordb, base_retriever)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting Structured Financial RAG API...")
    Config.create_directories()
    
    if initialize_components():
        logger.info("✅ API components initialized successfully")
    else:
        logger.warning("⚠️ Some components failed to initialize")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Structured Financial RAG API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Query documents with structured output",
            "/upload": "POST - Upload and process documents",
            "/status": "GET - System status",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status and component health."""
    try:
        # Check vector database
        vectordb_path = Path(Config.VECTOR_STORE_DIR)
        vector_database_ready = vectordb_path.exists() and any(vectordb_path.iterdir())
        
        # Check processed documents
        processed_dir = Path(Config.PROCESSED_DATA_DIR)
        documents_available = 0
        if processed_dir.exists():
            documents_available = len(list(processed_dir.glob("*.json")))
        
        # Check cache
        cache_dir = Path("cache")
        cache_status = "available" if cache_dir.exists() and any(cache_dir.iterdir()) else "empty"
        
        return SystemStatusResponse(
            status="operational" if vector_database_ready else "initializing",
            vector_database_ready=vector_database_ready,
            documents_available=documents_available,
            cache_status=cache_status
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate files
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
        
        # Process files
        files_data = []
        for file in files:
            content = await file.read()
            files_data.append((content, file.filename))
        
        # Process with PDF processor
        processed_data = pdf_processor.process_multiple_pdfs(files_data)
        
        if not processed_data:
            raise HTTPException(status_code=500, detail="Failed to process documents")
        
        # Save processed data
        for data in processed_data:
            filename = data['filename']
            json_path = Path(Config.PROCESSED_DATA_DIR) / f"{filename}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Create documents for embedding
        documents = enhanced_embedder.process_documents(processed_data)
        
        # Embed and store
        enhanced_embedder.embed_and_store(documents, force_reload=False)
        
        # Reinitialize components
        global _vectordb, _retriever
        _vectordb = None
        _retriever = None
        initialize_components()
        
        return DocumentUploadResponse(
            status="success",
            message=f"Successfully processed {len(processed_data)} documents",
            documents_processed=len(processed_data),
            embeddings_created=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Failed to upload documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload documents: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents and return structured response."""
    try:
        # Validate query
        is_valid, error_msg = InputValidator.validate_query(request.query)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid query: {error_msg}")
        
        sanitized_question = InputValidator.sanitize_text(request.query)
        
        # Initialize components if needed
        if not initialize_components():
            raise HTTPException(status_code=500, detail="Failed to initialize system components")
        
        # Initialize LLM client
        if request.model.lower() == "openai":
            llm_client = get_openai_client()
            use_openai = True
        else:
            llm_client = get_gemini_client()
            use_openai = False
        
        # Process query
        start_time = time.time()
        
        # Select retrieval method
        if request.search_type == "enhanced":
            docs = _retriever.get_relevant_documents(sanitized_question, k=request.k_value)
        elif request.search_type == "diverse":
            docs = _retriever.get_diverse_documents(sanitized_question, k=request.k_value)
        else:
            docs = _vectordb.as_retriever(search_kwargs={"k": request.k_value}).get_relevant_documents(sanitized_question)
        
        # Create context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        if use_openai:
            openai_llm = llm_client.get_llm()
            prompt = f"""
You are an expert financial analyst assistant. Answer the following question based ONLY on the provided document snippets.

CRITICAL INSTRUCTIONS FOR STRUCTURED OUTPUT:
1. Quote EXACT numbers, percentages, and figures when available
2. Include specific time periods (Q1FY26, Q2FY25, etc.) in your answer
3. If the information is not in the snippets, say "Information not available in the provided documents"
4. Be precise with financial metrics and avoid approximations
5. Format your response clearly with specific values and units

DOCUMENT SNIPPETS:
{context}

QUESTION: {sanitized_question}

Provide a clear, factual answer with specific numbers and time periods:
"""
            result = openai_llm.invoke(prompt)
            raw_answer = result.content
        else:
            prompt = f"""
You are a financial analyst expert. Answer the question based ONLY on the provided document snippets.

IMPORTANT RULES FOR STRUCTURED OUTPUT:
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
            raw_answer = llm_client.query(prompt)
        
        # Enhance answer
        enhanced_answer = answer_enhancer.enhance_answer(raw_answer, docs, sanitized_question)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Extract sources
        sources = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            filename = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', '?')
            sources.append(f"{filename} - {source} (Page {page})")
        
        # Parse into structured format
        structured_response = response_parser.parse_response(
            sanitized_question,
            enhanced_answer,
            sources,
            processing_time
        )
        
        # Format response based on requested format
        if request.output_format == "json":
            formatted_response = output_formatter.format_for_agent(structured_response)
        elif request.output_format == "markdown":
            formatted_response = {"markdown": output_formatter.format_for_markdown(structured_response)}
        elif request.output_format == "csv":
            formatted_response = {"csv": output_formatter.format_for_csv(structured_response)}
        else:  # agent format
            formatted_response = output_formatter.format_for_agent(structured_response)
        
        # Log successful query
        logger.log_query(sanitized_question, request.model, True)
        
        return QueryResponse(**formatted_response)
        
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        logger.log_query(request.query, request.model, False)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.post("/batch_query")
async def batch_query_documents(queries: List[str]):
    """Process multiple queries in batch."""
    try:
        results = []
        for query in queries:
            try:
                request = QueryRequest(query=query)
                result = await query_documents(request)
                results.append(result.dict())
            except Exception as e:
                results.append({
                    "status": "error",
                    "query": query,
                    "error": str(e)
                })
        
        return {"results": results, "total_queries": len(queries)}
        
    except Exception as e:
        logger.error(f"Failed to process batch queries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process batch queries: {str(e)}")


@app.post("/bulk_questions")
async def process_bulk_questions(file: UploadFile = File(...)):
    """Process bulk questions from Excel file."""
    try:
        # Validate file
        if not file.filename.lower().endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")
        
        # Read file content
        content = await file.read()
        
        # Validate Excel file
        is_valid, validation_msg = bulk_processor.validate_excel_file(content, file.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_msg)
        
        # Parse questions
        questions = bulk_processor.parse_excel_questions(content)
        
        # Initialize components
        initialize_components()
        
        # Define query function
        def query_function(question, **kwargs):
            try:
                # Use the existing query logic
                request = QueryRequest(query=question)
                result = query_documents_sync(request)
                return result
            except Exception as e:
                logger.error(f"Error in bulk query function: {str(e)}")
                raise
        
        # Process bulk questions
        bulk_result = bulk_processor.process_bulk_questions(questions, query_function)
        
        # Convert to dict for JSON response
        result_dict = {
            "total_questions": bulk_result.total_questions,
            "successful_questions": bulk_result.successful_questions,
            "failed_questions": bulk_result.failed_questions,
            "total_processing_time": bulk_result.total_processing_time,
            "summary": bulk_result.summary,
            "timestamp": bulk_result.timestamp,
            "results": []
        }
        
        # Convert results to dict
        for result in bulk_result.results:
            result_dict["results"].append({
                "question": result.question,
                "question_type": result.question_type,
                "figure_type": result.figure_type,
                "period": result.period,
                "figure_value": result.figure_value,
                "row_number": result.row_number,
                "answer": result.answer,
                "processing_time": result.processing_time,
                "status": result.status,
                "error_message": result.error_message,
                "structured_response": asdict(result.structured_response) if result.structured_response else None
            })
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Failed to process bulk questions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process bulk questions: {str(e)}")


def query_documents_sync(request: QueryRequest) -> QueryResponse:
    """Synchronous version of query_documents for bulk processing."""
    try:
        # Validate input
        is_valid, error_msg = InputValidator.validate_query(request.query)
        if not is_valid:
            raise ValueError(f"Invalid query: {error_msg}")
        
        sanitized_query = InputValidator.sanitize_text(request.query)
        
        # Initialize components if needed
        initialize_components()
        
        # Get relevant documents
        docs = _retriever.get_relevant_documents(sanitized_query, k=8)
        
        if not docs:
            return QueryResponse(
                query=request.query,
                answer="No relevant documents found.",
                confidence=0.0,
                sources=[],
                processing_time=0.0,
                status="no_documents"
            )
        
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer using OpenAI (default for API)
        openai_client = get_openai_client()
        openai_llm = openai_client.get_llm()
        
        prompt = f"""You are a financial analyst assistant. Answer the following question based on the provided context.

Context:
{context}

Question: {sanitized_query}

Instructions:
1. Provide a clear, concise answer based on the context
2. If the information is not available in the context, say "Information not found in the provided documents"
3. Use specific numbers and metrics when available
4. Be precise and factual

Answer:"""
        
        response = openai_llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Enhance answer
        enhanced_answer = answer_enhancer.enhance_answer(answer, docs, sanitized_query)
        
        # Parse into structured format
        structured_response = response_parser.parse_response(sanitized_query, enhanced_answer)
        
        # Extract source information
        sources = []
        for doc in docs:
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            })
        
        return QueryResponse(
            query=request.query,
            answer=enhanced_answer,
            confidence=structured_response.confidence,
            sources=sources,
            processing_time=0.0,  # Will be calculated by bulk processor
            status="success",
            structured_response=asdict(structured_response)
        )
        
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise

if __name__ == "__main__":
    uvicorn.run(
        "api.structured_rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 