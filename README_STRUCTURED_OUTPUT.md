# Structured Financial RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that provides **structured, agent-ready responses** for financial document analysis. The system can interpret initial responses and format them appropriately for consumption by other AI agents.

## 🎯 Key Features

### 📊 Structured Output
- **Response Classification**: Automatically categorizes responses (revenue, cost, capacity, trend, etc.)
- **Metric Extraction**: Extracts financial metrics with values, units, and time periods
- **Confidence Scoring**: Provides confidence levels for response quality
- **Source Attribution**: Links responses to specific document sources and pages

### 🤖 Agent-Ready Formats
- **JSON**: Structured data for programmatic consumption
- **Markdown**: Human-readable formatted reports
- **CSV**: Tabular data for analysis tools
- **Agent Format**: Optimized for AI agent consumption

### 📊 Bulk Question Processing
- **Excel Upload**: Process multiple questions from Excel files
- **Batch Processing**: Handle large question sets efficiently
- **Progress Tracking**: Real-time progress monitoring
- **Export Results**: Download results in Excel, JSON, or CSV formats
- **Validation**: Automatic Excel file format validation

### 🚀 Multiple Interfaces
- **Streamlit Web App**: Interactive UI for document upload and querying
- **REST API**: FastAPI endpoint for programmatic access
- **Client Library**: Python client for easy integration

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Vector        │    │   LLM           │
│   Processing    │───▶│   Database      │───▶│   Generation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Structured    │    │   Response      │    │   Output        │
│   Output        │◀───│   Parser        │◀───│   Formatter     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd financial_rag
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp env_example.txt .env
# Edit .env with your API keys
```

## 🚀 Quick Start

### Option 1: Streamlit Web App
```bash
streamlit run ui/structured_rag_app.py
```
- Upload PDF documents
- Ask financial questions (single or bulk from Excel)
- Get structured responses in multiple formats
- Download results for agent consumption

#### Bulk Question Processing
1. **Create Excel file** with columns: `Question`, `Type`, `Figure in cr`, `Period`, `Figure`
2. **Upload Excel file** in the "Bulk Questions" tab
3. **Process all questions** automatically
4. **Download results** in Excel, JSON, or CSV format

Example Excel format:
| Question | Type | Figure in cr | Period | Figure |
|----------|------|--------------|--------|--------|
| Revenue guidance for the next few years | Figure | cr | 2 years | 3500 |
| Guidance on R&D expense | Range | % | Annual | 1.9 to 2 |

### Option 2: REST API
```bash
python api/structured_rag_api.py
```
- API runs on `http://localhost:8000`
- Interactive docs at `http://localhost:8000/docs`
- Programmatic access for agents

### Option 3: Direct Python Usage
```python
from utils.structured_output import response_parser, output_formatter

# Parse a response
structured_response = response_parser.parse_response(
    query="What was the revenue in Q1FY26?",
    raw_response="Revenue in Q1FY26 was 1,234 cr",
    sources=["document.pdf"],
    processing_time=2.5
)

# Format for agent consumption
agent_data = output_formatter.format_for_agent(structured_response)
```

## 📋 API Endpoints

### Query Documents
```http
POST /query
Content-Type: application/json

{
  "query": "What was the sales volume in Q1FY25?",
  "model": "gemini",
  "search_type": "enhanced",
  "output_format": "agent",
  "k_value": 6,
  "include_sources": true
}
```

### Upload Documents
```http
POST /upload
Content-Type: multipart/form-data

files: [PDF files]
```

### Bulk Questions from Excel
```http
POST /bulk_questions
Content-Type: multipart/form-data

file: [Excel file with questions]
```

**Excel Format Required:**
- Columns: `Question`, `Type`, `Figure in cr`, `Period`, `Figure`
- File types: `.xlsx`, `.xls`
- Maximum size: 50MB

### System Status
```http
GET /status
```

## 🤖 Agent Consumption Examples

### Python Client
```python
from examples.agent_client_example import StructuredRAGClient

# Initialize client
client = StructuredRAGClient("http://localhost:8000")

# Query documents
response = client.query_documents(
    query="What was the revenue growth in Q1FY26?",
    output_format="agent"
)

# Process structured response
if response["status"] == "success":
    metrics = response["metrics"]
    confidence = response["confidence"]
    
    for metric in metrics:
        print(f"{metric['metric_name']}: {metric['value']} {metric['unit']}")
```

### Financial Analysis Agent
```python
from examples.agent_client_example import FinancialAnalysisAgent

# Initialize agent
agent = FinancialAnalysisAgent(client)

# Analyze multiple metrics
queries = [
    "What was the sales volume in Q1FY25?",
    "What is the current capacity?",
    "What was the cost per ton in Q1FY26?"
]

results = agent.analyze_financial_metrics(queries)
report = agent.generate_report(results)
```

## 📊 Response Structure

### Agent Format Response
```json
{
  "status": "success",
  "query": "What was the sales volume in Q1FY25?",
  "response_type": "capacity",
  "answer": "The sales volume in Q1FY25 was 7.4 mnT.",
  "metrics": [
    {
      "metric_name": "volume",
      "value": 7.4,
      "unit": "mnT",
      "time_period": "Q1FY25",
      "confidence": 0.8
    }
  ],
  "confidence": 0.85,
  "sources": ["investor_presentation.pdf - Q1FY25 Results (Page 3)"],
  "timestamp": "2025-08-06T08:30:00",
  "processing_time": 2.5,
  "metadata": {
    "total_metrics": 1,
    "has_sources": true,
    "response_quality": "high"
  }
}
```

## 🔧 Configuration

### Environment Variables
```bash
# API Keys
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key

# Model Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=200

# Performance
MAX_WORKERS=4
BATCH_SIZE=32
GPU_ENABLED=false
```

### Response Types
- `revenue`: Revenue, sales, income metrics
- `cost`: Cost, expense, expenditure metrics
- `capacity`: Production capacity, volume metrics
- `trend`: Growth, increase, decrease patterns
- `comparison`: Comparative analysis
- `general`: General information
- `not_found`: Information not available

## 🧪 Testing

### Test Structured Output System
```bash
python test_structured_output.py
```

### Test API Endpoints
```bash
python examples/agent_client_example.py
```

### Test ChromaDB
```bash
python test_chroma.py
```

## 📈 Performance Optimization

### Caching
- Document processing cache
- Embedding cache
- Response cache

### Parallel Processing
- Multi-threaded PDF processing
- Batch embedding generation
- Concurrent API requests

### Memory Management
- Streaming document processing
- Efficient vector storage
- Garbage collection optimization

## 🔍 Troubleshooting

### Common Issues

1. **ChromaDB Issues**:
```bash
python check_vector_db.py reset
python check_vector_db.py rebuild
```

2. **API Connection Issues**:
```bash
# Check if API is running
curl http://localhost:8000/health
```

3. **Document Processing Issues**:
```bash
# Clear cache and reprocess
python clear_data.py
```

### Logs
- Application logs: `logs/financial_rag.log`
- API logs: Console output
- Error logs: `logs/error.log`

## 🤝 Integration Examples

### Integration with Trading Systems
```python
# Get real-time financial metrics
response = client.query_documents("What is the current capacity?")
if response["confidence"] > 0.8:
    capacity = response["metrics"][0]["value"]
    # Use in trading algorithm
```

### Integration with Reporting Systems
```python
# Generate quarterly reports
queries = [
    "Q1 revenue growth",
    "Q1 cost per unit",
    "Q1 capacity utilization"
]

results = agent.analyze_financial_metrics(queries)
report = agent.generate_report(results)
# Send to reporting system
```

### Integration with Alert Systems
```python
# Monitor key metrics
response = client.query_documents("What is the current cost per ton?")
current_cost = response["metrics"][0]["value"]

if current_cost > threshold:
    # Send alert
    send_alert(f"Cost per ton increased to {current_cost}")
```

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) (when API is running)
- [Configuration Guide](config.py)
- [Testing Guide](test_structured_output.py)
- [Client Examples](examples/agent_client_example.py)

## 🎯 Use Cases

1. **Financial Analysis**: Extract and analyze financial metrics from documents
2. **Compliance Monitoring**: Track regulatory compliance metrics
3. **Performance Tracking**: Monitor key performance indicators
4. **Risk Assessment**: Analyze financial risk factors
5. **Reporting Automation**: Generate automated financial reports
6. **Trading Systems**: Provide real-time financial data for algorithms
7. **Audit Support**: Extract audit-relevant information
8. **Due Diligence**: Analyze company financials for M&A

## 🔮 Future Enhancements

- [ ] Real-time document processing
- [ ] Multi-language support
- [ ] Advanced financial entity recognition
- [ ] Integration with external data sources
- [ ] Machine learning model fine-tuning
- [ ] Advanced visualization capabilities
- [ ] Blockchain integration for audit trails
- [ ] Mobile application support

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation
- Test with the provided examples 