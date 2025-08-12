#!/usr/bin/env python3
"""
Example client for consuming the Structured Financial RAG API
Demonstrates how other agents can use the structured output.
"""

import requests
import json
import time
from typing import Dict, List, Any
from pathlib import Path

class StructuredRAGClient:
    """Client for consuming the Structured Financial RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        try:
            response = self.session.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def upload_documents(self, pdf_files: List[str]) -> Dict[str, Any]:
        """Upload and process documents."""
        try:
            files = []
            for pdf_file in pdf_files:
                if Path(pdf_file).exists():
                    files.append(('files', (Path(pdf_file).name, open(pdf_file, 'rb'), 'application/pdf')))
            
            response = self.session.post(f"{self.base_url}/upload", files=files)
            response.raise_for_status()
            
            # Close file handles
            for _, (_, file_handle, _) in files:
                file_handle.close()
            
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def query_documents(self, query: str, model: str = "gemini", 
                       search_type: str = "enhanced", output_format: str = "agent") -> Dict[str, Any]:
        """Query documents and get structured response."""
        try:
            payload = {
                "query": query,
                "model": model,
                "search_type": search_type,
                "output_format": output_format,
                "k_value": 6,
                "include_sources": True
            }
            
            response = self.session.post(f"{self.base_url}/query", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def batch_query(self, queries: List[str]) -> Dict[str, Any]:
        """Process multiple queries in batch."""
        try:
            response = self.session.post(f"{self.base_url}/batch_query", json=queries)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

class FinancialAnalysisAgent:
    """Example agent that consumes structured RAG responses."""
    
    def __init__(self, rag_client: StructuredRAGClient):
        self.rag_client = rag_client
        self.analysis_history = []
    
    def analyze_financial_metrics(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze financial metrics using structured RAG responses."""
        print("🔍 Starting Financial Analysis...")
        
        analysis_results = {
            "timestamp": time.time(),
            "queries": queries,
            "responses": [],
            "summary": {},
            "confidence_scores": []
        }
        
        for query in queries:
            print(f"📊 Querying: {query}")
            
            # Get structured response
            response = self.rag_client.query_documents(query)
            
            if response.get("status") == "success":
                analysis_results["responses"].append(response)
                analysis_results["confidence_scores"].append(response.get("confidence", 0))
                
                # Extract metrics for summary
                metrics = response.get("metrics", [])
                for metric in metrics:
                    metric_name = metric.get("metric_name", "unknown")
                    value = metric.get("value", "N/A")
                    unit = metric.get("unit", "")
                    time_period = metric.get("time_period", "")
                    
                    if metric_name not in analysis_results["summary"]:
                        analysis_results["summary"][metric_name] = []
                    
                    analysis_results["summary"][metric_name].append({
                        "value": value,
                        "unit": unit,
                        "time_period": time_period,
                        "confidence": response.get("confidence", 0)
                    })
                
                print(f"✅ Response: {response.get('answer', 'No answer')[:100]}...")
                print(f"🎯 Confidence: {response.get('confidence', 0):.2f}")
            else:
                print(f"❌ Failed to get response: {response.get('message', 'Unknown error')}")
        
        # Calculate overall confidence
        if analysis_results["confidence_scores"]:
            avg_confidence = sum(analysis_results["confidence_scores"]) / len(analysis_results["confidence_scores"])
            analysis_results["overall_confidence"] = avg_confidence
        else:
            analysis_results["overall_confidence"] = 0
        
        self.analysis_history.append(analysis_results)
        return analysis_results
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a human-readable report from analysis results."""
        report = "# Financial Analysis Report\n\n"
        report += f"**Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Overall Confidence:** {analysis_results.get('overall_confidence', 0):.2f}\n\n"
        
        # Summary of metrics
        if analysis_results.get("summary"):
            report += "## Key Metrics Summary\n\n"
            for metric_name, values in analysis_results["summary"].items():
                report += f"### {metric_name.title()}\n"
                for value_info in values:
                    report += f"- {value_info['value']} {value_info['unit']} ({value_info['time_period']}) "
                    report += f"[Confidence: {value_info['confidence']:.2f}]\n"
                report += "\n"
        
        # Detailed responses
        report += "## Detailed Responses\n\n"
        for i, response in enumerate(analysis_results.get("responses", []), 1):
            report += f"### Query {i}: {response.get('query', 'Unknown')}\n"
            report += f"**Answer:** {response.get('answer', 'No answer')}\n"
            report += f"**Confidence:** {response.get('confidence', 0):.2f}\n"
            report += f"**Response Type:** {response.get('response_type', 'Unknown')}\n\n"
        
        return report
    
    def export_to_json(self, analysis_results: Dict[str, Any], filename: str = None) -> str:
        """Export analysis results to JSON file."""
        if filename is None:
            filename = f"financial_analysis_{int(time.time())}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        return filename

def main():
    """Example usage of the Structured RAG Client and Financial Analysis Agent."""
    
    print("🚀 Structured Financial RAG Client Example")
    print("=" * 50)
    
    # Initialize client
    client = StructuredRAGClient()
    
    # Check health
    print("🔍 Checking API health...")
    health = client.health_check()
    print(f"Health Status: {health}")
    
    if health.get("status") != "healthy":
        print("❌ API is not healthy. Please start the API server first.")
        print("Run: python api/structured_rag_api.py")
        return
    
    # Get system status
    print("\n📊 Getting system status...")
    status = client.get_system_status()
    print(f"System Status: {status}")
    
    # Initialize analysis agent
    agent = FinancialAnalysisAgent(client)
    
    # Example financial queries
    queries = [
        "What was the sales volume in Q1FY25?",
        "What was the revenue growth in Q1FY26?",
        "What is the current capacity?",
        "What was the cost per ton in Q1FY26?",
        "What is the capex for new expansion?"
    ]
    
    print(f"\n📋 Analyzing {len(queries)} financial queries...")
    
    # Perform analysis
    analysis_results = agent.analyze_financial_metrics(queries)
    
    # Generate report
    print("\n📄 Generating analysis report...")
    report = agent.generate_report(analysis_results)
    
    # Save report
    report_filename = f"financial_analysis_report_{int(time.time())}.md"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Export to JSON
    json_filename = agent.export_to_json(analysis_results)
    
    print(f"\n✅ Analysis completed!")
    print(f"📄 Report saved to: {report_filename}")
    print(f"📊 Data exported to: {json_filename}")
    print(f"🎯 Overall confidence: {analysis_results.get('overall_confidence', 0):.2f}")
    
    # Display summary
    print("\n📋 Analysis Summary:")
    for metric_name, values in analysis_results.get("summary", {}).items():
        print(f"  {metric_name.title()}: {len(values)} values found")
    
    print("\n🤖 Agent consumption example completed!")

if __name__ == "__main__":
    main() 