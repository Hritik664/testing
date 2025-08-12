import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.retrievers import BaseRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from config import Config
from utils.logger import logger

class FinancialQueryAnalyzer:
    """Analyze financial queries for better retrieval."""
    
    def __init__(self):
        self.financial_keywords = {
            'revenue': ['revenue', 'sales', 'income', 'top-line'],
            'profit': ['profit', 'earnings', 'net income', 'bottom-line'],
            'margin': ['margin', 'profitability', 'gross margin', 'operating margin'],
            'growth': ['growth', 'increase', 'decrease', 'change', 'trend'],
            'quarter': ['quarter', 'Q1', 'Q2', 'Q3', 'Q4', 'quarterly'],
            'year': ['year', 'annual', 'fiscal year', 'FY'],
            'region': ['region', 'geography', 'country', 'market'],
            'segment': ['segment', 'business unit', 'division', 'product line']
        }
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query for financial context and requirements."""
        query_lower = query.lower()
        
        analysis = {
            'query_type': self._classify_query_type(query_lower),
            'financial_entities': self._extract_financial_entities(query_lower),
            'time_period': self._extract_time_period(query_lower),
            'comparison_type': self._extract_comparison_type(query_lower),
            'specific_metrics': self._extract_specific_metrics(query_lower),
            'complexity_score': self._calculate_complexity_score(query_lower)
        }
        
        return analysis
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of financial query."""
        if any(word in query for word in ['revenue', 'sales', 'income']):
            return 'revenue'
        elif any(word in query for word in ['profit', 'earnings', 'margin']):
            return 'profitability'
        elif any(word in query for word in ['growth', 'increase', 'decrease']):
            return 'growth'
        elif any(word in query for word in ['quarter', 'Q1', 'Q2', 'Q3', 'Q4']):
            return 'quarterly'
        elif any(word in query for word in ['year', 'annual', 'fiscal']):
            return 'annual'
        elif any(word in query for word in ['region', 'geography', 'market']):
            return 'geographic'
        else:
            return 'general'
    
    def _extract_financial_entities(self, query: str) -> List[str]:
        """Extract financial entities from query."""
        entities = []
        for category, keywords in self.financial_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    entities.append(category)
                    break
        return list(set(entities))
    
    def _extract_time_period(self, query: str) -> Optional[str]:
        """Extract time period from query."""
        time_patterns = {
            r'Q[1-4]\s+\d{4}': 'quarter',
            r'FY\s+\d{4}': 'fiscal_year',
            r'\d{4}': 'year',
            r'last\s+quarter': 'previous_quarter',
            r'this\s+quarter': 'current_quarter',
            r'next\s+quarter': 'next_quarter'
        }
        
        for pattern, period_type in time_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return period_type
        return None
    
    def _extract_comparison_type(self, query: str) -> Optional[str]:
        """Extract comparison type from query."""
        if any(word in query for word in ['vs', 'versus', 'compared', 'comparison']):
            return 'comparison'
        elif any(word in query for word in ['trend', 'over time', 'growth']):
            return 'trend'
        elif any(word in query for word in ['forecast', 'projection', 'outlook']):
            return 'forecast'
        return None
    
    def _extract_specific_metrics(self, query: str) -> List[str]:
        """Extract specific financial metrics from query."""
        metrics = []
        metric_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',
            r'[\d,]+(?:\.\d{2})?%',
            r'[\d,]+(?:\.\d{2})?\s*(?:million|billion|trillion)'
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            metrics.extend(matches)
        
        return metrics
    
    def _calculate_complexity_score(self, query: str) -> float:
        """Calculate query complexity score."""
        score = 0.0
        
        # Length factor
        score += min(len(query.split()) / 10.0, 1.0)
        
        # Financial terms factor
        financial_terms = sum(1 for category in self.financial_keywords.values() 
                            for term in category if term in query)
        score += min(financial_terms / 5.0, 1.0)
        
        # Numbers factor
        numbers = len(re.findall(r'\d+', query))
        score += min(numbers / 3.0, 1.0)
        
        return min(score / 3.0, 1.0)

class EnhancedRetriever(BaseRetriever):
    """Enhanced retrieval system with better search strategies."""
    
    vectordb: Optional[Any] = None
    base_retriever: Optional[Any] = None
    query_analyzer: Optional[Any] = None
    
    def __init__(self, vectordb, base_retriever):
        super().__init__()
        self.vectordb = vectordb
        self.base_retriever = base_retriever
        self.query_analyzer = FinancialQueryAnalyzer()
        
    def _rerank_documents(self, documents: List[Document], query: str, query_analysis: Dict) -> List[Document]:
        """Rerank documents based on query analysis and financial relevance."""
        if not documents:
            return documents
        
        scored_docs = []
        
        for doc in documents:
            score = self._calculate_relevance_score(doc, query, query_analysis)
            scored_docs.append((doc, score))
        
        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs]
    
    def _calculate_relevance_score(self, doc: Document, query: str, query_analysis: Dict) -> float:
        """Calculate relevance score for a document."""
        score = 0.0
        doc_text = doc.page_content.lower()
        doc_metadata = doc.metadata
        
        # Base similarity score
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            vectors = vectorizer.fit_transform([query.lower(), doc_text])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            score += similarity * 0.4  # 40% weight for text similarity
        except:
            score += 0.0
        
        # Financial entity matching
        query_entities = query_analysis.get('financial_entities', [])
        doc_entities = doc_metadata.get('financial_entities', [])
        
        if query_entities and doc_entities:
            entity_overlap = len(set(query_entities) & set(doc_entities))
            entity_score = entity_overlap / max(len(query_entities), 1)
            score += entity_score * 0.3  # 30% weight for entity matching
        
        # Metadata relevance
        if query_analysis.get('query_type') == 'quarterly' and 'Q' in doc_text:
            score += 0.2
        if query_analysis.get('query_type') == 'annual' and ('year' in doc_text or 'annual' in doc_text):
            score += 0.2
        
        # Number presence (for quantitative queries)
        if query_analysis.get('complexity_score', 0) > 0.5:
            if doc_metadata.get('has_numbers', False):
                score += 0.1
            if doc_metadata.get('has_percentages', False) and '%' in query:
                score += 0.1
            if doc_metadata.get('has_currency', False) and '$' in query:
                score += 0.1
        
        # Source relevance
        source = doc_metadata.get('source', '').lower()
        if 'transcript' in source and 'call' in query.lower():
            score += 0.1
        if 'presentation' in source and 'presentation' in query.lower():
            score += 0.1
        
        return min(score, 1.0)
    
    def get_relevant_documents(self, query: str, *, k: int = 8) -> List[Document]:
        """Get relevant documents with enhanced retrieval."""
        # Analyze query
        query_analysis = self.query_analyzer.analyze_query(query)
        
        # Adjust k based on query complexity
        adjusted_k = min(k * 2, 16) if query_analysis['complexity_score'] > 0.5 else k
        
        # Get initial documents
        try:
            documents = self.base_retriever.get_relevant_documents(query, k=adjusted_k)
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            return []
        
        # Rerank documents
        reranked_docs = self._rerank_documents(documents, query, query_analysis)
        
        # Return top k documents
        return reranked_docs[:k]
    
    def get_diverse_documents(self, query: str, k: int = 6) -> List[Document]:
        """Get diverse set of relevant documents."""
        documents = self.get_relevant_documents(query, k=k * 2)
        
        if not documents:
            return []
        
        # Ensure diversity by source and page
        diverse_docs = []
        seen_sources = set()
        seen_pages = set()
        
        for doc in documents:
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page', 0)
            source_page = f"{source}_{page}"
            
            if len(diverse_docs) >= k:
                break
            
            # Add if we haven't seen this source-page combination
            if source_page not in seen_pages:
                diverse_docs.append(doc)
                seen_sources.add(source)
                seen_pages.add(source_page)
            # Or if we need more diversity in sources
            elif source not in seen_sources and len(seen_sources) < k // 2:
                diverse_docs.append(doc)
                seen_sources.add(source)
                seen_pages.add(source_page)
        
        return diverse_docs

class AnswerEnhancer:
    """Enhance answer quality and accuracy."""
    
    def __init__(self):
        self.financial_patterns = {
            'currency': r'\$[\d,]+(?:\.\d{2})?',
            'percentage': r'[\d,]+(?:\.\d{2})?%',
            'large_numbers': r'[\d,]+(?:\.\d{2})?\s*(?:million|billion|trillion)',
            'quarters': r'Q[1-4]\s+\d{4}',
            'years': r'\d{4}'
        }
    
    def enhance_answer(self, answer: str, documents: List[Document], query: str) -> str:
        """Enhance answer with additional context and verification."""
        enhanced_answer = answer
        
        # Extract key numbers from documents
        key_numbers = self._extract_key_numbers(documents)
        
        # Add source citations
        source_citations = self._generate_source_citations(documents)
        
        # Add confidence indicators
        confidence = self._calculate_confidence(documents, query)
        
        # Enhance with key numbers if missing
        if key_numbers and not any(num in answer for num in key_numbers[:3]):
            enhanced_answer += f"\n\nKey figures mentioned: {', '.join(key_numbers[:3])}"
        
        # Add source information
        if source_citations:
            enhanced_answer += f"\n\nSources: {source_citations}"
        
        # Add confidence level
        if confidence < 0.7:
            enhanced_answer += "\n\nNote: Limited information available for this query."
        elif confidence > 0.9:
            enhanced_answer += "\n\nNote: High confidence based on multiple sources."
        
        return enhanced_answer
    
    def _extract_key_numbers(self, documents: List[Document]) -> List[str]:
        """Extract key financial numbers from documents."""
        numbers = []
        
        for doc in documents:
            text = doc.page_content
            
            # Extract different types of numbers
            for pattern_name, pattern in self.financial_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                numbers.extend(matches)
        
        # Return unique numbers, sorted by frequency
        from collections import Counter
        number_counts = Counter(numbers)
        return [num for num, count in number_counts.most_common(5)]
    
    def _generate_source_citations(self, documents: List[Document]) -> str:
        """Generate source citations for the answer."""
        sources = {}
        
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '?')
            
            if source not in sources:
                sources[source] = []
            sources[source].append(page)
        
        citations = []
        for source, pages in sources.items():
            unique_pages = sorted(list(set(pages)))
            page_str = f"pages {', '.join(map(str, unique_pages))}" if len(unique_pages) > 1 else f"page {unique_pages[0]}"
            citations.append(f"{source} ({page_str})")
        
        return "; ".join(citations)
    
    def _calculate_confidence(self, documents: List[Document], query: str) -> float:
        """Calculate confidence level for the answer."""
        if not documents:
            return 0.0
        
        # Factors for confidence calculation
        num_sources = len(set(doc.metadata.get('source', '') for doc in documents))
        total_relevance = sum(1 for doc in documents if any(word in doc.page_content.lower() 
                                                          for word in query.lower().split()))
        has_numbers = any(doc.metadata.get('has_numbers', False) for doc in documents)
        
        # Calculate confidence score
        confidence = 0.0
        confidence += min(num_sources / 3.0, 1.0) * 0.4  # Source diversity
        confidence += min(total_relevance / len(documents), 1.0) * 0.3  # Relevance
        confidence += 0.3 if has_numbers else 0.0  # Quantitative data
        
        return min(confidence, 1.0)

# Global instances
query_analyzer = FinancialQueryAnalyzer()
answer_enhancer = AnswerEnhancer() 