import os
import json
import fitz  # PyMuPDF
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import hashlib
import tempfile
from pathlib import Path

from config import Config
from utils.logger import logger
from utils.validators import FileValidator, PathValidator

class OptimizedPDFProcessor:
    """High-performance PDF processor with parallel processing and caching."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.cache_dir = Path("cache/pdf_processing")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, file_bytes: bytes, filename: str) -> str:
        """Generate cache key for file."""
        file_hash = hashlib.md5(file_bytes).hexdigest()
        return f"{filename}_{file_hash}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load processed data from cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save processed data to cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _extract_text_from_page(self, page_data: Tuple[int, fitz.Page]) -> Dict:
        """Extract text from a single page with optimization."""
        page_num, page = page_data
        
        # Optimize text extraction
        text = page.get_text("text")  # Use "text" mode for better performance
        
        # Clean and preprocess text
        text = self._clean_text(text)
        
        return {
            "page": page_num + 1,
            "text": text,
            "word_count": len(text.split())
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing."""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
        
        return text.strip()
    
    def _process_single_pdf(self, file_bytes: bytes, filename: str) -> Dict:
        """Process a single PDF with caching and optimization."""
        cache_key = self._get_cache_key(file_bytes, filename)
        
        # Check cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            logger.info(f"Loaded from cache: {filename}")
            return cached_data
        
        try:
            # Validate file
            is_valid, validation_message = FileValidator.validate_uploaded_file(file_bytes, filename)
            if not is_valid:
                raise ValueError(f"File validation failed: {validation_message}")
            
            # Process PDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            # Process pages in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                page_data = [(i, doc[i]) for i in range(len(doc))]
                futures = [executor.submit(self._extract_text_from_page, page_data[i]) 
                          for i in range(len(page_data))]
                
                pages = []
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc=f"Processing {filename}"):
                    pages.append(future.result())
            
            doc.close()
            
            # Sort pages by page number
            pages.sort(key=lambda x: x["page"])
            
            # Create result
            result = {
                "filename": filename,
                "source": self._infer_source_from_filename(filename),
                "pages": pages,
                "total_pages": len(pages),
                "total_words": sum(page["word_count"] for page in pages),
                "processed_at": datetime.now().isoformat()
            }
            
            # Save to cache
            self._save_to_cache(cache_key, result)
            
            logger.info(f"Successfully processed: {filename} ({len(pages)} pages)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}", exc_info=True)
            raise
    
    def _infer_source_from_filename(self, filename: str) -> str:
        """Infer document source from filename."""
        filename_lower = filename.lower()
        
        if "transcript" in filename_lower or "call" in filename_lower:
            return "Conference Call Transcript"
        elif "ppt" in filename_lower or "presentation" in filename_lower:
            return "Investor Presentation"
        elif "report" in filename_lower or "quarterly" in filename_lower:
            return "Financial Report"
        elif "filing" in filename_lower or "sec" in filename_lower:
            return "SEC Filing"
        else:
            return "Financial Document"
    
    def process_multiple_pdfs(self, files_data: List[Tuple[bytes, str]]) -> List[Dict]:
        """Process multiple PDFs in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_single_pdf, file_bytes, filename): (file_bytes, filename)
                for file_bytes, filename in files_data
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    file_bytes, filename = futures[future]
                    logger.error(f"Failed to process {filename}: {str(e)}")
        
        return results

# Global processor instance
pdf_processor = OptimizedPDFProcessor() 