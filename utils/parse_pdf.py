import os
import fitz  # PyMuPDF
import json
from datetime import datetime
from tqdm import tqdm
from config import Config
from utils.logger import logger
from utils.validators import FileValidator, PathValidator

# Create directories
Config.create_directories()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append({"page": page_num + 1, "text": text})
    return pages

def parse_and_save(pdf_path):
    try:
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Validate file path
        if not PathValidator.validate_directory_path(pdf_path):
            logger.error(f"Invalid file path: {pdf_path}")
            return False

        parsed = {
            "filename": filename,
            "source": infer_source_from_filename(base_name),
            "pages": extract_text_from_pdf(pdf_path)
        }

        output_path = os.path.join(Config.PROCESSED_DATA_DIR, f"{base_name}__{timestamp}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        logger.log_file_processing(filename, "parse_and_save", True)
        print(f"✅ Parsed and saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to parse and save {pdf_path}: {str(e)}", exc_info=True)
        print(f"❌ Failed to parse: {pdf_path}")
        return False

def infer_source_from_filename(name: str):
    if "transcript" in name.lower():
        return "Conference Call Transcript"
    elif "ppt" in name.lower() or "presentation" in name.lower():
        return "Investor Presentation"
    else:
        return "Unknown Document Type"

def parse_all_pdfs():
    try:
        pdf_files = [f for f in os.listdir(Config.RAW_DATA_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            logger.warning("No PDF files found in data/raw/")
            print("⚠️ No PDF files found in data/raw/")
            return

        logger.info(f"Found {len(pdf_files)} PDF(s) in {Config.RAW_DATA_DIR}")
        print(f"🔍 Found {len(pdf_files)} PDF(s) in {Config.RAW_DATA_DIR}")
        
        success_count = 0
        for pdf in tqdm(pdf_files, desc="Parsing PDFs"):
            full_path = os.path.join(Config.RAW_DATA_DIR, pdf)
            if parse_and_save(full_path):
                success_count += 1
        
        logger.info(f"Successfully parsed {success_count}/{len(pdf_files)} PDFs")
        print(f"✅ Successfully parsed {success_count}/{len(pdf_files)} PDFs")
        
    except Exception as e:
        logger.error(f"Failed to parse PDFs: {str(e)}", exc_info=True)
        print(f"❌ Failed to parse PDFs: {str(e)}")

if __name__ == "__main__":
    parse_all_pdfs()
