import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from config import Config
from utils.logger import logger
from utils.validators import FileValidator
from utils.structured_output import response_parser, output_formatter, ResponseType, StructuredResponse


@dataclass
class BulkQuestion:
    """Represents a single question from the Excel file."""
    question: str
    question_type: str
    figure_type: str
    period: str
    figure_value: str
    row_number: int


@dataclass
class BulkQuestionResult:
    """Represents the result of processing a bulk question."""
    question: str
    question_type: str
    figure_type: str
    period: str
    figure_value: str
    row_number: int
    answer: str
    structured_response: Optional[StructuredResponse]
    processing_time: float
    status: str  # "success", "error", "no_answer"
    error_message: Optional[str] = None


@dataclass
class BulkProcessingResult:
    """Represents the overall result of bulk question processing."""
    total_questions: int
    successful_questions: int
    failed_questions: int
    total_processing_time: float
    results: List[BulkQuestionResult]
    summary: Dict[str, Any]
    timestamp: str


class BulkQuestionProcessor:
    """Processes bulk questions from Excel files."""
    
    def __init__(self):
        self.expected_columns = [
            'Question', 'Type', 'Figure in cr', 'Period', 'Figure'
        ]
        self.column_mapping = {
            'Question': 'question',
            'Type': 'question_type', 
            'Figure in cr': 'figure_type',
            'Period': 'period',
            'Figure': 'figure_value'
        }
    
    def validate_excel_file(self, file_bytes: bytes, filename: str) -> tuple[bool, str]:
        """Validate the uploaded Excel file."""
        try:
            # Check file extension
            if not filename.lower().endswith(('.xlsx', '.xls')):
                return False, "File must be an Excel file (.xlsx or .xls)"
            
            # Check file size
            if len(file_bytes) > Config.MAX_FILE_SIZE:
                return False, f"File size exceeds maximum allowed size ({Config.MAX_FILE_SIZE / (1024*1024):.1f}MB)"
            
            # Try to read the Excel file
            df = pd.read_excel(file_bytes)
            
            # Check if required columns exist
            missing_columns = []
            for col in self.expected_columns:
                if col not in df.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"
            
            # Check if file has data
            if df.empty:
                return False, "Excel file is empty"
            
            # Check for required data in Question column
            if df['Question'].isna().all():
                return False, "No questions found in the 'Question' column"
            
            return True, "File validation successful"
            
        except Exception as e:
            return False, f"Error validating Excel file: {str(e)}"
    
    def parse_excel_questions(self, file_bytes: bytes) -> List[BulkQuestion]:
        """Parse questions from Excel file."""
        try:
            df = pd.read_excel(file_bytes)
            
            questions = []
            for index, row in df.iterrows():
                # Skip rows with empty questions
                if pd.isna(row['Question']) or str(row['Question']).strip() == '':
                    continue
                
                question = BulkQuestion(
                    question=str(row['Question']).strip(),
                    question_type=str(row['Type']).strip() if not pd.isna(row['Type']) else '',
                    figure_type=str(row['Figure in cr']).strip() if not pd.isna(row['Figure in cr']) else '',
                    period=str(row['Period']).strip() if not pd.isna(row['Period']) else '',
                    figure_value=str(row['Figure']).strip() if not pd.isna(row['Figure']) else '',
                    row_number=index + 2  # Excel rows are 1-indexed, +1 for header
                )
                questions.append(question)
            
            logger.info(f"Parsed {len(questions)} questions from Excel file")
            return questions
            
        except Exception as e:
            logger.error(f"Failed to parse Excel questions: {str(e)}", exc_info=True)
            raise
    
    def process_bulk_questions(
        self, 
        questions: List[BulkQuestion], 
        query_function,
        **query_kwargs
    ) -> BulkProcessingResult:
        """Process a list of questions using the provided query function."""
        start_time = time.time()
        results = []
        successful = 0
        failed = 0
        
        logger.info(f"Starting bulk processing of {len(questions)} questions")
        
        for i, question in enumerate(questions, 1):
            question_start_time = time.time()
            
            try:
                logger.info(f"Processing question {i}/{len(questions)}: {question.question[:50]}...")
                
                # Call the query function
                query_result = query_function(question.question, **query_kwargs)
                
                # Parse the response into structured format
                structured_response = None
                if hasattr(query_result, 'answer') and query_result.answer:
                    try:
                        structured_response = response_parser.parse_response(
                            question.question, 
                            query_result.answer
                        )
                    except Exception as parse_error:
                        logger.warning(f"Failed to parse response for question {i}: {parse_error}")
                
                processing_time = time.time() - question_start_time
                
                result = BulkQuestionResult(
                    question=question.question,
                    question_type=question.question_type,
                    figure_type=question.figure_type,
                    period=question.period,
                    figure_value=question.figure_value,
                    row_number=question.row_number,
                    answer=query_result.answer if hasattr(query_result, 'answer') else str(query_result),
                    structured_response=structured_response,
                    processing_time=processing_time,
                    status="success"
                )
                
                results.append(result)
                successful += 1
                
            except Exception as e:
                processing_time = time.time() - question_start_time
                error_msg = f"Failed to process question: {str(e)}"
                logger.error(f"Error processing question {i}: {error_msg}")
                
                result = BulkQuestionResult(
                    question=question.question,
                    question_type=question.question_type,
                    figure_type=question.figure_type,
                    period=question.period,
                    figure_value=question.figure_value,
                    row_number=question.row_number,
                    answer="",
                    structured_response=None,
                    processing_time=processing_time,
                    status="error",
                    error_message=error_msg
                )
                
                results.append(result)
                failed += 1
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(results, successful, failed, total_time)
        
        bulk_result = BulkProcessingResult(
            total_questions=len(questions),
            successful_questions=successful,
            failed_questions=failed,
            total_processing_time=total_time,
            results=results,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Bulk processing completed: {successful} successful, {failed} failed in {total_time:.2f}s")
        return bulk_result
    
    def _generate_summary(self, results: List[BulkQuestionResult], successful: int, failed: int, total_time: float) -> Dict[str, Any]:
        """Generate a summary of the bulk processing results."""
        # Extract metrics from successful responses
        metrics_summary = {}
        response_types = {}
        
        for result in results:
            if result.status == "success" and result.structured_response:
                # Count response types
                response_type = result.structured_response.response_type.value
                response_types[response_type] = response_types.get(response_type, 0) + 1
                
                # Extract metrics
                for metric in result.structured_response.metrics:
                    metric_name = metric.metric_name
                    if metric_name not in metrics_summary:
                        metrics_summary[metric_name] = {
                            'count': 0,
                            'values': [],
                            'units': set()
                        }
                    
                    metrics_summary[metric_name]['count'] += 1
                    if metric.value:
                        metrics_summary[metric_name]['values'].append(metric.value)
                    if metric.unit:
                        metrics_summary[metric_name]['units'].add(metric.unit)
        
        # Convert sets to lists for JSON serialization
        for metric_name in metrics_summary:
            metrics_summary[metric_name]['units'] = list(metrics_summary[metric_name]['units'])
        
        return {
            'success_rate': (successful / len(results)) * 100 if results else 0,
            'average_processing_time': total_time / len(results) if results else 0,
            'response_types': response_types,
            'metrics_extracted': metrics_summary,
            'total_metrics_found': sum(len(r.structured_response.metrics) for r in results if r.structured_response)
        }
    
    def export_results(self, bulk_result: BulkProcessingResult, output_format: str = "excel") -> bytes:
        """Export bulk processing results to various formats."""
        try:
            if output_format.lower() == "excel":
                return self._export_to_excel(bulk_result)
            elif output_format.lower() == "json":
                return self._export_to_json(bulk_result)
            elif output_format.lower() == "csv":
                return self._export_to_csv(bulk_result)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}", exc_info=True)
            raise
    
    def _export_to_excel(self, bulk_result: BulkProcessingResult) -> bytes:
        """Export results to Excel format."""
        # Create DataFrame for main results
        results_data = []
        for result in bulk_result.results:
            row = {
                'Row Number': result.row_number,
                'Question': result.question,
                'Question Type': result.question_type,
                'Figure Type': result.figure_type,
                'Period': result.period,
                'Expected Figure': result.figure_value,
                'Answer': result.answer,
                'Status': result.status,
                'Processing Time (s)': round(result.processing_time, 2),
                'Error Message': result.error_message or ''
            }
            
            # Add structured response data if available
            if result.structured_response:
                row.update({
                    'Response Type': result.structured_response.response_type.value,
                    'Confidence': result.structured_response.confidence,
                    'Metrics Count': len(result.structured_response.metrics),
                    'Sources Count': len(result.structured_response.sources)
                })
            else:
                row.update({
                    'Response Type': '',
                    'Confidence': '',
                    'Metrics Count': 0,
                    'Sources Count': 0
                })
            
            results_data.append(row)
        
        df_results = pd.DataFrame(results_data)
        
        # Create DataFrame for summary
        summary_data = {
            'Metric': [
                'Total Questions',
                'Successful Questions', 
                'Failed Questions',
                'Success Rate (%)',
                'Total Processing Time (s)',
                'Average Processing Time (s)',
                'Total Metrics Extracted'
            ],
            'Value': [
                bulk_result.total_questions,
                bulk_result.successful_questions,
                bulk_result.failed_questions,
                round(bulk_result.summary['success_rate'], 2),
                round(bulk_result.total_processing_time, 2),
                round(bulk_result.summary['average_processing_time'], 2),
                bulk_result.summary['total_metrics_found']
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        
        # Create Excel writer
        from io import BytesIO
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='Results', index=False)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Add response types summary
            if bulk_result.summary['response_types']:
                df_response_types = pd.DataFrame([
                    {'Response Type': rt, 'Count': count}
                    for rt, count in bulk_result.summary['response_types'].items()
                ])
                df_response_types.to_excel(writer, sheet_name='Response Types', index=False)
        
        output.seek(0)
        return output.getvalue()
    
    def _export_to_json(self, bulk_result: BulkProcessingResult) -> bytes:
        """Export results to JSON format."""
        # Convert dataclasses to dictionaries
        result_dict = asdict(bulk_result)
        
        # Convert to JSON
        json_str = json.dumps(result_dict, indent=2, ensure_ascii=False, default=str)
        return json_str.encode('utf-8')
    
    def _export_to_csv(self, bulk_result: BulkProcessingResult) -> bytes:
        """Export results to CSV format."""
        # Create DataFrame for main results
        results_data = []
        for result in bulk_result.results:
            row = {
                'Row Number': result.row_number,
                'Question': result.question,
                'Question Type': result.question_type,
                'Figure Type': result.figure_type,
                'Period': result.period,
                'Expected Figure': result.figure_value,
                'Answer': result.answer,
                'Status': result.status,
                'Processing Time (s)': round(result.processing_time, 2),
                'Error Message': result.error_message or '',
                'Response Type': result.structured_response.response_type.value if result.structured_response else '',
                'Confidence': result.structured_response.confidence if result.structured_response else '',
                'Metrics Count': len(result.structured_response.metrics) if result.structured_response else 0
            }
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        
        # Convert to CSV
        csv_str = df.to_csv(index=False)
        return csv_str.encode('utf-8')


# Global instance
bulk_processor = BulkQuestionProcessor() 