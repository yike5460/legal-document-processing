"""
Data Validator Utility
Validates input documents and extracted data
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import magic
import json

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates documents and extracted data
    Ensures data quality throughout the pipeline
    """
    
    SUPPORTED_FORMATS = {
        'application/pdf': '.pdf',
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/tiff': '.tiff',
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx'
    }
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self):
        """Initialize data validator"""
        logger.info("Data validator initialized")
    
    def validate_document(self, document_path: str) -> bool:
        """
        Validate input document
        
        Args:
            document_path: Path to document
            
        Returns:
            True if valid, False otherwise
        """
        
        try:
            path = Path(document_path)
            
            # Check file exists
            if not path.exists():
                logger.error(f"Document not found: {document_path}")
                return False
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                logger.error(f"File too large: {file_size} bytes")
                return False
            
            if file_size == 0:
                logger.error("File is empty")
                return False
            
            # Check file type
            try:
                mime = magic.from_file(str(path), mime=True)
                if mime not in self.SUPPORTED_FORMATS:
                    logger.error(f"Unsupported file type: {mime}")
                    return False
            except:
                # Fallback to extension check
                if path.suffix.lower() not in ['.pdf', '.jpg', '.png', '.doc', '.docx']:
                    logger.error(f"Unsupported file extension: {path.suffix}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Document validation error: {e}")
            return False
    
    def validate_extraction(self, extraction: Dict) -> Dict:
        """
        Validate extraction results
        
        Args:
            extraction: Extracted data
            
        Returns:
            Validation results
        """
        
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields - only entities and deadlines are truly required
        required_fields = ['entities', 'deadlines']
        for field in required_fields:
            if field not in extraction:
                results["valid"] = False
                results["errors"].append(f"Missing required field: {field}")
        
        # Validate entities
        if 'entities' in extraction:
            entity_validation = self._validate_entities(extraction['entities'])
            results["errors"].extend(entity_validation.get("errors", []))
            results["warnings"].extend(entity_validation.get("warnings", []))
        
        # Validate deadlines
        if 'deadlines' in extraction:
            deadline_validation = self._validate_deadlines(extraction['deadlines'])
            results["errors"].extend(deadline_validation.get("errors", []))
            results["warnings"].extend(deadline_validation.get("warnings", []))
        
        # Check for empty extraction
        if extraction.get('deadlines') == [] and not extraction.get('entities'):
            results["warnings"].append("No data extracted from document")
        
        return results
    
    def _validate_entities(self, entities: Dict) -> Dict:
        """Validate entity extraction"""
        
        results = {"errors": [], "warnings": []}
        
        # Check for empty entities
        if not entities or all(not v for v in entities.values()):
            results["warnings"].append("No entities extracted")
        
        # Validate case numbers format
        case_numbers = entities.get('case_numbers', [])
        for case_num in case_numbers:
            if not self._is_valid_case_number(case_num):
                results["warnings"].append(f"Unusual case number format: {case_num}")
        
        return results
    
    def _validate_deadlines(self, deadlines: List[Dict]) -> Dict:
        """Validate deadline extraction"""
        
        results = {"errors": [], "warnings": []}
        
        for idx, deadline in enumerate(deadlines):
            # Check required deadline fields
            if not deadline.get('action'):
                results["errors"].append(f"Deadline {idx} missing action")
            
            if not deadline.get('time_period') and not deadline.get('calculated_date'):
                results["warnings"].append(f"Deadline {idx} missing time period")
            
            # Validate confidence score
            confidence = deadline.get('confidence', 0)
            if confidence < 0 or confidence > 1:
                results["errors"].append(f"Invalid confidence score: {confidence}")
            
            # Check for ambiguity
            if deadline.get('ambiguity_flag'):
                results["warnings"].append(f"Deadline {idx} has ambiguity")
        
        return results
    
    def _is_valid_case_number(self, case_number: str) -> bool:
        """Check if case number follows common patterns"""
        
        import re
        
        # Common case number patterns
        patterns = [
            r'^\d{2,4}-[A-Z]{2,4}-\d{4,6}$',  # Federal format
            r'^\d{4,8}$',                       # Simple numeric
            r'^[A-Z]{2,4}\d{4,8}$',            # State format
        ]
        
        for pattern in patterns:
            if re.match(pattern, case_number):
                return True
        
        return False
    
    def validate_api_request(self, request: Dict) -> Dict:
        """
        Validate API request data
        
        Args:
            request: API request data
            
        Returns:
            Validation results
        """
        
        results = {
            "valid": True,
            "errors": []
        }
        
        # Check required fields
        if not request.get('document_url') and not request.get('document_content'):
            results["valid"] = False
            results["errors"].append("Either document_url or document_content required")
        
        # Validate document URL
        if request.get('document_url'):
            url = request['document_url']
            if not url.startswith(('s3://', 'http://', 'https://', '/')):
                results["valid"] = False
                results["errors"].append("Invalid document URL format")
        
        # Validate priority
        if request.get('priority'):
            if request['priority'] not in ['low', 'normal', 'high', 'critical']:
                results["errors"].append("Invalid priority value")
        
        return results