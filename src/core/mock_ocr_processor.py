"""
Mock OCR Processor for Testing
Simulates OCR functionality without requiring DOTS.OCR model
"""

import logging
from typing import Dict, List
from pathlib import Path
from PIL import Image
import pdf2image

logger = logging.getLogger(__name__)

class MockOCRProcessor:
    """
    Mock OCR processor for testing when DOTS.OCR is not available
    Simulates document processing with sample data
    """
    
    def __init__(self, model_name: str = "mock"):
        """Initialize mock OCR processor"""
        self.model_name = model_name
        self.device = "cpu"
        logger.info("Mock OCR processor initialized for testing")
    
    async def process(self, document_path: str) -> Dict:
        """
        Simulate document processing
        
        Args:
            document_path: Path to document
            
        Returns:
            Mock extracted data
        """
        
        # Simulate different responses based on document name
        path = Path(document_path)
        
        if "court_order" in path.name.lower():
            return self._mock_court_order()
        elif "motion" in path.name.lower():
            return self._mock_motion()
        elif "discovery" in path.name.lower():
            return self._mock_discovery()
        else:
            return self._mock_generic_document()
    
    def _mock_court_order(self) -> Dict:
        """Mock court order extraction in DOTS.OCR format"""
        
        # Simulated DOTS.OCR layout elements response for 2 pages
        page1_elements = [
            {
                "bbox": [72, 72, 540, 108],
                "category": "Page-header",
                "text": "UNITED STATES DISTRICT COURT\nEASTERN DISTRICT OF NEW YORK"
            },
            {
                "bbox": [72, 144, 540, 162],
                "category": "Text",
                "text": "Case No. 21-CV-12345"
            },
            {
                "bbox": [72, 198, 540, 270],
                "category": "Text",
                "text": "John Doe,\n    Plaintiff,\nv.\nABC Corporation,\n    Defendant."
            },
            {
                "bbox": [240, 306, 372, 324],
                "category": "Title",
                "text": "ORDER"
            },
            {
                "bbox": [72, 360, 540, 396],
                "category": "Text",
                "text": "Upon consideration of Defendant's Motion to Dismiss, it is hereby ORDERED that:"
            },
            {
                "bbox": [72, 432, 540, 540],
                "category": "List-item",
                "text": "1. Plaintiff shall file a response to the motion within 14 days of this order.\n2. Defendant may file a reply within 7 days of Plaintiff's response.\n3. A hearing on this matter is scheduled for 30 days from the date of this order."
            }
        ]
        
        page2_elements = [
            {
                "bbox": [72, 72, 540, 180],
                "category": "Table",
                "text": "<table><tr><th>Party</th><th>Deadline</th><th>Action Required</th></tr><tr><td>Plaintiff</td><td>14 days</td><td>File Response</td></tr><tr><td>Defendant</td><td>7 days after response</td><td>File Reply</td></tr></table>"
            },
            {
                "bbox": [72, 216, 180, 234],
                "category": "Text",
                "text": "IT IS SO ORDERED."
            },
            {
                "bbox": [72, 270, 200, 288],
                "category": "Text",
                "text": "Dated: August 16, 2025"
            },
            {
                "bbox": [72, 324, 270, 360],
                "category": "Page-footer",
                "text": "/s/ Judge Smith\nUnited States District Judge"
            }
        ]
        
        # Combine all elements
        all_elements = page1_elements + page2_elements
        
        # Extract text from all elements
        page1_text = "\n\n".join([elem["text"] for elem in page1_elements])
        page2_text = "\n\n".join([elem["text"] for elem in page2_elements])
        full_text = page1_text + "\n\n--- Page Break ---\n\n" + page2_text
        
        # Extract tables
        tables = []
        for elem in all_elements:
            if elem["category"] == "Table":
                tables.append({
                    "bbox": elem["bbox"],
                    "html": elem["text"],
                    "page": 2 if elem in page2_elements else 1
                })
        
        return {
            "text": full_text,
            "layout": {
                "pages": 2,
                "structure": all_elements
            },
            "tables": tables,
            "reading_order": [elem["text"] for elem in all_elements],
            "confidence": 0.95,
            "pages": [
                {"page": 1, "confidence": 0.95, "table_count": 0},
                {"page": 2, "confidence": 0.95, "table_count": 1}
            ]
        }
    
    def _mock_motion(self) -> Dict:
        """Mock motion document extraction in DOTS.OCR format"""
        
        layout_elements = [
            {
                "bbox": [180, 72, 432, 108],
                "category": "Title",
                "text": "MOTION TO DISMISS"
            },
            {
                "bbox": [72, 144, 540, 216],
                "category": "Text",
                "text": "Defendant ABC Corporation hereby moves to dismiss the complaint pursuant to Rule 12(b)(6)."
            },
            {
                "bbox": [72, 252, 540, 288],
                "category": "Text",
                "text": "The response deadline is 21 days from service of this motion."
            }
        ]
        
        full_text = "\n\n".join([elem["text"] for elem in layout_elements])
        
        return {
            "text": full_text,
            "layout": {"pages": 1, "structure": layout_elements},
            "tables": [],
            "reading_order": [elem["text"] for elem in layout_elements],
            "confidence": 0.92,
            "pages": [{"page": 1, "confidence": 0.92, "table_count": 0}]
        }
    
    def _mock_discovery(self) -> Dict:
        """Mock discovery document extraction in DOTS.OCR format"""
        
        layout_elements = [
            {
                "bbox": [180, 72, 432, 108],
                "category": "Title",
                "text": "DISCOVERY REQUEST"
            },
            {
                "bbox": [72, 144, 540, 180],
                "category": "Text",
                "text": "Plaintiff requests the following discovery:"
            },
            {
                "bbox": [72, 216, 540, 288],
                "category": "List-item",
                "text": "1. All documents related to the contract\n2. Communications between parties"
            },
            {
                "bbox": [72, 324, 540, 360],
                "category": "Text",
                "text": "Response due within 30 days of service."
            }
        ]
        
        full_text = "\n\n".join([elem["text"] for elem in layout_elements])
        
        return {
            "text": full_text,
            "layout": {"pages": 1, "structure": layout_elements},
            "tables": [],
            "reading_order": [elem["text"] for elem in layout_elements],
            "confidence": 0.90,
            "pages": [{"page": 1, "confidence": 0.90, "table_count": 0}]
        }
    
    def _mock_generic_document(self) -> Dict:
        """Mock generic legal document extraction in DOTS.OCR format"""
        
        layout_elements = [
            {
                "bbox": [72, 72, 540, 108],
                "category": "Text",
                "text": "Sample legal document text for testing purposes."
            }
        ]
        
        full_text = "\n\n".join([elem["text"] for elem in layout_elements])
        
        return {
            "text": full_text,
            "layout": {"pages": 1, "structure": layout_elements},
            "tables": [],
            "reading_order": [elem["text"] for elem in layout_elements],
            "confidence": 0.85,
            "pages": [{"page": 1, "confidence": 0.85, "table_count": 0}]
        }
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Mock OCR processor cleanup completed")