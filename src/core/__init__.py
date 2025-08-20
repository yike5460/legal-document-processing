"""
Core processing modules for legal document system
"""

from .ocr_processor import DOTSOCRProcessor
from .ai_extractor import ClaudeExtractor
from .rag_engine import RAGEngine
from .confidence_scorer import ConfidenceScorer
from .deadline_calculator import DeadlineCalculator

__all__ = [
    'DOTSOCRProcessor',
    'ClaudeExtractor',
    'RAGEngine',
    'ConfidenceScorer',
    'DeadlineCalculator'
]