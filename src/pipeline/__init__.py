"""
Pipeline orchestration for legal document processing
"""

from .main_pipeline import LegalDocumentPipeline, PipelineConfig, ProcessingResult

__all__ = [
    'LegalDocumentPipeline',
    'PipelineConfig',
    'ProcessingResult'
]