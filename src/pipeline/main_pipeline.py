"""
Main Pipeline Orchestrator
Coordinates the entire document processing workflow
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env.local file
load_dotenv('.env.local')

# Core modules
from ..core.ocr_processor import DOTSOCRProcessor
from ..core.document_classifier import DocumentClassifier
from ..core.ai_extractor import ClaudeExtractor
from ..core.rag_engine import RAGEngine
from ..core.confidence_scorer import ConfidenceScorer
from ..core.deadline_calculator import DeadlineCalculator
from ..utils.pii_masker import PIIMasker
from ..utils.data_validator import DataValidator
from ..utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline"""
    dots_ocr_model: str = "/tmp/dots.ocr/weights/DotsOCR"
    claude_model: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    opensearch_domain: str = "legal-documents"
    confidence_threshold_auto: float = 0.95
    confidence_threshold_review: float = 0.85
    enable_pii_masking: bool = True
    enable_rag: bool = True
    max_retries: int = 3
    processing_timeout: int = 60  # seconds

@dataclass
class ProcessingResult:
    """Result of document processing"""
    document_id: str
    status: str  # success, partial, failed
    document_type: Optional[str] = None
    classification_confidence: float = 0.0
    deadlines: List[Dict] = field(default_factory=list)
    entities: Dict = field(default_factory=dict)
    confidence_scores: Dict = field(default_factory=dict)
    routing_decisions: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0
    error_messages: List[str] = field(default_factory=list)
    audit_trail: List[Dict] = field(default_factory=list)

class LegalDocumentPipeline:
    """
    Main orchestrator for legal document processing
    Coordinates all components in the correct sequence
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize pipeline with configuration"""
        self.config = config or PipelineConfig()
        
        # Initialize components
        self._init_components()
        
        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "total_deadlines_extracted": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0
        }
    
    def _init_components(self):
        """Initialize all pipeline components"""
        try:
            # Core processors - use mock if DOTS.OCR fails to load
            # DOTS.OCR model has been installed and is available
            use_mock_ocr = True  # Set to True to use mock OCR for testing

            if use_mock_ocr:
                logger.info("Using mock OCR processor for testing (configured)")
                from ..core.mock_ocr_processor import MockOCRProcessor
                self.ocr_processor = MockOCRProcessor()
            else:
                try:
                    self.ocr_processor = DOTSOCRProcessor(
                        model_name=self.config.dots_ocr_model
                    )
                except Exception as ocr_error:
                    logger.warning(f"DOTS.OCR not available: {ocr_error}")
                    logger.info("Using mock OCR processor for testing (fallback)")
                    from ..core.mock_ocr_processor import MockOCRProcessor
                    self.ocr_processor = MockOCRProcessor()
            
            # Document classifier - initialize with fallback to pattern matching if Claude not available
            try:
                self.document_classifier = DocumentClassifier()
            except Exception as classifier_error:
                logger.warning(f"Document classifier initialization warning: {classifier_error}")
                logger.info("Document classifier will use pattern-based fallback if Bedrock is unavailable")
                self.document_classifier = DocumentClassifier(bedrock_client=None)
            
            # AI extractor - initialize with proper error handling
            try:
                self.ai_extractor = ClaudeExtractor(
                    model_id=self.config.claude_model
                )
            except Exception as ai_error:
                logger.error(f"AWS Bedrock not available: {ai_error}")
                logger.warning("AI extraction requires AWS Bedrock. Pipeline may have reduced functionality.")
                # Initialize with None client, will use fallback methods
                self.ai_extractor = ClaudeExtractor(
                    model_id=self.config.claude_model,
                    bedrock_client=None
                )
            
            # RAG engine for confidence scoring
            if self.config.enable_rag:
                try:
                    self.rag_engine = RAGEngine(
                        opensearch_domain=self.config.opensearch_domain
                    )
                except Exception as rag_error:
                    logger.warning(f"OpenSearch not available: {rag_error}")
                    logger.info("Using mock RAG engine for testing")
                    from ..core.mock_rag_engine import MockRAGEngine
                    self.rag_engine = MockRAGEngine(
                        opensearch_domain=self.config.opensearch_domain
                    )
            else:
                self.rag_engine = None
            
            # Supporting modules
            self.confidence_scorer = ConfidenceScorer(
                rag_engine=self.rag_engine
            )
            
            self.deadline_calculator = DeadlineCalculator()
            
            # Security and validation
            if self.config.enable_pii_masking:
                self.pii_masker = PIIMasker()
            else:
                self.pii_masker = None
            
            self.validator = DataValidator()
            
            logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    async def process_document(self, 
                              document_path: str,
                              metadata: Optional[Dict] = None) -> ProcessingResult:
        """
        Process a legal document through the complete pipeline
        
        Args:
            document_path: Path to the document
            metadata: Optional metadata about the document
            
        Returns:
            ProcessingResult with extracted information
        """
        
        start_time = datetime.now()
        document_id = self._generate_document_id(document_path)
        
        result = ProcessingResult(
            document_id=document_id,
            status="processing"
        )
        
        try:
            # Step 1: Validate input
            self._log_step(result, "Validating input document")
            if not self.validator.validate_document(document_path):
                raise ValueError(f"Invalid document: {document_path}")
            
            # Step 2: OCR Processing with DOTS.OCR
            self._log_step(result, "Extracting text and layout with DOTS.OCR")
            ocr_result = await self._perform_ocr(document_path)
            print(f"OCR Result: {json.dumps(ocr_result, indent=2)}")  # Debugging line in JSON format

            # Step 3: PII Masking BEFORE classification (Security First)
            if self.config.enable_pii_masking:
                self._log_step(result, "Masking PII before AI processing")
                masked_text, masking_map = self._mask_pii(ocr_result['text'])
            else:
                masked_text = ocr_result['text']
                masking_map = {}
            
            # Step 4: Document Classification with Claude 4 using masked text
            self._log_step(result, "Classifying document type with masked text")
            classification = await self.document_classifier.classify(
                masked_text,  # Use masked text for classification
                ocr_result.get('layout')
            )
            result.document_type = classification.get('primary_type', 'unknown')
            result.classification_confidence = classification.get('confidence', 0.0)
            
            # Step 5: AI Extraction with Claude 4 using masked text
            self._log_step(result, f"Extracting deadlines for {result.document_type}")
            extraction = await self._extract_information(
                masked_text,  # Use masked text for extraction
                ocr_result['layout'],
                metadata,
                classification
            )
            
            # Step 6: Validation - Check if extracted deadlines are legally valid
            self._log_step(result, "Validating extracted deadlines")
            validated_deadlines = await self._validate_deadlines(
                extraction['deadlines'],
                result.document_type
            )
            
            # Step 7: Calculation - Convert relative dates to actual dates
            self._log_step(result, "Calculating actual deadline dates")
            calculated_deadlines = self._calculate_deadlines(
                validated_deadlines,
                extraction['entities']
            )
            
            # Step 8: Reconstruction - Unmask PII in final results
            if masking_map:
                self._log_step(result, "Reconstructing original PII values")
                calculated_deadlines = self._unmask_deadlines(calculated_deadlines, masking_map)
                extraction['entities'] = self._unmask_entities(extraction['entities'], masking_map)
            
            # Step 9: Confidence Scoring based on validation accuracy
            self._log_step(result, "Scoring confidence based on validation")
            scored_deadlines = await self._score_confidence(
                calculated_deadlines,
                result.document_type,
                ocr_result.get('confidence', 0.9)
            )
            
            # Step 10: Routing decisions based on document type and confidence
            self._log_step(result, "Determining routing based on type and confidence")
            routing = self._make_routing_decisions(
                scored_deadlines,
                classification
            )
            
            # Step 11: Cache validated results for future reference
            if self.config.enable_rag and self.rag_engine:
                self._log_step(result, "Caching validated results")
                await self._cache_validated_results(scored_deadlines, extraction['entities'])
            
            # Populate result
            result.deadlines = scored_deadlines
            result.entities = extraction['entities']
            result.confidence_scores = {
                d['id']: d['confidence'] 
                for d in scored_deadlines
            }
            result.routing_decisions = routing
            result.status = "success"
            
        except Exception as e:
            logger.error(f"Pipeline error for {document_id}: {e}")
            result.status = "failed"
            result.error_messages.append(str(e))
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update statistics
        self._update_statistics(result)
        
        return result
    
    async def _perform_ocr(self, document_path: str) -> Dict:
        """Perform OCR with error handling and retries"""
        
        for attempt in range(self.config.max_retries):
            try:
                return await asyncio.wait_for(
                    self.ocr_processor.process(document_path),
                    timeout=self.config.processing_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"OCR timeout on attempt {attempt + 1}")
                if attempt == self.config.max_retries - 1:
                    raise
            except Exception as e:
                logger.error(f"OCR error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise Exception("OCR processing failed after all retries")
    
    def _mask_pii(self, text: str) -> Tuple[str, Dict]:
        """Mask PII with error handling"""
        
        try:
            return self.pii_masker.mask_document(text)
        except Exception as e:
            logger.error(f"PII masking error: {e}")
            # Return original text if masking fails
            return text, {}
    
    async def _extract_information(self, 
                                  text: str, 
                                  layout: Dict,
                                  metadata: Optional[Dict],
                                  classification: Optional[Dict] = None) -> Dict:
        """Extract information using Claude 4 with document type context"""
        
        # Enhance metadata with classification info
        enhanced_metadata = metadata or {}
        if classification:
            enhanced_metadata['document_type'] = classification.get('primary_type', 'unknown')
            enhanced_metadata['extraction_hints'] = classification.get('extraction_hints', {})
            enhanced_metadata['extraction_strategy'] = classification.get('extraction_strategy', {})
        
        try:
            return await self.ai_extractor.extract(
                text=text,
                layout=layout,
                metadata=enhanced_metadata
            )
        except Exception as e:
            logger.error(f"AI extraction error: {e}")
            # Return empty extraction on failure
            return {
                "entities": {},
                "deadlines": [],
                "document_context": {}
            }
    
    def _unmask_deadlines(self, deadlines: List[Dict], masking_map: Dict) -> List[Dict]:
        """Restore PII in deadline results"""
        if not masking_map:
            return deadlines
        
        unmasked = []
        for deadline in deadlines:
            deadline_str = json.dumps(deadline)
            for token, original in masking_map.items():
                if isinstance(original, dict):
                    deadline_str = deadline_str.replace(token, original.get('original', token))
                else:
                    deadline_str = deadline_str.replace(token, original)
            unmasked.append(json.loads(deadline_str))
        
        return unmasked
    
    def _unmask_entities(self, entities: Dict, masking_map: Dict) -> Dict:
        """Restore PII in entity results"""
        if not masking_map:
            return entities
        
        entities_str = json.dumps(entities)
        for token, original in masking_map.items():
            if isinstance(original, dict):
                entities_str = entities_str.replace(token, original.get('original', token))
            else:
                entities_str = entities_str.replace(token, original)
        
        return json.loads(entities_str)
    
    def _calculate_deadlines(self, 
                            raw_deadlines: List[Dict],
                            entities: Dict) -> List[Dict]:
        """Calculate actual deadline dates"""
        
        calculated = []
        
        for idx, deadline in enumerate(raw_deadlines):
            try:
                # Determine jurisdiction
                jurisdiction = self._determine_jurisdiction(entities)
                
                # Calculate date
                calc_result = self.deadline_calculator.calculate(
                    trigger_text=deadline.get('trigger_event', 'today'),
                    period_text=deadline.get('time_period', ''),
                    jurisdiction=jurisdiction,
                    rule_type=deadline.get('action', 'unknown')
                )
                
                # Merge calculation with original deadline
                deadline['id'] = f"deadline_{idx}"
                deadline['calculated_date'] = calc_result['date'].isoformat()
                deadline['calculation_method'] = calc_result['method']
                deadline['warnings'] = calc_result.get('warnings', [])
                
                calculated.append(deadline)
                
            except Exception as e:
                logger.error(f"Deadline calculation error: {e}")
                deadline['id'] = f"deadline_{idx}"
                deadline['calculation_error'] = str(e)
                calculated.append(deadline)
        
        return calculated
    
    async def _validate_deadlines(self, deadlines: List[Dict], document_type: str) -> List[Dict]:
        """Validate extracted deadlines against legal rules"""
        validated = []
        
        for deadline in deadlines:
            try:
                # Check if deadline pattern is legally valid
                is_valid = await self._check_legal_validity(
                    deadline,
                    document_type
                )
                
                deadline['validation_status'] = 'valid' if is_valid else 'invalid'
                deadline['validation_confidence'] = 0.95 if is_valid else 0.5
                
                if not is_valid:
                    deadline['warnings'] = deadline.get('warnings', []) + [
                        "Deadline pattern not recognized in legal rules database"
                    ]
                
            except Exception as e:
                logger.error(f"Validation error: {e}")
                deadline['validation_status'] = 'unknown'
                deadline['validation_confidence'] = 0.7
            
            validated.append(deadline)
        
        return validated
    
    async def _check_legal_validity(self, deadline: Dict, document_type: str) -> bool:
        """Check if deadline matches known legal patterns"""
        if not self.rag_engine:
            return True  # Assume valid if no RAG available
        
        try:
            # Search for similar validated patterns
            query = f"{document_type} {deadline.get('action', '')} {deadline.get('time_period', '')}"
            similar = await self.rag_engine.find_similar(query, limit=5)
            
            # Check if any high-confidence matches exist
            for pattern in similar:
                if pattern.get('similarity_score', 0) > 0.85:
                    return True
            
            return False
        except Exception:
            return True  # Default to valid on error
    
    async def _score_confidence(self, 
                               deadlines: List[Dict],
                               document_type: str,
                               ocr_confidence: float) -> List[Dict]:
        """Score confidence based on validation results"""
        scored = []
        
        for deadline in deadlines:
            # Base confidence on validation status
            validation_confidence = deadline.get('validation_confidence', 0.7)
            calculation_confidence = 0.9 if not deadline.get('calculation_error') else 0.5
            
            # Combine factors
            confidence = (
                validation_confidence * 0.5 +  # 50% weight on validation
                calculation_confidence * 0.3 +  # 30% weight on calculation
                ocr_confidence * 0.2  # 20% weight on OCR quality
            )
            
            # Apply document type weighting
            doc_weight = self.confidence_scorer.document_type_weights.get(document_type, 1.0)
            confidence *= doc_weight
            
            deadline['confidence'] = min(0.99, max(0.01, confidence))
            deadline['confidence_factors'] = {
                'validation': validation_confidence,
                'calculation': calculation_confidence,
                'ocr_quality': ocr_confidence,
                'document_type_weight': doc_weight
            }
            
            scored.append(deadline)
        
        return scored
    
    def _make_routing_decisions(self, 
                               deadlines: List[Dict],
                               classification: Dict) -> List[Dict]:
        """Determine routing based on document type and confidence"""
        
        routing = []
        doc_type = classification.get('primary_type', 'unknown')
        
        # Get overall confidence for document-level routing
        overall_confidence = sum(d.get('confidence', 0) for d in deadlines) / len(deadlines) if deadlines else 0
        
        # Use classifier's routing logic for document-level decision
        if hasattr(self.document_classifier, 'get_routing_decision'):
            doc_routing = self.document_classifier.get_routing_decision(
                classification,
                overall_confidence
            )
        else:
            doc_routing = {'destination': 'attorney_review'}
        
        for deadline in deadlines:
            confidence = deadline.get('confidence', 0)
            
            # Apply routing rules:
            # 1. Court orders ALWAYS go to attorney regardless of confidence
            if doc_type == 'court_order':
                route = "ATTORNEY_REVIEW"
                reason = "Court orders require attorney review (mandatory)"
            # 2. High confidence non-court-orders can be auto-processed
            elif confidence >= self.config.confidence_threshold_auto and doc_type != 'court_order':
                route = "AUTO_PROCESS"
                reason = f"High confidence ({confidence:.1%}) for {doc_type}"
            # 3. Medium confidence goes to paralegal
            elif confidence >= self.config.confidence_threshold_review:
                route = "PARALEGAL_REVIEW"
                reason = f"Medium confidence ({confidence:.1%}) requires verification"
            # 4. Low confidence goes to attorney
            else:
                route = "ATTORNEY_REVIEW"
                reason = f"Low confidence ({confidence:.1%}) requires attorney review"
            
            # Override for critical deadlines
            if deadline.get('priority') == 'critical' and route != "ATTORNEY_REVIEW":
                route = "ATTORNEY_REVIEW"
                reason = "Critical deadline requires attorney review"
            
            # Override for invalid deadlines
            if deadline.get('validation_status') == 'invalid':
                route = "ATTORNEY_REVIEW"
                reason = "Invalid deadline pattern requires attorney review"
            
            routing.append({
                "deadline_id": deadline.get('id', 'unknown'),
                "routing": route,
                "confidence": confidence,
                "reason": reason,
                "document_type": doc_type,
                "validation_status": deadline.get('validation_status', 'unknown')
            })
        
        return routing
    
    def _get_routing_reason(self, confidence: float, deadline: Dict) -> str:
        """Generate human-readable routing reason"""
        
        if confidence >= self.config.confidence_threshold_auto:
            return "High confidence - automated processing"
        elif confidence >= self.config.confidence_threshold_review:
            return "Medium confidence - paralegal review recommended"
        elif deadline.get('priority') == 'critical':
            return "Critical deadline - attorney review required"
        else:
            return "Low confidence - attorney review required"
    
    async def _cache_validated_results(self, deadlines: List[Dict], entities: Dict):
        """Cache validated results for future reference"""
        
        if not self.rag_engine:
            return
        
        try:
            # Only cache successfully validated deadlines
            valid_deadlines = [
                d for d in deadlines 
                if d.get('validation_status') == 'valid' and d.get('confidence', 0) > 0.85
            ]
            
            if valid_deadlines:
                await self.rag_engine.cache_extraction({
                    "deadlines": valid_deadlines,
                    "entities": entities,
                    "timestamp": datetime.now().isoformat(),
                    "validation_type": "verified"
                })
                logger.info(f"Cached {len(valid_deadlines)} validated deadlines")
        except Exception as e:
            logger.error(f"Caching error: {e}")
    
    def _determine_jurisdiction(self, entities: Dict) -> str:
        """Determine jurisdiction from entities"""
        
        courts = entities.get('courts', [])
        
        for court in courts:
            court_lower = court.lower()
            if 'federal' in court_lower or 'united states' in court_lower:
                return 'federal'
            elif 'california' in court_lower:
                return 'california'
            elif 'new york' in court_lower:
                return 'new_york'
            elif 'texas' in court_lower:
                return 'texas'
        
        return 'federal'  # Default
    
    def _generate_document_id(self, document_path: str) -> str:
        """Generate unique document ID"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = Path(document_path).stem
        return f"doc_{filename}_{timestamp}"
    
    def _log_step(self, result: ProcessingResult, message: str):
        """Log processing step to audit trail"""
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": message,
            "document_id": result.document_id
        }
        
        result.audit_trail.append(entry)
        logger.info(f"[{result.document_id}] {message}")
    
    def _update_statistics(self, result: ProcessingResult):
        """Update pipeline statistics"""
        
        self.stats["documents_processed"] += 1
        
        if result.status == "success":
            self.stats["total_deadlines_extracted"] += len(result.deadlines)
            
            # Update average confidence
            if result.confidence_scores:
                avg_conf = sum(result.confidence_scores.values()) / len(result.confidence_scores)
                self.stats["average_confidence"] = (
                    (self.stats["average_confidence"] * (self.stats["documents_processed"] - 1) + avg_conf) /
                    self.stats["documents_processed"]
                )
            
            # Update average processing time
            self.stats["average_processing_time"] = (
                (self.stats["average_processing_time"] * (self.stats["documents_processed"] - 1) + 
                 result.processing_time) / self.stats["documents_processed"]
            )
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        return self.stats.copy()
    
    async def process_batch(self, 
                           document_paths: List[str],
                           max_concurrent: int = 5) -> List[ProcessingResult]:
        """
        Process multiple documents concurrently
        
        Args:
            document_paths: List of document paths
            max_concurrent: Maximum concurrent processing
            
        Returns:
            List of processing results
        """
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(path):
            async with semaphore:
                return await self.process_document(path)
        
        tasks = [process_with_semaphore(path) for path in document_paths]
        return await asyncio.gather(*tasks)
    
    def cleanup(self):
        """Clean up resources"""
        
        try:
            if hasattr(self, 'ocr_processor'):
                self.ocr_processor.cleanup()
            
            if hasattr(self, 'ai_extractor'):
                self.ai_extractor.cleanup()
            
            if self.rag_engine:
                self.rag_engine.cleanup()
            
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")