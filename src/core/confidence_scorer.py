"""
Confidence Scorer
Calculates confidence scores for extracted deadlines
"""

import logging
from typing import Dict, List, Optional
import re

logger = logging.getLogger(__name__)

class ConfidenceScorer:
    """
    Calculates confidence scores for extracted deadlines
    Combines multiple factors including RAG similarity
    """
    
    def __init__(self, rag_engine=None):
        """Initialize confidence scorer"""
        
        self.rag_engine = rag_engine
        
        # Weight factors for confidence calculation
        self.weights = {
            "ocr_quality": 0.15,
            "extraction_confidence": 0.20,
            "pattern_similarity": 0.20,
            "historical_match": 0.15,
            "rule_alignment": 0.10,
            "document_type": 0.20
        }
        
        # Document type confidence weights
        self.document_type_weights = {
            "court_order": 1.2,      # Higher confidence for binding orders
            "motion": 0.9,           # Lower confidence for proposed deadlines
            "notice": 1.0,           # Standard confidence for informational
            "complaint": 0.95,       # Slightly lower for initial filings
            "answer": 0.95,          # Slightly lower for responses
            "discovery": 1.0,        # Standard for discovery requests
            "unknown": 0.8           # Lower confidence for unknown types
        }
        
        # Common legal deadline patterns
        self.known_patterns = {
            "motion_response": [
                r"respond.*motion",
                r"opposition.*due",
                r"file.*response"
            ],
            "discovery": [
                r"respond.*discovery",
                r"produce.*documents",
                r"answer.*interrogator"
            ],
            "appeal": [
                r"notice.*appeal",
                r"appeal.*deadline",
                r"file.*appeal"
            ],
            "hearing": [
                r"appear.*hearing",
                r"attend.*conference",
                r"court.*appearance"
            ]
        }
        
        logger.info("Confidence scorer initialized")
    
    def calculate(self,
                 deadline: Dict,
                 similar_patterns: List[Dict],
                 ocr_confidence: float,
                 document_type: Optional[str] = None) -> float:
        """
        Calculate confidence score for a deadline with document type awareness
        
        Args:
            deadline: Extracted deadline
            similar_patterns: Similar patterns from RAG
            ocr_confidence: OCR extraction confidence
            document_type: Document classification type
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        
        scores = {}
        
        # 1. OCR Quality Score
        scores['ocr_quality'] = min(1.0, ocr_confidence)
        
        # 2. Extraction Confidence (from Claude)
        scores['extraction_confidence'] = deadline.get('confidence', 0.5)
        
        # 3. Pattern Similarity Score
        scores['pattern_similarity'] = self._calculate_pattern_similarity(
            deadline,
            similar_patterns
        )
        
        # 4. Historical Match Score
        scores['historical_match'] = self._calculate_historical_match(
            similar_patterns
        )
        
        # 5. Rule Alignment Score
        scores['rule_alignment'] = self._calculate_rule_alignment(deadline)
        
        # 6. Document Type Score
        scores['document_type'] = self._calculate_document_type_score(document_type)
        
        # Calculate weighted average
        base_score = sum(
            scores.get(factor, 0) * self.weights[factor]
            for factor in self.weights
        )
        
        # Apply document type weighting
        type_weight = self.document_type_weights.get(document_type, 1.0) if document_type else 1.0
        total_score = min(1.0, base_score * type_weight)
        
        # Apply modifiers
        total_score = self._apply_modifiers(total_score, deadline)
        
        # Ensure score is in valid range
        total_score = max(0.0, min(1.0, total_score))
        
        # Log scoring details for transparency
        logger.debug(f"Confidence scoring for deadline: {deadline.get('action', 'unknown')}")
        logger.debug(f"  Component scores: {scores}")
        logger.debug(f"  Final score: {total_score:.3f}")
        
        return total_score
    
    def _calculate_pattern_similarity(self,
                                     deadline: Dict,
                                     similar_patterns: List[Dict]) -> float:
        """Calculate pattern similarity score"""
        
        if not similar_patterns:
            # Check against known patterns
            return self._check_known_patterns(deadline)
        
        # Use top similarity scores
        top_scores = [p.get('similarity_score', 0) for p in similar_patterns[:5]]
        
        if not top_scores:
            return 0.5
        
        # Weight by position (top matches more important)
        weighted_score = sum(
            score * (1.0 / (i + 1))
            for i, score in enumerate(top_scores)
        )
        
        # Normalize
        max_possible = sum(1.0 / (i + 1) for i in range(len(top_scores)))
        normalized = weighted_score / max_possible if max_possible > 0 else 0
        
        return min(1.0, normalized)
    
    def _check_known_patterns(self, deadline: Dict) -> float:
        """Check deadline against known patterns"""
        
        action = deadline.get('action', '').lower()
        source = deadline.get('source_text', '').lower()
        
        for pattern_type, patterns in self.known_patterns.items():
            for pattern in patterns:
                if re.search(pattern, action) or re.search(pattern, source):
                    return 0.8  # High confidence for known patterns
        
        return 0.5  # Neutral if no pattern match
    
    def _calculate_historical_match(self, similar_patterns: List[Dict]) -> float:
        """Calculate score based on historical verified patterns"""
        
        if not similar_patterns:
            return 0.5
        
        # Filter for verified patterns
        verified = [
            p for p in similar_patterns 
            if p.get('human_verified', False)
        ]
        
        if not verified:
            return 0.6  # Slightly above neutral if no verified patterns
        
        # Calculate based on verification count and confidence
        scores = []
        for pattern in verified[:3]:  # Top 3 verified patterns
            verification_count = pattern.get('verification_count', 0)
            pattern_confidence = pattern.get('confidence_score', 0.5)
            
            # Higher score for more verifications
            verification_factor = min(1.0, verification_count / 100)
            scores.append(pattern_confidence * verification_factor)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _calculate_rule_alignment(self, deadline: Dict) -> float:
        """Calculate alignment with legal rules"""
        
        score = 0.7  # Base score
        
        # Check for clear calculation method
        calc_method = deadline.get('calculation_method', '').lower()
        if calc_method in ['calendar_days', 'business_days', 'court_days']:
            score += 0.1
        
        # Check for explicit time period
        time_period = deadline.get('time_period', '')
        if re.search(r'\d+\s*(day|week|month)', time_period):
            score += 0.1
        
        # Check for clear trigger event
        trigger = deadline.get('trigger_event', '').lower()
        if trigger and trigger != 'unknown':
            score += 0.1
        
        # Penalty for ambiguity
        if deadline.get('ambiguity_flag', False):
            score -= 0.2
        
        # Penalty for missing critical information
        if not deadline.get('action'):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _calculate_document_type_score(self, document_type: Optional[str]) -> float:
        """Calculate confidence score based on document type"""
        
        if not document_type:
            return 0.7  # Default score for unknown type
        
        # Document type reliability scores
        type_scores = {
            "court_order": 0.95,     # High reliability - binding documents
            "motion": 0.85,          # Medium reliability - proposed actions
            "notice": 0.90,          # High reliability - informational
            "complaint": 0.85,       # Medium reliability - initial filings
            "answer": 0.85,          # Medium reliability - responses
            "discovery": 0.90,       # High reliability - procedural
            "unknown": 0.70          # Low reliability - unclassified
        }
        
        return type_scores.get(document_type, 0.70)
    
    def _apply_modifiers(self, base_score: float, deadline: Dict) -> float:
        """Apply modifiers based on special conditions"""
        
        score = base_score
        
        # Critical deadlines need higher confidence
        if deadline.get('priority') == 'critical':
            # Reduce score if not very high
            if score < 0.95:
                score *= 0.9
        
        # Boost for unambiguous deadlines
        if not deadline.get('ambiguity_flag', False):
            score = min(1.0, score * 1.05)
        
        # Penalty for calculation errors
        if deadline.get('calculation_error'):
            score *= 0.7
        
        # Penalty for too many warnings
        warnings = deadline.get('warnings', [])
        if len(warnings) > 2:
            score *= 0.85
        
        # Boost for exact date matches
        if deadline.get('calculated_date') and not deadline.get('relative_date'):
            score = min(1.0, score * 1.1)
        
        return score
    
    def batch_calculate(self,
                       deadlines: List[Dict],
                       ocr_confidence: float,
                       document_type: Optional[str] = None) -> List[Dict]:
        """
        Calculate confidence for multiple deadlines with document type awareness
        
        Args:
            deadlines: List of extracted deadlines
            ocr_confidence: OCR extraction confidence
            document_type: Document classification type
            
        Returns:
            Deadlines with confidence scores
        """
        
        scored_deadlines = []
        
        for deadline in deadlines:
            # Get similar patterns if RAG is available
            similar_patterns = []
            if self.rag_engine:
                try:
                    query = f"{deadline.get('trigger_event', '')} {deadline.get('action', '')}"
                    similar_patterns = self.rag_engine.find_similar(query, limit=10)
                except Exception as e:
                    logger.error(f"RAG search error: {e}")
            
            # Calculate confidence with document type
            confidence = self.calculate(deadline, similar_patterns, ocr_confidence, document_type)
            deadline['confidence'] = confidence
            
            # Add confidence breakdown for transparency
            deadline['confidence_factors'] = {
                "ocr_quality": ocr_confidence,
                "extraction_confidence": deadline.get('confidence', 0.5),
                "pattern_matches": len(similar_patterns),
                "has_warnings": len(deadline.get('warnings', [])) > 0
            }
            
            scored_deadlines.append(deadline)
        
        return scored_deadlines
    
    def get_confidence_statistics(self, deadlines: List[Dict]) -> Dict:
        """Get statistics about confidence scores"""
        
        if not deadlines:
            return {
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0
            }
        
        scores = [d.get('confidence', 0) for d in deadlines]
        
        return {
            "count": len(scores),
            "average": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "high_confidence": sum(1 for s in scores if s >= 0.95),
            "medium_confidence": sum(1 for s in scores if 0.85 <= s < 0.95),
            "low_confidence": sum(1 for s in scores if s < 0.85)
        }