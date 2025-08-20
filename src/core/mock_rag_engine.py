"""
Mock RAG Engine for Testing
Simulates RAG functionality without requiring OpenSearch
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MockRAGEngine:
    """
    Mock RAG engine for testing when OpenSearch is not available
    Simulates pattern matching and similarity search
    """
    
    def __init__(self, opensearch_domain: str = "mock"):
        """Initialize mock RAG engine"""
        self.domain = opensearch_domain
        logger.info("Mock RAG engine initialized for testing")
        
        # Mock pattern database
        self.patterns = [
            {
                "pattern_id": "federal_motion_response",
                "trigger_phrase": "motion filed",
                "action_required": "file opposition",
                "time_period": "14 days",
                "jurisdiction": "federal",
                "similarity_score": 0.95,
                "human_verified": True,
                "verification_count": 150
            },
            {
                "pattern_id": "discovery_response",
                "trigger_phrase": "discovery served",
                "action_required": "respond to discovery",
                "time_period": "30 days",
                "jurisdiction": "federal",
                "similarity_score": 0.88,
                "human_verified": True,
                "verification_count": 200
            },
            {
                "pattern_id": "appeal_notice",
                "trigger_phrase": "judgment entered",
                "action_required": "file notice of appeal",
                "time_period": "30 days",
                "jurisdiction": "federal",
                "similarity_score": 0.92,
                "human_verified": True,
                "verification_count": 75
            }
        ]
    
    async def find_similar(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Simulate similarity search
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            Mock similar patterns
        """
        
        # Simple keyword matching for simulation
        results = []
        
        query_lower = query.lower()
        
        for pattern in self.patterns:
            score = 0.0
            
            # Calculate mock similarity based on keyword overlap
            if pattern["trigger_phrase"] in query_lower:
                score += 0.4
            if pattern["action_required"] in query_lower:
                score += 0.3
            if pattern["time_period"] in query_lower:
                score += 0.2
            if "motion" in query_lower and "motion" in pattern["pattern_id"]:
                score += 0.1
            if "discovery" in query_lower and "discovery" in pattern["pattern_id"]:
                score += 0.1
            if "appeal" in query_lower and "appeal" in pattern["pattern_id"]:
                score += 0.1
            
            if score > 0:
                result = pattern.copy()
                result["similarity_score"] = min(score, 0.95)
                results.append(result)
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]
    
    async def cache_extraction(self, extraction: Dict) -> bool:
        """
        Simulate caching extraction
        
        Args:
            extraction: Data to cache
            
        Returns:
            Success status
        """
        
        logger.info("Mock caching extraction (no-op in test mode)")
        return True
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Mock RAG engine cleanup completed")