"""
RAG Engine using AWS OpenSearch
Handles similarity search and pattern matching
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import hashlib
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from opensearchpy.exceptions import NotFoundError

logger = logging.getLogger(__name__)

class RAGEngine:
    """
    RAG engine for legal pattern matching and similarity search
    Uses AWS OpenSearch for vector storage and retrieval
    """
    
    def __init__(self, 
                 opensearch_domain: str,
                 index_name: str = "legal-deadlines",
                 region: str = "us-east-1"):
        """Initialize OpenSearch connection"""
        
        self.domain = opensearch_domain
        self.index_name = index_name
        self.region = region
        
        try:
            # Initialize OpenSearch client
            self.client = OpenSearch(
                hosts=[{
                    'host': opensearch_domain,
                    'port': 443
                }],
                http_auth=self._get_auth(),
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=30
            )
            
            # Initialize Bedrock for embeddings
            self.bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=region
            )
            
            # Ensure index exists
            self._ensure_index()
            
            logger.info(f"RAG engine initialized with index: {index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise
    
    def _get_auth(self):
        """Get OpenSearch authentication"""
        
        # In production, use IAM roles or AWS Secrets Manager
        # This is a placeholder
        return ('admin', 'admin')
    
    def _ensure_index(self):
        """Ensure OpenSearch index exists with proper mappings"""
        
        index_body = {
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 1,
                "index.knn": True,
                "index.knn.space_type": "cosinesimil"
            },
            "mappings": {
                "properties": {
                    # Document identification
                    "document_id": {"type": "keyword"},
                    "pattern_id": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    
                    # Pattern content
                    "trigger_phrase": {"type": "text"},
                    "deadline_text": {"type": "text"},
                    "action_required": {"type": "text"},
                    "full_text": {"type": "text"},
                    
                    # Categorization
                    "jurisdiction": {"type": "keyword"},
                    "document_type": {"type": "keyword"},
                    "rule_type": {"type": "keyword"},
                    "priority": {"type": "keyword"},
                    
                    # Calculation details
                    "time_period": {"type": "keyword"},
                    "calculation_method": {"type": "keyword"},
                    "exceptions": {"type": "keyword"},
                    
                    # Verification and quality
                    "human_verified": {"type": "boolean"},
                    "verification_count": {"type": "integer"},
                    "confidence_score": {"type": "float"},
                    "accuracy_score": {"type": "float"},
                    
                    # Vector embedding
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1536,  # Titan embedding dimension
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss"
                        }
                    }
                }
            }
        }
        
        try:
            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(
                    index=self.index_name,
                    body=index_body
                )
                logger.info(f"Created index: {self.index_name}")
                
                # Seed with initial patterns
                self._seed_patterns()
            else:
                logger.info(f"Index already exists: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def _seed_patterns(self):
        """Seed index with common legal patterns"""
        
        patterns = [
            {
                "pattern_id": "federal_motion_response",
                "trigger_phrase": "motion filed",
                "deadline_text": "14 days to respond",
                "action_required": "file opposition",
                "jurisdiction": "federal",
                "document_type": "motion",
                "time_period": "14 days",
                "calculation_method": "calendar_days",
                "human_verified": True,
                "verification_count": 100,
                "confidence_score": 0.95
            },
            {
                "pattern_id": "federal_discovery_response",
                "trigger_phrase": "discovery request served",
                "deadline_text": "30 days to respond",
                "action_required": "respond to discovery",
                "jurisdiction": "federal",
                "document_type": "discovery",
                "time_period": "30 days",
                "calculation_method": "calendar_days",
                "human_verified": True,
                "verification_count": 150,
                "confidence_score": 0.98
            },
            {
                "pattern_id": "appeal_notice",
                "trigger_phrase": "judgment entered",
                "deadline_text": "30 days to file notice of appeal",
                "action_required": "file notice of appeal",
                "jurisdiction": "federal",
                "document_type": "judgment",
                "time_period": "30 days",
                "calculation_method": "calendar_days",
                "human_verified": True,
                "verification_count": 75,
                "confidence_score": 0.99
            }
        ]
        
        for pattern in patterns:
            try:
                # Generate embedding
                pattern['embedding'] = self._generate_embedding(
                    f"{pattern['trigger_phrase']} {pattern['action_required']}"
                )
                
                # Index pattern
                self.client.index(
                    index=self.index_name,
                    body=pattern,
                    id=pattern['pattern_id']
                )
                
            except Exception as e:
                logger.error(f"Error seeding pattern: {e}")
    
    async def find_similar(self, 
                          query: str,
                          limit: int = 10,
                          min_score: float = 0.7) -> List[Dict]:
        """
        Find similar patterns using vector search
        
        Args:
            query: Search query
            limit: Maximum results
            min_score: Minimum similarity score
            
        Returns:
            List of similar patterns
        """
        
        try:
            # Generate embedding for query
            query_embedding = self._generate_embedding(query)
            
            # Build KNN search query
            search_query = {
                "size": limit,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": limit
                        }
                    }
                },
                "min_score": min_score,
                "_source": {
                    "excludes": ["embedding"]  # Don't return embeddings
                }
            }
            
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=search_query
            )
            
            # Extract results
            results = []
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['similarity_score'] = hit['_score']
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return []
    
    async def find_by_jurisdiction(self, 
                                  jurisdiction: str,
                                  document_type: Optional[str] = None) -> List[Dict]:
        """Find patterns by jurisdiction and document type"""
        
        try:
            # Build query
            must_clauses = [
                {"term": {"jurisdiction": jurisdiction.lower()}}
            ]
            
            if document_type:
                must_clauses.append({"term": {"document_type": document_type.lower()}})
            
            query = {
                "size": 100,
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "filter": [
                            {"term": {"human_verified": True}}
                        ]
                    }
                },
                "sort": [
                    {"confidence_score": {"order": "desc"}},
                    {"verification_count": {"order": "desc"}}
                ]
            }
            
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            
            # Extract results
            results = []
            for hit in response['hits']['hits']:
                results.append(hit['_source'])
            
            return results
            
        except Exception as e:
            logger.error(f"Jurisdiction search error: {e}")
            return []
    
    async def cache_extraction(self, extraction: Dict):
        """Cache extraction results for future learning"""
        
        try:
            # Process each deadline
            for deadline in extraction.get('deadlines', []):
                # Create pattern document
                pattern = {
                    "document_id": f"doc_{datetime.now().timestamp()}",
                    "pattern_id": self._generate_pattern_id(deadline),
                    "timestamp": datetime.now().isoformat(),
                    "trigger_phrase": deadline.get('trigger_event', ''),
                    "deadline_text": deadline.get('time_period', ''),
                    "action_required": deadline.get('action', ''),
                    "full_text": deadline.get('source_text', ''),
                    "jurisdiction": self._extract_jurisdiction(extraction.get('entities', {})),
                    "document_type": extraction.get('document_context', {}).get('type', 'unknown'),
                    "time_period": deadline.get('time_period', ''),
                    "calculation_method": deadline.get('calculation_method', 'unknown'),
                    "priority": deadline.get('priority', 'standard'),
                    "human_verified": False,
                    "verification_count": 0,
                    "confidence_score": deadline.get('confidence', 0.5),
                    "accuracy_score": 0.0
                }
                
                # Generate embedding
                pattern['embedding'] = self._generate_embedding(
                    f"{pattern['trigger_phrase']} {pattern['action_required']}"
                )
                
                # Index pattern
                self.client.index(
                    index=self.index_name,
                    body=pattern
                )
            
            logger.info(f"Cached {len(extraction.get('deadlines', []))} patterns")
            
        except Exception as e:
            logger.error(f"Caching error: {e}")
    
    def update_verification(self, pattern_id: str, verified: bool, confidence: float):
        """Update pattern verification status"""
        
        try:
            # Update document
            self.client.update(
                index=self.index_name,
                id=pattern_id,
                body={
                    "script": {
                        "source": """
                            ctx._source.human_verified = params.verified;
                            ctx._source.verification_count += 1;
                            ctx._source.confidence_score = 
                                (ctx._source.confidence_score * ctx._source.verification_count + params.confidence) / 
                                (ctx._source.verification_count + 1);
                        """,
                        "params": {
                            "verified": verified,
                            "confidence": confidence
                        }
                    }
                }
            )
            
            logger.info(f"Updated verification for pattern: {pattern_id}")
            
        except Exception as e:
            logger.error(f"Verification update error: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Amazon Titan"""
        
        try:
            # Call Titan embedding model
            response = self.bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text})
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding', [])
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    def _generate_pattern_id(self, deadline: Dict) -> str:
        """Generate unique pattern ID"""
        
        content = f"{deadline.get('trigger_event', '')}_{deadline.get('action', '')}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _extract_jurisdiction(self, entities: Dict) -> str:
        """Extract jurisdiction from entities"""
        
        courts = entities.get('courts', [])
        
        for court in courts:
            if isinstance(court, dict):
                jurisdiction = court.get('jurisdiction', '').lower()
                if jurisdiction:
                    return jurisdiction
            elif isinstance(court, str):
                court_lower = court.lower()
                if 'federal' in court_lower or 'united states' in court_lower:
                    return 'federal'
                elif 'california' in court_lower:
                    return 'california'
                elif 'new york' in court_lower:
                    return 'new_york'
        
        return 'unknown'
    
    def get_statistics(self) -> Dict:
        """Get RAG engine statistics"""
        
        try:
            # Get index stats
            stats = self.client.indices.stats(index=self.index_name)
            
            # Count verified patterns
            verified_count = self.client.count(
                index=self.index_name,
                body={
                    "query": {"term": {"human_verified": True}}
                }
            )
            
            return {
                "total_patterns": stats['indices'][self.index_name]['total']['docs']['count'],
                "verified_patterns": verified_count['count'],
                "index_size_mb": stats['indices'][self.index_name]['total']['store']['size_in_bytes'] / 1024 / 1024
            }
            
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return {}
    
    def cleanup(self):
        """Clean up resources"""
        
        try:
            # Close OpenSearch connection
            if hasattr(self, 'client'):
                self.client.close()
            
            logger.info("RAG engine cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")