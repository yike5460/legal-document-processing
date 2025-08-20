"""
Document Classification using Claude 4 Sonnet
Intelligently categorizes legal documents for type-specific processing
"""

import json
import logging
from typing import Dict
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class DocumentClassifier:
    """
    AI-powered document classification using Claude 4 Sonnet.
    Provides intelligent categorization with extraction hints for downstream processing.
    """
    
    DOCUMENT_TYPES = {
        "court_order": {
            "description": "Binding directive from a judge",
            "indicators": ["IT IS ORDERED", "THE COURT ORDERS", "IT IS HEREBY ORDERED"],
            "entity_focus": ["judge_name", "case_number", "ordered_actions", "compliance_deadlines"],
            "deadline_sensitivity": "critical",
            "binding_level": "binding",
            "priority": "critical",
            "requires_attorney_review": True
        },
        "motion": {
            "description": "Request for court action",
            "indicators": ["MOTION TO", "MOTION FOR", "MOVES THIS COURT", "MOVES FOR"],
            "entity_focus": ["moving_party", "relief_sought", "hearing_date", "opposition_deadline"],
            "deadline_sensitivity": "high",
            "binding_level": "proposed",
            "priority": "high",
            "requires_attorney_review": False
        },
        "notice": {
            "description": "Informational filing",
            "indicators": ["NOTICE OF", "PLEASE TAKE NOTICE", "BE ADVISED"],
            "entity_focus": ["noticing_party", "event_description", "relevant_dates"],
            "deadline_sensitivity": "medium",
            "binding_level": "informational",
            "priority": "standard",
            "requires_attorney_review": False
        },
        "complaint": {
            "description": "Initial lawsuit filing",
            "indicators": ["COMPLAINT FOR", "PLAINTIFF ALLEGES", "CAUSES OF ACTION"],
            "entity_focus": ["plaintiff", "defendant", "claims", "response_deadline"],
            "deadline_sensitivity": "high",
            "binding_level": "proposed",
            "priority": "high",
            "requires_attorney_review": False
        },
        "answer": {
            "description": "Response to complaint",
            "indicators": ["ANSWER TO", "DEFENDANT RESPONDS", "AFFIRMATIVE DEFENSES"],
            "entity_focus": ["defendant", "admissions", "denials", "counterclaim_deadline"],
            "deadline_sensitivity": "high",
            "binding_level": "proposed",
            "priority": "high",
            "requires_attorney_review": False
        },
        "discovery": {
            "description": "Information request",
            "indicators": ["REQUEST FOR", "INTERROGATORIES", "PRODUCTION OF DOCUMENTS"],
            "entity_focus": ["requesting_party", "response_deadline", "items_requested"],
            "deadline_sensitivity": "high",
            "binding_level": "proposed",
            "priority": "high",
            "requires_attorney_review": False
        }
    }
    
    def __init__(self, bedrock_client=None, model_id=None):
        """Initialize with optional Bedrock client and model ID."""
        self.bedrock_client = bedrock_client
        if not self.bedrock_client:
            try:
                self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
            except Exception as e:
                logger.warning(f"Could not initialize Bedrock client: {e}")
                self.bedrock_client = None
        
        # Use Claude Sonnet 4 model ID (can be overridden with inference profile)
        self.model_id = model_id or "us.anthropic.claude-sonnet-4-20250514-v1:0"
        
    async def classify(self, text: str) -> Dict:
        """
        Classify legal document using Claude 4 Sonnet.
        
        Args:
            text: Document text (first 5000 chars used for classification)
            
        Returns:
            Classification result with confidence and extraction hints
        """
        try:
            # Use Claude 4 for intelligent classification
            classification = await self._classify_with_claude(text[:5000])
            
            # Enhance with type-specific strategies
            doc_type = classification.get("primary_type")
            if doc_type in self.DOCUMENT_TYPES:
                type_info = self.DOCUMENT_TYPES[doc_type]
                classification["extraction_strategy"] = {
                    "entity_focus": type_info["entity_focus"],
                    "deadline_sensitivity": type_info["deadline_sensitivity"],
                    "requires_attorney_review": type_info["requires_attorney_review"]
                }
                
                # Override extraction hints with known good values
                classification["extraction_hints"]["binding_level"] = type_info["binding_level"]
                classification["extraction_hints"]["priority"] = type_info["priority"]
            
            logger.info(f"Document classified as {doc_type} with confidence {classification.get('confidence', 0):.2%}")
            return classification
            
        except ClientError as e:
            logger.warning(f"Claude API error during classification: {e}")
            # Fallback to pattern-based classification
            return self._fallback_classification(text)
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._default_classification()
    
    async def _classify_with_claude(self, text: str) -> Dict:
        """Use Claude 4 Sonnet for intelligent classification."""
        classification_prompt = self._build_classification_prompt(text)
        
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": classification_prompt}],
                "max_tokens": 500,
                "temperature": 0.1  # Low temperature for consistent classification
            })
        )
        
        result = json.loads(response['body'].read())
        classification_text = result['content'][0]['text']
        
        # Parse JSON response
        try:
            classification = json.loads(classification_text)
        except json.JSONDecodeError:
            # Extract JSON from response if wrapped in text
            import re
            json_match = re.search(r'\{.*\}', classification_text, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse classification response")
        
        return classification
    
    def _build_classification_prompt(self, text: str) -> str:
        """Build detailed classification prompt for Claude."""
        return f"""Analyze this legal document and provide detailed classification.

Document text:
{text}

Classify into ONE of these categories:
1. court_order: Binding directive from a judge (look for "IT IS ORDERED", "THE COURT ORDERS", "IT IS HEREBY ORDERED")
2. motion: Request for court action (look for "MOTION TO", "MOTION FOR", "MOVES THIS COURT")
3. notice: Informational filing (look for "NOTICE OF", "PLEASE TAKE NOTICE", "BE ADVISED")
4. complaint: Initial lawsuit filing (look for "COMPLAINT FOR", "PLAINTIFF ALLEGES", "CAUSES OF ACTION")
5. answer: Response to complaint (look for "ANSWER TO", "DEFENDANT RESPONDS", "AFFIRMATIVE DEFENSES")
6. discovery: Information request (look for "REQUEST FOR", "INTERROGATORIES", "PRODUCTION OF DOCUMENTS")

Consider these factors:
- Document structure and formatting (headers, captions, signature blocks)
- Legal language and tone (mandatory vs permissive language)
- Issuing authority (court vs party filing)
- Presence of court caption and case number
- Binding language ("SHALL", "MUST", "ORDERED") vs proposed language ("requests", "seeks")

Return ONLY a JSON object with this structure:
{{
    "primary_type": "category_name",
    "confidence": 0.95,
    "reasoning": "Brief explanation of classification decision",
    "indicators": ["List of", "key phrases", "found in document"],
    "extraction_hints": {{
        "deadline_keywords": ["within", "days", "no later than"],
        "priority": "critical|high|standard",
        "requires_response": true,
        "binding_level": "binding|proposed|informational",
        "suggested_entities": ["entity types to extract"]
    }}
}}"""
    
    def _fallback_classification(self, text: str) -> Dict:
        """Pattern-based fallback classification when Claude is unavailable."""
        text_upper = text.upper()
        
        # Check each document type
        for doc_type, type_info in self.DOCUMENT_TYPES.items():
            if any(indicator in text_upper for indicator in type_info["indicators"]):
                return {
                    "primary_type": doc_type,
                    "confidence": 0.75,  # Lower confidence for pattern matching
                    "reasoning": f"Pattern matching found indicators: {type_info['indicators'][0]}",
                    "indicators": [ind for ind in type_info["indicators"] if ind in text_upper],
                    "extraction_hints": {
                        "deadline_keywords": ["within", "days", "no later than", "must", "shall"],
                        "priority": type_info["priority"],
                        "requires_response": doc_type in ["motion", "complaint", "discovery"],
                        "binding_level": type_info["binding_level"]
                    },
                    "extraction_strategy": {
                        "entity_focus": type_info["entity_focus"],
                        "deadline_sensitivity": type_info["deadline_sensitivity"],
                        "requires_attorney_review": type_info["requires_attorney_review"]
                    }
                }
        
        # Default to unknown
        return self._default_classification()
    
    def _default_classification(self) -> Dict:
        """Return default classification for unknown documents."""
        return {
            "primary_type": "unknown",
            "confidence": 0.0,
            "reasoning": "Could not determine document type",
            "indicators": [],
            "extraction_hints": {
                "deadline_keywords": ["within", "days", "no later than", "must", "shall"],
                "priority": "standard",
                "requires_response": False,
                "binding_level": "informational"
            },
            "extraction_strategy": {
                "entity_focus": ["parties", "dates", "case_number"],
                "deadline_sensitivity": "standard",
                "requires_attorney_review": True  # Unknown docs need review
            }
        }
    
    def get_routing_decision(self, classification: Dict, confidence_score: float) -> Dict:
        """
        Determine routing based on document type and confidence.
        
        Args:
            classification: Document classification result
            confidence_score: Overall confidence score
            
        Returns:
            Routing decision with destination and priority
        """
        doc_type = classification.get("primary_type", "unknown")
        type_info = self.DOCUMENT_TYPES.get(doc_type, {})
        
        # Court orders always go to attorney
        if doc_type == "court_order":
            return {
                "destination": "attorney_review",
                "priority": "critical",
                "reason": "Court orders require attorney review regardless of confidence",
                "sla_hours": 2
            }
        
        # High confidence threshold for auto-processing
        if confidence_score >= 0.95 and not type_info.get("requires_attorney_review", False):
            return {
                "destination": "auto_calendar",
                "priority": type_info.get("priority", "standard"),
                "reason": f"High confidence ({confidence_score:.1%}) for {doc_type}",
                "sla_hours": 24
            }
        
        # Medium confidence goes to paralegal
        if confidence_score >= 0.85:
            return {
                "destination": "paralegal_review",
                "priority": type_info.get("priority", "standard"),
                "reason": f"Medium confidence ({confidence_score:.1%}) requires human verification",
                "sla_hours": 8
            }
        
        # Low confidence or unknown goes to attorney
        return {
            "destination": "attorney_review",
            "priority": "high",
            "reason": f"Low confidence ({confidence_score:.1%}) or unknown document type",
            "sla_hours": 4
        }