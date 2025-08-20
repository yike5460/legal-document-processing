"""
AI Extractor using Claude 4 Sonnet
Handles entity recognition and deadline extraction
"""

import json
import logging
from typing import Dict, Optional
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class ClaudeExtractor:
    """
    Claude 3.5 Sonnet extractor for legal documents
    Unified extraction of entities and deadlines
    """
    
    def __init__(self, 
                 model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
                 region: str = "us-east-1",
                 bedrock_client=None):
        """Initialize Claude 3.5 Sonnet via Bedrock"""
        
        self.model_id = model_id
        self.region = region
        
        # Use provided client or create new one
        if bedrock_client:
            self.bedrock = bedrock_client
        else:
            try:
                # Initialize Bedrock client
                self.bedrock = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=region
                )
                logger.info(f"Claude 3.5 Sonnet initialized in {region}")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude 3.5 Sonnet: {e}")
                self.bedrock = None
        
        # Model parameters
        self.model_params = {
            "temperature": 0.1,
            "max_tokens": 8192,
            "anthropic_version": "bedrock-2023-05-31",
            "top_p": 0.95
        }
    
    async def extract(self, 
                     text: str,
                     layout: Dict,
                     metadata: Optional[Dict] = None) -> Dict:
        """
        Extract entities and deadlines from legal document
        
        Args:
            text: Document text (potentially masked)
            layout: Document layout information
            metadata: Optional document metadata
            
        Returns:
            Dictionary with entities, deadlines, and context
        """
        
        try:
            # Build extraction prompt
            prompt = self._build_prompt(text, layout, metadata)
            
            # Call Claude 4 Sonnet
            response = await self._call_claude(prompt)
            
            # Parse and validate response
            extraction = self._parse_response(response)
            
            # Post-process extraction
            extraction = self._post_process(extraction)
            
            return extraction
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            # Return empty extraction on failure
            return self._empty_extraction()
    
    def _build_prompt(self, text: str, layout: Dict, metadata: Optional[Dict]) -> str:
        """Build comprehensive extraction prompt"""
        
        # Include metadata context if available
        context = ""
        if metadata:
            context = f"\nDocument Metadata:\n{json.dumps(metadata, indent=2)}\n"
        
        prompt = f"""You are an expert legal document analyzer using Claude's advanced capabilities.
Perform comprehensive extraction from this legal document.

{context}

Document Text:
{text}

Layout Information:
{json.dumps(layout, indent=2) if layout else "Not available"}

EXTRACTION REQUIREMENTS:

1. ENTITIES - Extract all legal entities:
   - Case parties (plaintiff, defendant, petitioner, respondent, etc.)
   - Case numbers and identifiers (preserve exact format)
   - Judge names and titles
   - Court names and jurisdictions
   - Attorney names, bar numbers, and firms
   - Legal citations and references

2. DEADLINES - Extract ALL time-sensitive information:
   - Explicit dates (e.g., "March 15, 2024")
   - Relative periods (e.g., "within 14 days", "30 days from service")
   - Triggering events (e.g., "from date of this order", "upon filing")
   - Required actions (e.g., "file opposition", "respond to motion")
   - Conditional clauses (e.g., "unless otherwise ordered")
   - Calculation methods (calendar days vs business days vs court days)
   - Priority level (critical, standard, informational)

3. DOCUMENT CONTEXT - Identify:
   - Document type (motion, order, complaint, notice, etc.)
   - Procedural posture (pre-trial, discovery, post-judgment, etc.)
   - Applicable rules or statutes cited
   - Special conditions or exceptions noted

4. EXTRACTION RULES:
   - Be comprehensive - extract ALL potential deadlines, even if uncertain
   - Preserve exact phrasing from the document
   - Note any ambiguities or unclear references
   - Flag items requiring human verification
   - Include confidence level for each extraction

Return a JSON object with this exact structure:
{{
    "entities": {{
        "parties": [
            {{"name": "...", "role": "plaintiff|defendant|other", "type": "individual|corporation"}}
        ],
        "case_numbers": ["exact case number"],
        "judges": [{{"name": "...", "title": "..."}}],
        "courts": [{{"name": "...", "jurisdiction": "federal|state", "level": "district|appeals|supreme"}}],
        "attorneys": [{{"name": "...", "bar_number": "...", "firm": "...", "representing": "..."}}],
        "citations": ["legal citations"]
    }},
    "deadlines": [
        {{
            "trigger_event": "specific triggering event or 'document_date' if from this document",
            "time_period": "exact time period (e.g., '14 days', '30 calendar days')",
            "action": "specific action required",
            "source_text": "exact quote from document",
            "calculation_method": "calendar_days|business_days|court_days",
            "priority": "critical|standard|informational",
            "conditions": "any conditions or exceptions",
            "ambiguity_flag": true|false,
            "confidence": 0.0-1.0
        }}
    ],
    "document_context": {{
        "type": "document type",
        "date": "document date if present",
        "procedural_posture": "current stage of proceedings",
        "applicable_rules": ["rules or statutes cited"],
        "special_conditions": ["any special conditions noted"],
        "requires_attorney_review": true|false
    }},
    "extraction_metadata": {{
        "total_entities": 0,
        "total_deadlines": 0,
        "average_confidence": 0.0,
        "extraction_warnings": ["any warnings or issues"],
        "model": "claude-3.5-sonnet"
    }}
}}

IMPORTANT: Return ONLY valid JSON. No additional text or explanation."""
        
        return prompt
    
    async def _call_claude(self, prompt: str) -> str:
        """Call Claude 4 Sonnet via Bedrock"""
        
        if not self.bedrock:
            logger.warning("Bedrock client not available, returning fallback response")
            # Return a fallback response that the parser can handle
            return json.dumps({
                "entities": {},
                "deadlines": [],
                "extraction_confidence": 0.0,
                "warnings": ["Bedrock client not available"]
            })
        
        try:
            # Prepare request body
            request_body = {
                "anthropic_version": self.model_params["anthropic_version"],
                "max_tokens": self.model_params["max_tokens"],
                "temperature": self.model_params["temperature"],
                "top_p": self.model_params["top_p"],
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Make API call
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract content
            if 'content' in response_body and response_body['content']:
                return response_body['content'][0]['text']
            else:
                raise ValueError("Empty response from Claude")
            
        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Claude call error: {e}")
            raise
    
    def _parse_response(self, response: str) -> Dict:
        """Parse and validate Claude's response"""
        
        try:
            # Extract JSON from response
            # Claude might include explanation, so find JSON boundaries
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                extraction = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
            
            # Validate structure
            self._validate_extraction(extraction)
            
            return extraction
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response: {response[:500]}...")
            return self._empty_extraction()
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return self._empty_extraction()
    
    def _validate_extraction(self, extraction: Dict):
        """Validate extraction structure"""
        
        required_keys = ['entities', 'deadlines', 'document_context']
        
        for key in required_keys:
            if key not in extraction:
                extraction[key] = {} if key != 'deadlines' else []
        
        # Ensure entities has required sub-keys
        entity_keys = ['parties', 'case_numbers', 'judges', 'courts', 'attorneys', 'citations']
        if 'entities' not in extraction:
            extraction['entities'] = {}
        
        for key in entity_keys:
            if key not in extraction['entities']:
                extraction['entities'][key] = []
        
        # Ensure deadlines is a list
        if not isinstance(extraction.get('deadlines'), list):
            extraction['deadlines'] = []
        
        # Add extraction metadata if missing
        if 'extraction_metadata' not in extraction:
            extraction['extraction_metadata'] = {
                "total_entities": len(extraction.get('entities', {})),
                "total_deadlines": len(extraction.get('deadlines', [])),
                "model": "claude-3.5-sonnet"
            }
    
    def _post_process(self, extraction: Dict) -> Dict:
        """Post-process extraction results"""
        
        # Calculate statistics
        total_entities = sum(
            len(v) if isinstance(v, list) else 1 
            for v in extraction.get('entities', {}).values()
        )
        
        total_deadlines = len(extraction.get('deadlines', []))
        
        # Calculate average confidence
        confidences = [
            d.get('confidence', 0.5) 
            for d in extraction.get('deadlines', [])
        ]
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Update metadata
        extraction['extraction_metadata'] = {
            "total_entities": total_entities,
            "total_deadlines": total_deadlines,
            "average_confidence": avg_confidence,
            "extraction_warnings": extraction.get('extraction_metadata', {}).get('extraction_warnings', []),
            "model": "claude-3.5-sonnet",
            "timestamp": datetime.now().isoformat()
        }
        
        # Flag for review if needed
        needs_review = (
            avg_confidence < 0.85 or
            any(d.get('ambiguity_flag', False) for d in extraction.get('deadlines', [])) or
            total_deadlines == 0
        )
        
        extraction['document_context']['requires_attorney_review'] = needs_review
        
        return extraction
    
    def _empty_extraction(self) -> Dict:
        """Return empty extraction structure"""
        
        return {
            "entities": {
                "parties": [],
                "case_numbers": [],
                "judges": [],
                "courts": [],
                "attorneys": [],
                "citations": []
            },
            "deadlines": [],
            "document_context": {
                "type": "unknown",
                "requires_attorney_review": True
            },
            "extraction_metadata": {
                "total_entities": 0,
                "total_deadlines": 0,
                "average_confidence": 0.0,
                "extraction_warnings": ["Extraction failed"],
                "model": "claude-3.5-sonnet",
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def cleanup(self):
        """Clean up resources"""
        
        # Bedrock client doesn't need explicit cleanup
        logger.info("AI extractor cleanup completed")