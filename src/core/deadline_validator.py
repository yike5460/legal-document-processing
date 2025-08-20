"""
Deadline Validator
Validates extracted deadlines using Claude Sonnet 4 for intelligent pattern recognition,
semantic analysis, and jurisdiction rule validation
"""

import logging
from typing import Dict, List, Optional
import json
import boto3
from botocore.exceptions import ClientError
import re

logger = logging.getLogger(__name__)

class DeadlineValidator:
    """
    Validates extracted deadline phrases using Claude Sonnet 4 AI.
    Performs intelligent pattern matching, semantic validation, and jurisdiction rule checking.
    """
    
    def __init__(self, bedrock_client=None, model_id: str = None, rag_engine=None):
        """
        Initialize validator with Claude Sonnet 4 for intelligent validation
        
        Args:
            bedrock_client: AWS Bedrock client (will create if not provided)
            model_id: Claude model ID (defaults to Claude Sonnet 4)
            rag_engine: Optional RAG engine for pattern matching
        """
        self.rag_engine = rag_engine
        
        # Initialize Bedrock client for Claude
        if bedrock_client:
            self.bedrock_client = bedrock_client
        else:
            try:
                self.bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name='us-east-1'
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Bedrock client: {e}")
                self.bedrock_client = None
        
        # Model configuration - using Claude Sonnet 4
        self.model_id = model_id or "us.anthropic.claude-sonnet-4-20250514-v1:0"
        
        # Fallback patterns for when Claude is unavailable
        self.fallback_patterns = {
            "court_order": ["SHALL", "MUST", "ORDERED", "REQUIRED", "DIRECTED"],
            "motion": ["DUE", "DEADLINE", "OPPOSITION", "RESPONSE"],
            "notice": ["EFFECTIVE", "EXPIRES", "VALID", "PERIOD"],
            "complaint": ["ANSWER", "RESPONSE", "SUMMONS", "DEFENDANT"],
            "discovery": ["RESPOND", "PRODUCE", "INTERROGATORIES", "DEPOSITION"]
        }
        
        logger.info(f"Deadline validator initialized with model: {self.model_id}")
    
    async def validate(self, 
                      deadline: Dict,
                      document_type: str,
                      jurisdiction: Optional[str] = None,
                      document_context: Optional[str] = None) -> Dict:
        """
        Validate a deadline using Claude Sonnet 4 for intelligent analysis.
        
        Args:
            deadline: Extracted deadline information
            document_type: Type of legal document
            jurisdiction: Legal jurisdiction
            document_context: Optional surrounding document text for better understanding
            
        Returns:
            Validation result with status, confidence, and detailed analysis
        """
        
        validation_result = {
            "is_valid": False,
            "confidence": 0.0,
            "matched_patterns": [],
            "warnings": [],
            "suggestions": [],
            "ai_analysis": {},
            "method": "fallback"
        }
        
        try:
            # Use Claude for intelligent validation if available
            if self.bedrock_client:
                claude_result = await self._validate_with_claude(
                    deadline,
                    document_type,
                    jurisdiction,
                    document_context
                )
                
                if claude_result and self._validate_claude_response(claude_result):
                    validation_result.update(claude_result)
                    validation_result["method"] = "claude-ai"
                else:
                    logger.warning("Claude validation failed, using fallback")
                    validation_result = self._fallback_validation(
                        deadline, document_type, jurisdiction
                    )
            else:
                # Use fallback validation
                validation_result = self._fallback_validation(
                    deadline, document_type, jurisdiction
                )
            
            # Check with RAG if available
            if self.rag_engine:
                rag_check = await self._check_rag_validity(deadline, document_type)
                # Adjust confidence based on RAG results
                if rag_check["confidence"] > 0.8:
                    validation_result["confidence"] = min(
                        0.99, 
                        validation_result["confidence"] * 1.1
                    )
                validation_result["rag_similarity"] = rag_check["confidence"]
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            validation_result = self._fallback_validation(
                deadline, document_type, jurisdiction
            )
            validation_result["warnings"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    async def _validate_with_claude(self,
                                   deadline: Dict,
                                   document_type: str,
                                   jurisdiction: Optional[str],
                                   document_context: Optional[str]) -> Dict:
        """Use Claude Sonnet 4 for intelligent deadline validation"""
        
        # Build comprehensive validation prompt
        prompt = self._build_validation_prompt(
            deadline,
            document_type,
            jurisdiction,
            document_context
        )
        
        try:
            # Call Claude
            response = await self._invoke_claude(prompt)
            
            # Parse Claude's response
            result = self._parse_claude_validation(response)
            
            # Add metadata
            result["model"] = self.model_id
            result["document_type"] = document_type
            if jurisdiction:
                result["jurisdiction"] = jurisdiction
            
            return result
            
        except Exception as e:
            logger.error(f"Claude validation error: {e}")
            raise
    
    def _build_validation_prompt(self,
                                deadline: Dict,
                                document_type: str,
                                jurisdiction: Optional[str],
                                document_context: Optional[str]) -> str:
        """Build comprehensive prompt for Claude deadline validation"""
        
        prompt = f"""You are an expert legal deadline validator with deep knowledge of civil procedure and jurisdiction-specific rules. 
Analyze the following extracted deadline and provide a comprehensive validation assessment.

DEADLINE INFORMATION:
- Action: {deadline.get('action', 'Not specified')}
- Trigger Event: {deadline.get('trigger_event', 'Not specified')}
- Time Period: {deadline.get('time_period', 'Not specified')}
- Specific Date: {deadline.get('specific_date', 'Not specified')}
- Ambiguity Flag: {deadline.get('ambiguity_flag', False)}
- Raw Text: {deadline.get('raw_text', 'Not provided')}

DOCUMENT TYPE: {document_type}

JURISDICTION: {jurisdiction or 'Not specified'}

DOCUMENT CONTEXT:
{document_context[:2000] if document_context else 'Not provided'}

Please perform a comprehensive validation analysis and provide your assessment in the following JSON format:

```json
{{
    "is_valid": boolean,
    "confidence": float (0.0 to 1.0),
    "pattern_analysis": {{
        "recognized_pattern": "description of the legal pattern identified",
        "pattern_type": "statutory|court_rule|procedural|administrative|contractual",
        "binding_language": ["list of binding terms found"],
        "matched_patterns": ["list of specific pattern matches"]
    }},
    "semantic_analysis": {{
        "action_clarity": "clear|ambiguous|missing",
        "action_type": "filing|response|appearance|discovery|appeal|motion|other",
        "trigger_clarity": "clear|ambiguous|missing",
        "trigger_type": "order_date|service|filing|judgment|notice|other",
        "time_clarity": "clear|ambiguous|missing",
        "time_type": "calendar_days|business_days|court_days|specific_date|unclear",
        "completeness": float (0.0 to 1.0),
        "ambiguity_issues": ["list of ambiguous elements"]
    }},
    "jurisdiction_validation": {{
        "complies_with_rules": boolean,
        "applicable_rules": ["list of applicable procedural rules"],
        "time_calculation_method": "calendar|business|court_days",
        "minimum_time_met": boolean,
        "maximum_time_reasonable": boolean,
        "weekend_holiday_adjustment": "required|not_required|unclear",
        "specific_requirements": ["jurisdiction-specific requirements or notes"]
    }},
    "legal_validity": {{
        "enforceable": boolean,
        "mandatory_vs_directory": "mandatory|directory|unclear",
        "consequences_specified": boolean,
        "appeal_rights_mentioned": boolean,
        "modification_possible": boolean
    }},
    "warnings": ["list of validation warnings"],
    "suggestions": ["list of suggestions for improvement or clarification"],
    "red_flags": ["critical issues that require immediate attorney review"],
    "precedent_references": ["relevant case law or statutory references if known"]
}}
```

Consider the following in your analysis:

1. PATTERN RECOGNITION:
   - Is this a standard legal deadline pattern?
   - Does it match typical {document_type} deadlines?
   - Are there binding/mandatory terms present?
   
2. SEMANTIC VALIDATION:
   - Is the action to be taken clearly specified?
   - Is the triggering event unambiguous?
   - Is the time period calculable?
   - Are there conditional or contingent elements?
   
3. JURISDICTION RULES for {jurisdiction or 'the relevant jurisdiction'}:
   - Does the deadline comply with local rules?
   - What calculation method applies?
   - Are there minimum/maximum time limits?
   - How are weekends/holidays handled?
   
4. LEGAL ENFORCEABILITY:
   - Is this deadline legally binding?
   - What are the consequences of missing it?
   - Can it be extended or modified?
   
5. RED FLAGS to identify:
   - Impossibly short deadlines
   - Ambiguous trigger events
   - Missing critical information
   - Conflicting deadline information
   - Non-standard or suspicious patterns

Provide a thorough analysis focusing on legal accuracy and practical enforceability."""
        
        return prompt
    
    async def _invoke_claude(self, prompt: str) -> str:
        """Invoke Claude Sonnet 4 model"""
        
        if not self.bedrock_client:
            raise ValueError("Bedrock client not initialized")
        
        try:
            # Prepare the request
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 3000,
                "messages": messages,
                "temperature": 0.1,  # Low temperature for consistent validation
                "top_p": 0.95
            })
            
            # Invoke the model
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            # Parse response
            response_body = json.loads(response.get('body').read())
            return response_body.get('content', [{}])[0].get('text', '')
            
        except ClientError as e:
            logger.error(f"AWS Bedrock error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error invoking Claude: {e}")
            raise
    
    def _parse_claude_validation(self, response: str) -> Dict:
        """Parse Claude's validation response"""
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in Claude response")
            
            # Parse JSON
            result = json.loads(json_str)
            
            # Ensure required fields
            result.setdefault('is_valid', False)
            result.setdefault('confidence', 0.5)
            result.setdefault('warnings', [])
            result.setdefault('suggestions', [])
            
            # Extract matched patterns for compatibility
            if 'pattern_analysis' in result:
                result['matched_patterns'] = result['pattern_analysis'].get('matched_patterns', [])
            
            # Add AI analysis flag
            result['ai_analysis'] = {
                'pattern': result.get('pattern_analysis', {}),
                'semantic': result.get('semantic_analysis', {}),
                'jurisdiction': result.get('jurisdiction_validation', {}),
                'legal': result.get('legal_validity', {})
            }
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude JSON response: {e}")
            logger.debug(f"Response was: {response}")
            raise
        except Exception as e:
            logger.error(f"Error processing Claude response: {e}")
            raise
    
    def _validate_claude_response(self, result: Dict) -> bool:
        """Validate Claude's response structure"""
        
        try:
            # Check required fields
            required = ['is_valid', 'confidence']
            for field in required:
                if field not in result:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check confidence is reasonable
            if not (0 <= result['confidence'] <= 1):
                logger.warning(f"Invalid confidence: {result['confidence']}")
                return False
            
            # Check for at least some analysis
            if not any([
                result.get('pattern_analysis'),
                result.get('semantic_analysis'),
                result.get('jurisdiction_validation')
            ]):
                logger.warning("No analysis provided")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Response validation error: {e}")
            return False
    
    def _fallback_validation(self,
                            deadline: Dict,
                            document_type: str,
                            jurisdiction: Optional[str]) -> Dict:
        """Fallback validation using pattern matching"""
        
        result = {
            "is_valid": False,
            "confidence": 0.5,
            "matched_patterns": [],
            "warnings": [],
            "suggestions": ["Manual review recommended - AI validation unavailable"],
            "method": "fallback"
        }
        
        # Check for required components
        if not deadline.get("action"):
            result["warnings"].append("No action specified")
            result["confidence"] *= 0.5
        
        if not deadline.get("time_period") and not deadline.get("specific_date"):
            result["warnings"].append("No time period or date specified")
            result["confidence"] *= 0.5
        
        # Check for binding language
        deadline_text = f"{deadline.get('action', '')} {deadline.get('trigger_event', '')}".upper()
        binding_words = self.fallback_patterns.get(document_type, [])
        
        if any(word in deadline_text for word in binding_words):
            result["is_valid"] = True
            result["confidence"] = min(0.7, result["confidence"] * 1.3)
            result["matched_patterns"].append("Binding language detected")
        
        # Basic jurisdiction check
        if jurisdiction and deadline.get("time_period"):
            days_match = re.search(r"(\d+)\s*days?", deadline.get("time_period", ""))
            if days_match:
                days = int(days_match.group(1))
                if days < 3:
                    result["warnings"].append("Unusually short deadline")
                    result["confidence"] *= 0.7
                elif days > 180:
                    result["warnings"].append("Unusually long deadline")
                    result["confidence"] *= 0.8
        
        # Set validity based on confidence
        if result["confidence"] >= 0.6:
            result["is_valid"] = True
        
        return result
    
    async def _check_rag_validity(self, deadline: Dict, document_type: str) -> Dict:
        """Check deadline against RAG database of validated patterns"""
        
        result = {
            "is_valid": True,
            "confidence": 0.7
        }
        
        try:
            # Build query for RAG search
            query = f"{document_type} {deadline.get('action', '')} {deadline.get('time_period', '')}"
            
            # Search for similar validated patterns
            similar = await self.rag_engine.find_similar(
                query=query,
                index="verified-qa-pairs",
                limit=5
            )
            
            if similar:
                # Check similarity scores
                top_score = similar[0].get("similarity_score", 0)
                
                if top_score > 0.9:
                    result["confidence"] = 0.95
                elif top_score > 0.8:
                    result["confidence"] = 0.85
                elif top_score > 0.7:
                    result["confidence"] = 0.75
                else:
                    result["confidence"] = 0.6
                    
                # Check if similar patterns were verified
                verified_count = sum(
                    1 for s in similar 
                    if s.get("human_verified", False)
                )
                
                if verified_count > 0:
                    result["confidence"] = min(0.99, result["confidence"] * 1.1)
            else:
                result["confidence"] = 0.5
                
        except Exception as e:
            logger.error(f"RAG validation error: {e}")
            result["confidence"] = 0.6
        
        return result
    
    async def batch_validate(self,
                            deadlines: List[Dict],
                            document_type: str,
                            jurisdiction: Optional[str] = None,
                            document_context: Optional[str] = None) -> List[Dict]:
        """
        Validate multiple deadlines using Claude.
        
        Args:
            deadlines: List of extracted deadlines
            document_type: Type of legal document
            jurisdiction: Legal jurisdiction
            document_context: Optional document context
            
        Returns:
            Deadlines with validation results
        """
        
        validated = []
        
        for deadline in deadlines:
            # Run validation
            validation = await self.validate(
                deadline, 
                document_type, 
                jurisdiction,
                document_context
            )
            
            # Add validation results to deadline
            deadline["validation"] = validation
            deadline["is_valid"] = validation["is_valid"]
            deadline["validation_confidence"] = validation["confidence"]
            
            # Add warnings if any
            if validation["warnings"]:
                deadline["warnings"] = deadline.get("warnings", []) + validation["warnings"]
            
            # Add AI analysis if available
            if "ai_analysis" in validation:
                deadline["ai_analysis"] = validation["ai_analysis"]
            
            validated.append(deadline)
        
        return validated
    
    def get_validation_statistics(self, deadlines: List[Dict]) -> Dict:
        """Get statistics about validation results"""
        
        total = len(deadlines)
        if total == 0:
            return {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "average_confidence": 0.0,
                "ai_validated": 0,
                "fallback_validated": 0
            }
        
        valid = sum(1 for d in deadlines if d.get("is_valid", False))
        invalid = total - valid
        
        confidences = [
            d.get("validation_confidence", 0) 
            for d in deadlines
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Count validation methods
        ai_validated = sum(
            1 for d in deadlines 
            if d.get("validation", {}).get("method") == "claude-ai"
        )
        fallback_validated = sum(
            1 for d in deadlines 
            if d.get("validation", {}).get("method") == "fallback"
        )
        
        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "valid_percentage": (valid / total) * 100,
            "average_confidence": avg_confidence,
            "high_confidence": sum(1 for c in confidences if c >= 0.9),
            "medium_confidence": sum(1 for c in confidences if 0.7 <= c < 0.9),
            "low_confidence": sum(1 for c in confidences if c < 0.7),
            "ai_validated": ai_validated,
            "fallback_validated": fallback_validated,
            "ai_validation_rate": (ai_validated / total) * 100 if total > 0 else 0
        }