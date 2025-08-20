"""
Deadline Calculator
Intelligent deadline parsing and calculation using Claude 4 Sonnet
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import json
import boto3
import holidays
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class DeadlineCalculator:
    """
    Calculates actual deadline dates using Claude 4 Sonnet for intelligent parsing
    Handles jurisdiction-aware rules and various counting methods
    """
    
    def __init__(self, bedrock_client=None, model_id: str = None):
        """
        Initialize deadline calculator with Claude 4 Sonnet
        
        Args:
            bedrock_client: AWS Bedrock client (will create if not provided)
            model_id: Claude model ID (defaults to Claude 4 Sonnet)
        """
        
        # Initialize Bedrock client
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
        
        # Model configuration
        self.model_id = model_id or "us.anthropic.claude-sonnet-4-20250514-v1:0"
        
        # Initialize holiday calendars for fallback
        current_year = datetime.now().year
        years_range = range(current_year - 1, current_year + 3)
        
        self.holiday_calendars = {
            'federal': holidays.US(years=years_range),
            'california': holidays.US(state='CA', years=years_range),
            'new_york': holidays.US(state='NY', years=years_range),
            'texas': holidays.US(state='TX', years=years_range)
        }
        
        logger.info(f"Deadline calculator initialized with model: {self.model_id}")
    
    async def calculate(self,
                       trigger_text: str,
                       period_text: str,
                       jurisdiction: str,
                       rule_type: str,
                       metadata: Optional[Dict] = None,
                       document_context: Optional[str] = None) -> Dict:
        """
        Calculate deadline date using Claude for intelligent parsing
        
        Args:
            trigger_text: Triggering event description
            period_text: Time period text (e.g., "14 days", "two weeks")
            jurisdiction: Legal jurisdiction
            rule_type: Type of legal rule
            metadata: Optional additional context
            document_context: Optional surrounding document text for better understanding
            
        Returns:
            Dictionary with calculated date and metadata
        """
        
        try:
            # Use Claude to parse and calculate if available
            if self.bedrock_client:
                result = await self._calculate_with_claude(
                    trigger_text,
                    period_text,
                    jurisdiction,
                    rule_type,
                    metadata,
                    document_context
                )
                
                # Validate Claude's response
                if self._validate_claude_result(result):
                    return result
                else:
                    logger.warning("Claude result validation failed, using fallback")
            
            # Fallback to rule-based calculation
            return self._fallback_calculation(
                trigger_text,
                period_text,
                jurisdiction,
                rule_type,
                metadata
            )
            
        except Exception as e:
            logger.error(f"Deadline calculation error: {e}")
            return self._error_result(str(e))
    
    async def _calculate_with_claude(self,
                                    trigger_text: str,
                                    period_text: str,
                                    jurisdiction: str,
                                    rule_type: str,
                                    metadata: Optional[Dict],
                                    document_context: Optional[str]) -> Dict:
        """Use Claude 4 Sonnet for intelligent deadline calculation"""
        
        # Prepare the prompt
        prompt = self._build_claude_prompt(
            trigger_text,
            period_text,
            jurisdiction,
            rule_type,
            metadata,
            document_context
        )
        
        # Call Claude
        try:
            response = await self._invoke_claude(prompt)
            
            # Parse Claude's response
            result = self._parse_claude_response(response)
            
            # Add metadata
            result['method'] = 'claude-ai'
            result['model'] = self.model_id
            result['jurisdiction'] = jurisdiction
            
            return result
            
        except Exception as e:
            logger.error(f"Claude invocation error: {e}")
            raise
    
    def _build_claude_prompt(self,
                           trigger_text: str,
                           period_text: str,
                           jurisdiction: str,
                           rule_type: str,
                           metadata: Optional[Dict],
                           document_context: Optional[str]) -> str:
        """Build prompt for Claude deadline calculation"""
        
        # Current date for reference
        today = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""You are an expert legal deadline calculator. Calculate the deadline date based on the following information.

CURRENT DATE: {today}

TRIGGERING EVENT: {trigger_text}

TIME PERIOD: {period_text}

JURISDICTION: {jurisdiction}

RULE TYPE: {rule_type}

ADDITIONAL CONTEXT:
{json.dumps(metadata, indent=2) if metadata else "None"}

DOCUMENT CONTEXT:
{document_context[:1000] if document_context else "None"}

Please analyze this legal deadline and provide your calculation in the following JSON format:

```json
{{
    "trigger_date": "YYYY-MM-DD",
    "trigger_date_reasoning": "explanation of how you determined the trigger date",
    "days_to_add": integer,
    "time_period_reasoning": "explanation of how you parsed the time period",
    "counting_method": "calendar_days|business_days|court_days",
    "counting_method_reasoning": "why this counting method applies",
    "exclude_weekends": boolean,
    "exclude_holidays": boolean,
    "deadline_date": "YYYY-MM-DD",
    "calculation_steps": ["step 1", "step 2", ...],
    "warnings": ["any warnings or caveats"],
    "confidence": float between 0 and 1,
    "legal_citations": ["relevant rules or statutes if known"]
}}
```

Consider the following when calculating:

1. TRIGGER DATE INTERPRETATION:
   - "date of this order" or "hereof" typically means today
   - "service" dates may require adding time for service completion
   - Look for specific dates mentioned in the text
   - Consider if the trigger is a future event

2. TIME PERIOD PARSING:
   - Convert written numbers to integers (e.g., "fourteen days" = 14)
   - Handle different units (days, weeks, months, years)
   - Watch for modifiers like "no less than", "within", "after"
   - Consider if the period is inclusive or exclusive

3. COUNTING METHODS by jurisdiction:
   - Federal: Often uses calendar days, excludes federal holidays for certain rules
   - California: May use court days (excluding weekends/holidays) for motions
   - New York: Often uses business days for commercial matters
   - Texas: Typically calendar days with weekend extension rules

4. SPECIAL RULES:
   - Motion responses often have specific day counts
   - Discovery deadlines may have different rules
   - Appeal deadlines are typically strict
   - Government parties may get extended deadlines

5. WEEKEND/HOLIDAY RULES:
   - If deadline falls on weekend/holiday, typically extends to next business day
   - Some jurisdictions count differently for short deadlines (<11 days)

Provide a thorough analysis and be conservative in your calculation when uncertain."""
        
        return prompt
    
    async def _invoke_claude(self, prompt: str) -> str:
        """Invoke Claude 4 Sonnet model"""
        
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
                "max_tokens": 2000,
                "messages": messages,
                "temperature": 0.1,  # Low temperature for consistency
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
    
    def _parse_claude_response(self, response: str) -> Dict:
        """Parse Claude's JSON response"""
        
        try:
            # Extract JSON from response (Claude may include explanation text)
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
            
            # Convert date strings to datetime objects
            if 'trigger_date' in result:
                result['trigger_date'] = datetime.strptime(result['trigger_date'], "%Y-%m-%d")
            
            if 'deadline_date' in result:
                result['date'] = datetime.strptime(result['deadline_date'], "%Y-%m-%d")
            else:
                # Calculate if not provided
                trigger = result.get('trigger_date', datetime.now())
                days = result.get('days_to_add', 30)
                result['date'] = self._apply_counting_method(
                    trigger,
                    days,
                    result.get('counting_method', 'calendar_days'),
                    result.get('exclude_weekends', False),
                    result.get('exclude_holidays', False)
                )
            
            # Ensure required fields
            result.setdefault('confidence', 0.8)
            result.setdefault('warnings', [])
            result.setdefault('days_added', result.get('days_to_add', 0))
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude JSON response: {e}")
            logger.debug(f"Response was: {response}")
            raise
        except Exception as e:
            logger.error(f"Error processing Claude response: {e}")
            raise
    
    def _apply_counting_method(self,
                              start_date: datetime,
                              days: int,
                              method: str,
                              exclude_weekends: bool,
                              exclude_holidays: bool) -> datetime:
        """Apply the counting method to calculate deadline"""
        
        if method == 'business_days' or method == 'court_days':
            return self._add_business_days(
                start_date, days, 'federal', exclude_holidays
            )
        elif method == 'calendar_days':
            if exclude_holidays:
                return self._add_calendar_days_exclude_holidays(
                    start_date, days, 'federal'
                )
            else:
                deadline = start_date + timedelta(days=days)
                # Extend past weekend if needed
                if exclude_weekends and deadline.weekday() >= 5:
                    deadline = self._extend_past_weekend(deadline)
                return deadline
        else:
            # Default to calendar days
            return start_date + timedelta(days=days)
    
    def _validate_claude_result(self, result: Dict) -> bool:
        """Validate Claude's calculation result"""
        
        try:
            # Check required fields
            required = ['date', 'trigger_date', 'days_added', 'confidence']
            for field in required:
                if field not in result:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check date validity
            if not isinstance(result['date'], datetime):
                logger.warning("Invalid deadline date type")
                return False
            
            # Check if date is reasonable (not too far in past or future)
            days_diff = (result['date'] - datetime.now()).days
            if days_diff < -365 or days_diff > 1095:  # 3 years
                logger.warning(f"Deadline date seems unreasonable: {days_diff} days")
                return False
            
            # Check confidence threshold
            if result.get('confidence', 0) < 0.5:
                logger.warning(f"Low confidence: {result.get('confidence')}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def _fallback_calculation(self,
                             trigger_text: str,
                             period_text: str,
                             jurisdiction: str,
                             rule_type: str,
                             metadata: Optional[Dict]) -> Dict:
        """Fallback to simple rule-based calculation"""
        
        # Parse trigger date (simple approach)
        trigger_date = self._simple_parse_trigger(trigger_text, metadata)
        
        # Parse period (extract number)
        import re
        numbers = re.findall(r'\d+', period_text)
        days = int(numbers[0]) if numbers else 30
        
        # Simple calculation
        deadline = self._add_business_days(
            trigger_date,
            days,
            jurisdiction.lower(),
            True  # exclude holidays
        )
        
        return {
            'date': deadline,
            'method': 'fallback_rules',
            'days_added': days,
            'trigger_date': trigger_date,
            'jurisdiction': jurisdiction,
            'warnings': ['Calculated using fallback rules - manual review recommended'],
            'confidence': 0.6
        }
    
    def _simple_parse_trigger(self, trigger_text: str, metadata: Optional[Dict]) -> datetime:
        """Simple trigger date parsing for fallback"""
        
        trigger_lower = trigger_text.lower()
        
        # Check metadata first
        if metadata:
            if 'document_date' in metadata:
                try:
                    return datetime.fromisoformat(metadata['document_date'])
                except:
                    pass
            if 'service_date' in metadata:
                try:
                    return datetime.fromisoformat(metadata['service_date'])
                except:
                    pass
        
        # Simple keyword matching
        if any(phrase in trigger_lower for phrase in ['today', 'this order', 'hereof', 'date of']):
            return datetime.now()
        
        if 'service' in trigger_lower:
            # Assume service takes 3 days
            return datetime.now() + timedelta(days=3)
        
        # Default to today
        return datetime.now()
    
    def _add_business_days(self,
                          start_date: datetime,
                          days: int,
                          jurisdiction: str,
                          exclude_holidays: bool) -> datetime:
        """Add business days excluding weekends and optionally holidays"""
        
        holidays_cal = self.holiday_calendars.get(jurisdiction, self.holiday_calendars['federal'])
        current_date = start_date
        days_added = 0
        
        while days_added < days:
            current_date += timedelta(days=1)
            
            # Skip weekends
            if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                continue
            
            # Skip holidays if required
            if exclude_holidays and current_date.date() in holidays_cal:
                continue
            
            days_added += 1
        
        return current_date
    
    def _add_calendar_days_exclude_holidays(self,
                                           start_date: datetime,
                                           days: int,
                                           jurisdiction: str) -> datetime:
        """Add calendar days excluding holidays"""
        
        holidays_cal = self.holiday_calendars.get(jurisdiction, self.holiday_calendars['federal'])
        current_date = start_date
        days_added = 0
        
        while days_added < days:
            current_date += timedelta(days=1)
            
            # Skip holidays but count weekends
            if current_date.date() in holidays_cal:
                continue
            
            days_added += 1
        
        return current_date
    
    def _extend_past_weekend(self, date: datetime) -> datetime:
        """Extend deadline past weekend if it falls on one"""
        
        weekday = date.weekday()
        
        if weekday == 5:  # Saturday
            return date + timedelta(days=2)
        elif weekday == 6:  # Sunday
            return date + timedelta(days=1)
        
        return date
    
    def _error_result(self, error_message: str) -> Dict:
        """Generate error result"""
        
        return {
            'date': datetime.now() + timedelta(days=30),  # Conservative 30 days
            'method': 'error_fallback',
            'days_added': 30,
            'trigger_date': datetime.now(),
            'warnings': [f"Calculation error: {error_message}", 
                        "Using conservative 30-day deadline - manual review required"],
            'confidence': 0.3,
            'error': error_message
        }