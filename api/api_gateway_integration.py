"""
API Gateway Integration Layer
Provides REST API endpoints for legal document processing system
"""

import json
import boto3
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import LegalDocumentPipeline, PipelineConfig, ProcessingResult
from src.utils.logger import get_logger

# Initialize AWS services
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

# Configure logging
logger = get_logger(__name__)

@dataclass
class APIResponse:
    """Standardized API response structure"""
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class APIGatewayHandler:
    """
    Main handler for API Gateway integration
    Routes requests to appropriate processing functions
    """
    
    def __init__(self):
        self.deadlines_table = dynamodb.Table('legal-deadlines')
        self.audit_table = dynamodb.Table('audit-trail')
        self.api_keys_table = dynamodb.Table('api-keys')
        
        # Initialize pipeline with configuration
        self.pipeline_config = PipelineConfig(
            opensearch_domain=os.environ.get('OPENSEARCH_DOMAIN', 'legal-docs.us-east-1.es.amazonaws.com'),
            bedrock_region=os.environ.get('BEDROCK_REGION', 'us-east-1'),
            enable_pii_masking=True,
            batch_size=5,
            max_retries=3
        )
        
        self.pipeline = LegalDocumentPipeline(self.pipeline_config)
    
    def lambda_handler(self, event: Dict, context) -> Dict:
        """
        Main Lambda handler for API Gateway events
        """
        
        try:
            # Extract request details
            path = event.get('path', '')
            method = event.get('httpMethod', '')
            headers = event.get('headers', {})
            body = json.loads(event.get('body', '{}')) if event.get('body') else {}
            
            # Authenticate request
            if not self._authenticate(headers):
                return self._error_response(401, "Unauthorized")
            
            # Route to appropriate handler
            if path == '/api/v1/documents/process' and method == 'POST':
                return self.process_document(body, headers)
            
            elif path.startswith('/api/v1/deadlines/') and method == 'GET':
                deadline_id = event['pathParameters']['id']
                return self.get_deadline(deadline_id)
            
            elif path == '/api/v1/deadlines/verify' and method == 'POST':
                return self.verify_deadline(body)
            
            elif path.startswith('/api/v1/audit/') and method == 'GET':
                document_id = event['pathParameters']['document_id']
                return self.get_audit_trail(document_id)
            
            elif path == '/api/v1/calendar/sync' and method == 'POST':
                return self.sync_calendar(body)
            
            elif path == '/api/v1/webhooks/register' and method == 'POST':
                return self.register_webhook(body)
            
            else:
                return self._error_response(404, "Endpoint not found")
                
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return self._error_response(500, "Internal server error")
    
    def process_document(self, body: Dict, headers: Dict) -> Dict:
        """
        Process a new legal document using the modular pipeline
        
        Expected body:
        {
            "document_url": "s3://bucket/path/to/document.pdf",
            "document_type": "court_order",
            "case_number": "21-CV-12345",
            "priority": "high",
            "metadata": {...}
        }
        """
        
        # Validate input
        if not body.get('document_url'):
            return self._error_response(400, "document_url is required")
        
        # Generate processing ID
        processing_id = f"proc_{datetime.now().timestamp()}"
        
        # Download document from S3 to temporary location
        document_url = body['document_url']
        if document_url.startswith('s3://'):
            # Parse S3 URL
            parts = document_url.replace('s3://', '').split('/', 1)
            if len(parts) != 2:
                return self._error_response(400, "Invalid S3 URL format")
            
            bucket, key = parts
            local_path = f"/tmp/{processing_id}.pdf"
            
            try:
                s3.download_file(bucket, key, local_path)
            except Exception as e:
                logger.error(f"Failed to download from S3: {str(e)}")
                return self._error_response(500, "Failed to download document")
        else:
            return self._error_response(400, "Only S3 URLs are supported")
        
        # Process document asynchronously
        try:
            # Run async pipeline in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            metadata = {
                "document_type": body.get('document_type', 'unknown'),
                "case_number": body.get('case_number'),
                "priority": body.get('priority', 'normal'),
                "processing_id": processing_id,
                "api_key": headers.get('x-api-key')
            }
            
            result = loop.run_until_complete(
                self.pipeline.process_document(local_path, metadata)
            )
            
            # Store results in DynamoDB
            self._store_results(processing_id, result, metadata)
            
            # Log to audit trail
            self._log_audit({
                "action": "document_processed",
                "processing_id": processing_id,
                "document_url": body['document_url'],
                "timestamp": datetime.now().isoformat(),
                "api_key": headers.get('x-api-key'),
                "success": result.success
            })
            
            # Clean up temporary file
            if os.path.exists(local_path):
                os.remove(local_path)
            
            # Return response
            return self._success_response({
                "processing_id": processing_id,
                "status": "completed" if result.success else "failed",
                "deadlines_found": len(result.deadlines),
                "confidence_scores": [d.confidence for d in result.deadlines],
                "routing_decisions": result.routing_decisions,
                "poll_url": f"/api/v1/deadlines/{processing_id}"
            })
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return self._error_response(500, f"Processing failed: {str(e)}")
        finally:
            # Clean up
            if 'loop' in locals():
                loop.close()
    
    def get_deadline(self, deadline_id: str) -> Dict:
        """
        Retrieve extracted deadline information
        """
        
        try:
            response = self.deadlines_table.get_item(
                Key={'deadline_id': deadline_id}
            )
            
            if 'Item' not in response:
                return self._error_response(404, "Deadline not found")
            
            deadline = response['Item']
            
            # Format response based on consumer needs
            formatted_response = {
                "deadline_id": deadline['deadline_id'],
                "status": deadline['status'],
                "extracted_deadlines": deadline.get('deadlines', []),
                "entities": deadline.get('entities', {}),
                "confidence_scores": deadline.get('confidence_scores', {}),
                "requires_review": deadline.get('requires_review', False),
                "processing_time": deadline.get('processing_time'),
                "created_at": deadline.get('created_at')
            }
            
            return self._success_response(formatted_response)
            
        except Exception as e:
            logger.error(f"Error retrieving deadline: {str(e)}")
            return self._error_response(500, "Error retrieving deadline")
    
    def verify_deadline(self, body: Dict) -> Dict:
        """
        Submit human verification for extracted deadlines
        
        Expected body:
        {
            "deadline_id": "...",
            "verification": {
                "deadline_index": 0,
                "is_correct": true,
                "corrected_date": "2024-04-15",
                "corrected_description": "...",
                "reviewer_id": "attorney_123",
                "notes": "..."
            }
        }
        """
        
        deadline_id = body.get('deadline_id')
        verification = body.get('verification', {})
        
        if not deadline_id or not verification:
            return self._error_response(400, "deadline_id and verification required")
        
        # Update deadline with verification
        try:
            self.deadlines_table.update_item(
                Key={'deadline_id': deadline_id},
                UpdateExpression="""
                    SET verifications = list_append(
                        if_not_exists(verifications, :empty_list), 
                        :verification
                    ),
                    last_verified = :timestamp,
                    verification_status = :status
                """,
                ExpressionAttributeValues={
                    ':empty_list': [],
                    ':verification': [verification],
                    ':timestamp': datetime.now().isoformat(),
                    ':status': 'verified'
                }
            )
            
            # Update OpenSearch with verified pattern for future learning
            # This would be done through the pipeline's RAG engine
            asyncio.run(self.pipeline.rag_engine.update_verified_pattern(
                deadline_id, 
                verification
            ))
            
            # Log verification
            self._log_audit({
                "action": "deadline_verified",
                "deadline_id": deadline_id,
                "reviewer_id": verification.get('reviewer_id'),
                "timestamp": datetime.now().isoformat()
            })
            
            return self._success_response({
                "message": "Verification recorded successfully",
                "deadline_id": deadline_id
            })
            
        except Exception as e:
            logger.error(f"Error recording verification: {str(e)}")
            return self._error_response(500, "Error recording verification")
    
    def get_audit_trail(self, document_id: str) -> Dict:
        """
        Retrieve complete audit trail for a document
        """
        
        try:
            response = self.audit_table.query(
                KeyConditionExpression='document_id = :doc_id',
                ExpressionAttributeValues={
                    ':doc_id': document_id
                }
            )
            
            audit_entries = response.get('Items', [])
            
            return self._success_response({
                "document_id": document_id,
                "audit_trail": audit_entries,
                "total_events": len(audit_entries)
            })
            
        except Exception as e:
            logger.error(f"Error retrieving audit trail: {str(e)}")
            return self._error_response(500, "Error retrieving audit trail")
    
    def sync_calendar(self, body: Dict) -> Dict:
        """
        Sync extracted deadlines with external calendar system
        
        Expected body:
        {
            "deadline_ids": ["..."],
            "calendar_type": "google|outlook|ical",
            "calendar_config": {
                "calendar_id": "...",
                "auth_token": "..."
            },
            "sync_options": {
                "create_reminders": true,
                "reminder_days": [7, 3, 1],
                "include_description": true
            }
        }
        """
        
        deadline_ids = body.get('deadline_ids', [])
        calendar_type = body.get('calendar_type')
        
        if not deadline_ids or not calendar_type:
            return self._error_response(400, "deadline_ids and calendar_type required")
        
        # Retrieve deadlines
        deadlines_to_sync = []
        for deadline_id in deadline_ids:
            response = self.deadlines_table.get_item(
                Key={'deadline_id': deadline_id}
            )
            if 'Item' in response:
                deadlines_to_sync.append(response['Item'])
        
        # Format for calendar system
        calendar_events = self._format_for_calendar(
            deadlines_to_sync,
            calendar_type,
            body.get('sync_options', {})
        )
        
        # Invoke calendar sync Lambda
        sync_result = lambda_client.invoke(
            FunctionName=f'calendar-sync-{calendar_type}',
            InvocationType='RequestResponse',
            Payload=json.dumps({
                "events": calendar_events,
                "config": body.get('calendar_config', {})
            })
        )
        
        sync_response = json.loads(sync_result['Payload'].read())
        
        return self._success_response({
            "synced_count": len(calendar_events),
            "calendar_type": calendar_type,
            "sync_result": sync_response
        })
    
    def register_webhook(self, body: Dict) -> Dict:
        """
        Register webhook for async notifications
        
        Expected body:
        {
            "webhook_url": "https://...",
            "events": ["processing_complete", "deadline_extracted", "error"],
            "auth": {
                "type": "bearer",
                "token": "..."
            }
        }
        """
        
        webhook_url = body.get('webhook_url')
        events = body.get('events', [])
        
        if not webhook_url or not events:
            return self._error_response(400, "webhook_url and events required")
        
        # Store webhook configuration
        webhook_id = f"webhook_{datetime.now().timestamp()}"
        
        self.api_keys_table.update_item(
            Key={'api_key': body.get('api_key')},
            UpdateExpression="SET webhooks = list_append(if_not_exists(webhooks, :empty), :webhook)",
            ExpressionAttributeValues={
                ':empty': [],
                ':webhook': [{
                    "webhook_id": webhook_id,
                    "url": webhook_url,
                    "events": events,
                    "auth": body.get('auth', {}),
                    "created_at": datetime.now().isoformat()
                }]
            }
        )
        
        return self._success_response({
            "webhook_id": webhook_id,
            "registered_events": events
        })
    
    def _authenticate(self, headers: Dict) -> bool:
        """
        Authenticate API request using API key
        """
        
        api_key = headers.get('x-api-key')
        
        if not api_key:
            return False
        
        try:
            # Validate API key
            response = self.api_keys_table.get_item(
                Key={'api_key': api_key}
            )
            return 'Item' in response and response['Item'].get('active', False)
        except:
            return False
    
    def _store_results(self, processing_id: str, result: ProcessingResult, metadata: Dict):
        """
        Store processing results in DynamoDB
        """
        
        try:
            item = {
                'deadline_id': processing_id,
                'status': 'completed' if result.success else 'failed',
                'deadlines': [
                    {
                        'date': d.date.isoformat() if d.date else None,
                        'description': d.description,
                        'trigger_event': d.trigger_event,
                        'jurisdiction': d.jurisdiction,
                        'confidence': d.confidence,
                        'source_text': d.source_text[:500]  # Truncate for storage
                    }
                    for d in result.deadlines
                ],
                'entities': result.entities,
                'confidence_scores': [d.confidence for d in result.deadlines],
                'requires_review': any(d.confidence < 0.95 for d in result.deadlines),
                'processing_time': result.processing_time,
                'created_at': datetime.now().isoformat(),
                'metadata': metadata
            }
            
            if result.error:
                item['error'] = str(result.error)
            
            self.deadlines_table.put_item(Item=item)
            
        except Exception as e:
            logger.error(f"Failed to store results: {str(e)}")
    
    def _format_for_calendar(self, deadlines: List[Dict], calendar_type: str, options: Dict) -> List[Dict]:
        """
        Format deadlines for specific calendar system
        """
        
        events = []
        
        for deadline_data in deadlines:
            for deadline in deadline_data.get('deadlines', []):
                event = {
                    "title": f"Legal Deadline: {deadline['description']}",
                    "date": deadline['date'],
                    "all_day": True
                }
                
                if options.get('include_description'):
                    event["description"] = f"""
                    Case: {deadline_data.get('metadata', {}).get('case_number', 'N/A')}
                    Action Required: {deadline['description']}
                    Confidence: {deadline.get('confidence', 0):.0%}
                    Source: {deadline.get('source_text', '')[:200]}
                    """
                
                if options.get('create_reminders'):
                    event["reminders"] = [
                        {"days_before": days}
                        for days in options.get('reminder_days', [7, 3, 1])
                    ]
                
                events.append(event)
        
        return events
    
    def _log_audit(self, entry: Dict):
        """
        Log action to audit trail
        """
        
        entry['timestamp'] = datetime.now().isoformat()
        self.audit_table.put_item(Item=entry)
    
    def _success_response(self, data: Dict) -> Dict:
        """
        Format successful API response
        """
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
        }
    
    def _error_response(self, status_code: int, message: str) -> Dict:
        """
        Format error API response
        """
        
        return {
            'statusCode': status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': message,
                'timestamp': datetime.now().isoformat()
            })
        }

# Lambda handler
handler = APIGatewayHandler()
lambda_handler = handler.lambda_handler