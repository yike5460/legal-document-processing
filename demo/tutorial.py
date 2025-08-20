"""
Legal Document Processing System - Comprehensive Demo Tutorial
Demonstrates all core components and features
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
import sys
from dotenv import load_dotenv

# Load environment variables from .env.local file
load_dotenv('.env.local')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.main_pipeline import LegalDocumentPipeline, PipelineConfig
from src.utils.logger import setup_logger, AuditLogger

# Setup logging
logger = setup_logger(__name__)
audit_logger = AuditLogger()

def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

async def demo_basic_processing():
    """Demonstrate basic document processing"""
    
    print_section("DEMO 1: Basic Document Processing\n")
    
    # Initialize pipeline with default config
    config = PipelineConfig(
        enable_pii_masking=True,
        enable_rag=True,
        confidence_threshold_auto=0.95
    )
    
    pipeline = LegalDocumentPipeline(config)
    
    # Sample document path (you'll need to provide an actual document)
    document_path = "samples/court_order_sample.pdf"
    
    print(f"Processing document: {document_path}")
    
    # Process document
    result = await pipeline.process_document(
        document_path,
        metadata={
            "case_number": "21-CV-12345",
            "document_date": datetime.now().isoformat(),
            "jurisdiction": "federal"
        }
    )
    
    # Display results
    print(f"\nProcessing Status: {result.status}")
    print(f"Processing Time: {result.processing_time:.2f} seconds")
    print(f"Document ID: {result.document_id}")
    
    print("\n--- Extracted Entities ---")
    for entity_type, entities in result.entities.items():
        if entities:
            print(f"{entity_type}: {entities}")
    
    print("\n--- Extracted Deadlines ---")
    for deadline in result.deadlines:
        print(f"• {deadline.get('action', 'Unknown action')}")
        print(f"  Date: {deadline.get('calculated_date', 'Not calculated')}")
        print(f"  Confidence: {deadline.get('confidence', 0):.2%}")
        print(f"  Routing: {deadline.get('routing', 'Unknown')}")
    
    print("\n--- Statistics ---")
    stats = pipeline.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Cleanup
    pipeline.cleanup()

async def demo_pii_masking():
    """Demonstrate PII masking capabilities"""
    
    print_section("DEMO 2: PII Masking")
    
    from src.utils.pii_masker import PIIMasker
    
    # Sample legal text with PII
    sample_text = """
    UNITED STATES DISTRICT COURT
    Case No. 21-CV-12345
    
    John Smith (SSN: 123-45-6789)
    123 Main Street, New York, NY 10001
    Phone: (555) 123-4567
    Email: jsmith@email.com
    
    Plaintiff,
    
    v.
    
    ABC Corporation
    Defendant.
    
    ORDER TO RESPOND
    
    Defendant shall file a response within 14 days of this order.
    Account Number: 1234567890
    Credit Card: 4111-1111-1111-1111
    """

    print("Original Text:\n", sample_text)
    
    # Initialize masker
    masker = PIIMasker()
    
    # Mask PII
    masked_text, masking_map = masker.mask_document(sample_text)

    print("\nMasked Text:\n", masked_text)
    
    print("\nMasking Map (first 3 entries):")
    for token, info in list(masking_map.items())[:3]:
        print(f"  {token}: {info['type']}")
    
    # Validate masking
    validation = masker.validate_masking(sample_text, masked_text)
    print(f"\nMasking Validation:")
    print(f"  Success: {validation['success']}")
    print(f"  Entities Masked: {validation['entities_masked']}")
    print(f"  Potential Leaks: {len(validation['potential_leaks'])}")
    
    # Unmask for demonstration
    unmasked = masker.unmask_document(masked_text, masking_map)
    print(f"\nUnmasking successful: {unmasked == sample_text}")

async def demo_deadline_calculation():
    """Demonstrate deadline calculation"""
    
    print_section("DEMO 3: Deadline Calculation")
    
    from src.core.deadline_calculator import DeadlineCalculator
    
    calculator = DeadlineCalculator()
    
    # Test cases
    test_cases = [
        {
            "trigger": "date of this order",
            "period": "14 days",
            "jurisdiction": "federal",
            "rule_type": "motion_response"
        },
        {
            "trigger": "service of complaint",
            "period": "30 days",
            "jurisdiction": "california",
            "rule_type": "discovery_response"
        },
        {
            "trigger": "judgment entered",
            "period": "30 days",
            "jurisdiction": "federal",
            "rule_type": "appeal_notice"
        }
    ]
    
    for case in test_cases:
        print(f"\nCalculating: {case['period']} from '{case['trigger']}'")
        print(f"Jurisdiction: {case['jurisdiction']}")
        print(f"Rule Type: {case['rule_type']}")
        
        result = await calculator.calculate(
            trigger_text=case['trigger'],
            period_text=case['period'],
            jurisdiction=case['jurisdiction'],
            rule_type=case['rule_type']
        )
        
        print(f"Result:")
        print(f"  Date: {result['date'].strftime('%Y-%m-%d')}")
        print(f"  Method: {result['method']}")
        print(f"  Warnings: {result.get('warnings', [])}")

async def demo_confidence_scoring():
    """Demonstrate confidence scoring"""
    
    print_section("DEMO 4: Confidence Scoring")
    
    from src.core.confidence_scorer import ConfidenceScorer
    
    scorer = ConfidenceScorer()
    
    # Sample deadlines with varying quality
    deadlines = [
        {
            "action": "file opposition to motion",
            "trigger_event": "date of this order",
            "time_period": "14 days",
            "confidence": 0.9,  # From Claude
            "ambiguity_flag": False
        },
        {
            "action": "respond",  # Vague action
            "trigger_event": "unknown",
            "time_period": "some days",  # Vague period
            "confidence": 0.4,
            "ambiguity_flag": True
        },
        {
            "action": "file notice of appeal",
            "trigger_event": "judgment entered",
            "time_period": "30 days",
            "confidence": 0.95,
            "ambiguity_flag": False,
            "priority": "critical"
        }
    ]
    
    ocr_confidence = 0.92
    
    print("Scoring deadlines...")
    
    for idx, deadline in enumerate(deadlines, 1):
        print(f"\nDeadline {idx}: {deadline['action']}")
        
        # Calculate confidence
        confidence = scorer.calculate(
            deadline=deadline,
            similar_patterns=[],  # No RAG patterns for demo
            ocr_confidence=ocr_confidence
        )
        
        print(f"  Final Confidence: {confidence:.2%}")
        print(f"  Factors:")
        print(f"    - OCR Quality: {ocr_confidence:.2%}")
        print(f"    - Extraction Confidence: {deadline['confidence']:.2%}")
        print(f"    - Has Ambiguity: {deadline.get('ambiguity_flag', False)}")
        print(f"    - Priority: {deadline.get('priority', 'standard')}")

async def demo_rag_search():
    """Demonstrate RAG similarity search"""
    
    print_section("DEMO 5: RAG Pattern Matching (Simulated)")
    
    # Note: This is simulated since it requires OpenSearch setup
    print("Simulating RAG pattern search...")
    
    queries = [
        "motion filed respond within 14 days",
        "discovery request 30 days produce documents",
        "notice of appeal deadline"
    ]
    
    # Simulated similar patterns
    simulated_patterns = {
        queries[0]: [
            {
                "pattern_id": "federal_motion_response",
                "trigger_phrase": "motion filed",
                "action_required": "file opposition",
                "time_period": "14 days",
                "jurisdiction": "federal",
                "similarity_score": 0.95,
                "human_verified": True,
                "verification_count": 150
            }
        ],
        queries[1]: [
            {
                "pattern_id": "discovery_response",
                "trigger_phrase": "discovery served",
                "action_required": "respond to discovery",
                "time_period": "30 days",
                "jurisdiction": "federal",
                "similarity_score": 0.88,
                "human_verified": True,
                "verification_count": 200
            }
        ],
        queries[2]: [
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
    }
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        patterns = simulated_patterns[query]
        
        if patterns:
            print("Similar patterns found:")
            for pattern in patterns:
                print(f"  • {pattern['pattern_id']}")
                print(f"    Similarity: {pattern['similarity_score']:.2%}")
                print(f"    Verified: {pattern['human_verified']}")
                print(f"    Used {pattern['verification_count']} times")

async def demo_document_classification():
    """Demonstrate document classification using Claude 4 Sonnet"""
    
    print_section("DEMO 6: Document Classification")
    
    from src.core.document_classifier import DocumentClassifier
    import boto3
    
    # Initialize with real Claude Sonnet 4 classifier
    try:
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name='us-east-1'
        )
        classifier = DocumentClassifier(bedrock_client=bedrock_client)
        print("Using Claude Sonnet 4 for intelligent classification")
    except Exception as e:
        print(f"Note: Could not initialize Bedrock client: {e}")
        print("Falling back to pattern-based classification")
        classifier = DocumentClassifier(bedrock_client=None)
    
    # Sample documents for classification
    test_documents = [
        {
            "name": "Court Order",
            "text": """
                UNITED STATES DISTRICT COURT
                SOUTHERN DISTRICT OF NEW YORK
                
                Case No. 23-CV-1234
                
                ORDER
                
                Upon consideration of the Motion for Summary Judgment filed by Plaintiff,
                and having reviewed all submissions,
                
                IT IS HEREBY ORDERED that:
                
                1. Plaintiff's Motion for Summary Judgment is GRANTED.
                2. Defendant shall pay damages in the amount of $500,000 within 30 days
                   of the date of this order.
                3. Defendant shall file any notice of appeal within 30 days.
                
                This constitutes the decision and order of the Court.
                
                Dated: January 15, 2024
                /s/ Judge Smith
            """
        },
        {
            "name": "Motion",
            "text": """
                MOTION FOR PROTECTIVE ORDER
                
                Defendant ABC Corp. respectfully MOVES THIS COURT for a Protective Order
                pursuant to Federal Rule of Civil Procedure 26(c), and in support thereof
                states as follows:
                
                1. Plaintiff has served overly broad discovery requests.
                2. The requested documents contain trade secrets.
                3. Disclosure would cause competitive harm.
                
                WHEREFORE, Defendant requests this Court enter a Protective Order
                limiting disclosure of confidential information.
                
                Opposition to this motion must be filed within 14 days.
            """
        },
        {
            "name": "Notice",
            "text": """
                NOTICE OF DEPOSITION
                
                PLEASE TAKE NOTICE that pursuant to Rule 30 of the Federal Rules
                of Civil Procedure, Plaintiff will take the deposition of John Doe,
                CEO of ABC Corporation.
                
                Date: February 20, 2024
                Time: 10:00 AM
                Location: 123 Main Street, New York, NY
                
                The deposition will be recorded by stenographic means and may
                continue from day to day until completed.
            """
        },
        {
            "name": "Complaint",
            "text": """
                COMPLAINT FOR DAMAGES
                
                Plaintiff XYZ Inc. alleges against Defendant ABC Corp. as follows:
                
                FIRST CAUSE OF ACTION
                (Breach of Contract)
                
                1. Plaintiff and Defendant entered into a contract on January 1, 2023.
                2. Plaintiff fully performed all obligations under the contract.
                3. Defendant breached the contract by failing to pay $1,000,000.
                
                WHEREFORE, Plaintiff demands judgment against Defendant for damages
                in the amount of $1,000,000, plus interest and costs.
                
                Defendant must answer this complaint within 21 days of service.
            """
        }
    ]
    
    # Process each document
    for doc in test_documents:
        print(f"\n{doc['name']}:")
        print("-" * 40)
        
        # Classify the document
        classification = await classifier.classify(doc['text'])
        
        # Display classification results
        print(f"Document Type: {classification['primary_type']}")
        print(f"Confidence: {classification['confidence']:.1%}")
        print(f"Reasoning: {classification['reasoning']}")
        
        # Show indicators found
        if classification['indicators']:
            print(f"Indicators Found: {', '.join(classification['indicators'])}")
        
        # Show extraction hints
        hints = classification.get('extraction_hints', {})
        print(f"Priority Level: {hints.get('priority', 'unknown')}")
        print(f"Binding Level: {hints.get('binding_level', 'unknown')}")
        print(f"Requires Response: {hints.get('requires_response', False)}")
        
        # Show extraction strategy
        strategy = classification.get('extraction_strategy', {})
        if strategy:
            print(f"Entity Focus: {', '.join(strategy.get('entity_focus', []))}")
            print(f"Deadline Sensitivity: {strategy.get('deadline_sensitivity', 'unknown')}")
            print(f"Attorney Review Required: {strategy.get('requires_attorney_review', False)}")
        
        # Get routing decision
        routing = classifier.get_routing_decision(classification, classification['confidence'])
        print(f"\nRouting Decision:")
        print(f"  Destination: {routing['destination']}")
        print(f"  Priority: {routing['priority']}")
        print(f"  Reason: {routing['reason']}")
        print(f"  SLA: {routing['sla_hours']} hours")
    
    # Demonstrate confidence-based routing
    print("\n" + "="*50)
    print("Confidence-Based Routing Examples:")
    print("-" * 50)
    
    routing_examples = [
        {"type": "court_order", "confidence": 0.98, "desc": "High-confidence court order"},
        {"type": "motion", "confidence": 0.96, "desc": "High-confidence motion"},
        {"type": "motion", "confidence": 0.88, "desc": "Medium-confidence motion"},
        {"type": "unknown", "confidence": 0.45, "desc": "Low-confidence unknown document"}
    ]
    
    for example in routing_examples:
        classification = {
            "primary_type": example["type"],
            "confidence": example["confidence"]
        }
        routing = classifier.get_routing_decision(classification, example["confidence"])
        
        print(f"\n{example['desc']}:")
        print(f"  Type: {example['type']}, Confidence: {example['confidence']:.1%}")
        print(f"  → Routes to: {routing['destination']}")
        print(f"  → Priority: {routing['priority']}")
        print(f"  → SLA: {routing['sla_hours']} hours")

async def main():
    """Run all demos"""
    
    print("\n" + "="*60)
    print("  LEGAL DOCUMENT PROCESSING SYSTEM")
    print("  Comprehensive Demo Tutorial")
    print("="*60)
    
    demos = [
        ("Basic Processing", demo_basic_processing),
        ("PII Masking", demo_pii_masking),
        ("Deadline Calculation", demo_deadline_calculation),
        ("Confidence Scoring", demo_confidence_scoring),
        ("RAG Pattern Matching", demo_rag_search),
        ("Document Classification", demo_document_classification)
    ]
    
    print("\nAvailable Demos:")
    for idx, (name, _) in enumerate(demos, 1):
        print(f"  {idx}. {name}")
    
    print("\nNote: Some demos are simulated as they require")
    print("AWS services (Bedrock, OpenSearch) to be configured.")
    
    # Run selected demos
    run_all = input("\nRun all demos? (y/n): ").lower() == 'y'
    
    if run_all:
        for name, demo_func in demos:
            try:
                await demo_func()
            except Exception as e:
                print(f"\nError in {name}: {e}")
                print("Continuing with next demo...")
    else:
        while True:
            choice = input("\nEnter demo number (1-6) or 'q' to quit\n 1. Basic Processing\n 2. PII Masking\n 3. Deadline Calculation\n 4. Confidence Scoring\n 5. RAG Pattern Matching\n 6. Document Classification\n")
            
            if choice.lower() == 'q':
                break
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(demos):
                    name, demo_func = demos[idx]
                    await demo_func()
                else:
                    print("Invalid choice")
            except (ValueError, Exception) as e:
                print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("  Demo completed successfully!")
    print("="*60)

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())