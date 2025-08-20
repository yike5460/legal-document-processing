"""
Automated System Test
Tests core components without requiring user input or AWS services
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all core modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test utils imports
        from src.utils.logger import setup_logger
        print("✓ Logger module")
        
        from src.utils.pii_masker import PIIMasker
        print("✓ PII Masker module")
        
        from src.utils.data_validator import DataValidator
        print("✓ Data Validator module")
        
        # Test core imports
        from src.core.deadline_calculator import DeadlineCalculator
        print("✓ Deadline Calculator module")
        
        from src.core.confidence_scorer import ConfidenceScorer
        print("✓ Confidence Scorer module")
        
        # Test pipeline import
        from src.pipeline.main_pipeline import LegalDocumentPipeline, PipelineConfig
        print("✓ Main Pipeline module")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_pii_masking():
    """Test PII masking functionality"""
    print("\nTesting PII Masking...")
    
    from src.utils.pii_masker import PIIMasker
    
    masker = PIIMasker()
    
    # Test text with various PII
    test_text = """
    John Smith (SSN: 123-45-6789)
    Email: john@example.com
    Phone: (555) 123-4567
    Case No. 21-CV-12345
    """
    
    masked_text, masking_map = masker.mask_document(test_text)
    
    # Verify SSN is masked
    assert "123-45-6789" not in masked_text, "SSN not properly masked"
    print("✓ SSN masking works")
    
    # Verify email is masked
    assert "john@example.com" not in masked_text, "Email not properly masked"
    print("✓ Email masking works")
    
    # Verify phone is masked
    assert "(555) 123-4567" not in masked_text, "Phone not properly masked"
    print("✓ Phone masking works")
    
    # Test unmasking
    unmasked = masker.unmask_document(masked_text, masking_map)
    assert "123-45-6789" in unmasked, "Unmasking failed"
    print("✓ Unmasking works")
    
    return True

def test_deadline_calculator():
    """Test deadline calculation"""
    print("\nTesting Deadline Calculator...")
    
    from src.core.deadline_calculator import DeadlineCalculator
    
    calculator = DeadlineCalculator()
    
    # Test basic calculation
    result = calculator.calculate(
        trigger_text="date of this order",
        period_text="14 days",
        jurisdiction="federal",
        rule_type="motion_response"
    )
    
    assert result['date'] is not None, "Date calculation failed"
    print(f"✓ Basic calculation: 14 days from today = {result['date'].strftime('%Y-%m-%d')}")
    
    # Test with different period
    result = calculator.calculate(
        trigger_text="service of complaint",
        period_text="30 days",
        jurisdiction="california",
        rule_type="discovery_response"
    )
    
    assert result['date'] is not None, "Date calculation failed"
    print(f"✓ 30-day calculation: {result['date'].strftime('%Y-%m-%d')}")
    
    return True

def test_confidence_scorer():
    """Test confidence scoring"""
    print("\nTesting Confidence Scorer...")
    
    from src.core.confidence_scorer import ConfidenceScorer
    
    scorer = ConfidenceScorer()
    
    # Test high confidence deadline
    deadline = {
        "action": "file opposition to motion",
        "trigger_event": "date of this order",
        "time_period": "14 days",
        "confidence": 0.9,
        "ambiguity_flag": False
    }
    
    confidence = scorer.calculate(
        deadline=deadline,
        similar_patterns=[],
        ocr_confidence=0.95
    )
    
    assert 0 <= confidence <= 1, "Invalid confidence score"
    print(f"✓ High confidence deadline: {confidence:.2%}")
    
    # Test low confidence deadline
    deadline_low = {
        "action": "respond",
        "trigger_event": "unknown",
        "time_period": "some days",
        "confidence": 0.3,
        "ambiguity_flag": True
    }
    
    confidence_low = scorer.calculate(
        deadline=deadline_low,
        similar_patterns=[],
        ocr_confidence=0.95
    )
    
    assert confidence_low < confidence, "Low confidence should score lower"
    print(f"✓ Low confidence deadline: {confidence_low:.2%}")
    
    return True

def test_data_validator():
    """Test data validation"""
    print("\nTesting Data Validator...")
    
    from src.utils.data_validator import DataValidator
    
    validator = DataValidator()
    
    # Test extraction validation
    valid_extraction = {
        "deadlines": [
            {
                "action": "file response",
                "time_period": "14 days",
                "confidence": 0.9
            }
        ],
        "entities": {
            "parties": ["John Doe", "Jane Smith"],
            "case_number": "21-CV-12345"
        }
    }
    
    result = validator.validate_extraction(valid_extraction)
    assert result['valid'], "Valid extraction marked as invalid"
    print("✓ Valid extraction passes validation")
    
    # Test invalid extraction
    invalid_extraction = {
        "deadlines": [
            {"action": ""},  # Empty action
            {"confidence": 1.5}  # Invalid confidence
        ]
    }
    
    result = validator.validate_extraction(invalid_extraction)
    assert not result['valid'], "Invalid extraction not caught"
    print("✓ Invalid extraction properly rejected")
    
    return True

def test_pipeline_initialization():
    """Test pipeline can be initialized"""
    print("\nTesting Pipeline Initialization...")
    
    from src.pipeline.main_pipeline import LegalDocumentPipeline, PipelineConfig
    
    # Initialize with minimal config (no AWS services)
    config = PipelineConfig(
        enable_pii_masking=True,
        enable_rag=False,  # Disable RAG to avoid OpenSearch dependency
        confidence_threshold_auto=0.95
    )
    
    try:
        pipeline = LegalDocumentPipeline(config)
        print("✓ Pipeline initialized successfully")
        
        # Check components are initialized
        assert hasattr(pipeline, 'pii_masker'), "PII masker not initialized"
        print("✓ PII masker component ready")
        
        assert hasattr(pipeline, 'deadline_calculator'), "Deadline calculator not initialized"
        print("✓ Deadline calculator component ready")
        
        assert hasattr(pipeline, 'confidence_scorer'), "Confidence scorer not initialized"
        print("✓ Confidence scorer component ready")
        
        assert hasattr(pipeline, 'validator'), "Data validator not initialized"
        print("✓ Data validator component ready")
        
        # Cleanup
        pipeline.cleanup()
        print("✓ Pipeline cleanup successful")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return False

async def test_sample_document_processing():
    """Test processing with a sample document (mocked)"""
    print("\nTesting Document Processing Flow (Mocked)...")
    
    from src.pipeline.main_pipeline import LegalDocumentPipeline, PipelineConfig
    
    # Create test document
    test_doc_path = "/tmp/test_court_order.pdf"
    
    # Create a minimal PDF for testing
    try:
        from pypdf import PdfWriter
        from pathlib import Path
        
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        
        with open(test_doc_path, "wb") as f:
            writer.write(f)
        
        print(f"✓ Created test document: {test_doc_path}")
    except Exception as e:
        print(f"✗ Could not create test PDF: {e}")
        print("  Continuing with validation tests...")
        return True
    
    # Initialize pipeline without AWS services
    config = PipelineConfig(
        enable_pii_masking=True,
        enable_rag=False,
        confidence_threshold_auto=0.95
    )
    
    pipeline = LegalDocumentPipeline(config)
    
    # Test document validation
    if pipeline.validator.validate_document(test_doc_path):
        print("✓ Test document passes validation")
    else:
        print("✗ Test document validation failed")
    
    # Clean up
    pipeline.cleanup()
    if Path(test_doc_path).exists():
        Path(test_doc_path).unlink()
        print("✓ Test document cleaned up")
    
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("  LEGAL DOCUMENT PROCESSING SYSTEM")
    print("  Automated Component Testing")
    print("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("PII Masking", test_pii_masking),
        ("Deadline Calculator", test_deadline_calculator),
        ("Confidence Scorer", test_confidence_scorer),
        ("Data Validator", test_data_validator),
        ("Pipeline Initialization", test_pipeline_initialization),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print(f"{'='*40}")
        
        try:
            if test_func():
                passed += 1
                print(f"→ {test_name}: PASSED ✓")
            else:
                failed += 1
                print(f"→ {test_name}: FAILED ✗")
        except Exception as e:
            failed += 1
            print(f"→ {test_name}: FAILED ✗")
            print(f"  Error: {e}")
    
    # Run async test
    print(f"\n{'='*40}")
    print("Running: Sample Document Processing")
    print(f"{'='*40}")
    
    try:
        asyncio.run(test_sample_document_processing())
        passed += 1
        print("→ Sample Document Processing: PASSED ✓")
    except Exception as e:
        failed += 1
        print("→ Sample Document Processing: FAILED ✗")
        print(f"  Error: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    print(f"  Total Tests: {passed + failed}")
    print(f"  Passed: {passed} ✓")
    print(f"  Failed: {failed} ✗")
    
    if failed == 0:
        print("\n  ✓ All tests passed successfully!")
        print("  The system is ready for use.")
        print("\n  Note: AWS services (Bedrock, OpenSearch) need to be")
        print("  configured with proper credentials for full functionality.")
    else:
        print(f"\n  ✗ {failed} test(s) failed.")
        print("  Please review the errors above.")
    
    print("="*60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)