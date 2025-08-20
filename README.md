# Legal Document Processing System

An AI-powered solution for automating legal docketing and deadline extraction from legal documents, addressing critical pain points in law firm operations where missing court-mandated deadlines can result in malpractice claims, client losses, and severe financial penalties.

## Overview

This system uses state-of-the-art AI models including DOTS.OCR for document processing and Claude 4 Sonnet for intelligent extraction to automate the processing of legal documents, extract critical deadlines, and route them appropriately for review.

## Key Features

- **Automated Document Processing**: Handles PDFs, images, Word documents with varying quality
- **PII Protection**: Masks sensitive information before AI processing
- **Intelligent Classification**: Uses Claude 4 Sonnet to classify documents and extract entities
- **Deadline Calculation**: Jurisdiction-specific deadline calculations with holiday handling
- **Confidence Scoring**: Multi-factor scoring to determine routing (auto-process, paralegal review, attorney review)
- **RAG Enhancement**: Optional pattern matching using AWS OpenSearch for improved accuracy

## Architecture

```
Document Input → PII Masking → OCR (DOTS.OCR) → AI Extraction (Claude 4 Sonnet) 
    → RAG Enhancement → Confidence Scoring → Routing Decision → Output
```

## Installation

### Prerequisites

```bash
# macOS
brew install cmake pkg-config

# For Apple M1/M2 Macs
xcode-select --install
```

### Setup

```bash
# Clone repository
git clone https://github.com/yike5460/legal-document-processing.git
cd legal-document-processing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Create .env.local file with AWS credentials (optional for full functionality)
cat > .env.local << EOF
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_actual_key
AWS_SECRET_ACCESS_KEY=your_actual_secret
OPENSEARCH_DOMAIN=your_domain  # optional
EOF
```

## Quick Start

### Run Interactive Demos

```bash
# Run the interactive tutorial (6 demos available)
python demo/tutorial.py
```

Available demos:
1. Basic Processing - End-to-end document pipeline
2. PII Masking - SSN, email, phone masking/unmasking
3. Deadline Calculation - Jurisdiction-specific calculations
4. Confidence Scoring - Multi-factor assessment
5. RAG Pattern Matching - Simulated similarity search
6. Document Classification - Claude 4 Sonnet classification

### Run Automated Tests

```bash
# Run all component tests
python demo/test_system.py
```

### Process Sample Documents

```python
from src.pipeline.main_pipeline import LegalDocumentPipeline, PipelineConfig
import asyncio

# Configure pipeline
config = PipelineConfig(
    enable_pii_masking=True,
    enable_rag=False,  # Set True if OpenSearch configured
    confidence_threshold_auto=0.95
)

# Process document
async def process():
    pipeline = LegalDocumentPipeline(config)
    result = await pipeline.process_document("samples/court_order_sample.pdf")
    print(f"Deadlines found: {len(result.deadlines)}")
    for deadline in result.deadlines:
        print(f"  - {deadline['action']}: {deadline['calculated_date']}")

asyncio.run(process())
```

## Project Structure

```
legal-document-processing/
├── src/
│   ├── core/              # Core processing modules
│   │   ├── ai_extractor.py
│   │   ├── confidence_scorer.py
│   │   ├── deadline_calculator.py
│   │   ├── document_classifier.py
│   │   └── ocr_processor.py
│   ├── pipeline/          # Main pipeline orchestration
│   │   └── main_pipeline.py
│   └── utils/            # Utility modules
│       ├── pii_masker.py
│       ├── data_validator.py
│       └── logger.py
├── demo/                 # Demo and test files
│   ├── tutorial.py       # Interactive demos
│   └── test_system.py    # Automated tests
├── samples/              # Sample legal documents
├── api/                  # API integration
└── requirements.txt
```

## AWS Services Required

- **AWS Bedrock**: Claude 4 Sonnet model access (required for full AI capabilities)
- **AWS OpenSearch**: Vector storage for RAG (optional, enhances accuracy)
- **AWS S3**: Document storage (optional)

## Configuration

The system can be configured via `PipelineConfig`:

```python
from src.pipeline.main_pipeline import PipelineConfig

config = PipelineConfig(
    dots_ocr_model="rednote-hilab/dots.ocr",
    claude_model="anthropic.claude-4-sonnet-20241022-v1:0",
    confidence_threshold_auto=0.95,
    enable_pii_masking=True,
    enable_rag=False  # Set True if OpenSearch available
)
```

## Performance

- Average processing time: 8 seconds per document
- Supports batch processing with configurable concurrency
- OpenSearch caching reduces repeated pattern lookups
- PII masking adds ~0.5 seconds overhead

## Contributing

Please read the development guidelines in the proposal document for detailed information about the architecture and implementation.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Documentation

For detailed technical documentation, see [legal-document-processing-proposal.md](legal-document-processing-proposal.md)