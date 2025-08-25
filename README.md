# PDF Data Extraction and Form Generation System

This project provides a comprehensive system for extracting data from various document types and using it to fill out forms via RAG (Retrieval-Augmented Generation).

## Current Status

The system currently supports **PDF data extraction** with the following capabilities:
- PDF type detection (digital vs. scanned)
- Element classification and extraction
- Text, table, and image processing
- OCR processing for scanned documents

## Project Structure

```
src/
├── extraction/          # Document data extraction (currently PDF-focused)
│   ├── classifiers/     # Document type and element classification
│   ├── extractors/      # Data extraction from document elements
│   ├── processors/      # Text, image, table, and OCR processing
│   └── routers/         # Element routing and processing coordination
├── generation/          # Form generation and RAG functionality (planned)
│   └── __init__.py      # TODO: Implement form filling and RAG
└── api/                 # REST API endpoints (planned)
    └── __init__.py      # TODO: Implement document upload and processing API
```

## Future Plans

### Phase 2: Multi-format Support
- **CSV files**: Tabular data extraction and parsing
- **TXT files**: Plain text processing and analysis
- **DOCX files**: Word document content extraction
- **MD files**: Markdown parsing and structure analysis

### Phase 3: Form Generation System
- **JSON specification**: Define form fields and requirements
- **RAG integration**: Intelligent form completion using extracted data
- **Business logic**: Domain-specific form filling (e.g., business plans, applications)

### Phase 4: API Development
- **Document upload**: Handle multiple file types
- **Form specification**: Process JSON form definitions
- **Response generation**: Return completed forms with extracted data

## Usage

### Current PDF Processing

```python
import sys
sys.path.append('src')

from extraction import PDFTypeDetector, DigitalElementClassifier, ElementRouter

# Detect PDF type
detector = PDFTypeDetector()
pdf_type = detector.detect('path/to/document.pdf')

if pdf_type == 'digital':
    # Classify elements
    classifier = DigitalElementClassifier()
    pages = classifier.classify('path/to/document.pdf')
    
    # Route elements to processors
    router = ElementRouter()
    router.route_elements(pages)
```

## Development

The project uses a modular architecture designed for easy extension:

- **Extraction modules**: Handle different document types and formats
- **Generation modules**: Process extracted data and generate forms
- **API modules**: Provide web interfaces for the system

Each module is self-contained with clear interfaces, making it easy to add new document types or processing capabilities.

## Requirements

See `requirements.txt` for current dependencies. The system is designed to work with Python 3.9+.

