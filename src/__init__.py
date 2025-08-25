"""
PDF Data Extraction and Form Generation System

This package provides:
- PDF data extraction capabilities
- Form generation and filling via RAG
- API endpoints for document processing
"""

# Import extraction functionality
from .extraction import (
    PDFTypeDetector,
    DigitalElementClassifier,
    ElementRouter,
)

# TODO: Import generation and API modules when implemented
# from .generation import FormGenerator, RAGProcessor
# from .api import DocumentProcessor, FormCompletionAPI

__all__ = [
    # Extraction
    'PDFTypeDetector',
    'DigitalElementClassifier',
    'ElementRouter',
    # TODO: Add generation and API exports when implemented
]
