"""Classifier utilities for document processing.

This package exposes PDF-related classifiers via the ``pdf`` submodule. The
structure allows future file types (e.g. CSV, DOCX) to provide their own
classifiers without altering the public API.
"""

from .pdf import DigitalElementClassifier, PDFTypeDetector

__all__ = ["PDFTypeDetector", "DigitalElementClassifier"]
