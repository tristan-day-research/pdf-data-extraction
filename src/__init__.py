"""Top-level package for document data extraction utilities.

The current implementation targets PDFs, but the project structure is designed
to support additional file types in the future. The package exposes
convenience functions for common operations.
"""

from .classifiers import DigitalElementClassifier, PdfTypeDetector
from .routers import route_elements


def classify(pdf_path: str):
    """Classify elements in a digital PDF.

    This is a convenience wrapper around :class:`DigitalElementClassifier`.
    """
    return DigitalElementClassifier().classify(pdf_path)


__all__ = [
    "classify",
    "route_elements",
    "DigitalElementClassifier",
    "PdfTypeDetector",
]
