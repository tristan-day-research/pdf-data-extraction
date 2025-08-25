"""Top-level package for PDF data extraction utilities.

Provides convenience functions for common operations."""

from .digital_element_classifier import DigitalElementClassifier
from .element_router import route_elements


def classify(pdf_path: str):
    """Classify elements in a digital PDF.

    This is a convenience wrapper around :class:`DigitalElementClassifier`.
    """
    return DigitalElementClassifier().classify(pdf_path)


__all__ = ["classify", "route_elements", "DigitalElementClassifier"]
