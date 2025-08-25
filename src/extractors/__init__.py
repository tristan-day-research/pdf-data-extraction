"""Extraction utilities for supplementary document data.

The :mod:`extractors.pdf` subpackage provides PDF-specific helpers. This layout
leaves room for extractors targeting other file formats while maintaining a
consistent API.
"""

from .pdf import extract_annotations, extract_forms, extract_metadata

__all__ = [
    "extract_metadata",
    "extract_annotations",
    "extract_forms",
]
