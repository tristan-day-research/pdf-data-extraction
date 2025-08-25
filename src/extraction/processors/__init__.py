"""Processing utilities for document element types.

Currently this package exposes PDF-specific processors via the ``pdf``
subpackage. The structure accommodates additional file formats in future
iterations while keeping a stable public API.
"""

from .pdf import process_image, process_ocr, process_table, process_text

__all__ = [
    "process_text",
    "process_table",
    "process_image",
    "process_ocr",
]
