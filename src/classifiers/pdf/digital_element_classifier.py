"""Identification utilities for elements within digitally generated PDFs.

The functions in this module categorise high-level page elements such as
text blocks, tables and images. No processing of the extracted content is
performed here; downstream modules should handle each element type
separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pdfplumber

ElementType = Dict[str, List]


@dataclass
class DigitalElementClassifier:
    """Locate simple elements on each page of a digital PDF."""

    def classify(self, pdf_path: str) -> List[ElementType]:
        """Return a list with one entry per page describing element types.

        Each entry contains the keys ``text``, ``tables`` and ``images`` with
        metadata describing the location of each element. The goal is merely to
        identify their presence; detailed processing is delegated elsewhere.
        """
        results: List[ElementType] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_data: ElementType = {
                    "text": page.extract_words(),
                    "tables": [table.bbox for table in page.find_tables()],
                    "images": page.images,
                }
                results.append(page_data)

        return results
