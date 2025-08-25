"""Utilities for determining whether a PDF file is digitally generated or scanned.

This module only detects the type of PDF. It does not attempt to perform
any OCR or further extraction. The simple heuristic used here checks
whether any text can be extracted from the first page of the document.
If text is present, the PDF is assumed to be digital; otherwise it is
likely a scanned document.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pdfminer.high_level import extract_text

PdfType = Literal["digital", "scanned"]


@dataclass
class PdfTypeDetector:
    """Detect whether a PDF is digital or scanned."""

    sample_pages: int = 1

    def detect(self, pdf_path: str) -> PdfType:
        """Return ``"digital"`` if text is found, otherwise ``"scanned"``.

        The detector only examines a small number of pages (``sample_pages``)
        for performance reasons. Any error during extraction falls back to a
        ``"scanned"`` result because scanned PDFs typically lack parseable
        text streams.
        """
        try:
            text = extract_text(pdf_path, maxpages=self.sample_pages)
        except Exception:
            return "scanned"

        return "digital" if text and text.strip() else "scanned"
