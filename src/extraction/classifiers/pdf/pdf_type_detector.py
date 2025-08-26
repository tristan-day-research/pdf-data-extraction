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
class PDFTypeDetector:
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


# import pymupdf
# from dataclasses import dataclass
# from typing import Dict, List, Tuple

# @dataclass
# class PDFRouter:
#     def analyze(self, pdf_path: str) -> Dict[str, float]:
#         doc = pymupdf.open(pdf_path)
#         sandwich_hits = 0
#         vector_ops = 0
#         raster_imgs = 0
#         figure_tokens = 0
#         widths = []

#         for pno in range(len(doc)):
#             page = doc[pno]
#             blocks = page.get_text("blocks") or []
#             text_blocks = [b for b in blocks if len(b) > 6 and b[6] == 0]
#             img_rects = [r for x in page.get_images(full=True) for r in page.get_image_bbox(x[0])]
#             # sandwich check
#             if text_blocks and img_rects:
#                 page_area = page.rect.width * page.rect.height
#                 max_img_area = max((r.width * r.height for r in img_rects), default=0)
#                 if max_img_area / max(page_area,1) > 0.60:
#                     sandwich_hits += 1
#             # vector vs raster proxy
#             # display-list analysis: count draw ops
#             dl = page.get_displaylist()
#             tp = dl.get_textpage()     # text only (cheap)
#             # use 'page.get_text("rawdict")' for spans, or:
#             # count raster images:
#             raster_imgs += len(page.get_images(full=True))
#             # rough proxy for vectors: length of annotations in draw list
#             vector_ops += len(page.get_drawings())

#             # column width proxy
#             for b in text_blocks:
#                 widths.append(b[2]-b[0])
#             # figure tokens
#             # fast check in blocks text (case-insensitive)
#             txt = "\n".join((b[4] or "") for b in text_blocks).lower()
#             if "figure" in txt or "fig." in txt or "table " in txt:
#                 figure_tokens += 1

#         doc.close()
#         n = max(len(widths), 1)
#         return {
#             "sandwich_ratio": sandwich_hits / max(len(widths),1),
#             "vector_to_raster": vector_ops / max(raster_imgs, 1),
#             "narrow_block_ratio": sum(1 for w in widths if w < 0.45*max(widths)) / n,
#             "figure_token_pages": figure_tokens / max(1, len(widths))
#         }

#     def route(self, pdf_path: str) -> str:
#         stats = self.analyze(pdf_path)
#         if stats["sandwich_ratio"] > 0.3:
#             return "sandwich"
#         if stats["vector_to_raster"] > 2.0 or stats["figure_token_pages"] > 0.2:
#             return "journal"
#         return "generic"


import pymupdf
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class PDFRouter:
    def analyze(self, pdf_path: str) -> Dict[str, float]:
        doc = pymupdf.open(pdf_path)
        sandwich_hits = 0
        vector_ops = 0
        raster_imgs = 0
        figure_tokens = 0
        widths = []
        
        # Get page count BEFORE processing pages
        page_count = len(doc)

        for pno in range(page_count):
            page = doc[pno]
            blocks = page.get_text("blocks") or []
            text_blocks = [b for b in blocks if len(b) > 6 and b[6] == 0]
            
            # Get image bounding boxes correctly
            img_rects = []
            for img_info in page.get_images(full=True):
                # img_info[7] contains the image name (string)
                if img_info[7]:  # Check if image has a name
                    try:
                        bbox = page.get_image_bbox(img_info[7])
                        img_rects.append(bbox)
                    except ValueError:
                        # Skip images without proper names
                        pass
            
            # sandwich check
            if text_blocks and img_rects:
                page_area = page.rect.width * page.rect.height
                max_img_area = max((r.width * r.height for r in img_rects), default=0)
                if max_img_area / max(page_area, 1) > 0.60:
                    sandwich_hits += 1
            
            # vector vs raster proxy
            # display-list analysis: count draw ops
            dl = page.get_displaylist()
            tp = dl.get_textpage()     # text only (cheap)
            
            # count raster images:
            raster_imgs += len(page.get_images(full=True))
            
            # rough proxy for vectors: length of annotations in draw list
            vector_ops += len(page.get_drawings())

            # column width proxy
            for b in text_blocks:
                widths.append(b[2] - b[0])
            
            # figure tokens
            # fast check in blocks text (case-insensitive)
            txt = "\n".join((b[4] or "") for b in text_blocks).lower()
            if "figure" in txt or "fig." in txt or "table " in txt:
                figure_tokens += 1

        doc.close()
        n = max(len(widths), 1)
        return {
            "sandwich_ratio": sandwich_hits / max(page_count, 1),  # Use pre-calculated page_count
            "vector_to_raster": vector_ops / max(raster_imgs, 1),
            "narrow_block_ratio": sum(1 for w in widths if w < 0.45 * max(widths)) / n if widths else 0,
            "figure_token_pages": figure_tokens / max(1, page_count)  # Use pre-calculated page_count
        }

    def route(self, pdf_path: str) -> str:
        stats = self.analyze(pdf_path)
        if stats["sandwich_ratio"] > 0.3:
            return "sandwich"
        if stats["vector_to_raster"] > 2.0 or stats["figure_token_pages"] > 0.2:
            return "journal"
        return "generic"