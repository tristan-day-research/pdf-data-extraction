from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, TypedDict
import math

# ---- Optional external tools ----
# - pdfplumber: required (geometry, words, images)
# - camelot: optional (table regions via lattice/stream); code handles absence
try:
    import camelot  # type: ignore
    _HAVE_CAMELOT = True
except Exception:
    _HAVE_CAMELOT = False

import pdfplumber  # type: ignore


# -----------------------------
# Types
# -----------------------------
BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1) with PDF coordinates
PageBBox = Dict[str, Any]  # {"page": int, "bbox": BBox}

class ElementDict(TypedDict, total=False):
    id: str
    kind: str                  # "text" | "table" | "image"
    page_range: Tuple[int, int]
    bboxes_per_page: List[PageBBox]
    metadata: Dict[str, Any]   # small, e.g., column_edges, header_signature
    text: str                  # text content (for text elements)

ElementType = Dict[str, List[ElementDict]]


# -----------------------------
# Utility geometry
# -----------------------------
def _iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _expand(b: BBox, px: float) -> BBox:
    x0, y0, x1, y1 = b
    return (x0 - px, y0 - px, x1 + px, y1 + px)


def _nearly_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _horiz_overlap(a: BBox, b: BBox) -> float:
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    return max(0.0, min(ax1, bx1) - max(ax0, bx0))


def _width(b: BBox) -> float:
    return max(0.0, b[2] - b[0])


def _height(b: BBox) -> float:
    return max(0.0, b[3] - b[1])


# -----------------------------
# Table detection (regions only)
# -----------------------------
def _detect_table_regions(pdf_path: str) -> Dict[int, List[ElementDict]]:
    """
    Return {page_index: [table_element,...]} with bbox metadata only.
    Tries Camelot (lattice + stream) if available; otherwise falls back to line/word density heuristic via pdfplumber.
    """
    tables_by_page: Dict[int, List[ElementDict]] = {}

    if _HAVE_CAMELOT:
        # Camelot returns per-page tables; we call twice and merge.
        try:
            latt = camelot.read_pdf(pdf_path, pages="all", flavor="lattice", suppress_stdout=True)
        except Exception:
            latt = []
        try:
            stre = camelot.read_pdf(pdf_path, pages="all", flavor="stream", suppress_stdout=True)
        except Exception:
            stre = []

        def add_tbls(objs):
            for t in getattr(objs, "tables", objs):
                p = (t.page or 1) - 1
                x0, y0, x1, y1 = t._bbox  # Camelot stores bbox in PDF coords (x0, y0, x1, y1)
                # Normalize to (x0, y0, x1, y1) with origin at top-left per pdfplumber; Camelot already uses PDF coords
                el: ElementDict = {
                    "id": f"table_p{p}_{len(tables_by_page.get(p, []))}",
                    "kind": "table",
                    "page_range": (p, p),
                    "bboxes_per_page": [{"page": p, "bbox": (x0, y0, x1, y1)}],
                    "metadata": {
                        "source": "camelot",
                        "shape": getattr(t, "shape", None),
                        "cols": getattr(t, "cols", None),
                        "rows": getattr(t, "rows", None),
                    },
                }
                tables_by_page.setdefault(p, []).append(el)

        add_tbls(latt)
        add_tbls(stre)

        # De-duplicate overlapping detections (lattice vs stream)
        for p, arr in tables_by_page.items():
            keep: List[ElementDict] = []
            for el in arr:
                bb = el["bboxes_per_page"][0]["bbox"]
                if any(_iou(bb, k["bboxes_per_page"][0]["bbox"]) > 0.6 for k in keep):
                    continue
                keep.append(el)
            tables_by_page[p] = keep

        return tables_by_page

    # Fallback heuristic (when Camelot unavailable): dense numeric blocks with column alignment
    with pdfplumber.open(pdf_path) as pdf:
        for p, page in enumerate(pdf.pages):
            words = page.extract_words(x_tolerance=2, y_tolerance=2, use_text_flow=True) or []
            if not words:
                continue
            # Simple grid-ish grouping: bucket words into vertical bands (columns) then cluster dense y-bands
            xs = sorted([(w["x0"], w["x1"]) for w in words], key=lambda t: t[0])
            if not xs:
                continue

            # Build rough column edges via greedy merge
            columns: List[List[Tuple[float, float]]] = []
            for x0, x1 in xs:
                if not columns or x0 - columns[-1][-1][1] > 8:  # gap threshold
                    columns.append([(x0, x1)])
                else:
                    columns[-1].append((x0, x1))

            # If we find ≥3 columns with many numeric tokens, mark a table region (very conservative)
            numeric_words = [w for w in words if any(ch.isdigit() for ch in w["text"])]
            if len(columns) >= 3 and len(numeric_words) >= 20:
                # Compute bbox of dense area
                x0 = min(w["x0"] for w in numeric_words)
                x1 = max(w["x1"] for w in numeric_words)
                y0 = min(w["top"] for w in numeric_words)
                y1 = max(w["bottom"] for w in numeric_words)
                el: ElementDict = {
                    "id": f"table_p{p}_0",
                    "kind": "table",
                    "page_range": (p, p),
                    "bboxes_per_page": [{"page": p, "bbox": (x0, y0, x1, y1)}],
                    "metadata": {"source": "heuristic", "note": "low-confidence"},
                }
                tables_by_page.setdefault(p, []).append(el)
    return tables_by_page


# -----------------------------
# Image detection (raster XObjects)
# -----------------------------
def _detect_images(pdf_path: str, save_images: bool = True, document_id: str = None, output_dir: str = None) -> Dict[int, List[ElementDict]]:
    """
    Detect and optionally save images from PDF.
    
    Args:
        pdf_path: Path to PDF file
        save_images: Whether to extract and save images
        document_id: Document ID for organizing images in document structure
        output_dir: Directory to save images (overrides document_id if provided)
    """
    import os
    from PIL import Image
    import io
    
    imgs_by_page: Dict[int, List[ElementDict]] = {}
    
    if save_images:
        if output_dir is None and document_id is not None:
            # Use document structure - create it if it doesn't exist
            from config import settings
            settings.data.create_document_structure(document_id)
            output_dir = settings.data.get_document_elements_path(document_id) / "images"
        elif output_dir is None:
            # If no document_id provided, don't save images to avoid cluttering source directory
            save_images = False
        
        if save_images and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    with pdfplumber.open(pdf_path) as pdf:
        for p, page in enumerate(pdf.pages):
            # pdfplumber exposes raster images via page.images (x0, top, x1, bottom, width, height, ...)
            for i, im in enumerate(page.images or []):
                x0, x1 = im.get("x0"), im.get("x1")
                top, bottom = im.get("top"), im.get("bottom")
                if None in (x0, x1, top, bottom):
                    continue
                    
                bbox: BBox = (float(x0), float(top), float(x1), float(bottom))
                
                # Extract image data
                image_data = None
                image_path = None
                
                if save_images and im.get("stream"):
                    try:
                        # Get image stream data
                        stream_data = im["stream"].get_data()
                        
                        # Try to open with PIL to validate and get format
                        img = Image.open(io.BytesIO(stream_data))
                        
                        # Generate filename
                        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                        img_filename = f"{pdf_name}_p{p}_img{i}.png"
                        img_path = os.path.join(output_dir, img_filename)
                        
                        # Save image
                        img.save(img_path)
                        image_path = img_path
                        image_data = {
                            "format": img.format,
                            "mode": img.mode,
                            "size": img.size
                        }
                        
                    except Exception as e:
                        # If PIL fails, try to save raw stream data
                        try:
                            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                            img_filename = f"{pdf_name}_p{p}_img{i}.raw"
                            img_path = os.path.join(output_dir, img_filename)
                            
                            with open(img_path, 'wb') as f:
                                f.write(stream_data)
                            image_path = img_path
                            image_data = {"note": "saved_as_raw_stream"}
                            
                        except Exception as e2:
                            image_data = {"error": f"Failed to save: {str(e2)}"}
                
                el: ElementDict = {
                    "id": f"image_p{p}_{i}",
                    "kind": "image",
                    "page_range": (p, p),
                    "bboxes_per_page": [{"page": p, "bbox": bbox}],
                    "metadata": {
                        "width": float(im.get("width", _width(bbox))),
                        "height": float(im.get("height", _height(bbox))),
                        "name": im.get("name"),
                        "stream": im.get("stream"),
                        "image_path": image_path,  # Path to saved image file
                        "image_data": image_data,  # Additional image metadata
                    },
                }
                imgs_by_page.setdefault(p, []).append(el)
    return imgs_by_page


# -----------------------------
# Text block detection
# -----------------------------
def _extract_all_text_blocks(pdf_path: str,
                            occupied_masks: Dict[int, List[BBox]]) -> Dict[int, List[ElementDict]]:
    """
    Extract all text content from PDF, properly identifying text vs non-text elements.
    Returns comprehensive text blocks with proper content and formatting metadata.
    """
    txt_by_page: Dict[int, List[ElementDict]] = {}
    
    with pdfplumber.open(pdf_path) as pdf:
        for p, page in enumerate(pdf.pages):
            # Get all characters with their properties
            chars = page.chars or []
            if not chars:
                continue
                
            # Group characters into words first
            words = page.extract_words(
                x_tolerance=2, y_tolerance=2, use_text_flow=True, keep_blank_chars=False
            ) or []
            
            # Remove words inside occupied regions
            occ = [_expand(b, 1.5) for b in occupied_masks.get(p, [])]
            keep_words = []
            for w in words:
                wb = (w["x0"], w["top"], w["x1"], w["bottom"])
                if any(_iou(wb, b) > 0.05 for b in occ):
                    continue
                keep_words.append(w)
            
            if not keep_words:
                continue
            
            # Sort words by position (top to bottom, left to right)
            keep_words.sort(key=lambda w: (w["top"], w["x0"]))
            
            # Group words into logical text blocks
            text_blocks = []
            current_block = []
            
            for word in keep_words:
                # Check if this word should start a new block
                if current_block:
                    last_word = current_block[-1]
                    
                    # Check if we should continue the current block
                    vertical_gap = word["top"] - last_word["bottom"]
                    horizontal_overlap = _horiz_overlap(
                        (word["x0"], word["top"], word["x1"], word["bottom"]),
                        (last_word["x0"], last_word["top"], last_word["x1"], last_word["bottom"])
                    )
                    
                    # Continue block if: small vertical gap, horizontal overlap, similar formatting
                    font_similar = word.get("fontname", "") == last_word.get("fontname", "")
                    size_similar = abs(word.get("size", 0) - last_word.get("size", 0)) < 1
                    reasonable_gap = vertical_gap < 15  # Allow larger gaps for paragraph breaks
                    
                    if reasonable_gap and (horizontal_overlap > 0 or font_similar and size_similar):
                        current_block.append(word)
                    else:
                        # End current block and start new one
                        if current_block:
                            text_blocks.append(_create_text_block(current_block, p, len(text_blocks)))
                        current_block = [word]
                else:
                    current_block = [word]
            
            # Don't forget the last block
            if current_block:
                text_blocks.append(_create_text_block(current_block, p, len(text_blocks)))
            
            txt_by_page[p] = text_blocks
    
    return txt_by_page

def _create_text_block(words: List[Dict], page_num: int, block_index: int) -> ElementDict:
    """Create a text block element from a list of words."""
    if not words:
        return {}
    
    # Extract all text content
    text_content = " ".join(w["text"] for w in words)
    
    # Calculate bounding box
    x0 = min(w["x0"] for w in words)
    y0 = min(w["top"] for w in words)
    x1 = max(w["x1"] for w in words)
    y1 = max(w["bottom"] for w in words)
    
    # Determine if this is a heading based on font size and position
    font_sizes = [w.get("size", 0) for w in words]
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
    
    # Check if this appears to be a heading (larger font, possibly centered)
    is_heading = False
    if avg_font_size > 12:  # Larger than typical body text
        # Check if it's centered or positioned differently
        page_width = words[0].get("page_width", 612)  # Default page width
        block_center = (x0 + x1) / 2
        page_center = page_width / 2
        if abs(block_center - page_center) < 50:  # Roughly centered
            is_heading = True
    
    # Get dominant font properties
    fonts = [w.get("fontname", "") for w in words]
    dominant_font = max(set(fonts), key=fonts.count) if fonts else ""
    
    return {
        "id": f"text_p{page_num}_{block_index}",
        "kind": "text",
        "page_range": (page_num, page_num),
        "bboxes_per_page": [{"page": page_num, "bbox": (x0, y0, x1, y1)}],
        "metadata": {
            "font": dominant_font,
            "font_size": avg_font_size,
            "is_heading": is_heading,
            "word_count": len(words),
            "column_span": None
        },
        "text": text_content
    }


# -----------------------------
# Stitching across pages
# -----------------------------
def _stitch_text_across_pages(text_by_page: Dict[int, List[ElementDict]],
                              page_sizes: List[Tuple[float, float]],
                              top_tol: float = 24.0,
                              bottom_tol: float = 24.0,
                              edge_tol: float = 8.0) -> List[ElementDict]:
    """
    Merge text elements that likely continue across page breaks:
    - block ends near bottom of page p and next block starts near top of page p+1
    - left/right edges align within edge_tol
    """
    stitched: List[ElementDict] = []
    # Index by page
    max_page = max(text_by_page.keys(), default=-1)
    visited = set()

    for p in range(max_page + 1):
        for i, el in enumerate(text_by_page.get(p, [])):
            if (p, i) in visited:
                continue
            chain = [el]
            # attempt to follow onto next pages
            curr = el
            cp_end = curr["bboxes_per_page"][-1]["bbox"]
            while True:
                np = curr["page_range"][1] + 1
                if np not in text_by_page:
                    break
                H_next = page_sizes[np][1] if np < len(page_sizes) else None
                if H_next is None:
                    break
                candidates = text_by_page[np]
                # bottom near page end?
                if (page_sizes[curr["page_range"][1]][1] - cp_end[3]) > bottom_tol:
                    break
                # find top-near candidates
                top_cands = []
                for j, nxt in enumerate(candidates):
                    nb = nxt["bboxes_per_page"][0]["bbox"]
                    if nb[1] > top_tol:
                        continue
                    # edge alignment
                    if _nearly_equal(cp_end[0], nb[0], edge_tol) or _nearly_equal(cp_end[2], nb[2], edge_tol):
                        top_cands.append((j, nxt))
                if not top_cands:
                    break
                # take the first reasonably aligned candidate (greedy)
                j, nxt = top_cands[0]
                chain.append(nxt)
                visited.add((np, j))
                curr = nxt
                cp_end = curr["bboxes_per_page"][-1]["bbox"]

            # merge chain into one element
            if len(chain) == 1:
                stitched.append(chain[0])
            else:
                start_p = chain[0]["page_range"][0]
                end_p = chain[-1]["page_range"][1]
                bpps: List[PageBBox] = []
                for node in chain:
                    bpps.extend(node["bboxes_per_page"])
                merged: ElementDict = {
                    "id": f"text_p{start_p}-{end_p}_{len(stitched)}",
                    "kind": "text",
                    "page_range": (start_p, end_p),
                    "bboxes_per_page": bpps,
                    "metadata": {"stitched_pages": len(chain)},
                }
                stitched.append(merged)

    return stitched


def _stitch_tables_across_pages(tables_by_page: Dict[int, List[ElementDict]],
                                page_sizes: List[Tuple[float, float]],
                                edge_tol: float = 8.0) -> List[ElementDict]:
    """
    Merge tables that continue across pages by comparing column edges and vertical proximity to page boundaries.
    For Camelot detections, we rely on bbox alignment; header text/structure can be added when available.
    """
    stitched: List[ElementDict] = []
    visited = set()
    max_page = max(tables_by_page.keys(), default=-1)

    def col_signature(b: BBox, n_bins: int = 6) -> Tuple[int, int]:
        # crude proxy: quantize left/right to bins across page width to compare alignment
        x0, _, x1, _ = b
        return (round(x0 / 10), round(x1 / 10))

    for p in range(max_page + 1):
        arr = tables_by_page.get(p, [])
        for i, el in enumerate(arr):
            if (p, i) in visited:
                continue
            chain = [el]
            curr = el
            while True:
                np = curr["page_range"][1] + 1
                if np not in tables_by_page:
                    break
                # current ends near bottom?
                cb = curr["bboxes_per_page"][-1]["bbox"]
                page_h = page_sizes[curr["page_range"][1]][1] if curr["page_range"][1] < len(page_sizes) else None
                if page_h is None or (page_h - cb[3]) > 28.0:
                    break
                # find next page table whose left/right edges roughly align
                sig = col_signature(cb)
                found = None
                for j, cand in enumerate(tables_by_page[np]):
                    nb = cand["bboxes_per_page"][0]["bbox"]
                    if nb[1] > 36.0:  # should start near top
                        continue
                    if col_signature(nb) == sig or _nearly_equal(cb[0], nb[0], edge_tol) and _nearly_equal(cb[2], nb[2], edge_tol):
                        found = (np, j, cand); break
                if not found:
                    break
                npg, j, cand = found
                chain.append(cand)
                visited.add((npg, j))
                curr = cand

            if len(chain) == 1:
                stitched.append(chain[0])
            else:
                start_p = chain[0]["page_range"][0]
                end_p = chain[-1]["page_range"][1]
                bpps: List[PageBBox] = []
                for node in chain:
                    bpps.extend(node["bboxes_per_page"])
                merged: ElementDict = {
                    "id": f"table_p{start_p}-{end_p}_{len(stitched)}",
                    "kind": "table",
                    "page_range": (start_p, end_p),
                    "bboxes_per_page": bpps,
                    "metadata": {"stitched_pages": len(chain)},
                }
                stitched.append(merged)

    return stitched


# -----------------------------
# Public API
# -----------------------------
@dataclass
class DigitalElementClassifier:

    def classify(self, pdf_path: str, document_id: str = None) -> ElementType:
        """
        Return a dictionary describing all element types in the PDF.
        Contains the keys ``text``, ``tables`` and ``images`` with metadata
        describing the location of each element. The goal is merely to
        identify their presence; detailed processing is delegated elsewhere.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Optional document ID for organizing extracted data
        """
        # 1) Detect tables and images per page
        tables_by_page = _detect_table_regions(pdf_path)
        images_by_page = _detect_images(pdf_path, save_images=True, document_id=document_id)

        # Build occupied masks to avoid overlapping text classification
        occupied_masks: Dict[int, List[BBox]] = {}
        for p, arr in tables_by_page.items():
            for el in arr:
                occupied_masks.setdefault(p, []).append(el["bboxes_per_page"][0]["bbox"])
        for p, arr in images_by_page.items():
            for el in arr:
                occupied_masks.setdefault(p, []).append(el["bboxes_per_page"][0]["bbox"])

        # 2) Text blocks from remaining regions
        text_by_page = _extract_all_text_blocks(pdf_path, occupied_masks)

        # 3) Page sizes for stitching
        page_sizes: List[Tuple[float, float]] = []
        with pdfplumber.open(pdf_path) as pdf:
            page_sizes = [(pg.width, pg.height) for pg in pdf.pages]

        # 4) Stitch across pages
        all_tables = _stitch_tables_across_pages(tables_by_page, page_sizes)
        # Add any unstitched tables (pages with no stitching step)
        stitched_ids = {(el["id"], el["page_range"]) for el in all_tables}
        for p, arr in tables_by_page.items():
            for el in arr:
                # identify unique by bbox+page_range if ID differs
                key = (el["id"], el["page_range"])
                if key not in stitched_ids:
                    all_tables.append(el)

        all_text = _stitch_text_across_pages(text_by_page, page_sizes)
        stitched_text_keys = {(tuple(x["page_range"]), tuple((bp["page"], *bp["bbox"]) for bp in x["bboxes_per_page"])) for x in all_text}
        for p, arr in text_by_page.items():
            for el in arr:
                key = (tuple(el["page_range"]), tuple((bp["page"], *bp["bbox"]) for bp in el["bboxes_per_page"]))
                if key not in stitched_text_keys:
                    all_text.append(el)

        # Images rarely stitch; flatten to list
        all_images: List[ElementDict] = []
        for p, arr in images_by_page.items():
            all_images.extend(arr)

        # Stable sort by page range then x,y for readability
        def page_key(e: ElementDict) -> Tuple[int, float, float]:
            pg = e["page_range"][0]
            b0 = e["bboxes_per_page"][0]["bbox"]
            return (pg, b0[1], b0[0])

        all_text.sort(key=page_key)
        all_tables.sort(key=page_key)
        all_images.sort(key=page_key)

        return {
            "text": all_text,
            "tables": all_tables,
            "images": all_images,
        }
        """
        Save extracted elements to the document's organized folder structure.
        
        Args:
            document_id: Document identifier
            elements: Dictionary containing text, tables, images, etc.
        """
        import json
        from config import settings
        
        # Ensure document structure exists
        settings.data.create_document_structure(document_id)
        
        # Get paths
        elements_path = settings.data.get_document_elements_path(document_id)
        
        # Save text blocks
        if 'text' in elements:
            text_path = elements_path / "text_blocks.json"
            with open(text_path, 'w') as f:
                json.dump(elements['text'], f, indent=2)
        
        # Save tables
        if 'tables' in elements:
            tables_path = elements_path / "tables.json"
            with open(tables_path, 'w') as f:
                json.dump(elements['tables'], f, indent=2)
        
        # Save images metadata (images themselves are saved by _detect_images)
        if 'images' in elements:
            images_metadata = []
            for img in elements['images']:
                img_meta = {
                    "id": img.get("id"),
                    "bbox": img.get("bboxes_per_page", [{}])[0].get("bbox"),
                    "page": img.get("page_range", [0])[0],
                    "metadata": img.get("metadata", {})
                }
                images_metadata.append(img_meta)
            
            images_path = elements_path / "images_metadata.json"
            with open(images_path, 'w') as f:
                json.dump(images_metadata, f, indent=2)
        
        # Create basic metadata
        metadata = {
            "document_id": document_id,
            "extracted_at": __import__('datetime').datetime.now().isoformat(),
            "text_blocks": len(elements.get('text', [])),
            "tables": len(elements.get('tables', [])),
            "images": len(elements.get('images', [])),
            "status": "elements_extracted"
        }
        
        metadata_path = settings.data.get_document_index_path(document_id) / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
