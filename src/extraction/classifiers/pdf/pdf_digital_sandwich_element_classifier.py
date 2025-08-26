from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, TypedDict
import os, io, hashlib, math, difflib

import pymupdf  # PyMuPDF
import camelot  # mandatory for tables

# ---------------- Types & geometry ----------------
BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1)

class ElementDict(TypedDict, total=False):
    id: str
    kind: str               # "text" | "table" | "image"
    page_range: Tuple[int, int]
    bboxes_per_page: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    text: str

ElementType = Dict[str, List[ElementDict]]

def _iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0.0, min(ay1, by1) - max(ay1, by1))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))  # fix typo
    inter = iw * ih
    if inter <= 0: return 0.0
    aarea = (ax1 - ax0) * (ay1 - ay0); barea = (bx1 - bx0) * (by1 - by0)
    return inter / (aarea + barea - inter)

def _expand(b: BBox, px: float) -> BBox:
    x0, y0, x1, y1 = b
    return (x0 - px, y0 - px, x1 + px, y1 + px)

def _nearly_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol

def _width(b: BBox) -> float: return max(0.0, b[2]-b[0])
def _height(b: BBox) -> float: return max(0.0, b[3]-b[1])

# ---------------- Column binning (per page) ----------------
def _bin_columns(blocks: List[Tuple[BBox, str]], page_width: float) -> List[int]:
    if not blocks:
        return []
    centers = [ (bb[0][0] + bb[0][2]) / 2.0 for bb in blocks ]
    bin_w = max(12.0, page_width / 20.0)
    bins: Dict[int, int] = {}
    for c in centers:
        bins[int(c // bin_w)] = bins.get(int(c // bin_w), 0) + 1
    if not bins:
        return [0] * len(blocks)
    top_bins = sorted(bins.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    centers2 = [ (k*bin_w + bin_w/2.0) for k,_ in top_bins ]
    out = []
    for c in centers:
        out.append(min(range(len(centers2)), key=lambda i: abs(c - centers2[i])))
    return out

# ---------------- Sandwich page test ----------------
def _is_sandwich_page(page: pymupdf.Page, image_bboxes: List[BBox]) -> bool:
    if not image_bboxes:
        return False
    pw, ph = page.rect.width, page.rect.height
    page_area = pw * ph
    max_area = max(( (x1-x0)*(y1-y0) for (x0,y0,x1,y1) in image_bboxes ), default=0.0)
    # treat as sandwich if a single image covers most of the page
    return (max_area / max(page_area, 1.0)) > 0.60 and bool(page.get_text("text").strip())

# ---------------- Images (sandwich-aware, render fallback) ----------------
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# def _detect_images_sandwich(
#     doc: pymupdf.Document,
#     pdf_path: str,
#     *,
#     save_images: bool = True,
#     output_dir: Optional[str] = None,
#     min_w: int = 16,
#     min_h: int = 16,
#     skip_bg_thresh: float = 0.60,
#     render_dpi: int = 300,
# ) -> Dict[int, List[ElementDict]]:
#     imgs_by_page: Dict[int, List[ElementDict]] = {}
#     if save_images and output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#     seen_hashes = set()

#     for pno in range(len(doc)):
#         page = doc[pno]
#         pw, ph = page.rect.width, page.rect.height
#         page_area = pw * ph

#         page_images = page.get_images(full=True)
#         placements: List[Tuple[int, pymupdf.Rect, str]] = []
#         for x in page_images:
#             xref, _, _, _, _, _, _, name, flt = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], (x[8] or "")
#             for rect in page.get_image_bbox(x[0]):
#                 placements.append((x[0], rect, flt.upper()))

#         is_sandwich = _is_sandwich_page(page, [tuple(r) for _, r, _ in placements])

#         for i, (xref, rect, flt) in enumerate(placements):
#             bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
#             w, h = int(rect.width), int(rect.height)
#             area = rect.width * rect.height

#             # skip tiny tiles
#             if w < min_w or h < min_h:
#                 continue
#             # skip background scan if sandwich
#             if is_sandwich and (area / page_area) > skip_bg_thresh:
#                 el = {
#                     "id": f"image_p{pno}_{i}",
#                     "kind": "image",
#                     "page_range": (pno, pno),
#                     "bboxes_per_page": [{"page": pno, "bbox": bbox}],
#                     "metadata": {"extractable": False, "reason": "background_scan"},
#                 }
#                 imgs_by_page.setdefault(pno, []).append(el)
#                 continue

#             saved_path = None
#             img_bytes: Optional[bytes] = None
#             meta: Dict[str, Any] = {"filters": flt}

#             # native extract if JPEG; otherwise render the on-page appearance
#             try:
#                 info = doc.extract_image(xref)
#                 if info and info.get("ext", "").lower() in ("jpg", "jpeg"):
#                     img_bytes = info["image"]
#                     meta.update({"ext": info.get("ext"), "width": info.get("width"), "height": info.get("height")})
#                 else:
#                     pix = page.get_pixmap(matrix=pymupdf.Matrix(render_dpi/72, render_dpi/72), clip=rect, annots=False)
#                     img_bytes = pix.tobytes("png")
#                     meta.update({"ext": "png", "rendered": True, "dpi": render_dpi})
#             except Exception as e:
#                 meta.update({"extractable": False, "error": str(e)})

#             if save_images and img_bytes:
#                 hsh = _sha256(img_bytes)
#                 if hsh in seen_hashes:
#                     meta["dedup"] = True
#                 else:
#                     seen_hashes.add(hsh)
#                     base = os.path.splitext(os.path.basename(pdf_path))[0]
#                     ext = ".jpg" if meta.get("ext") in ("jpg", "jpeg") else ".png"
#                     fname = f"{base}_p{pno}_img{i}{ext}"
#                     if output_dir:
#                         saved_path = os.path.join(output_dir, fname)
#                         with open(saved_path, "wb") as f:
#                             f.write(img_bytes)
#                     meta["sha256"] = hsh

#             el: ElementDict = {
#                 "id": f"image_p{pno}_{i}",
#                 "kind": "image",
#                 "page_range": (pno, pno),
#                 "bboxes_per_page": [{"page": pno, "bbox": bbox}],
#                 "metadata": {**meta, "path": saved_path} if saved_path or meta else {},
#             }
#             imgs_by_page.setdefault(pno, []).append(el)

#     return imgs_by_page


def _detect_images_sandwich(
    doc: pymupdf.Document,
    pdf_path: str,
    *,
    save_images: bool = True,
    output_dir: Optional[str] = None,
    min_w: int = 16,
    min_h: int = 16,
    skip_bg_thresh: float = 0.60,
    render_dpi: int = 300,
) -> Dict[int, List[ElementDict]]:
    imgs_by_page: Dict[int, List[ElementDict]] = {}
    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    seen_hashes = set()

    for pno in range(len(doc)):
        page = doc[pno]
        pw, ph = page.rect.width, page.rect.height
        page_area = pw * ph

        page_images = page.get_images(full=True)
        placements: List[Tuple[int, pymupdf.Rect, str]] = []
        
        # FIXED: Correct image bounding box extraction
        for img_info in page_images:
            xref, name, flt = img_info[0], img_info[7], (img_info[8] or "") if len(img_info) > 8 else ""
            
            # Use image NAME instead of xref for get_image_bbox()
            if name:  # Only process images that have names
                try:
                    rect = page.get_image_bbox(name)
                    placements.append((xref, rect, flt.upper()))
                except ValueError:
                    # Skip images that can't be processed due to bad names
                    continue
            # Alternative: For images without names, you could try other methods
            # else:
            #     # Fallback: try to estimate position or use different approach
            #     pass

        is_sandwich = _is_sandwich_page(page, [tuple(r) for _, r, _ in placements])

        for i, (xref, rect, flt) in enumerate(placements):
            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
            w, h = int(rect.width), int(rect.height)
            area = rect.width * rect.height

            # skip tiny tiles
            if w < min_w or h < min_h:
                continue
            # skip background scan if sandwich
            if is_sandwich and (area / page_area) > skip_bg_thresh:
                el = {
                    "id": f"image_p{pno}_{i}",
                    "kind": "image",
                    "page_range": (pno, pno),
                    "bboxes_per_page": [{"page": pno, "bbox": bbox}],
                    "metadata": {"extractable": False, "reason": "background_scan"},
                }
                imgs_by_page.setdefault(pno, []).append(el)
                continue

            saved_path = None
            img_bytes: Optional[bytes] = None
            meta: Dict[str, Any] = {"filters": flt}

            # native extract if JPEG; otherwise render the on-page appearance
            try:
                info = doc.extract_image(xref)
                if info and info.get("ext", "").lower() in ("jpg", "jpeg"):
                    img_bytes = info["image"]
                    meta.update({"ext": info.get("ext"), "width": info.get("width"), "height": info.get("height")})
                else:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(render_dpi/72, render_dpi/72), clip=rect, annots=False)
                    img_bytes = pix.tobytes("png")
                    meta.update({"ext": "png", "rendered": True, "dpi": render_dpi})
            except Exception as e:
                meta.update({"extractable": False, "error": str(e)})

            if save_images and img_bytes:
                hsh = _sha256(img_bytes)
                if hsh in seen_hashes:
                    meta["dedup"] = True
                else:
                    seen_hashes.add(hsh)
                    base = os.path.splitext(os.path.basename(pdf_path))[0]
                    ext = ".jpg" if meta.get("ext") in ("jpg", "jpeg") else ".png"
                    fname = f"{base}_p{pno}_img{i}{ext}"
                    if output_dir:
                        saved_path = os.path.join(output_dir, fname)
                        with open(saved_path, "wb") as f:
                            f.write(img_bytes)
                    meta["sha256"] = hsh

            el: ElementDict = {
                "id": f"image_p{pno}_{i}",
                "kind": "image",
                "page_range": (pno, pno),
                "bboxes_per_page": [{"page": pno, "bbox": bbox}],
                "metadata": {**meta, "path": saved_path} if saved_path or meta else {},
            }
            imgs_by_page.setdefault(pno, []).append(el)

    return imgs_by_page

# ---------------- Text blocks (column-aware) ----------------
def _extract_text_blocks_sandwich(doc: pymupdf.Document,
                                  occupied_masks: Dict[int, List[BBox]],
                                  top_margin_ratio=0.06,
                                  bottom_margin_ratio=0.06) -> Dict[int, List[ElementDict]]:
    out: Dict[int, List[ElementDict]] = {}
    for pno in range(len(doc)):
        page = doc[pno]
        pw, ph = page.rect.width, page.rect.height
        top_band = ph * top_margin_ratio
        bot_band = ph * (1 - bottom_margin_ratio)

        raw_blocks = page.get_text("blocks") or []
        text_blocks: List[Tuple[BBox, str]] = []
        for b in raw_blocks:
            if len(b) < 7 or b[6] != 0:
                continue
            bbox = (b[0], b[1], b[2], b[3])
            txt = (b[4] or "").strip()
            # exclude overlaps with tables/images
            occ = [_expand(m, 1.5) for m in occupied_masks.get(pno, [])]
            if any(_iou(bbox, m) > 0.05 for m in occ):
                continue
            # suppress thin header/footer furniture
            if (bbox[1] < top_band or bbox[3] > bot_band) and _height(bbox) < 14:
                continue
            if txt:
                text_blocks.append((bbox, txt))

        if not text_blocks:
            continue

        cols = _bin_columns(text_blocks, pw)
        zipped = list(zip(text_blocks, cols))
        zipped.sort(key=lambda z: (z[1], z[0][0][1], z[0][0][0]))

        elems: List[ElementDict] = []
        for idx, ((bbox, txt), col) in enumerate(zipped):
            elems.append({
                "id": f"text_p{pno}_{idx}",
                "kind": "text",
                "page_range": (pno, pno),
                "bboxes_per_page": [{"page": pno, "bbox": bbox}],
                "metadata": {"column_index": col},
                "text": txt,
            })
        out[pno] = elems
    return out

# ---------------- Text stitching across pages ----------------
def _stitch_text(text_by_page: Dict[int, List[ElementDict]],
                 page_sizes: List[Tuple[float, float]],
                 top_tol: float = 28.0,
                 bottom_tol: float = 28.0,
                 edge_tol: float = 8.0) -> List[ElementDict]:
    stitched: List[ElementDict] = []
    visited = set()
    max_page = max(text_by_page.keys(), default=-1)

    def lr(bb: BBox) -> Tuple[float, float]: return (bb[0], bb[2])

    for p in range(max_page + 1):
        for i, el in enumerate(text_by_page.get(p, [])):
            if (p, i) in visited: 
                continue
            chain = [el]
            curr = el
            curr_bb = curr["bboxes_per_page"][-1]["bbox"]
            curr_col = curr["metadata"].get("column_index", 0)
            while True:
                np = curr["page_range"][1] + 1
                if np not in text_by_page: break
                ph = page_sizes[curr["page_range"][1]][1]
                if (ph - curr_bb[3]) > bottom_tol: break
                matches = []
                for j, nxt in enumerate(text_by_page[np]):
                    nb = nxt["bboxes_per_page"][0]["bbox"]
                    if nb[1] > top_tol: continue
                    if nxt["metadata"].get("column_index", 0) != curr_col: continue
                    l0,r0 = lr(curr_bb); l1,r1 = lr(nb)
                    if _nearly_equal(l0,l1,edge_tol) or _nearly_equal(r0,r1,edge_tol):
                        matches.append((j, nxt))
                if not matches: break
                j, best = matches[0]
                visited.add((np, j))
                chain.append(best)
                curr = best
                curr_bb = curr["bboxes_per_page"][-1]["bbox"]

            if len(chain) == 1:
                stitched.append(chain[0])
            else:
                start_p = chain[0]["page_range"][0]
                end_p = chain[-1]["page_range"][1]
                bpps = []
                full_text = []
                for node in chain:
                    bpps.extend(node["bboxes_per_page"])
                    full_text.append(node.get("text",""))
                stitched.append({
                    "id": f"text_p{start_p}-{end_p}_{len(stitched)}",
                    "kind": "text",
                    "page_range": (start_p, end_p),
                    "bboxes_per_page": bpps,
                    "metadata": {"stitched_pages": len(chain), "column_index": curr_col},
                    "text": "\n".join(t for t in full_text if t)
                })
    return stitched

# ---------------- Mandatory tables via Camelot ----------------
def _header_signature(df) -> str:
    # Try to build a textual header signature from first non-empty row(s)
    if df is None or df.shape[0] == 0:
        return ""
    first = df.iloc[0].astype(str).tolist()
    # normalize whitespace/case
    norm = [ " ".join(x.strip().split()).lower() for x in first ]
    return "|".join(norm)

def _detect_tables_mandatory(pdf_path: str) -> Dict[int, List[ElementDict]]:
    """
    Mandatory table detection using Camelot (both flavors). Merges and dedupes.
    Adds lightweight metadata to support stitching (col edges, header sig).
    """
    by_page: Dict[int, List[ElementDict]] = {}
    sets = []
    # Lattice first (high precision), then Stream (recall)
    try:
        sets.append(camelot.read_pdf(pdf_path, pages="all", flavor="lattice", suppress_stdout=True))
    except Exception:
        pass
    try:
        sets.append(camelot.read_pdf(pdf_path, pages="all", flavor="stream", suppress_stdout=True))
    except Exception:
        pass

    for tblset in sets:
        for t in getattr(tblset, "tables", []):
            p = (t.page or 1) - 1
            x0, y0, x1, y1 = t._bbox
            # Camelot sometimes flips y; but bbox is in PDF coords (x0,y0,x1,y1) with origin bottom-left.
            # pymupdf uses top-left origin in "rect" textual coords; we keep Camelot bbox as-is for consistency here.
            # We'll compare left/right edges only for stitching.
            sig = _header_signature(getattr(t, "df", None))
            cols = getattr(t, "cols", None)
            meta = {
                "source": "camelot",
                "flavor": getattr(t, "flavor", None),
                "n_rows": getattr(t, "shape", (None, None))[0],
                "n_cols": getattr(t, "shape", (None, None))[1],
                "header_signature": sig,
                "col_edges": list(cols) if cols is not None else None,
            }
            el: ElementDict = {
                "id": f"table_p{p}_{len(by_page.get(p, []))}",
                "kind": "table",
                "page_range": (p, p),
                "bboxes_per_page": [{"page": p, "bbox": (x0, y0, x1, y1)}],
                "metadata": meta,
            }
            by_page.setdefault(p, []).append(el)

    # De-duplicate overlaps per page (lattice vs stream)
    for p, arr in list(by_page.items()):
        keep: List[ElementDict] = []
        for el in arr:
            bb = el["bboxes_per_page"][0]["bbox"]
            if any(_iou(bb, k["bboxes_per_page"][0]["bbox"]) > 0.60 for k in keep):
                # prefer lattice over stream if flavors conflict
                if el["metadata"].get("flavor") == "lattice":
                    # replace earlier stream if overlapping
                    for idx, k in enumerate(keep):
                        if _iou(bb, k["bboxes_per_page"][0]["bbox"]) > 0.60 and k["metadata"].get("flavor") != "lattice":
                            keep[idx] = el
                            break
                    continue
                else:
                    continue
            keep.append(el)
        by_page[p] = keep

    return by_page

def _stitch_tables_across_pages(tables_by_page: Dict[int, List[ElementDict]],
                                page_sizes: List[Tuple[float, float]],
                                edge_tol: float = 8.0,
                                header_sim_thresh: float = 0.55) -> List[ElementDict]:
    """
    Merge tables across consecutive pages using:
      - End near bottom / start near top,
      - Left/right edge alignment OR similar column edges,
      - Header similarity (first row text) if present.
    """
    stitched: List[ElementDict] = []
    visited = set()
    max_page = max(tables_by_page.keys(), default=-1)

    def col_sig(edges: Optional[List[float]]) -> Optional[Tuple[int, ...]]:
        if not edges: 
            return None
        # quantize to 10pt bins for fuzzy matching
        return tuple(int(round(e/10.0)) for e in edges)

    def header_sim(a: str, b: str) -> float:
        if not a and not b: return 1.0
        if not a or not b: return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    for p in range(max_page + 1):
        curr_list = tables_by_page.get(p, [])
        for i, el in enumerate(curr_list):
            if (p, i) in visited:
                continue
            chain = [el]
            curr = el
            while True:
                np = curr["page_range"][1] + 1
                if np not in tables_by_page:
                    break
                cb = curr["bboxes_per_page"][-1]["bbox"]
                ph = page_sizes[curr["page_range"][1]][1]
                # must end near bottom
                if (ph - cb[3]) > 28.0:
                    break
                # find top-near candidate on next page
                found = None
                for j, nxt in enumerate(tables_by_page[np]):
                    nb = nxt["bboxes_per_page"][0]["bbox"]
                    if nb[1] > 36.0:  # must start near top
                        continue
                    # alignment tests
                    align = (_nearly_equal(cb[0], nb[0], edge_tol) and _nearly_equal(cb[2], nb[2], edge_tol))
                    # or column edges similarity if we have them
                    cs0 = col_sig(curr["metadata"].get("col_edges"))
                    cs1 = col_sig(nxt["metadata"].get("col_edges"))
                    cols_match = (cs0 is not None and cs1 is not None and cs0 == cs1)
                    # header similarity (helps when widths drift)
                    hs = header_sim(curr["metadata"].get("header_signature",""), nxt["metadata"].get("header_signature",""))
                    if align or cols_match or hs >= header_sim_thresh:
                        found = (np, j, nxt)
                        break
                if not found:
                    break
                npg, j, cand = found
                visited.add((npg, j))
                chain.append(cand)
                curr = cand

            if len(chain) == 1:
                stitched.append(chain[0])
            else:
                start_p = chain[0]["page_range"][0]
                end_p = chain[-1]["page_range"][1]
                bpps: List[Dict[str, Any]] = []
                for node in chain:
                    bpps.extend(node["bboxes_per_page"])
                stitched.append({
                    "id": f"table_p{start_p}-{end_p}_{len(stitched)}",
                    "kind": "table",
                    "page_range": (start_p, end_p),
                    "bboxes_per_page": bpps,
                    "metadata": {
                        "stitched_pages": len(chain),
                        "source": "camelot",
                        "flavor": "mixed",
                    }
                })
    return stitched

# ---------------- Sandwich classifier (tables mandatory) ----------------
@dataclass
class PDFSandwichElementClassifier:
    save_images: bool = True
    images_output_dir: Optional[str] = None  # e.g., "./out_images"

    def classify(self, pdf_path: str) -> ElementType:
        """
        Assumes the document has already been routed as 'sandwich'.
        Returns {'text': [...], 'tables': [...], 'images': [...]} with stitched elements.
        """
        # --- open doc
        doc = pymupdf.open(pdf_path)

        # --- tables (mandatory)
        tables_by_page = _detect_tables_mandatory(pdf_path)

        # --- images (sandwich-aware)
        images_by_page = _detect_images_sandwich(
            doc,
            pdf_path,
            save_images=self.save_images,
            output_dir=self.images_output_dir,
        )

        # --- occupied masks for text
        occupied: Dict[int, List[BBox]] = {}
        for p, arr in tables_by_page.items():
            for el in arr:
                occupied.setdefault(p, []).append(el["bboxes_per_page"][0]["bbox"])
        for p, arr in images_by_page.items():
            for el in arr:
                occupied.setdefault(p, []).append(el["bboxes_per_page"][0]["bbox"])

        # --- text (column-aware, sandwich policy)
        text_by_page = _extract_text_blocks_sandwich(doc, occupied)

        # --- page sizes
        page_sizes = [(doc[p].rect.width, doc[p].rect.height) for p in range(len(doc))]

        # --- stitch text and tables across pages
        all_text = _stitch_text(text_by_page, page_sizes)
        # ensure any singletons are included
        stitched_text_keys = {
            (tuple(e["page_range"]), tuple((bp["page"], *bp["bbox"]) for bp in e["bboxes_per_page"]))
            for e in all_text
        }
        for p, arr in text_by_page.items():
            for el in arr:
                key = (tuple(el["page_range"]), tuple((bp["page"], *bp["bbox"]) for bp in el["bboxes_per_page"]))
                if key not in stitched_text_keys:
                    all_text.append(el)

        all_tables = _stitch_tables_across_pages(tables_by_page, page_sizes)
        stitched_table_ids = {(e["id"], e["page_range"]) for e in all_tables}
        for p, arr in tables_by_page.items():
            for el in arr:
                key = (el["id"], el["page_range"])
                if key not in stitched_table_ids:
                    all_tables.append(el)

        # --- flatten images
        all_images: List[ElementDict] = []
        for p, arr in images_by_page.items():
            all_images.extend(arr)

        # --- sort
        def sort_key(e: ElementDict) -> Tuple[int, float, float]:
            pg = e["page_range"][0]
            bb = e["bboxes_per_page"][0]["bbox"]
            return (pg, bb[1], bb[0])

        all_text.sort(key=sort_key)
        all_tables.sort(key=sort_key)
        all_images.sort(key=sort_key)

        doc.close()
        return {
            "text": all_text,
            "tables": all_tables,
            "images": all_images,
        }
