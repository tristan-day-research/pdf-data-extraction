from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Literal, Optional

Policy = Literal["sandwich", "journal", "multi_column", "generic", "unknown"]

@dataclass
class RouterThresholds:
    # Sandwich: big background image + text on many pages
    sandwich_ratio: float = 0.30

    # Journal-ish: vector drawings dominate over raster images
    # (Your examples are ~10x and ~638x; even 2–3x is often enough)
    vector_to_raster_journal: float = 2.0

    # Pages with “Figure/Fig./Table” tokens near graphics
    figure_token_pages: float = 0.20

    # Narrow blocks imply multi-column/sidebars
    narrow_block_ratio_multi_col: float = 0.35  # flexible; journals often exceed 0.45+

@dataclass
class RouteDecision:
    policy: Policy
    # Feature flags the processor can use to toggle behavior
    flags: Dict[str, Any]
    # Confidence score in [0,1] for the chosen policy (heuristic)
    confidence: float

def route_pdf_format(
    stats: Dict[str, float],
    *,
    thresholds: RouterThresholds = RouterThresholds(),
    # Optional extra signals if you compute them elsewhere:
    tagged_pdf: Optional[bool] = None,   # True if /StructTreeRoot present
    has_rtl_or_vertical: Optional[bool] = None,  # True for RTL/vertical scripts
) -> RouteDecision:
    """
    Route a PDF to an extraction policy given quick-format stats.

    Expected keys in `stats`:
        - sandwich_ratio: float in [0,1]
        - vector_to_raster: float (draw ops / raster images)
        - narrow_block_ratio: float in [0,1]
        - figure_token_pages: float in [0,1]
    """
    sr = float(stats.get("sandwich_ratio", 0.0))
    v2r = float(stats.get("vector_to_raster", 0.0))
    nbr = float(stats.get("narrow_block_ratio", 0.0))
    ftp = float(stats.get("figure_token_pages", 0.0))

    flags: Dict[str, Any] = {
        "column_aware_text": True,          # always recommended
        "render_non_jpeg_images": True,     # fallback renderer for JPX/JBIG2/CCITT/SMask
        "dedupe_images": True,
        "skip_background_scans": False,
        "caption_linking": False,
        "footnote_suppression": False,
        "render_vector_figures": False,     # render path-dense regions as PNGs
        "prefer_tag_order": False,          # for tagged PDFs
        "rtl_or_vertical": bool(has_rtl_or_vertical) if has_rtl_or_vertical is not None else False,
    }

    # 1) Sandwich takes precedence
    if sr >= thresholds.sandwich_ratio:
        flags["skip_background_scans"] = True
        # Column-aware text still helpful; images via renderer to avoid .raw dumps
        return RouteDecision(
            policy="sandwich",
            flags=flags,
            confidence=min(1.0, 0.6 + 0.4 * (sr - thresholds.sandwich_ratio) / max(1e-6, 1 - thresholds.sandwich_ratio)),
        )

    # 2) Tagged PDFs (if you detect them) – not mutually exclusive with others, but we expose a hint
    if tagged_pdf:
        flags["prefer_tag_order"] = True

    # 3) Journal: vector-heavy and figure tokens common
    journal_score = 0.0
    if v2r >= thresholds.vector_to_raster_journal:
        journal_score += 0.6
        flags["render_vector_figures"] = True
    if ftp >= thresholds.figure_token_pages:
        journal_score += 0.4
        flags["caption_linking"] = True
        flags["footnote_suppression"] = True
    # Narrow blocks support multi-column/sidebars typical of journals
    multi_col_hint = nbr >= thresholds.narrow_block_ratio_multi_col
    if multi_col_hint:
        journal_score += 0.1  # weak bonus, capped below

    if journal_score >= 0.7:
        # Clamp confidence to [0,1]
        conf = min(1.0, journal_score)
        return RouteDecision(policy="journal", flags=flags, confidence=conf)

    # 4) Multi-column narra
